#!/usr/bin/env python3
"""Minimal silent-video -> audio inference using LipVoicer."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf

PIPE_ROOT = Path(__file__).resolve().parents[1]
LV_ROOT = PIPE_ROOT / "third_party" / "LipVoicer"
sys.path.insert(0, str(LV_ROOT))

from PIL import Image
import numpy as np
import torchvision.transforms as T

import ASR.asr_models as asr_models
from dataloaders.lipreading_utils import Compose, Normalize, CenterCrop
from dataloaders.stft import denormalise_mel
from dataloaders.video_reader import VideoReader
from hifi_gan import utils as voc_utils
from hifi_gan.env import AttrDict
from hifi_gan.generator import Generator as Vocoder
from inference_real_video import sampling
from models.audiovisual_model import AudioVisualModel
from models.model_builder import ModelBuilder
from mouthroi_processing import crop_and_infer
from utils import calc_diffusion_hyperparams


def load_face(video_path: str) -> torch.Tensor:
    vr = VideoReader(video_path, 1)
    start_pts, _, n = vr._compute_video_stats()
    clip = vr.read_video_only(start_pts, 1) if n <= 0 else vr.read_video_only(0, 1)
    img = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert("RGB")
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


@torch.no_grad()
def infer(video_path: str, out_wav: str, ckpt: str | None = None,
          w_video: float = 2.0, w_asr: float = 1.5, asr_start: int = 270) -> str:
    video_path = str(Path(video_path).resolve())
    out_wav = str(Path(out_wav).resolve())
    os.chdir(LV_ROOT)

    cfg = OmegaConf.load(LV_ROOT / "configs" / "config.yaml")
    ckpt_path = ckpt or str(LV_ROOT / cfg.generate.ckpt_path)

    torch.cuda.set_device(0)

    dh = calc_diffusion_hyperparams(**cfg.diffusion, fast=True)

    b = ModelBuilder()
    net = AudioVisualModel((
        b.build_lipreadingnet(),
        b.build_facial(fc_out=128, with_fc=True),
        b.build_diffwave_model(cfg.melgen),
    )).cuda().eval()
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"])

    asr_net, tok, _ = asr_models.get_models("LRS3")
    dec = lambda *_a, **_k: [""]  # ctcdecode unavailable; only used for debug print

    out_dir = Path(out_wav).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mouthroi, text = crop_and_infer.main(video_path, str(out_dir))

    roi_tf = Compose([Normalize(0.0, 255.0), CenterCrop((88, 88)), Normalize(0.421, 0.165)])
    mouthroi = roi_tf(mouthroi).unsqueeze(0).unsqueeze(0).cuda()
    face = load_face(video_path).cuda()

    mel = sampling(net, dh, w_video,
                   condition=(mouthroi, face),
                   asr_guidance_net=asr_net, w_asr=w_asr, asr_start=asr_start,
                   guidance_text=text, tokenizer=tok, decoder=dec)
    mel = denormalise_mel(mel)

    with open(LV_ROOT / "hifi_gan" / "config.json") as f:
        h = AttrDict(json.load(f))
    voc = Vocoder(h).cuda()
    sd = voc_utils.load_checkpoint(str(LV_ROOT / "hifi_gan" / "g_02400000"), "cuda")
    voc.load_state_dict(sd["generator"])
    voc.eval()
    voc.remove_weight_norm()

    audio = voc(mel).squeeze()
    audio = (audio / 1.1 / audio.abs().max()).cpu().numpy()
    sf.write(out_wav, audio, 16000)
    return out_wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="silent input video (.mp4)")
    ap.add_argument("-o", "--out", default="out.wav", help="output wav path")
    ap.add_argument("--ckpt", default=None, help="MelGen checkpoint override")
    args = ap.parse_args()
    path = infer(os.path.abspath(args.video), os.path.abspath(args.out), args.ckpt)
    print(path)


if __name__ == "__main__":
    main()
