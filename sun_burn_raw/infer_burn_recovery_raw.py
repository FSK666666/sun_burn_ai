#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from burn_recovery_net import BurnRecoveryNet
from raw_io import denormalize_u16, parse_shape, read_u16_bin, save_preview_png, write_u16_bin


def apply_saved_config(cfg: dict, saved_cfg: dict) -> dict:
    merged = dict(cfg)
    for key, value in saved_cfg.items():
        if key in {"raw_shape", "image_size"} and value is not None:
            value = tuple(value)
        merged[key] = value
    return merged


def load_frames(paths: list[Path], shape: tuple[int, int] | None, endian: str, raw_max: float):
    frames_u16 = [read_u16_bin(path, shape=shape, endian=endian) for path in paths]
    base_shape = frames_u16[0].shape
    for path, frame in zip(paths, frames_u16):
        if frame.shape != base_shape:
            raise ValueError(f"Shape mismatch: {path} is {frame.shape}, expected {base_shape}")
    frames = np.stack([frame.astype(np.float32) / raw_max for frame in frames_u16], axis=0)
    return np.clip(frames, 0.0, 1.0).astype(np.float32), frames_u16


def resize_stack(frames: np.ndarray, image_size: tuple[int, int] | None):
    if image_size is None:
        return frames
    h, w = image_size
    return np.stack(
        [cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in frames],
        axis=0,
    ).astype(np.float32)


def resize_to_original(image: np.ndarray, shape: tuple[int, int]):
    if tuple(image.shape[-2:]) == tuple(shape):
        return image
    h, w = shape
    return cv2.resize(np.squeeze(image), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer burn recovery on five uint16 raw bin frames.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", nargs=5, type=Path, required=True, help="Five .bin/.raw frames in temporal order.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-shape", type=str, default=None, help="HxW. If omitted, infer from file size.")
    parser.add_argument("--image-size", type=str, default=None, help="Override inference resize HxW.")
    parser.add_argument("--endian", choices=["little", "big"], default=None)
    parser.add_argument("--raw-max", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-gate", action="store_true", help="Use direct restored = last - correction.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_cfg = checkpoint.get("config", {})

    raw_shape = parse_shape(args.raw_shape) if args.raw_shape else saved_cfg.get("raw_shape")
    if raw_shape is not None:
        raw_shape = tuple(raw_shape)
    image_size = parse_shape(args.image_size) if args.image_size else saved_cfg.get("image_size")
    if image_size is not None:
        image_size = tuple(image_size)
    endian = args.endian or saved_cfg.get("endian", "little")
    raw_max = float(args.raw_max if args.raw_max is not None else saved_cfg.get("raw_max", 65535.0))
    base_channels = int(saved_cfg.get("base_channels", 32))

    frames, frames_u16 = load_frames(args.frames, raw_shape, endian, raw_max)
    original_shape = frames.shape[-2:]
    frames_for_net = resize_stack(frames, image_size)
    x = torch.from_numpy(frames_for_net[None, :, :, :]).float().to(device)

    model = BurnRecoveryNet(in_frames=5, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        prob_logits, correction = model(x)
        prob = torch.sigmoid(prob_logits)
        if args.no_gate:
            restored = torch.clamp(x[:, -1:, :, :] - correction, 0.0, 1.0)
        else:
            restored = torch.clamp(x[:, -1:, :, :] - (prob > args.threshold).float() * correction, 0.0, 1.0)

    prob_np = resize_to_original(prob[0, 0].cpu().numpy(), original_shape)
    correction_np = resize_to_original(correction[0, 0].cpu().numpy(), original_shape)
    restored_np = resize_to_original(restored[0, 0].cpu().numpy(), original_shape)
    correction_u16 = denormalize_u16(correction_np, raw_max)
    restored_u16 = denormalize_u16(restored_np, raw_max)
    mask_u16 = np.where(prob_np > args.threshold, 65535, 0).astype(np.uint16)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_u16_bin(out / "restored_last_u16.bin", restored_u16, endian=endian)
    write_u16_bin(out / "pred_correction_u16.bin", correction_u16, endian=endian)
    write_u16_bin(out / "pred_mask_u16.bin", mask_u16, endian=endian)
    np.save(out / "pred_probability.npy", prob_np.astype(np.float32))
    np.save(out / "pred_correction_u16.npy", correction_u16)
    save_preview_png(out / "input_last_preview.png", frames_u16[-1], raw_max=raw_max)
    save_preview_png(out / "restored_last_preview.png", restored_u16, raw_max=raw_max)
    cv2.imwrite(str(out / "pred_probability_preview.png"), np.clip(prob_np * 255.0, 0, 255).astype(np.uint8))
    save_preview_png(out / "pred_correction_preview.png", correction_u16, raw_max=raw_max)

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"input shape: {original_shape}")
    print(f"network image_size: {image_size}")
    print(f"output dir: {out}")
    print(f"prob max: {float(prob_np.max()):.6f}")
    print(f"correction max raw: {int(correction_u16.max())}")


if __name__ == "__main__":
    main()
