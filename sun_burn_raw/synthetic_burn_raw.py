#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from raw_io import RAW_SHAPES, infer_raw_shape, parse_shape, save_preview_png, write_u16_bin


IMAGE_STEMS = ("000", "001", "002", "003", "004")
RAW_SUFFIXES = {".bin", ".raw"}


@dataclass
class BurnRawConfig:
    raw_max: int = 65535
    min_peak: int = 1800
    max_peak: int = 9000
    active_threshold: int = 64
    size_multiplier: float = 1.0
    point_edge_softness: float = 0.18
    stripe_edge_softness: float = 0.34
    stripe_probability: float = 0.62
    horizontal_bias: bool = True
    max_stripe_angle_deg: float = 12.0
    disc_overlap_ratio: float = 0.78
    seed: int = 2026


def _draw_soft_disc(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    peak: float,
    edge_softness: float,
):
    h, w = canvas.shape
    pad = max(2, int(radius * (1.0 + edge_softness) + 3))
    x0 = max(0, int(cx) - pad)
    x1 = min(w, int(cx) + pad + 1)
    y0 = max(0, int(cy) - pad)
    y1 = min(h, int(cy) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = radius * max(0.20, 1.0 - edge_softness)
    outer = radius * (1.0 + edge_softness)
    alpha = np.ones_like(dist, dtype=np.float32)
    edge = dist > inner
    alpha[edge] = 1.0 - (dist[edge] - inner) / max(outer - inner, 1e-6)
    alpha = np.clip(alpha, 0.0, 1.0)
    value = peak * alpha * alpha * (3.0 - 2.0 * alpha)
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], value)


def _sample_angle(rng: np.random.Generator, cfg: BurnRawConfig) -> float:
    if cfg.horizontal_bias:
        return np.deg2rad(rng.uniform(-cfg.max_stripe_angle_deg, cfg.max_stripe_angle_deg))
    return rng.uniform(0.0, np.pi)


def generate_burn_trace_u16(
    h: int,
    w: int,
    cfg: BurnRawConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    burn = np.zeros((h, w), dtype=np.float32)
    peak = float(rng.uniform(cfg.min_peak, cfg.max_peak))

    if rng.random() < cfg.stripe_probability:
        angle = _sample_angle(rng, cfg)
        length = rng.uniform(60.0, 210.0) * cfg.size_multiplier * (w / 640.0)
        radius = rng.uniform(7.0, 15.0) * cfg.size_multiplier * (w / 640.0)
        spacing = max(1.0, radius * cfg.disc_overlap_ratio)
        cx = rng.uniform(0.18 * w, 0.82 * w)
        cy = rng.uniform(0.15 * h, 0.85 * h)
        dx = np.cos(angle)
        dy = np.sin(angle)
        count = max(2, int(length / spacing))
        offsets = np.linspace(-0.5 * length, 0.5 * length, count)
        for offset in offsets:
            local_peak = peak * rng.uniform(0.82, 1.12)
            local_radius = radius * rng.uniform(0.90, 1.12)
            _draw_soft_disc(
                burn,
                cx + dx * offset,
                cy + dy * offset,
                local_radius,
                local_peak,
                cfg.stripe_edge_softness,
            )
    else:
        n_points = int(rng.integers(1, 4))
        for _ in range(n_points):
            radius = rng.uniform(8.0, 19.0) * cfg.size_multiplier * (w / 640.0)
            cx = rng.uniform(radius, max(radius + 1, w - radius))
            cy = rng.uniform(radius, max(radius + 1, h - radius))
            _draw_soft_disc(
                burn,
                cx,
                cy,
                radius,
                peak * rng.uniform(0.75, 1.10),
                cfg.point_edge_softness,
            )

    burn = np.where(burn >= cfg.active_threshold, burn, 0.0)
    return np.clip(np.round(burn), 0, cfg.raw_max).astype(np.uint16)


def list_sample_dirs(root: Path) -> list[Path]:
    return sorted([p for p in Path(root).iterdir() if p.is_dir()], key=lambda p: p.name)


def sample_has_raw_frames(sample_dir: Path) -> bool:
    names = {p.stem for p in sample_dir.iterdir() if p.suffix.lower() in RAW_SUFFIXES}
    return all(stem in names for stem in IMAGE_STEMS)


def write_burn_artifacts(
    sample_dir: Path,
    cfg: BurnRawConfig,
    rng: np.random.Generator,
    shape: tuple[int, int] | None = None,
    endian: str = "little",
) -> list[Path]:
    first_raw = next(
        p for p in sample_dir.iterdir()
        if p.stem == IMAGE_STEMS[0] and p.suffix.lower() in RAW_SUFFIXES
    )
    if shape is None:
        shape = infer_raw_shape(first_raw, RAW_SHAPES)
    h, w = shape
    burn = generate_burn_trace_u16(h, w, cfg, rng)
    mask = np.where(burn > 0, 65535, 0).astype(np.uint16)

    burn_npy = sample_dir / "synthetic_burn_trace_u16.npy"
    burn_bin = sample_dir / "synthetic_burn_trace_u16.bin"
    mask_bin = sample_dir / "synthetic_burn_mask_u16.bin"
    preview = sample_dir / "synthetic_burn_trace_preview.png"
    metadata = sample_dir / "synthetic_burn_metadata_raw.txt"

    np.save(burn_npy, burn)
    write_u16_bin(burn_bin, burn, endian=endian)
    write_u16_bin(mask_bin, mask, endian=endian)
    save_preview_png(preview, burn, raw_max=cfg.raw_max)
    metadata.write_text(
        "\n".join(
            [
                f"shape={h}x{w}",
                f"raw_max={cfg.raw_max}",
                f"min_peak={cfg.min_peak}",
                f"max_peak={cfg.max_peak}",
                f"active_threshold={cfg.active_threshold}",
                f"size_multiplier={cfg.size_multiplier}",
                f"point_edge_softness={cfg.point_edge_softness}",
                f"stripe_edge_softness={cfg.stripe_edge_softness}",
                f"stripe_probability={cfg.stripe_probability}",
                f"horizontal_bias={cfg.horizontal_bias}",
                f"max_stripe_angle_deg={cfg.max_stripe_angle_deg}",
            ]
        ),
        encoding="utf-8",
    )
    return [burn_npy, burn_bin, mask_bin, preview, metadata]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic burn labels for uint16 raw bin samples.")
    parser.add_argument("--root", type=Path, required=True, help="Root containing sample_xxxxxx folders.")
    parser.add_argument("--shape", type=str, default=None, help="HxW. If omitted, infer from file size.")
    parser.add_argument("--endian", choices=["little", "big"], default="little")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-peak", type=int, default=1800)
    parser.add_argument("--max-peak", type=int, default=9000)
    parser.add_argument("--active-threshold", type=int, default=64)
    parser.add_argument("--size-multiplier", type=float, default=1.0)
    parser.add_argument("--stripe-probability", type=float, default=0.62)
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    shape = parse_shape(args.shape)
    cfg = BurnRawConfig(
        min_peak=args.min_peak,
        max_peak=args.max_peak,
        active_threshold=args.active_threshold,
        size_multiplier=args.size_multiplier,
        stripe_probability=args.stripe_probability,
        seed=args.seed,
    )
    sample_dirs = [p for p in list_sample_dirs(args.root) if sample_has_raw_frames(p)]
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]
    if not sample_dirs:
        raise RuntimeError(f"No raw samples found under {args.root}")

    rng = np.random.default_rng(args.seed)
    for index, sample_dir in enumerate(sample_dirs, start=1):
        write_burn_artifacts(sample_dir, cfg, rng, shape=shape, endian=args.endian)
        if index == 1 or index == len(sample_dirs) or index % args.print_every == 0:
            print(f"[{index}/{len(sample_dirs)}] wrote burn artifacts: {sample_dir.name}", flush=True)


if __name__ == "__main__":
    main()
