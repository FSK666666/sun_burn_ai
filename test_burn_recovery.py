#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from burn_recovery_net import BurnRecoveryNet
from train_burn_recovery import BurnRecoveryDataset, TrainConfig, run_one_epoch, set_seed


def apply_saved_config(cfg: TrainConfig, saved_cfg: dict):
    for key, value in saved_cfg.items():
        if not hasattr(cfg, key):
            continue
        if key.endswith("_root") or key in {"output_dir", "resume_checkpoint"}:
            value = Path(value) if value is not None else None
        elif key == "image_size" and value is not None:
            value = tuple(value)
        setattr(cfg, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BurnRecoveryNet checkpoints.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt"),
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path(r"C:\Users\17874\Documents\python\datasets\burn_recovery\val"),
            Path(r"C:\Users\17874\Documents\python\datasets\burn_recovery\test"),
            Path(r"C:\Users\17874\Documents\python\datasets\burn_recovery\test_flir"),
        ],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--generate-missing-burn",
        action="store_true",
        help="Generate missing burn traces on the fly. Off by default for reproducible tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    apply_saved_config(cfg, checkpoint.get("config", {}))

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise ValueError("Set both --image-height and --image-width.")
        cfg.image_size = (args.image_height, args.image_width)

    model = BurnRecoveryNet(in_frames=5, base_channels=cfg.base_channels).to(device)
    model.load_state_dict(checkpoint["model"])

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"base_channels: {cfg.base_channels}")
    print(f"image_size: {cfg.image_size}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"mask_threshold: {cfg.mask_threshold:.6f}")

    for root in args.roots:
        dataset = BurnRecoveryDataset(
            root=root,
            image_size=cfg.image_size,
            temporal_scale_range=(1.0, 1.0),
            generate_missing_burn=args.generate_missing_burn,
            max_samples=args.max_samples,
            no_burn_probability=0.0,
            mask_threshold=cfg.mask_threshold,
            saturation_percentile=cfg.saturation_percentile,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        metrics = run_one_epoch(model, loader, None, device, cfg)
        print(f"\n[{root.name}] samples={len(dataset)}")
        for key in [
            "loss",
            "dice",
            "iou",
            "precision",
            "recall",
            "active_mae",
            "active_rmse",
            "bg_mae",
            "bg_changed",
            "bg_changed_2",
            "global_mae",
            "max_error",
            "bg_max",
        ]:
            print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
