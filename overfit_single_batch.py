#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from burn_recovery_net import BurnRecoveryNet
from train_burn_recovery import (
    BurnRecoveryDataset,
    TrainConfig,
    run_one_epoch,
    save_checkpoint,
    save_visualizations,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed single-batch overfit test.")
    parser.add_argument("--root", type=Path, default=TrainConfig.train_root)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\17874\Documents\python\checkpoints\burn_recovery_overfit"),
    )
    return parser.parse_args()


def print_label_stats(dataset: BurnRecoveryDataset):
    print("fixed batch label check:")
    for index in range(len(dataset)):
        item = dataset[index]
        mask = item["p"].numpy()[0]
        correction = item["c"].numpy()[0]
        active = mask > 0

        if np.any(correction[~active] != 0):
            raise RuntimeError(f"{item['sample']} has nonzero correction outside mask")
        if not active.any():
            raise RuntimeError(f"{item['sample']} has empty mask; use burn samples for overfit")

        print(
            f"  {item['sample']}: "
            f"mask_ratio={mask.mean():.6f} "
            f"c_min={correction.min():.6f} "
            f"c_max={correction.max():.6f} "
            f"c_mean={correction.mean():.6f} "
            f"active_c_mean={correction[active].mean():.6f}"
        )


def main():
    args = parse_args()
    cfg = TrainConfig()
    cfg.train_root = args.root
    cfg.val_root = args.root
    cfg.output_dir = args.output_dir
    cfg.max_train_samples = args.samples
    cfg.max_val_samples = args.samples
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.weight_decay = 0.0
    cfg.use_scheduler = False
    cfg.train_no_burn_probability = 0.0
    cfg.val_no_burn_probability = 0.0
    cfg.generate_missing_burn = False
    cfg.val_generate_missing_burn = False
    cfg.temporal_scale_range = (1.0, 1.0)
    cfg.num_workers = 0
    cfg.progress_print_every = 1
    cfg.base_channels = args.base_channels
    cfg.c_bg_loss_weight = 1.0
    cfg.c_global_loss_weight = 0.5
    cfg.image_size = (args.image_height, args.image_width)
    cfg.save_every_epoch = False
    cfg.vis_every_epoch = True
    cfg.vis_num_samples = min(args.samples, 6)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixed_dataset = BurnRecoveryDataset(
        root=cfg.train_root,
        image_size=cfg.image_size,
        temporal_scale_range=cfg.temporal_scale_range,
        generate_missing_burn=False,
        max_samples=cfg.max_train_samples,
        no_burn_probability=0.0,
        mask_threshold=cfg.mask_threshold,
        aggregation=cfg.aggregation,
        validate_samples=True,
    )
    print_label_stats(fixed_dataset)

    loader = DataLoader(
        fixed_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = BurnRecoveryNet(in_frames=5, base_channels=cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(cfg.use_amp and device.type == "cuda"),
    )

    print(f"device: {device}")
    print(f"samples: {len(fixed_dataset)}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"epochs: {cfg.epochs}")
    print(f"image_size: {cfg.image_size}")
    print(f"loss weights: prob=1 active=1 bg={cfg.c_bg_loss_weight} global={cfg.c_global_loss_weight} grad={cfg.gradient_loss_weight}")

    best_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_one_epoch(model, loader, optimizer, device, cfg, scaler)
        val_metrics = run_one_epoch(model, loader, None, device, cfg)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]

        print(
            f"epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_dice={train_metrics['dice']:.6f} "
            f"train_active_mae={train_metrics['active_mae']:.6f} "
            f"train_bg_mae={train_metrics['bg_mae']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_dice={val_metrics['dice']:.6f} "
            f"val_active_mae={val_metrics['active_mae']:.6f} "
            f"val_bg_mae={val_metrics['bg_mae']:.6f} "
            f"prob_max={val_metrics['prob_max']:.6f} "
            f"prob_active={val_metrics['prob_active_mean']:.6f} "
            f"prob_bg={val_metrics['prob_bg_mean']:.6f}"
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        cfg.output_dir / "overfit_last.pt",
        model,
        optimizer,
        scaler,
        None,
        cfg.epochs,
        best_loss,
        cfg,
    )
    save_visualizations(model, loader, device, cfg, cfg.epochs)
    print(f"saved: {cfg.output_dir / 'overfit_last.pt'}")


if __name__ == "__main__":
    main()
