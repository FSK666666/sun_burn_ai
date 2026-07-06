#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from train_burn_recovery import TrainConfig, train_with_config


def make_small_scale_config() -> TrainConfig:
    cfg = TrainConfig()

    cfg.output_dir = Path(r"C:\Users\17874\Documents\python\checkpoints\burn_recovery_small")

    cfg.max_train_samples = 200
    cfg.max_val_samples = 100
    cfg.batch_size = 6
    cfg.epochs = 10

    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.use_scheduler = True
    cfg.num_workers = 0

    cfg.generate_missing_burn = True
    cfg.val_generate_missing_burn = False
    cfg.train_no_burn_probability = 0.15
    cfg.val_no_burn_probability = 0.15

    cfg.c_bg_loss_weight = 1.0
    cfg.c_global_loss_weight = 0.5
    cfg.gradient_loss_weight = 0.1

    cfg.progress_print_every = 10
    cfg.save_every_epoch = True
    cfg.vis_every_epoch = True
    cfg.vis_num_samples = 6

    return cfg


def main():
    cfg = make_small_scale_config()
    train_with_config(cfg)


if __name__ == "__main__":
    main()
