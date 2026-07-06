#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from train_burn_recovery_raw import TrainRawConfig, train_with_config


def make_small_raw_config() -> TrainRawConfig:
    cfg = TrainRawConfig()
    cfg.output_dir = Path(r"C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw_small")

    cfg.max_train_samples = 1000
    cfg.max_val_samples = 300
    cfg.batch_size = 2
    cfg.epochs = 10

    cfg.raw_shape = None
    cfg.image_size = None
    cfg.raw_max = 65535.0
    cfg.endian = "little"

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
    train_with_config(make_small_raw_config())


if __name__ == "__main__":
    main()
