#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练 BurnRecoveryNet。

数据要求：
每个 sample_xxxxxx 文件夹内至少包含：
    000.jpg
    001.jpg
    002.jpg
    003.jpg
    004.jpg
    synthetic_burn_trace.npy

训练输入：
    clean_frames + synthetic_burn_trace -> [5, H, W]

训练标签：
    P_target: synthetic_burn_trace > 0
    C_target: synthetic_burn_trace / 255
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from burn_recovery_net import BurnRecoveryNet

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


@dataclass
class TrainConfig:
    train_root: Path = Path(r"C:\Users\17874\Documents\python\datasets\burn_recovery\train")
    val_root: Path = Path(r"C:\Users\17874\Documents\python\datasets\burn_recovery\val")
    output_dir: Path = Path(r"C:\Users\17874\Documents\python\checkpoints\burn_recovery")

    image_size: tuple[int, int] | None = None  # 例如 (256, 320)，None 表示保持原尺寸
    # Use a smaller default resolution so an 8GB GPU can train without OOM.
    # Set image_size=None and batch_size=1 if you want full-resolution training.
    # image_size: tuple[int, int] | None = (256, 320)
    batch_size: int = 6
    num_workers: int = 0  # Windows 下先用 0，稳定
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True

    base_channels: int = 32
    p_loss_weight: float = 1.0
    c_loss_weight: float = 1.0
    c_active_loss_weight: float = 4.0

    temporal_scale_range: tuple[float, float] = (0.92, 1.08)
    train_no_burn_probability: float = 0.15
    val_no_burn_probability: float = 0.15
    generate_missing_burn: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = 500
    val_subset_seed: int = 20260705
    seed: int = 2026
    save_every_epoch: bool = True
    use_tensorboard: bool = True
    vis_every_epoch: bool = True
    vis_num_samples: int = 4
    progress_print_every: int = 50


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(path: Path):
    parts = []
    text = path.stem
    number = ""
    for ch in text:
        if ch.isdigit():
            number += ch
        else:
            if number:
                parts.append(int(number))
                number = ""
            parts.append(ch)
    if number:
        parts.append(int(number))
    return parts


def make_fixed_random_subset(dataset: Dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


class BurnRecoveryDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int] | None = None,
        temporal_scale_range: tuple[float, float] = (1.0, 1.0),
        generate_missing_burn: bool = True,
        max_samples: int | None = None,
        no_burn_probability: float = 0.0,
        no_burn_seed: int = 0,
        deterministic_no_burn: bool = False,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.temporal_scale_range = temporal_scale_range
        self.generate_missing_burn = generate_missing_burn
        self.no_burn_probability = float(no_burn_probability)
        self.no_burn_seed = int(no_burn_seed)
        self.deterministic_no_burn = deterministic_no_burn

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        sample_dirs = sorted(
            [p for p in self.root.iterdir() if p.is_dir()],
            key=natural_key,
        )
        if generate_missing_burn:
            valid_dirs = sample_dirs
        else:
            valid_dirs = [
                p for p in sample_dirs
                if (p / "synthetic_burn_trace.npy").is_file()
            ]

        if max_samples is not None:
            valid_dirs = valid_dirs[:max_samples]

        if not valid_dirs:
            raise RuntimeError(
                f"No samples with synthetic_burn_trace.npy found under {self.root}. "
                "请先生成标签，或把 generate_missing_burn 设置为 True。"
            )

        self.sample_dirs = valid_dirs

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _use_no_burn_sample(self, index: int) -> bool:
        if self.no_burn_probability <= 0:
            return False
        if self.no_burn_probability >= 1:
            return True

        if self.deterministic_no_burn:
            rng = np.random.default_rng(self.no_burn_seed + index)
            return bool(rng.random() < self.no_burn_probability)

        return bool(np.random.random() < self.no_burn_probability)

    def _load_frames(self, sample_dir: Path) -> np.ndarray:
        frame_paths = sorted(
            [
                p for p in sample_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_SUFFIXES
                and p.name.startswith(("000", "001", "002", "003", "004"))
            ],
            key=natural_key,
        )[:5]

        if len(frame_paths) != 5:
            raise RuntimeError(f"{sample_dir} should contain 5 frame images, got {len(frame_paths)}")

        frames = []
        for path in frame_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if self.image_size is not None:
                h, w = self.image_size
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(img.astype(np.float32) / 255.0)

        return np.stack(frames, axis=0)

    def _load_burn_trace(self, sample_dir: Path, target_hw: tuple[int, int]) -> np.ndarray:
        burn_path = sample_dir / "synthetic_burn_trace.npy"
        if burn_path.is_file():
            burn = np.load(burn_path).astype(np.float32)
        elif self.generate_missing_burn:
            # 兜底：训练时临时生成一个标签，不写入磁盘。
            from preview_burn_on_training_groups import generate_burn_pattern, BURN_ACTIVE_THRESHOLD

            h, w = target_hw
            burn = generate_burn_pattern(h=h, w=w, pattern_type="auto")
            burn = burn.astype(np.float32)
            burn = np.where(burn > BURN_ACTIVE_THRESHOLD, burn, 0.0).astype(np.float32)
        else:
            raise FileNotFoundError(f"Missing burn trace: {burn_path}")

        if self.image_size is not None:
            h, w = target_hw
            burn = cv2.resize(burn, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.maximum(burn, 0.0) / 255.0

    def __getitem__(self, index: int):
        sample_dir = self.sample_dirs[index]
        clean = self._load_frames(sample_dir)
        _, h, w = clean.shape

        if self._use_no_burn_sample(index):
            correction = np.zeros((h, w), dtype=np.float32)
        else:
            correction = self._load_burn_trace(sample_dir, (h, w))
        mask = (correction > 0).astype(np.float32)

        low, high = self.temporal_scale_range
        scales = np.random.uniform(low, high, size=(5, 1, 1)).astype(np.float32)
        burned = np.clip(clean + correction[None, :, :] * scales, 0.0, 1.0)

        return {
            "x": torch.from_numpy(burned).float(),
            "clean": torch.from_numpy(clean).float(),
            "p": torch.from_numpy(mask[None, :, :]).float(),
            "c": torch.from_numpy(correction[None, :, :]).float(),
            "sample": sample_dir.name,
        }


def compute_loss(
    prob: torch.Tensor,
    correction: torch.Tensor,
    p_target: torch.Tensor,
    c_target: torch.Tensor,
    cfg: TrainConfig,
):
    bce = nn.functional.binary_cross_entropy(prob, p_target)
    l1_all = nn.functional.l1_loss(correction, c_target)

    active = p_target > 0.5
    if active.any():
        l1_active = nn.functional.l1_loss(correction[active], c_target[active])
    else:
        l1_active = correction.new_tensor(0.0)

    loss = (
        cfg.p_loss_weight * bce
        + cfg.c_loss_weight * l1_all
        + cfg.c_active_loss_weight * l1_active
    )
    return loss, {
        "bce": float(bce.detach().cpu()),
        "l1_all": float(l1_all.detach().cpu()),
        "l1_active": float(l1_active.detach().cpu()),
    }


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: TrainConfig,
    scaler: torch.amp.GradScaler | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_bce = 0.0
    total_l1 = 0.0
    total_active = 0.0
    n_batches = 0

    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        p_target = batch["p"].to(device, non_blocking=True)
        c_target = batch["c"].to(device, non_blocking=True)

        amp_enabled = bool(cfg.use_amp and device.type == "cuda")
        autocast_context = (
            torch.amp.autocast("cuda", enabled=True)
            if amp_enabled
            else nullcontext()
        )

        with torch.set_grad_enabled(is_train), autocast_context:
            prob, correction = model(x)

        prob = prob.float()
        correction = correction.float()
        loss, parts = compute_loss(prob, correction, p_target, c_target, cfg)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_bce += parts["bce"]
        total_l1 += parts["l1_all"]
        total_active += parts["l1_active"]
        n_batches += 1

        if (
            cfg.progress_print_every > 0
            and (batch_idx % cfg.progress_print_every == 0 or batch_idx == total_batches)
        ):
            phase = "train" if is_train else "val"
            print(
                f"{phase} batch {batch_idx}/{total_batches} "
                f"loss={total_loss / max(n_batches, 1):.5f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "bce": total_bce / max(n_batches, 1),
        "l1_all": total_l1 / max(n_batches, 1),
        "l1_active": total_active / max(n_batches, 1),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    cfg: TrainConfig,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )


def to_u8_image(x: np.ndarray) -> np.ndarray:
    x = np.squeeze(x)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    if img.ndim == 2:
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img.copy()
    cv2.putText(
        canvas,
        label,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


@torch.no_grad()
def save_visualizations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
):
    model.eval()
    vis_dir = cfg.output_dir / "vis" / f"epoch_{epoch:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        p_target = batch["p"].to(device, non_blocking=True)
        c_target = batch["c"].to(device, non_blocking=True)
        samples = batch["sample"]

        prob, correction = model(x)
        restored = torch.clamp(x[:, -1:, :, :] - (prob > 0.5).float() * correction, 0.0, 1.0)

        batch_size = x.shape[0]
        for i in range(batch_size):
            if saved >= cfg.vis_num_samples:
                return

            panels = [
                add_label(to_u8_image(x[i, -1].cpu().numpy()), "input burned"),
                add_label(to_u8_image(clean[i, -1].cpu().numpy()), "target clean"),
                add_label(to_u8_image(restored[i, 0].cpu().numpy()), "restored"),
                add_label(to_u8_image(prob[i, 0].cpu().numpy()), "pred P"),
                add_label(to_u8_image(p_target[i, 0].cpu().numpy()), "target mask"),
                add_label(to_u8_image(correction[i, 0].cpu().numpy()), "pred C"),
                add_label(to_u8_image(c_target[i, 0].cpu().numpy()), "target C"),
            ]

            h = panels[0].shape[0]
            separator = np.full((h, 6, 3), 255, dtype=np.uint8)
            row = panels[0]
            for panel in panels[1:]:
                row = np.concatenate([row, separator, panel], axis=1)

            out_path = vis_dir / f"{saved:02d}_{samples[i]}.png"
            cv2.imwrite(str(out_path), row)
            saved += 1


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_dataset = BurnRecoveryDataset(
        root=cfg.train_root,
        image_size=cfg.image_size,
        temporal_scale_range=cfg.temporal_scale_range,
        generate_missing_burn=cfg.generate_missing_burn,
        max_samples=cfg.max_train_samples,
        no_burn_probability=cfg.train_no_burn_probability,
        no_burn_seed=cfg.seed,
        deterministic_no_burn=False,
    )
    val_dataset = BurnRecoveryDataset(
        root=cfg.val_root,
        image_size=cfg.image_size,
        temporal_scale_range=(1.0, 1.0),
        generate_missing_burn=cfg.generate_missing_burn,
        max_samples=None,
        no_burn_probability=cfg.val_no_burn_probability,
        no_burn_seed=cfg.val_subset_seed,
        deterministic_no_burn=True,
    )
    full_val_samples = len(val_dataset)
    val_dataset = make_fixed_random_subset(
        val_dataset,
        cfg.max_val_samples,
        cfg.val_subset_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
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

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)} / {full_val_samples}")
    print(f"output dir: {cfg.output_dir}")
    print(f"image_size: {cfg.image_size}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"amp: {bool(cfg.use_amp and device.type == 'cuda')}")
    print(f"train no-burn probability: {cfg.train_no_burn_probability}")
    print(f"val no-burn probability: {cfg.val_no_burn_probability}")

    writer = None
    if cfg.use_tensorboard:
        if SummaryWriter is None:
            print("TensorBoard is not available; scalar logging is disabled.")
        else:
            writer = SummaryWriter(log_dir=str(cfg.output_dir / "runs"))
            print(f"tensorboard log dir: {cfg.output_dir / 'runs'}")

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, cfg, scaler)
        val_metrics = run_one_epoch(model, val_loader, None, device, cfg)
        elapsed = time.time() - start

        print(
            f"epoch {epoch:03d}/{cfg.epochs} "
            f"time={elapsed:.1f}s "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_bce={val_metrics['bce']:.5f} "
            f"val_l1_active={val_metrics['l1_active']:.5f}"
        )

        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()

        if cfg.vis_every_epoch:
            save_visualizations(model, val_loader, device, cfg, epoch)

        if cfg.save_every_epoch:
            save_checkpoint(
                cfg.output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                best_val,
                cfg,
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                cfg.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val,
                cfg,
            )
            print(f"  saved best.pt, best_val={best_val:.5f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
