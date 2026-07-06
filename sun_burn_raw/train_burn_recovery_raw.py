#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from burn_recovery_net import BurnRecoveryNet
from raw_io import RAW_SHAPES, parse_shape, read_u16_bin, resize_float, save_preview_png, write_u16_bin
from synthetic_burn_raw import BurnRawConfig, generate_burn_trace_u16

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


IMAGE_STEMS = ("000", "001", "002", "003", "004")
RAW_SUFFIXES = {".bin", ".raw"}


@dataclass
class TrainRawConfig:
    train_root: Path = Path(r"C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train")
    val_root: Path = Path(r"C:\Users\17874\Documents\python\datasets_raw\burn_recovery\val")
    output_dir: Path = Path(r"C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw")

    raw_shape: tuple[int, int] | None = None
    image_size: tuple[int, int] | None = None
    endian: str = "little"
    raw_max: float = 65535.0

    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    grad_clip_norm: float = 5.0
    use_scheduler: bool = True
    resume_checkpoint: Path | None = None
    pretrained_checkpoint: Path | None = None

    base_channels: int = 32
    p_loss_weight: float = 1.0
    c_active_loss_weight: float = 1.0
    c_bg_loss_weight: float = 1.0
    c_global_loss_weight: float = 0.50
    gradient_loss_weight: float = 0.10
    dice_loss_weight: float = 0.50
    focal_alpha: float = 0.50
    focal_gamma: float = 2.0
    mask_threshold: float = 64.0 / 65535.0
    background_change_threshold: float = 32.0 / 65535.0
    aggregation: str = "median"

    temporal_scale_range: tuple[float, float] = (0.92, 1.08)
    train_no_burn_probability: float = 0.15
    val_no_burn_probability: float = 0.15
    generate_missing_burn: bool = True
    val_generate_missing_burn: bool = False
    validate_samples_on_init: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = 500
    val_subset_seed: int = 20260705
    seed: int = 2026
    save_every_epoch: bool = True
    use_tensorboard: bool = True
    vis_every_epoch: bool = True
    vis_num_samples: int = 4
    progress_print_every: int = 50

    synthetic_min_peak: int = 1800
    synthetic_max_peak: int = 9000
    synthetic_size_multiplier: float = 1.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(path: Path):
    parts = []
    number = ""
    for ch in path.stem:
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


def apply_saved_config_for_resume(cfg: TrainRawConfig, saved_cfg: dict):
    runtime_keys = {"train_root", "val_root", "output_dir", "resume_checkpoint", "pretrained_checkpoint"}
    for key, value in saved_cfg.items():
        if key in runtime_keys or not hasattr(cfg, key):
            continue
        if key.endswith("_root") or key in {"output_dir", "resume_checkpoint"}:
            value = Path(value) if value is not None else None
        elif key in {"image_size", "raw_shape"} and value is not None:
            value = tuple(value)
        setattr(cfg, key, value)


class BurnRecoveryRawDataset(Dataset):
    def __init__(
        self,
        root: Path,
        raw_shape: tuple[int, int] | None = None,
        image_size: tuple[int, int] | None = None,
        endian: str = "little",
        raw_max: float = 65535.0,
        temporal_scale_range: tuple[float, float] = (1.0, 1.0),
        generate_missing_burn: bool = True,
        max_samples: int | None = None,
        no_burn_probability: float = 0.0,
        no_burn_seed: int = 0,
        deterministic_no_burn: bool = False,
        mask_threshold: float = 64.0 / 65535.0,
        aggregation: str = "median",
        validate_samples: bool = False,
        synthetic_cfg: BurnRawConfig | None = None,
    ):
        self.root = Path(root)
        self.raw_shape = raw_shape
        self.image_size = image_size
        self.endian = endian
        self.raw_max = float(raw_max)
        self.temporal_scale_range = temporal_scale_range
        self.generate_missing_burn = generate_missing_burn
        self.no_burn_probability = float(no_burn_probability)
        self.no_burn_seed = int(no_burn_seed)
        self.deterministic_no_burn = deterministic_no_burn
        self.mask_threshold = float(mask_threshold)
        self.aggregation = str(aggregation)
        self.synthetic_cfg = synthetic_cfg or BurnRawConfig()

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        sample_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()], key=natural_key)
        if not generate_missing_burn:
            sample_dirs = [p for p in sample_dirs if self._burn_path(p).is_file()]
        if max_samples is not None:
            sample_dirs = sample_dirs[:max_samples]
        if validate_samples:
            sample_dirs, invalid = self._validate_sample_dirs(sample_dirs)
            print(f"{self.root}: valid={len(sample_dirs)}, invalid={len(invalid)}")
            for folder, reason in invalid[:20]:
                print(f"  invalid {folder}: {reason}")
        if not sample_dirs:
            raise RuntimeError(f"No raw samples found under {self.root}")
        self.sample_dirs = sample_dirs

    def _burn_path(self, sample_dir: Path) -> Path:
        return sample_dir / "synthetic_burn_trace_u16.npy"

    def _list_frame_paths(self, sample_dir: Path) -> list[Path]:
        paths = []
        for stem in IMAGE_STEMS:
            matches = [
                p for p in sample_dir.iterdir()
                if p.is_file() and p.stem == stem and p.suffix.lower() in RAW_SUFFIXES
            ]
            if len(matches) != 1:
                raise RuntimeError(f"{sample_dir} should contain exactly one {stem}.bin/.raw file")
            paths.append(matches[0])
        return paths

    def _validate_sample_dirs(self, sample_dirs: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
        valid = []
        invalid = []
        for sample_dir in sample_dirs:
            try:
                frame_paths = self._list_frame_paths(sample_dir)
                shape = None
                for path in frame_paths:
                    arr = read_u16_bin(path, self.raw_shape, self.endian)
                    if shape is None:
                        shape = arr.shape
                    elif arr.shape != shape:
                        raise RuntimeError(f"shape mismatch at {path.name}")
                burn_path = self._burn_path(sample_dir)
                if not burn_path.is_file() and not self.generate_missing_burn:
                    raise RuntimeError("missing synthetic_burn_trace_u16.npy")
                if burn_path.is_file():
                    burn = np.load(burn_path, mmap_mode="r")
                    if shape is not None and tuple(burn.shape) != tuple(shape):
                        raise RuntimeError(f"burn shape {tuple(burn.shape)} does not match frame shape {shape}")
                valid.append(sample_dir)
            except Exception as exc:
                invalid.append((sample_dir, str(exc)))
        return valid, invalid

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
        frames = []
        for path in self._list_frame_paths(sample_dir):
            img = read_u16_bin(path, self.raw_shape, self.endian).astype(np.float32) / self.raw_max
            img = resize_float(img, self.image_size)
            frames.append(np.clip(img, 0.0, 1.0))
        return np.stack(frames, axis=0).astype(np.float32)

    def _load_burn_trace(self, sample_dir: Path, target_hw: tuple[int, int], index: int) -> np.ndarray:
        burn_path = self._burn_path(sample_dir)
        if burn_path.is_file():
            burn = np.load(burn_path).astype(np.float32)
            if not np.isfinite(burn).all():
                raise ValueError(f"Non-finite burn label: {burn_path}")
            if float(burn.min()) < 0:
                raise ValueError(f"Negative burn label: {burn_path}")
            if float(burn.max()) > self.raw_max + 1e-3:
                raise ValueError(f"Burn label exceeds raw_max: {burn_path}")
        elif self.generate_missing_burn:
            rng = np.random.default_rng(self.no_burn_seed + index)
            h, w = target_hw
            burn = generate_burn_trace_u16(h, w, self.synthetic_cfg, rng).astype(np.float32)
        else:
            raise FileNotFoundError(f"Missing burn trace: {burn_path}")

        if self.image_size is not None:
            h, w = target_hw
            burn = cv2.resize(burn, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(burn / self.raw_max, 0.0, 1.0).astype(np.float32)

    def _aggregate_effective_correction(self, effective_correction: np.ndarray) -> np.ndarray:
        if self.aggregation == "median":
            return np.median(effective_correction, axis=0).astype(np.float32)
        if self.aggregation == "mean":
            return np.mean(effective_correction, axis=0).astype(np.float32)
        if self.aggregation == "max":
            return np.max(effective_correction, axis=0).astype(np.float32)
        if self.aggregation == "last":
            return effective_correction[-1].astype(np.float32)
        raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

    def __getitem__(self, index: int):
        sample_dir = self.sample_dirs[index]
        clean = self._load_frames(sample_dir)
        _, h, w = clean.shape
        if self._use_no_burn_sample(index):
            raw_correction = np.zeros((h, w), dtype=np.float32)
        else:
            raw_correction = self._load_burn_trace(sample_dir, (h, w), index)

        low, high = self.temporal_scale_range
        scales = np.random.uniform(low, high, size=(5, 1, 1)).astype(np.float32)
        requested_correction = raw_correction[None, :, :] * scales
        effective_correction = np.minimum(requested_correction, 1.0 - clean)
        effective_correction = np.maximum(effective_correction, 0.0).astype(np.float32)
        burned = (clean + effective_correction).astype(np.float32)
        correction = self._aggregate_effective_correction(effective_correction)
        correction = np.where(correction >= self.mask_threshold, correction, 0.0).astype(np.float32)
        mask = (correction > 0.0).astype(np.float32)

        return {
            "x": torch.from_numpy(burned).float(),
            "clean": torch.from_numpy(clean).float(),
            "p": torch.from_numpy(mask[None, :, :]).float(),
            "c": torch.from_numpy(correction[None, :, :]).float(),
            "sample": sample_dir.name,
        }


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dim=dims)
    pred_sum = prob.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    normal_loss = 1.0 - (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    empty_target_loss = prob.mean(dim=dims)
    return torch.where(target_sum > 0, normal_loss, empty_target_loss).mean()


def gradient_l1(pred: torch.Tensor, target: torch.Tensor):
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return nn.functional.l1_loss(pred_dx, target_dx) + nn.functional.l1_loss(pred_dy, target_dy)


def compute_loss(prob_logits, correction, p_target, c_target, cfg: TrainRawConfig):
    prob = torch.sigmoid(prob_logits)
    loss_focal = focal_loss_with_logits(prob_logits, p_target, cfg.focal_alpha, cfg.focal_gamma)
    loss_dice = dice_loss(prob, p_target)
    loss_prob = loss_focal + cfg.dice_loss_weight * loss_dice
    error = torch.abs(correction - c_target)
    active = p_target > 0.5
    background = ~active
    l1_active = error[active].mean() if active.any() else correction.new_tensor(0.0)
    l1_bg = torch.abs(correction[background]).mean() if background.any() else correction.new_tensor(0.0)
    l1_global = error.mean()
    grad = gradient_l1(correction, c_target)
    loss = (
        cfg.p_loss_weight * loss_prob
        + cfg.c_active_loss_weight * l1_active
        + cfg.c_bg_loss_weight * l1_bg
        + cfg.c_global_loss_weight * l1_global
        + cfg.gradient_loss_weight * grad
    )
    return loss, {
        "focal": float(loss_focal.detach().cpu()),
        "dice_loss": float(loss_dice.detach().cpu()),
        "l1_active": float(l1_active.detach().cpu()),
        "l1_bg": float(l1_bg.detach().cpu()),
        "l1_global": float(l1_global.detach().cpu()),
        "gradient": float(grad.detach().cpu()),
    }


@torch.no_grad()
def compute_metric_sums(prob, correction, p_target, c_target, cfg: TrainRawConfig):
    pred_mask = prob >= 0.5
    target_mask = p_target > 0.5
    background = ~target_mask
    tp = (pred_mask & target_mask).sum().float()
    fp = (pred_mask & background).sum().float()
    fn = ((~pred_mask) & target_mask).sum().float()
    tn = ((~pred_mask) & background).sum().float()
    error = torch.abs(correction - c_target)
    active_count = target_mask.sum().float()
    bg_count = background.sum().float()
    active_abs = error[target_mask].sum() if active_count > 0 else correction.new_tensor(0.0)
    active_sq = (correction[target_mask] - c_target[target_mask]).pow(2).sum() if active_count > 0 else correction.new_tensor(0.0)
    prob_active_sum = prob[target_mask].sum() if active_count > 0 else correction.new_tensor(0.0)
    if bg_count > 0:
        bg_pred_abs = torch.abs(correction[background])
        bg_abs = bg_pred_abs.sum()
        bg_changed_1 = (bg_pred_abs > cfg.background_change_threshold).float().sum()
        bg_changed_2 = (bg_pred_abs > 2.0 * cfg.background_change_threshold).float().sum()
        bg_max = bg_pred_abs.max()
        prob_bg_sum = prob[background].sum()
    else:
        bg_abs = bg_changed_1 = bg_changed_2 = bg_max = prob_bg_sum = correction.new_tensor(0.0)
    global_abs = error.sum()
    global_count = torch.tensor(error.numel(), device=error.device, dtype=error.dtype)
    error_hist = torch.histc(error.detach().float(), bins=256, min=0.0, max=1.0)
    result = {
        "tp": float(tp.cpu()),
        "fp": float(fp.cpu()),
        "fn": float(fn.cpu()),
        "tn": float(tn.cpu()),
        "active_abs": float(active_abs.cpu()),
        "active_sq": float(active_sq.cpu()),
        "active_count": float(active_count.cpu()),
        "bg_abs": float(bg_abs.cpu()),
        "bg_count": float(bg_count.cpu()),
        "bg_changed_1": float(bg_changed_1.cpu()),
        "bg_changed_2": float(bg_changed_2.cpu()),
        "bg_max": float(bg_max.cpu()),
        "prob_sum": float(prob.sum().cpu()),
        "prob_max": float(prob.max().cpu()),
        "prob_active_sum": float(prob_active_sum.cpu()),
        "prob_bg_sum": float(prob_bg_sum.cpu()),
        "global_abs": float(global_abs.cpu()),
        "global_count": float(global_count.cpu()),
        "max_error": float(error.max().cpu()),
    }
    for idx, value in enumerate(error_hist.cpu().tolist()):
        result[f"error_hist_{idx}"] = float(value)
    return result


def finalize_metrics(sums: dict[str, float]):
    eps = 1e-6
    tp = sums.get("tp", 0.0)
    fp = sums.get("fp", 0.0)
    fn = sums.get("fn", 0.0)
    tn = sums.get("tn", 0.0)
    precision = 1.0 if tp + fp <= 0 else tp / (tp + fp + eps)
    recall = 1.0 if tp + fn <= 0 else tp / (tp + fn + eps)
    dice = 1.0 if 2 * tp + fp + fn <= 0 else 2 * tp / (2 * tp + fp + fn + eps)
    iou = 1.0 if tp + fp + fn <= 0 else tp / (tp + fp + fn + eps)
    active_count = sums.get("active_count", 0.0)
    bg_count = sums.get("bg_count", 0.0)
    global_count = sums.get("global_count", 0.0)
    hist = [sums.get(f"error_hist_{idx}", 0.0) for idx in range(256)]
    hist_total = sum(hist)
    p95_error = 0.0
    if hist_total > 0:
        cutoff = 0.95 * hist_total
        cumulative = 0.0
        for idx, count in enumerate(hist):
            cumulative += count
            if cumulative >= cutoff:
                p95_error = (idx + 0.5) / 256.0
                break
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "false_positive_rate": 0.0 if fp + tn <= 0 else fp / (fp + tn + eps),
        "active_mae": 0.0 if active_count <= 0 else sums.get("active_abs", 0.0) / (active_count + eps),
        "active_rmse": 0.0 if active_count <= 0 else (sums.get("active_sq", 0.0) / (active_count + eps)) ** 0.5,
        "bg_mae": 0.0 if bg_count <= 0 else sums.get("bg_abs", 0.0) / (bg_count + eps),
        "bg_changed": 0.0 if bg_count <= 0 else sums.get("bg_changed_1", 0.0) / (bg_count + eps),
        "bg_changed_2": 0.0 if bg_count <= 0 else sums.get("bg_changed_2", 0.0) / (bg_count + eps),
        "global_mae": 0.0 if global_count <= 0 else sums.get("global_abs", 0.0) / (global_count + eps),
        "p95_error": p95_error,
        "max_error": sums.get("max_error", 0.0),
        "bg_max": sums.get("bg_max", 0.0),
        "prob_mean": 0.0 if global_count <= 0 else sums.get("prob_sum", 0.0) / (global_count + eps),
        "prob_max": sums.get("prob_max", 0.0),
        "prob_active_mean": 0.0 if active_count <= 0 else sums.get("prob_active_sum", 0.0) / (active_count + eps),
        "prob_bg_mean": 0.0 if bg_count <= 0 else sums.get("prob_bg_sum", 0.0) / (bg_count + eps),
    }


def run_one_epoch(model, loader, optimizer, device, cfg: TrainRawConfig, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_parts: dict[str, float] = {}
    total_metric_sums: dict[str, float] = {}
    n_batches = 0
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        p_target = batch["p"].to(device, non_blocking=True)
        c_target = batch["c"].to(device, non_blocking=True)
        amp_enabled = bool(cfg.use_amp and device.type == "cuda")
        autocast_context = torch.amp.autocast("cuda", enabled=True) if amp_enabled else nullcontext()
        with torch.set_grad_enabled(is_train), autocast_context:
            prob_logits, correction = model(x)
        prob_logits = prob_logits.float()
        prob = torch.sigmoid(prob_logits)
        correction = correction.float()
        loss, parts = compute_loss(prob_logits, correction, p_target, c_target, cfg)
        metric_sums = compute_metric_sums(prob, correction, p_target, c_target, cfg)
        with torch.no_grad():
            active = p_target > 0.5
            background = ~active
            prob_active_mean = prob[active].mean() if active.any() else prob.new_tensor(0.0)
            prob_bg_mean = prob[background].mean() if background.any() else prob.new_tensor(0.0)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        for key, value in parts.items():
            total_parts[key] = total_parts.get(key, 0.0) + value
        for key, value in metric_sums.items():
            if key in {"max_error", "bg_max", "prob_max"}:
                total_metric_sums[key] = max(total_metric_sums.get(key, 0.0), value)
            else:
                total_metric_sums[key] = total_metric_sums.get(key, 0.0) + value
        n_batches += 1
        if cfg.progress_print_every > 0 and (batch_idx % cfg.progress_print_every == 0 or batch_idx == total_batches):
            phase = "train" if is_train else "val"
            print(
                f"{phase} batch {batch_idx}/{total_batches} "
                f"loss={total_loss / max(n_batches, 1):.5f} "
                f"focal={parts['focal']:.5f} dice_loss={parts['dice_loss']:.5f} "
                f"active={parts['l1_active']:.5f} bg={parts['l1_bg']:.5f} "
                f"global={parts['l1_global']:.5f} grad={parts['gradient']:.5f} "
                f"prob_mean={float(prob.mean().detach().cpu()):.5f} "
                f"prob_max={float(prob.max().detach().cpu()):.5f} "
                f"prob_active={float(prob_active_mean.detach().cpu()):.5f} "
                f"prob_bg={float(prob_bg_mean.detach().cpu()):.5f}",
                flush=True,
            )
    denom = max(n_batches, 1)
    result = {"loss": total_loss / denom}
    result.update({key: value / denom for key, value in total_parts.items()})
    result.update(finalize_metrics(total_metric_sums))
    return result


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, best_val, cfg: TrainRawConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": asdict(cfg),
        },
        path,
    )


def to_u8_image(x: np.ndarray) -> np.ndarray:
    return np.clip(np.squeeze(x) * 255.0, 0, 255).astype(np.uint8)


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


@torch.no_grad()
def save_visualizations(model, loader, device, cfg: TrainRawConfig, epoch: int):
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
        prob_logits, correction = model(x)
        prob = torch.sigmoid(prob_logits)
        restored_gated = torch.clamp(x[:, -1:, :, :] - (prob > 0.5).float() * correction, 0.0, 1.0)
        abs_error = torch.abs(correction - c_target)
        for i in range(x.shape[0]):
            if saved >= cfg.vis_num_samples:
                return
            panels = [
                add_label(to_u8_image(x[i, 0].cpu().numpy()), "input first"),
                add_label(to_u8_image(x[i, -1].cpu().numpy()), "input last"),
                add_label(to_u8_image(clean[i, -1].cpu().numpy()), "target clean"),
                add_label(to_u8_image(restored_gated[i, 0].cpu().numpy()), "restored gated"),
                add_label(to_u8_image(prob[i, 0].cpu().numpy()), "pred P"),
                add_label(to_u8_image(p_target[i, 0].cpu().numpy()), "target mask"),
                add_label(to_u8_image(correction[i, 0].cpu().numpy()), "pred C"),
                add_label(to_u8_image(c_target[i, 0].cpu().numpy()), "target C"),
                add_label(to_u8_image(abs_error[i, 0].cpu().numpy()), "abs error"),
            ]
            h = panels[0].shape[0]
            sep = np.full((h, 6, 3), 255, dtype=np.uint8)
            row = panels[0]
            for panel in panels[1:]:
                row = np.concatenate([row, sep, panel], axis=1)
            cv2.imwrite(str(vis_dir / f"{saved:02d}_{samples[i]}.png"), row)
            saved += 1


def train_with_config(cfg: TrainRawConfig):
    resume_data = None
    if cfg.resume_checkpoint is not None:
        resume_data = torch.load(cfg.resume_checkpoint, map_location="cpu")
        apply_saved_config_for_resume(cfg, resume_data.get("config", {}))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    synthetic_cfg = BurnRawConfig(
        raw_max=int(cfg.raw_max),
        min_peak=cfg.synthetic_min_peak,
        max_peak=cfg.synthetic_max_peak,
        active_threshold=max(1, int(round(cfg.mask_threshold * cfg.raw_max))),
        size_multiplier=cfg.synthetic_size_multiplier,
        seed=cfg.seed,
    )
    train_dataset = BurnRecoveryRawDataset(
        root=cfg.train_root,
        raw_shape=cfg.raw_shape,
        image_size=cfg.image_size,
        endian=cfg.endian,
        raw_max=cfg.raw_max,
        temporal_scale_range=cfg.temporal_scale_range,
        generate_missing_burn=cfg.generate_missing_burn,
        max_samples=cfg.max_train_samples,
        no_burn_probability=cfg.train_no_burn_probability,
        no_burn_seed=cfg.seed,
        deterministic_no_burn=False,
        mask_threshold=cfg.mask_threshold,
        aggregation=cfg.aggregation,
        validate_samples=cfg.validate_samples_on_init,
        synthetic_cfg=synthetic_cfg,
    )
    val_dataset_full = BurnRecoveryRawDataset(
        root=cfg.val_root,
        raw_shape=cfg.raw_shape,
        image_size=cfg.image_size,
        endian=cfg.endian,
        raw_max=cfg.raw_max,
        temporal_scale_range=(1.0, 1.0),
        generate_missing_burn=cfg.val_generate_missing_burn,
        max_samples=None,
        no_burn_probability=cfg.val_no_burn_probability,
        no_burn_seed=cfg.val_subset_seed,
        deterministic_no_burn=True,
        mask_threshold=cfg.mask_threshold,
        aggregation=cfg.aggregation,
        validate_samples=cfg.validate_samples_on_init,
        synthetic_cfg=synthetic_cfg,
    )
    full_val_samples = len(val_dataset_full)
    val_dataset = make_fixed_random_subset(val_dataset_full, cfg.max_val_samples, cfg.val_subset_seed)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.use_amp and device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs) if cfg.use_scheduler else None
    start_epoch = 1
    best_val = float("inf")
    if resume_data is None and cfg.pretrained_checkpoint is not None:
        pretrained = torch.load(cfg.pretrained_checkpoint, map_location=device)
        state_dict = pretrained.get("model", pretrained)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"loaded pretrained model weights: {cfg.pretrained_checkpoint}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    if resume_data is not None:
        model.load_state_dict(resume_data["model"])
        optimizer.load_state_dict(resume_data["optimizer"])
        if resume_data.get("scaler") is not None:
            scaler.load_state_dict(resume_data["scaler"])
        if scheduler is not None and resume_data.get("scheduler") is not None:
            scheduler.load_state_dict(resume_data["scheduler"])
        start_epoch = int(resume_data.get("epoch", 0)) + 1
        best_val = float(resume_data.get("best_val", best_val))
        print(f"resumed checkpoint: {cfg.resume_checkpoint}")

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)} / {full_val_samples}")
    print(f"output dir: {cfg.output_dir}")
    print(f"raw_shape: {cfg.raw_shape}")
    print(f"image_size: {cfg.image_size}")
    print(f"raw_max: {cfg.raw_max}")
    print(f"pretrained_checkpoint: {cfg.pretrained_checkpoint}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"amp: {bool(cfg.use_amp and device.type == 'cuda')}")
    print(f"mask threshold: {cfg.mask_threshold:.8f} ({cfg.mask_threshold * cfg.raw_max:.1f} raw)")
    print(f"aggregation: {cfg.aggregation}")
    print(
        "loss weights: "
        f"p={cfg.p_loss_weight}, active={cfg.c_active_loss_weight}, "
        f"bg={cfg.c_bg_loss_weight}, global={cfg.c_global_loss_weight}, "
        f"grad={cfg.gradient_loss_weight}, dice={cfg.dice_loss_weight}"
    )
    writer = None
    if cfg.use_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(cfg.output_dir / "runs"))
        print(f"tensorboard log dir: {cfg.output_dir / 'runs'}")
    elif cfg.use_tensorboard:
        print("TensorBoard is not available; scalar logging is disabled.")

    for epoch in range(start_epoch, cfg.epochs + 1):
        start = time.time()
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, cfg, scaler)
        val_metrics = run_one_epoch(model, val_loader, None, device, cfg)
        elapsed = time.time() - start
        print(
            f"epoch {epoch:03d}/{cfg.epochs} time={elapsed:.1f}s "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_dice={val_metrics['dice']:.5f} val_iou={val_metrics['iou']:.5f} "
            f"val_active_mae={val_metrics['active_mae']:.6f} "
            f"val_bg_mae={val_metrics['bg_mae']:.6f} "
            f"val_active_raw={val_metrics['active_mae'] * cfg.raw_max:.1f} "
            f"val_bg_raw={val_metrics['bg_mae'] * cfg.raw_max:.1f}"
        )
        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()
        if scheduler is not None:
            scheduler.step()
        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
        if cfg.vis_every_epoch:
            save_visualizations(model, val_loader, device, cfg, epoch)
        if cfg.save_every_epoch:
            save_checkpoint(cfg.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, scheduler, epoch, best_val, cfg)
        if is_best:
            save_checkpoint(cfg.output_dir / "best.pt", model, optimizer, scaler, scheduler, epoch, best_val, cfg)
            print(f"  saved best.pt, best_val={best_val:.5f}")
    if writer is not None:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train BurnRecoveryNet on uint16 raw bin data.")
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--val-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--raw-shape", type=str, default=None, help="HxW, e.g. 512x640 or 1024x1280.")
    parser.add_argument("--image-size", type=str, default=None, help="Optional resize HxW for training.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=None, help="Load model weights only for transfer training.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainRawConfig()
    if args.train_root is not None:
        cfg.train_root = args.train_root
    if args.val_root is not None:
        cfg.val_root = args.val_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.raw_shape is not None:
        cfg.raw_shape = parse_shape(args.raw_shape)
    if args.image_size is not None:
        cfg.image_size = parse_shape(args.image_size)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
    if args.max_val_samples is not None:
        cfg.max_val_samples = args.max_val_samples
    if args.resume is not None:
        cfg.resume_checkpoint = args.resume
    if args.pretrained is not None:
        cfg.pretrained_checkpoint = args.pretrained
    train_with_config(cfg)


if __name__ == "__main__":
    main()
