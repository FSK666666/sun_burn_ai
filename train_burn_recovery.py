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
from dataclasses import asdict, dataclass
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
    grad_clip_norm: float = 5.0
    use_scheduler: bool = True
    resume_checkpoint: Path | None = None

    base_channels: int = 32
    p_loss_weight: float = 1.0
    c_active_loss_weight: float = 1.0
    c_bg_loss_weight: float = 0.10
    c_global_loss_weight: float = 0.20
    gradient_loss_weight: float = 0.10
    dice_loss_weight: float = 0.50
    focal_alpha: float = 0.50
    focal_gamma: float = 2.0
    mask_threshold: float = 2.0 / 255.0
    background_change_threshold: float = 1.0 / 255.0
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


def apply_saved_config_for_resume(cfg: TrainConfig, saved_cfg: dict):
    runtime_keys = {
        "train_root",
        "val_root",
        "output_dir",
        "resume_checkpoint",
    }
    for key, value in saved_cfg.items():
        if key in runtime_keys or not hasattr(cfg, key):
            continue
        if key == "image_size" and value is not None:
            value = tuple(value)
        setattr(cfg, key, value)


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
        mask_threshold: float = 8.0 / 255.0,
        aggregation: str = "median",
        validate_samples: bool = False,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.temporal_scale_range = temporal_scale_range
        self.generate_missing_burn = generate_missing_burn
        self.no_burn_probability = float(no_burn_probability)
        self.no_burn_seed = int(no_burn_seed)
        self.deterministic_no_burn = deterministic_no_burn
        self.mask_threshold = float(mask_threshold)
        self.aggregation = str(aggregation)

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

        if validate_samples:
            valid_dirs, invalid = self._validate_sample_dirs(valid_dirs)
            print(
                f"{self.root}: valid samples={len(valid_dirs)}, "
                f"invalid samples={len(invalid)}"
            )
            for folder, reason in invalid[:20]:
                print(f"  invalid {folder}: {reason}")

        if not valid_dirs:
            raise RuntimeError(
                f"No samples with synthetic_burn_trace.npy found under {self.root}. "
                "请先生成标签，或把 generate_missing_burn 设置为 True。"
            )

        self.sample_dirs = valid_dirs

    def _list_frame_paths(self, sample_dir: Path) -> list[Path]:
        expected_stems = {"000", "001", "002", "003", "004"}
        frame_paths = sorted(
            [
                p for p in sample_dir.iterdir()
                if p.is_file()
                and p.stem in expected_stems
                and p.suffix.lower() in IMAGE_SUFFIXES
            ],
            key=natural_key,
        )

        if len(frame_paths) != 5:
            raise RuntimeError(
                f"{sample_dir} should contain exactly 5 frame images, got {len(frame_paths)}"
            )
        return frame_paths

    def _validate_sample_dirs(self, sample_dirs: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
        valid = []
        invalid = []

        for sample_dir in sample_dirs:
            try:
                frame_paths = self._list_frame_paths(sample_dir)
                shape = None
                for path in frame_paths:
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise RuntimeError(f"failed to read {path.name}")
                    if shape is None:
                        shape = img.shape
                    elif img.shape != shape:
                        raise RuntimeError(f"shape mismatch at {path.name}")

                burn_path = sample_dir / "synthetic_burn_trace.npy"
                if not burn_path.is_file() and not self.generate_missing_burn:
                    raise RuntimeError("missing synthetic_burn_trace.npy")
                if burn_path.is_file():
                    burn = np.load(burn_path, mmap_mode="r")
                    if shape is not None and tuple(burn.shape) != tuple(shape):
                        raise RuntimeError(
                            f"burn shape {tuple(burn.shape)} does not match frame shape {shape}"
                        )
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
        frame_paths = self._list_frame_paths(sample_dir)

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
            if not np.isfinite(burn).all():
                raise ValueError(f"Non-finite burn label: {burn_path}")
            if float(burn.min()) < 0:
                raise ValueError(f"Negative burn label: {burn_path}")
            if float(burn.max()) > 255.0 + 1e-3:
                raise ValueError(f"Burn label exceeds 255: {burn_path}")
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
            raw_correction = self._load_burn_trace(sample_dir, (h, w))

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


def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dim=dims)
    pred_sum = prob.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    normal_loss = 1.0 - (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    empty_target_loss = prob.mean(dim=dims)
    loss = torch.where(target_sum > 0, normal_loss, empty_target_loss)
    return loss.mean()


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return (
        nn.functional.l1_loss(pred_dx, target_dx)
        + nn.functional.l1_loss(pred_dy, target_dy)
    )


def compute_loss(
    prob_logits: torch.Tensor,
    correction: torch.Tensor,
    p_target: torch.Tensor,
    c_target: torch.Tensor,
    cfg: TrainConfig,
):
    prob = torch.sigmoid(prob_logits)
    loss_focal = focal_loss_with_logits(
        prob_logits,
        p_target,
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
    )
    loss_dice = dice_loss(prob, p_target)
    loss_prob = loss_focal + cfg.dice_loss_weight * loss_dice

    error = torch.abs(correction - c_target)
    active = p_target > 0.5
    background = ~active

    if active.any():
        l1_active = error[active].mean()
    else:
        l1_active = correction.new_tensor(0.0)

    if background.any():
        l1_bg = torch.abs(correction[background]).mean()
    else:
        l1_bg = correction.new_tensor(0.0)

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
        "loss_prob": float(loss_prob.detach().cpu()),
        "focal": float(loss_focal.detach().cpu()),
        "dice_loss": float(loss_dice.detach().cpu()),
        "l1_active": float(l1_active.detach().cpu()),
        "l1_bg": float(l1_bg.detach().cpu()),
        "l1_global": float(l1_global.detach().cpu()),
        "gradient": float(grad.detach().cpu()),
    }


@torch.no_grad()
def compute_metric_sums(
    prob: torch.Tensor,
    correction: torch.Tensor,
    p_target: torch.Tensor,
    c_target: torch.Tensor,
    cfg: TrainConfig,
) -> dict[str, float]:
    pred_mask = prob >= 0.5
    target_mask = p_target > 0.5
    background = ~target_mask

    tp = (pred_mask & target_mask).sum().float()
    fp = (pred_mask & background).sum().float()
    fn = ((~pred_mask) & target_mask).sum().float()
    tn = ((~pred_mask) & background).sum().float()
    error = torch.abs(correction - c_target)

    active_count = target_mask.sum().float()
    if active_count > 0:
        active_abs = error[target_mask].sum()
        active_sq = (correction[target_mask] - c_target[target_mask]).pow(2).sum()
        prob_active_sum = prob[target_mask].sum()
    else:
        active_abs = correction.new_tensor(0.0)
        active_sq = correction.new_tensor(0.0)
        prob_active_sum = correction.new_tensor(0.0)

    bg_count = background.sum().float()
    if bg_count > 0:
        bg_pred_abs = torch.abs(correction[background])
        bg_abs = bg_pred_abs.sum()
        bg_changed_1 = (bg_pred_abs > cfg.background_change_threshold).float().sum()
        bg_changed_2 = (bg_pred_abs > 2.0 * cfg.background_change_threshold).float().sum()
        bg_max = bg_pred_abs.max()
        prob_bg_sum = prob[background].sum()
    else:
        bg_abs = correction.new_tensor(0.0)
        bg_changed_1 = correction.new_tensor(0.0)
        bg_changed_2 = correction.new_tensor(0.0)
        bg_max = correction.new_tensor(0.0)
        prob_bg_sum = correction.new_tensor(0.0)

    global_abs = error.sum()
    global_count = torch.tensor(error.numel(), device=error.device, dtype=error.dtype)
    max_error = error.max()
    prob_sum = prob.sum()
    prob_max = prob.max()
    error_hist = torch.histc(error.detach().float(), bins=256, min=0.0, max=1.0)

    result = {
        "tp": float(tp.detach().cpu()),
        "fp": float(fp.detach().cpu()),
        "fn": float(fn.detach().cpu()),
        "tn": float(tn.detach().cpu()),
        "active_abs": float(active_abs.detach().cpu()),
        "active_sq": float(active_sq.detach().cpu()),
        "active_count": float(active_count.detach().cpu()),
        "bg_abs": float(bg_abs.detach().cpu()),
        "bg_count": float(bg_count.detach().cpu()),
        "bg_changed_1": float(bg_changed_1.detach().cpu()),
        "bg_changed_2": float(bg_changed_2.detach().cpu()),
        "bg_max": float(bg_max.detach().cpu()),
        "prob_sum": float(prob_sum.detach().cpu()),
        "prob_max": float(prob_max.detach().cpu()),
        "prob_active_sum": float(prob_active_sum.detach().cpu()),
        "prob_bg_sum": float(prob_bg_sum.detach().cpu()),
        "global_abs": float(global_abs.detach().cpu()),
        "global_count": float(global_count.detach().cpu()),
        "max_error": float(max_error.detach().cpu()),
    }
    for idx, value in enumerate(error_hist.cpu().tolist()):
        result[f"error_hist_{idx}"] = float(value)
    return result


def finalize_metrics(sums: dict[str, float]) -> dict[str, float]:
    eps = 1e-6
    tp = sums.get("tp", 0.0)
    fp = sums.get("fp", 0.0)
    fn = sums.get("fn", 0.0)
    tn = sums.get("tn", 0.0)

    precision = 1.0 if tp + fp <= 0 else tp / (tp + fp + eps)
    recall = 1.0 if tp + fn <= 0 else tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = 1.0 if 2 * tp + fp + fn <= 0 else 2 * tp / (2 * tp + fp + fn + eps)
    iou = 1.0 if tp + fp + fn <= 0 else tp / (tp + fp + fn + eps)
    false_positive_rate = 0.0 if fp + tn <= 0 else fp / (fp + tn + eps)

    active_count = sums.get("active_count", 0.0)
    bg_count = sums.get("bg_count", 0.0)
    global_count = sums.get("global_count", 0.0)

    active_mae = 0.0 if active_count <= 0 else sums.get("active_abs", 0.0) / (active_count + eps)
    active_rmse = 0.0 if active_count <= 0 else (sums.get("active_sq", 0.0) / (active_count + eps)) ** 0.5
    bg_mae = 0.0 if bg_count <= 0 else sums.get("bg_abs", 0.0) / (bg_count + eps)
    bg_changed = 0.0 if bg_count <= 0 else sums.get("bg_changed_1", 0.0) / (bg_count + eps)
    bg_changed_2 = 0.0 if bg_count <= 0 else sums.get("bg_changed_2", 0.0) / (bg_count + eps)
    global_mae = 0.0 if global_count <= 0 else sums.get("global_abs", 0.0) / (global_count + eps)
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
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "false_positive_rate": false_positive_rate,
        "active_mae": active_mae,
        "active_rmse": active_rmse,
        "bg_mae": bg_mae,
        "bg_changed": bg_changed,
        "bg_changed_2": bg_changed_2,
        "global_mae": global_mae,
        "p95_error": p95_error,
        "max_error": sums.get("max_error", 0.0),
        "bg_max": sums.get("bg_max", 0.0),
        "prob_mean": 0.0 if global_count <= 0 else sums.get("prob_sum", 0.0) / (global_count + eps),
        "prob_max": sums.get("prob_max", 0.0),
        "prob_active_mean": 0.0 if active_count <= 0 else sums.get("prob_active_sum", 0.0) / (active_count + eps),
        "prob_bg_mean": 0.0 if bg_count <= 0 else sums.get("prob_bg_sum", 0.0) / (bg_count + eps),
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
    total_parts: dict[str, float] = {}
    total_metric_sums: dict[str, float] = {}
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
            prob_logits, correction = model(x)

        prob_logits = prob_logits.float()
        prob = torch.sigmoid(prob_logits)
        correction = correction.float()
        loss, parts = compute_loss(prob_logits, correction, p_target, c_target, cfg)
        metric_sums = compute_metric_sums(prob, correction, p_target, c_target, cfg)
        with torch.no_grad():
            active = p_target > 0.5
            background = ~active
            prob_active_mean = (
                prob[active].mean()
                if active.any()
                else prob.new_tensor(0.0)
            )
            prob_bg_mean = (
                prob[background].mean()
                if background.any()
                else prob.new_tensor(0.0)
            )

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
            if key in {"max_error", "bg_max"}:
                total_metric_sums[key] = max(total_metric_sums.get(key, 0.0), value)
            else:
                total_metric_sums[key] = total_metric_sums.get(key, 0.0) + value
        n_batches += 1

        if (
            cfg.progress_print_every > 0
            and (batch_idx % cfg.progress_print_every == 0 or batch_idx == total_batches)
        ):
            phase = "train" if is_train else "val"
            print(
                f"{phase} batch {batch_idx}/{total_batches} "
                f"loss={total_loss / max(n_batches, 1):.5f} "
                f"focal={parts['focal']:.5f} "
                f"dice_loss={parts['dice_loss']:.5f} "
                f"active={parts['l1_active']:.5f} "
                f"bg={parts['l1_bg']:.5f} "
                f"global={parts['l1_global']:.5f} "
                f"grad={parts['gradient']:.5f} "
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler,
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
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": asdict(cfg),
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

        prob_logits, correction = model(x)
        prob = torch.sigmoid(prob_logits)
        restored_direct = torch.clamp(x[:, -1:, :, :] - correction, 0.0, 1.0)
        restored_gated = torch.clamp(
            x[:, -1:, :, :] - (prob > 0.5).float() * correction,
            0.0,
            1.0,
        )
        abs_error = torch.abs(correction - c_target)

        batch_size = x.shape[0]
        for i in range(batch_size):
            if saved >= cfg.vis_num_samples:
                return

            panels = [
                add_label(to_u8_image(x[i, 0].cpu().numpy()), "input first"),
                add_label(to_u8_image(x[i, -1].cpu().numpy()), "input last"),
                add_label(to_u8_image(clean[i, -1].cpu().numpy()), "target clean"),
                add_label(to_u8_image(restored_direct[i, 0].cpu().numpy()), "restored direct"),
                add_label(to_u8_image(restored_gated[i, 0].cpu().numpy()), "restored gated"),
                add_label(to_u8_image(prob[i, 0].cpu().numpy()), "pred P"),
                add_label(to_u8_image(p_target[i, 0].cpu().numpy()), "target mask"),
                add_label(to_u8_image(correction[i, 0].cpu().numpy()), "pred C"),
                add_label(to_u8_image(c_target[i, 0].cpu().numpy()), "target C"),
                add_label(to_u8_image(abs_error[i, 0].cpu().numpy()), "abs error"),
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
    resume_data = None
    if cfg.resume_checkpoint is not None:
        resume_data = torch.load(cfg.resume_checkpoint, map_location="cpu")
        apply_saved_config_for_resume(cfg, resume_data.get("config", {}))

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
        mask_threshold=cfg.mask_threshold,
        aggregation=cfg.aggregation,
        validate_samples=cfg.validate_samples_on_init,
    )
    val_dataset = BurnRecoveryDataset(
        root=cfg.val_root,
        image_size=cfg.image_size,
        temporal_scale_range=(1.0, 1.0),
        generate_missing_burn=cfg.val_generate_missing_burn,
        max_samples=None,
        no_burn_probability=cfg.val_no_burn_probability,
        no_burn_seed=cfg.val_subset_seed,
        deterministic_no_burn=True,
        mask_threshold=cfg.mask_threshold,
        aggregation=cfg.aggregation,
        validate_samples=cfg.validate_samples_on_init,
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
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.use_scheduler
        else None
    )
    start_epoch = 1
    best_val = float("inf")

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
    print(f"image_size: {cfg.image_size}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"amp: {bool(cfg.use_amp and device.type == 'cuda')}")
    print(f"train no-burn probability: {cfg.train_no_burn_probability}")
    print(f"val no-burn probability: {cfg.val_no_burn_probability}")
    print(f"mask threshold: {cfg.mask_threshold:.6f}")
    print(f"aggregation: {cfg.aggregation}")
    print(f"grad clip norm: {cfg.grad_clip_norm}")
    print(f"scheduler: {scheduler.__class__.__name__ if scheduler is not None else 'None'}")

    writer = None
    if cfg.use_tensorboard:
        if SummaryWriter is None:
            print("TensorBoard is not available; scalar logging is disabled.")
        else:
            writer = SummaryWriter(log_dir=str(cfg.output_dir / "runs"))
            print(f"tensorboard log dir: {cfg.output_dir / 'runs'}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        start = time.time()
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, cfg, scaler)
        val_metrics = run_one_epoch(model, val_loader, None, device, cfg)
        elapsed = time.time() - start

        print(
            f"epoch {epoch:03d}/{cfg.epochs} "
            f"time={elapsed:.1f}s "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_dice={val_metrics['dice']:.5f} "
            f"val_iou={val_metrics['iou']:.5f} "
            f"val_active_mae={val_metrics['active_mae']:.5f} "
            f"val_bg_mae={val_metrics['bg_mae']:.5f}"
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
            save_checkpoint(
                cfg.output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scaler,
                scheduler,
                epoch,
                best_val,
                cfg,
            )

        if is_best:
            save_checkpoint(
                cfg.output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                scheduler,
                epoch,
                best_val,
                cfg,
            )
            print(f"  saved best.pt, best_val={best_val:.5f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
