#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从PNG/JPG训练样本重建灼烧输入，执行推理并显示预测与真值。"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from burn_recovery_net import BurnRecoveryNet


# ============================ 可调参数 ============================
INPUT_DIR = Path(
    r"C:\Users\17874\Documents\python\datasets\burn_recovery\val\sample_001005"
)
CHECKPOINT_PATH = Path(
    r"C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt"
)

# 留空字符串表示不保存；填写目录时保存12张单图、总览图和预测数组。
OUTPUT_DIR = r""

MASK_THRESHOLD = 0.5
DEVICE = "auto"  # "auto"、"cuda"或"cpu"
SHOW_WINDOW = True
FIGURE_SIZE = (16, 12)
FIGURE_DPI = 150

# 仅影响展示和保存的第1/5帧输入、纠正图，不改变网络实际输入。
ENABLE_HISTOGRAM_STRETCH = True
STRETCH_LOW_PERCENTILE = 1.0
STRETCH_HIGH_PERCENTILE = 99.0

# 合成灼烧时5帧强度的随机范围，需与训练数据生成脚本保持一致。
TEMPORAL_SCALE_RANGE = (1, 1)
# ================================================================

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.is_file():
        return metadata
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def find_clean_frames(sample_dir: Path) -> list[Path]:
    expected_stems = {f"{index:03d}" for index in range(5)}
    paths = sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file()
        and path.stem in expected_stems
        and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) != 5:
        raise RuntimeError(
            f"{sample_dir} 中应包含000到004共5张图，实际找到{len(paths)}张"
        )
    return paths


def load_clean_frames(paths: list[Path]) -> np.ndarray:
    frames = []
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"无法读取图像：{path}")
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(
                f"输入尺寸不一致：{path.name}为{image.shape}，期望{expected_shape}"
            )
        frames.append(image)
    return np.stack(frames).astype(np.float32) / 255.0


def get_temporal_scales(metadata: dict[str, str]) -> np.ndarray:
    if metadata.get("temporal_scales"):
        scales = np.asarray(
            [float(value) for value in metadata["temporal_scales"].split(",")],
            dtype=np.float32,
        )
        if scales.size != 5:
            raise ValueError("元数据中的temporal_scales必须包含5个值")
        return scales
    if "seed" not in metadata:
        raise ValueError("元数据缺少temporal_scales和seed，无法重建灼烧输入")
    rng = np.random.default_rng(int(metadata["seed"]) + 1)
    return rng.uniform(*TEMPORAL_SCALE_RANGE, size=5).astype(np.float32)


def build_burned_input(
    sample_dir: Path,
    clean: np.ndarray,
    metadata: dict[str, str],
    target_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace_name = metadata.get("burn_trace_npy", "synthetic_burn_trace.npy")
    trace_path = sample_dir / trace_name
    if not trace_path.is_file():
        raise FileNotFoundError(f"找不到灼烧真值：{trace_path}")

    trace = np.load(trace_path).astype(np.float32)
    if trace.shape != clean.shape[1:] or not np.isfinite(trace).all():
        raise ValueError(
            f"灼烧真值尺寸或数值无效：trace={trace.shape}，图像={clean.shape[1:]}"
        )

    active_threshold = float(metadata.get("burn_active_threshold", "0.5"))
    trace = np.where(trace > active_threshold, trace, 0.0) / 255.0
    scales = get_temporal_scales(metadata)[:, None, None]
    requested = trace[None, :, :] * scales
    effective = np.minimum(requested, 1.0 - clean)
    effective = np.maximum(effective, 0.0).astype(np.float32)
    burned = np.clip(clean + effective, 0.0, 1.0).astype(np.float32)

    gt_correction = np.median(effective, axis=0).astype(np.float32)
    gt_correction = np.where(
        gt_correction >= target_threshold, gt_correction, 0.0
    ).astype(np.float32)
    gt_mask = gt_correction > 0.0
    return burned, gt_mask, gt_correction


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch检测不到可用CUDA设备")
    return torch.device(name)


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[BurnRecoveryNet, dict]:
    # checkpoint来自本项目本地训练，config中含Path，需关闭weights_only模式。
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"checkpoint格式错误，缺少model字段：{checkpoint_path}")
    config = checkpoint.get("config", {})
    model = BurnRecoveryNet(
        in_frames=5, base_channels=int(config.get("base_channels", 32))
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def histogram_stretch(image: np.ndarray) -> np.ndarray:
    low = float(np.percentile(image, STRETCH_LOW_PERCENTILE))
    high = float(np.percentile(image, STRETCH_HIGH_PERCENTILE))
    if high <= low:
        return image.copy()
    stretched = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def prepare_display_frame(image: np.ndarray) -> np.ndarray:
    image_u8 = to_u8(image)
    if ENABLE_HISTOGRAM_STRETCH:
        return histogram_stretch(image_u8)
    return image_u8


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"写入失败：{path}")


def main() -> None:
    sample_dir = INPUT_DIR.expanduser().resolve()
    checkpoint_path = CHECKPOINT_PATH.expanduser().resolve()
    output_dir = (
        Path(OUTPUT_DIR).expanduser().resolve()
        if OUTPUT_DIR.strip()
        else None
    )

    if not sample_dir.is_dir():
        raise FileNotFoundError(f"训练样本文件夹不存在：{sample_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型权重不存在：{checkpoint_path}")
    if not 0.0 <= MASK_THRESHOLD <= 1.0:
        raise ValueError("MASK_THRESHOLD必须在0到1之间")
    if DEVICE not in {"auto", "cuda", "cpu"}:
        raise ValueError('DEVICE必须为"auto"、"cuda"或"cpu"')
    if not 0.0 <= STRETCH_LOW_PERCENTILE < STRETCH_HIGH_PERCENTILE <= 100.0:
        raise ValueError(
            "拉伸百分位必须满足0 <= STRETCH_LOW_PERCENTILE "
            "< STRETCH_HIGH_PERCENTILE <= 100"
        )

    device = select_device(DEVICE)
    model, config = load_model(checkpoint_path, device)
    target_threshold = float(config.get("mask_threshold", 2.0 / 255.0))

    clean = load_clean_frames(find_clean_frames(sample_dir))
    metadata = read_metadata(sample_dir / "synthetic_burn_metadata.txt")
    burned, gt_mask, gt_correction = build_burned_input(
        sample_dir, clean, metadata, target_threshold
    )

    input_tensor = torch.from_numpy(burned).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits, correction_tensor = model(input_tensor)
        probability_tensor = torch.sigmoid(logits)

    probability = probability_tensor[0, 0].cpu().numpy()
    pred_correction = correction_tensor[0, 0].cpu().numpy()
    pred_mask = probability >= MASK_THRESHOLD
    gated_correction = pred_correction * pred_mask.astype(np.float32)
    restored_first = np.clip(burned[0] - gated_correction, 0.0, 1.0)
    restored_last = np.clip(burned[-1] - gated_correction, 0.0, 1.0)
    mask_error = np.abs(
        gt_mask.astype(np.float32) - pred_mask.astype(np.float32)
    )
    correction_error = np.abs(gt_correction - pred_correction)

    images = [
        prepare_display_frame(burned[0]),
        prepare_display_frame(burned[-1]),
        pred_mask.astype(np.uint8) * 255,
        to_u8(pred_correction),
        prepare_display_frame(restored_first),
        prepare_display_frame(restored_last),
        gt_mask.astype(np.uint8) * 255,
        to_u8(gt_correction),
        prepare_display_frame(clean[0]),
        prepare_display_frame(clean[-1]),
        to_u8(mask_error),
        to_u8(correction_error),
    ]
    names = [
        "burned_frame_001",
        "burned_frame_005",
        "predicted_mask",
        "predicted_correction",
        "restored_frame_001",
        "restored_frame_005",
        "ground_truth_mask",
        "ground_truth_correction",
        "clean_frame_001",
        "clean_frame_005",
        "mask_absolute_error",
        "correction_absolute_error",
    ]
    titles = [
        "Burned frame 1",
        "Burned frame 5",
        f"Predicted mask (P >= {MASK_THRESHOLD:g})",
        "Predicted correction",
        "Restored frame 1",
        "Restored frame 5",
        "Ground-truth mask",
        "Ground-truth correction",
        "Clean frame 1",
        "Clean frame 5",
        "Mask absolute error",
        "Correction absolute error",
    ]

    plt.figure(figsize=FIGURE_SIZE)
    for index, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(3, 4, index)
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, image in zip(names, images):
            save_image(output_dir / f"{name}.png", image)
        np.save(
            output_dir / "predicted_probability.npy",
            probability.astype(np.float32),
        )
        np.save(
            output_dir / "predicted_correction.npy",
            pred_correction.astype(np.float32),
        )
        plt.savefig(
            output_dir / "inference_visualization.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        print(f"结果已保存：{output_dir}")
    else:
        print("输出目录为空：本次仅显示，不保存图片")

    print(f"device: {device}")
    print(f"sample: {sample_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"histogram stretch: {ENABLE_HISTOGRAM_STRETCH}")
    print(f"predicted mask pixels: {int(pred_mask.sum())}")
    print(f"ground-truth mask pixels: {int(gt_mask.sum())}")
    print(
        f"predicted correction max: "
        f"{float(pred_correction.max() * 255.0):.3f} gray levels"
    )

    if SHOW_WINDOW:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
