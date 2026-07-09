#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将真实灼烧场景的5张连续PNG送入网络并显示恢复结果。"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from burn_recovery_net import BurnRecoveryNet


# ============================ 可调参数 ============================
INPUT_DIR = Path(
    r"C:\Users\17874\Documents\python\datasets\burn_recovery\灼烧真实场景测试\test3"
)
CHECKPOINT_PATH = Path(
    r"C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt"
)

# 留空表示不保存；填写目录时保存6张单图、总览图和预测浮点数组。
OUTPUT_DIR = r""

MASK_THRESHOLD = 0.5
DEVICE = "auto"  # "auto"、"cuda"或"cpu"
SHOW_WINDOW = True
FIGURE_SIZE = (15, 8)
FIGURE_DPI = 150

# 仅改变显示和保存效果，不改变送入网络的数据。
ENABLE_HISTOGRAM_STRETCH = True
STRETCH_LOW_PERCENTILE = 1.0
STRETCH_HIGH_PERCENTILE = 99.0
# ================================================================

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError(f"图像文件名必须是数字：{path.name}") from error


def find_frames(input_dir: Path) -> list[Path]:
    paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    paths.sort(key=numeric_stem)
    if len(paths) != 5:
        raise RuntimeError(
            f"{input_dir} 中应恰好包含5张连续帧，实际找到{len(paths)}张"
        )
    return paths


def load_frames(paths: list[Path]) -> np.ndarray:
    frames = []
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"无法读取图像：{path}")
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(
                f"5张图尺寸必须一致：{path.name}为{image.shape}，"
                f"期望{expected_shape}"
            )
        frames.append(image)
    return np.stack(frames).astype(np.float32) / 255.0


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch检测不到可用CUDA设备")
    return torch.device(name)


def load_model(checkpoint_path: Path, device: torch.device) -> BurnRecoveryNet:
    # checkpoint由本项目生成，config中含Path，需显式关闭weights_only模式。
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"checkpoint格式错误，缺少model字段：{checkpoint_path}")
    config = checkpoint.get("config", {})
    model = BurnRecoveryNet(
        in_frames=5,
        base_channels=int(config.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


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
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"写入失败：{path}")
    encoded.tofile(path)


def main() -> None:
    input_dir = INPUT_DIR.expanduser().resolve()
    checkpoint_path = CHECKPOINT_PATH.expanduser().resolve()
    output_dir = (
        Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR.strip() else None
    )

    if not input_dir.is_dir():
        raise FileNotFoundError(f"真实灼烧输入目录不存在：{input_dir}")
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

    frame_paths = find_frames(input_dir)
    frames = load_frames(frame_paths)
    device = select_device(DEVICE)
    model = load_model(checkpoint_path, device)

    input_tensor = torch.from_numpy(frames).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits, correction_tensor = model(input_tensor)
        probability_tensor = torch.sigmoid(logits)

    probability = probability_tensor[0, 0].cpu().numpy()
    correction = correction_tensor[0, 0].cpu().numpy()
    mask = probability >= MASK_THRESHOLD
    gated_correction = correction * mask.astype(np.float32)
    restored_first = np.clip(frames[0] - gated_correction, 0.0, 1.0)
    restored_last = np.clip(frames[-1] - gated_correction, 0.0, 1.0)

    images = [
        prepare_display_frame(frames[0]),
        prepare_display_frame(frames[-1]),
        mask.astype(np.uint8) * 255,
        to_u8(correction),
        prepare_display_frame(restored_first),
        prepare_display_frame(restored_last),
    ]
    names = [
        "input_frame_001",
        "input_frame_005",
        "predicted_mask",
        "predicted_correction",
        "restored_frame_001",
        "restored_frame_005",
    ]
    titles = [
        "Real burned frame 1",
        "Real burned frame 5",
        f"Predicted mask (P >= {MASK_THRESHOLD:g})",
        "Predicted correction",
        "Restored frame 1",
        "Restored frame 5",
    ]

    plt.figure(figsize=FIGURE_SIZE)
    for index, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(2, 3, index)
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
            correction.astype(np.float32),
        )
        plt.savefig(
            output_dir / "real_burn_inference_visualization.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        print(f"结果已保存：{output_dir}")
    else:
        print("输出目录为空：本次仅显示，不保存图片")

    print(f"device: {device}")
    print(f"input: {input_dir}")
    print(f"frames: {', '.join(path.name for path in frame_paths)}")
    print(f"histogram stretch: {ENABLE_HISTOGRAM_STRETCH}")
    print(f"predicted mask pixels: {int(mask.sum())} / {mask.size}")
    print(
        f"predicted correction max: "
        f"{float(correction.max() * 255.0):.3f} gray levels"
    )

    if SHOW_WINDOW:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
