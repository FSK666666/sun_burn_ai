#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量预览训练集文件夹中的 5 帧红外图像，并随机叠加灼烧痕迹。

目录结构示例：
TRAIN_ROOT/
  sample_000001/
    000.jpg
    001.jpg
    002.jpg
    003.jpg
    004.jpg
  sample_000002/
    ...

功能：
1. 读取训练集根目录 TRAIN_ROOT；
2. 自动遍历其中的样本子文件夹；
3. 每个样本子文件夹读取 5 帧图像；
4. 对同一组 5 帧叠加同一位置、同一形状的随机灼烧痕迹；
5. 每帧灼烧强度允许小幅随机波动；
6. 使用 2x2 图窗显示灼烧后第一帧、灼烧后最后一帧、帧间差和灼烧图；
7. 按任意键或鼠标点击显示下一组，关闭图窗结束；
8. 自动写入模式支持多线程并行生成和保存；
9. 可选择保存合成结果。

注意：
- 当前代码按 8-bit 灰度 JPG/PNG 处理；
- 每次重新运行都会生成不同的随机灼烧；
- 同一组 5 帧中的灼烧位置和形状固定。
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


def configure_matplotlib_chinese_font():
    """配置 Matplotlib 中文字体，避免中文标题显示乱码。"""
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]

    for font_path in font_candidates:
        if Path(font_path).is_file():
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    return None


configure_matplotlib_chinese_font()


# ============================================================
# 用户配置区
# ============================================================

# 总训练集目录：该目录下每个子文件夹是一组 5 帧
TRAIN_ROOT = r"C:\Users\17874\Documents\python\datasets\burn_recovery\train"

# 是否递归查找所有包含图片的子文件夹
RECURSIVE_SEARCH = True

# 是否随机打乱样本显示顺序
SHUFFLE_GROUPS = True

# 每组期望帧数
EXPECTED_FRAMES = 5

# 图片扩展名
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 是否保存合成结果
SAVE_OUTPUT = True

# 保存根目录
OUTPUT_ROOT = r"./synthetic_burn_preview"

# 是否保存原始 5 帧
SAVE_CLEAN_FRAMES = False

# 自动写入模式：True 时不显示图窗，直接为每个样本生成并保存灼烧信息
AUTO_WRITE_BURN_ARTIFACTS = True

# 自动写入最多处理多少个样本；None 表示处理全部
AUTO_WRITE_MAX_GROUPS = None

# 是否启用多线程。多线程只用于自动写入模式；Matplotlib 交互预览仍在主线程执行。
ENABLE_MULTITHREADING = True

# 最大线程数。None 表示自动设置为 min(32, CPU核心数+4)。
# 机械硬盘建议 4~8；SSD 可尝试 8~16。
MAX_WORKERS = None

# 启用 Python 多线程时，关闭 OpenCV 自身线程，避免线程过度嵌套。
DISABLE_OPENCV_INTERNAL_THREADS = True

# 灼烧尺寸控制
# 256x192 可先用 0.5；512x640 可先用 1.0
# Burn mark size controls. Increase BURN_SIZE_MULTIPLIER to enlarge traces.
# Examples: 0.7 smaller, 1.0 baseline, 1.5 larger, 2.0 much larger.
BURN_BASE_SIZE_SCALE = 1.5
BURN_SIZE_MULTIPLIER = 1.0
BURN_BLUR_WITH_SIZE = False
BURN_DISC_RADIUS_MULTIPLIER = 0.5
BURN_DISC_OVERLAP_RATIO = 0.8
BURN_HORIZONTAL_BIAS = True
BURN_MAX_STRIPE_ANGLE_DEG = 12
BURN_DISC_ROUNDNESS_JITTER = 0.12
BURN_DWELL_INTENSITY_VARIATION = 0.35

# 灼烧图生成参数
EDGE_FALLOFF = 22
MIN_EDGE_WEIGHT = 0.25
BLUR_SIGMA = 0.8
BURN_CLUSTER_EDGE_BLUR_MULTIPLIER = 0.25
BURN_STRIPE_EDGE_BLUR_MULTIPLIER = 1.0
BURN_STRIPE_DISC_EDGE_SOFTNESS = 0.32

# mask 阈值：偏置大于该值的像素标记为灼烧区域
MASK_THRESHOLD = 8.0
BURN_ACTIVE_THRESHOLD = 0.5

# 5 帧之间的强度波动范围
TEMPORAL_SCALE_RANGE = (0.92, 1.08)

# 是否优先把灼烧放在图像上方
PREFER_TOP_REGION = True

# 是否允许自动组合多个灼烧类型
ALLOW_MULTI_PATTERN = True


# ============================================================
# 基础工具
# ============================================================

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_group_dirs(root: Path) -> list[Path]:
    """
批量预览训练集文件夹中的 5 帧红外图像，并随机叠加灼烧痕迹。

目录结构示例：
TRAIN_ROOT/
  sample_000001/
    000.jpg
    001.jpg
    002.jpg
    003.jpg
    004.jpg
  sample_000002/
    ...

功能：
1. 读取训练集根目录 TRAIN_ROOT；
2. 自动遍历其中的样本子文件夹；
3. 每个样本子文件夹读取 5 帧图像；
4. 对同一组 5 帧叠加同一位置、同一形状的随机灼烧痕迹；
5. 每帧灼烧强度允许小幅随机波动；
6. 使用 2x2 图窗显示灼烧后第一帧、灼烧后最后一帧、帧间差和灼烧图；
7. 按任意键或鼠标点击显示下一组，关闭图窗结束；
8. 可选择保存合成结果。

注意：
- 当前代码按 8-bit 灰度 JPG/PNG 处理；
- 每次重新运行都会生成不同的随机灼烧；
- 同一组 5 帧中的灼烧位置和形状固定。
"""
    if RECURSIVE_SEARCH:
        candidates = [p for p in root.rglob("*") if p.is_dir()]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()]

    group_dirs = []

    for folder in candidates:
        image_files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if image_files:
            group_dirs.append(folder)

    return sorted(group_dirs, key=natural_key)


def load_group_images(group_dir: Path) -> tuple[list[Path], list[np.ndarray]]:
    """
批量预览训练集文件夹中的 5 帧红外图像，并随机叠加灼烧痕迹。

目录结构示例：
TRAIN_ROOT/
  sample_000001/
    000.jpg
    001.jpg
    002.jpg
    003.jpg
    004.jpg
  sample_000002/
    ...

功能：
1. 读取训练集根目录 TRAIN_ROOT；
2. 自动遍历其中的样本子文件夹；
3. 每个样本子文件夹读取 5 帧图像；
4. 对同一组 5 帧叠加同一位置、同一形状的随机灼烧痕迹；
5. 每帧灼烧强度允许小幅随机波动；
6. 使用 2x2 图窗显示灼烧后第一帧、灼烧后最后一帧、帧间差和灼烧图；
7. 按任意键或鼠标点击显示下一组，关闭图窗结束；
8. 可选择保存合成结果。

注意：
- 当前代码按 8-bit 灰度 JPG/PNG 处理；
- 每次重新运行都会生成不同的随机灼烧；
- 同一组 5 帧中的灼烧位置和形状固定。
"""
    paths = sorted(
        [
            p for p in group_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=natural_key,
    )

    if len(paths) < EXPECTED_FRAMES:
        raise ValueError(
            f"{group_dir} has only {len(paths)} images; "
            f"expected at least {EXPECTED_FRAMES}."
        )

    # 若超过 5 张，只取前 5 张
    paths = paths[:EXPECTED_FRAMES]

    images = []
    shape = None

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            raise ValueError(
                f"Image shape mismatch in {group_dir}: "
                f"{path.name} has shape {img.shape}, expected {shape}."
            )

        images.append(img)

    return paths, images


def _distance_to_edge_map(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum.reduce([
        xx,
        yy,
        w - 1 - xx,
        h - 1 - yy
    ]).astype(np.float32)
    return dist


def _edge_weight_map(h, w, edge_falloff=25, min_weight=0.25):
    dist = _distance_to_edge_map(h, w)
    weight = 1.0 - np.exp(-dist / edge_falloff)
    weight = min_weight + (1.0 - min_weight) * weight
    return weight.astype(np.float32)


def get_burn_size_scale() -> float:
    return float(BURN_BASE_SIZE_SCALE) * float(BURN_SIZE_MULTIPLIER)


def get_effective_blur_sigma(pattern_type: str = "stripe") -> float:
    if pattern_type == "cluster":
        multiplier = BURN_CLUSTER_EDGE_BLUR_MULTIPLIER
    else:
        multiplier = BURN_STRIPE_EDGE_BLUR_MULTIPLIER

    blur_sigma = float(BLUR_SIGMA) * float(multiplier)
    if BURN_BLUR_WITH_SIZE:
        blur_sigma *= max(float(BURN_SIZE_MULTIPLIER), 0.1)
    return blur_sigma


def set_burn_size(multiplier: float, base_scale: float | None = None) -> float:
    """交互式调整灼烧痕迹尺寸，返回实际尺寸系数。"""
    global BURN_BASE_SIZE_SCALE, BURN_SIZE_MULTIPLIER

    if base_scale is not None:
        BURN_BASE_SIZE_SCALE = float(base_scale)

    BURN_SIZE_MULTIPLIER = float(multiplier)
    return get_burn_size_scale()

def _gaussian_blob(h, w, cx, cy, sigma_x, sigma_y, amplitude=1.0):
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(
        -(
            ((xx - cx) ** 2) / (2 * sigma_x ** 2)
            + ((yy - cy) ** 2) / (2 * sigma_y ** 2)
        )
    ).astype(np.float32)
    return amplitude * g


def _soft_disc_blob(
    h,
    w,
    cx,
    cy,
    radius_x,
    radius_y,
    amplitude=1.0,
    edge_softness=0.35,
):
    yy, xx = np.mgrid[0:h, 0:w]
    radius_x = max(float(radius_x), 1.0)
    radius_y = max(float(radius_y), 1.0)
    norm_dist = np.sqrt(
        ((xx - cx) / radius_x) ** 2
        + ((yy - cy) / radius_y) ** 2
    ).astype(np.float32)
    disc = np.ones((h, w), dtype=np.float32)
    outside = norm_dist > 1.0
    disc[outside] = np.exp(
        -((norm_dist[outside] - 1.0) ** 2)
        / (2 * edge_softness ** 2)
    )
    return amplitude * disc


def _sample_near_top_region(rng, h, w):
    margin_x = max(4, min(20, w // 10))
    margin_y = max(4, min(10, h // 10))

    x_low = margin_x
    x_high = max(x_low + 1, w - margin_x)

    if PREFER_TOP_REGION:
        y_low = margin_y
        y_high = max(y_low + 1, max(20, h // 4))
    else:
        y_low = margin_y
        y_high = max(y_low + 1, h - margin_y)

    x = rng.integers(x_low, x_high)
    y = rng.integers(y_low, y_high)

    return float(x), float(y)


def _sample_stripe_angle(rng, jitter_deg=None):
    if jitter_deg is None:
        jitter_deg = BURN_MAX_STRIPE_ANGLE_DEG

    if not BURN_HORIZONTAL_BIAS:
        return rng.uniform(-1.0, 1.0)

    direction = rng.choice([-1.0, 1.0])
    angle = np.deg2rad(rng.uniform(-jitter_deg, jitter_deg))
    return angle if direction > 0 else np.pi + angle


# ============================================================
# 灼烧形态生成
# ============================================================

def _generate_cluster_pattern(h, w, rng, size_scale):
    burn = np.zeros((h, w), dtype=np.float32)

    cx, cy = _sample_near_top_region(rng, h, w)
    n_pts = rng.integers(4, 10)

    for k in range(n_pts):
        if k == 0:
            px = cx + rng.normal(0, 3 * size_scale)
            py = cy + rng.normal(0, 3 * size_scale)
            amp = rng.uniform(43, 50)
            sx = rng.uniform(2.0, 5.0) * size_scale
            sy = rng.uniform(2.0, 5.0) * size_scale
        else:
            px = cx + rng.normal(0, 18 * size_scale)
            py = cy + rng.normal(0, 12 * size_scale)
            amp = rng.uniform(16, 43)
            sx = rng.uniform(2.0, 4.0) * size_scale
            sy = rng.uniform(2.0, 4.0) * size_scale

        burn += _gaussian_blob(h, w, px, py, sx, sy, amp)

    return burn


def _generate_broken_chain_pattern(h, w, rng, size_scale):
    burn = np.zeros((h, w), dtype=np.float32)

    start_x, start_y = _sample_near_top_region(rng, h, w)
    n_ctrl = rng.integers(3, 5)
    ctrl_pts = [(start_x, start_y)]

    angle = _sample_stripe_angle(rng)
    step_len = rng.uniform(30, 55) * size_scale

    for _ in range(n_ctrl - 1):
        last_x, last_y = ctrl_pts[-1]
        if BURN_HORIZONTAL_BIAS:
            angle = _sample_stripe_angle(rng, jitter_deg=BURN_MAX_STRIPE_ANGLE_DEG)
        else:
            angle += rng.uniform(-0.6, 0.6)
        length = step_len * rng.uniform(0.8, 1.3)

        nx = np.clip(last_x + length * np.cos(angle), 3, w - 4)

        max_y = h // 3 if PREFER_TOP_REGION else h - 4
        ny = np.clip(last_y + length * np.sin(angle), 3, max_y)

        ctrl_pts.append((nx, ny))

    ctrl_pts = np.array(ctrl_pts, dtype=np.float32)

    seg_lens = np.sqrt(np.sum(np.diff(ctrl_pts, axis=0) ** 2, axis=1))
    total_len = np.sum(seg_lens)

    if total_len < 1e-6:
        return burn

    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])

    disc_radius = rng.uniform(5.5, 8.5) * size_scale * BURN_DISC_RADIUS_MULTIPLIER
    sample_spacing = max(1.0, disc_radius * BURN_DISC_OVERLAP_RATIO)
    n_samples = max(4, int(np.ceil(total_len / sample_spacing)) + 1)

    sample_pos = np.linspace(0, total_len, n_samples)

    sampled_pts = []

    for s in sample_pos:
        seg_idx = np.searchsorted(cum_len, s, side="right") - 1
        seg_idx = min(seg_idx, len(seg_lens) - 1)

        p0 = ctrl_pts[seg_idx]
        p1 = ctrl_pts[seg_idx + 1]

        t = (s - cum_len[seg_idx]) / (seg_lens[seg_idx] + 1e-8)
        sampled_pts.append(p0 * (1 - t) + p1 * t)

    sampled_pts = np.array(sampled_pts, dtype=np.float32)

    keep_prob = rng.uniform(0.72, 0.92)
    keep_mask = rng.random(len(sampled_pts)) < keep_prob

    if np.sum(keep_mask) < 2:
        indices = rng.choice(len(keep_mask), size=min(2, len(keep_mask)), replace=False)
        keep_mask[indices] = True

    for keep, (px, py) in zip(keep_mask, sampled_pts):
        if not keep:
            continue

        dwell = rng.uniform(
            1.0 - BURN_DWELL_INTENSITY_VARIATION,
            1.0 + BURN_DWELL_INTENSITY_VARIATION,
        )
        amp = rng.uniform(30, 48) * dwell
        roundness = rng.uniform(
            1.0 - BURN_DISC_ROUNDNESS_JITTER,
            1.0 + BURN_DISC_ROUNDNESS_JITTER,
        )
        radius_x = disc_radius * roundness
        radius_y = disc_radius / roundness
        disc = _soft_disc_blob(
            h,
            w,
            px,
            py,
            radius_x,
            radius_y,
            amp,
            edge_softness=BURN_STRIPE_DISC_EDGE_SOFTNESS,
        )
        burn = np.maximum(burn, disc)

    return burn


def _generate_short_polyline_pattern(h, w, rng, size_scale):
    burn = np.zeros((h, w), dtype=np.float32)

    start_x, start_y = _sample_near_top_region(rng, h, w)
    n_ctrl = rng.integers(3, 6)
    ctrl_pts = [(start_x, start_y)]

    angle = _sample_stripe_angle(rng)
    step_len = rng.uniform(18, 38) * size_scale

    for _ in range(n_ctrl - 1):
        last_x, last_y = ctrl_pts[-1]
        if BURN_HORIZONTAL_BIAS:
            angle = _sample_stripe_angle(rng, jitter_deg=BURN_MAX_STRIPE_ANGLE_DEG)
        else:
            angle += rng.uniform(-0.5, 0.5)
        length = step_len * rng.uniform(0.9, 1.2)

        nx = np.clip(last_x + length * np.cos(angle), 3, w - 4)

        max_y = h // 3 if PREFER_TOP_REGION else h - 4
        ny = np.clip(last_y + length * np.sin(angle), 3, max_y)

        ctrl_pts.append((nx, ny))

    ctrl_pts = np.array(ctrl_pts, dtype=np.float32)

    seg_lens = np.sqrt(np.sum(np.diff(ctrl_pts, axis=0) ** 2, axis=1))
    total_len = np.sum(seg_lens)

    if total_len < 1e-6:
        return burn

    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])

    disc_radius = rng.uniform(4.8, 7.5) * size_scale * BURN_DISC_RADIUS_MULTIPLIER
    sample_spacing = max(1.0, disc_radius * BURN_DISC_OVERLAP_RATIO)
    n_samples = max(5, int(np.ceil(total_len / sample_spacing)) + 1)

    sample_pos = np.linspace(0, total_len, n_samples)

    for s in sample_pos:
        seg_idx = np.searchsorted(cum_len, s, side="right") - 1
        seg_idx = min(seg_idx, len(seg_lens) - 1)

        p0 = ctrl_pts[seg_idx]
        p1 = ctrl_pts[seg_idx + 1]

        t = (s - cum_len[seg_idx]) / (seg_lens[seg_idx] + 1e-8)
        px, py = p0 * (1 - t) + p1 * t

        dwell = rng.uniform(
            1.0 - BURN_DWELL_INTENSITY_VARIATION,
            1.0 + BURN_DWELL_INTENSITY_VARIATION,
        )
        amp = rng.uniform(32, 48) * dwell
        roundness = rng.uniform(
            1.0 - BURN_DISC_ROUNDNESS_JITTER,
            1.0 + BURN_DISC_ROUNDNESS_JITTER,
        )
        radius_x = disc_radius * roundness
        radius_y = disc_radius / roundness
        disc = _soft_disc_blob(
            h,
            w,
            px,
            py,
            radius_x,
            radius_y,
            amp,
            edge_softness=BURN_STRIPE_DISC_EDGE_SOFTNESS,
        )
        burn = np.maximum(burn, disc)

    return burn


def generate_burn_pattern(
    h,
    w,
    pattern_type="auto",
    edge_falloff=22,
    min_edge_weight=0.25,
    blur_sigma=None,
    seed=None,
    size_scale=None,
):
    rng = np.random.default_rng(seed)
    if size_scale is None:
        size_scale = get_burn_size_scale()

    if pattern_type == "auto":
        if ALLOW_MULTI_PATTERN:
            pattern_types = np.array(["cluster", "broken_chain", "polyline"])
            n_patterns = int(rng.integers(1, len(pattern_types) + 1))
            selected_types = rng.choice(
                pattern_types,
                size=n_patterns,
                replace=False,
            )

            burn = np.zeros((h, w), dtype=np.float32)

            for selected_type in selected_types:
                burn += generate_burn_pattern(
                    h=h,
                    w=w,
                    pattern_type=str(selected_type),
                    edge_falloff=edge_falloff,
                    min_edge_weight=min_edge_weight,
                    blur_sigma=None,
                    seed=int(rng.integers(0, 10**9)),
                    size_scale=size_scale,
                )
            return burn.astype(np.float32)
        else:
            selected_type = rng.choice(
                ["cluster", "broken_chain", "polyline"]
            )
            return generate_burn_pattern(
                h=h,
                w=w,
                pattern_type=str(selected_type),
                edge_falloff=edge_falloff,
                min_edge_weight=min_edge_weight,
                blur_sigma=None,
                seed=int(rng.integers(0, 10**9)),
                size_scale=size_scale,
            )

    elif pattern_type == "cluster":
        burn = _generate_cluster_pattern(h, w, rng, size_scale)

    elif pattern_type == "broken_chain":
        burn = _generate_broken_chain_pattern(h, w, rng, size_scale)

    elif pattern_type == "polyline":
        burn = _generate_short_polyline_pattern(h, w, rng, size_scale)

    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

    if blur_sigma is None:
        if pattern_type == "cluster":
            blur_sigma = get_effective_blur_sigma("cluster")
        else:
            blur_sigma = get_effective_blur_sigma("stripe")

    burn *= _edge_weight_map(
        h,
        w,
        edge_falloff=edge_falloff,
        min_weight=min_edge_weight,
    )

    if blur_sigma > 0:
        burn = cv2.GaussianBlur(burn, (0, 0), blur_sigma)

    return burn.astype(np.float32)


# ============================================================
# 灼烧叠加与显示
# ============================================================

def apply_burn_to_group(
    clean_images: list[np.ndarray],
    seed: int,
):
    h, w = clean_images[0].shape
    size_scale = get_burn_size_scale()

    burn_map = generate_burn_pattern(
        h=h,
        w=w,
        pattern_type="auto",
        edge_falloff=EDGE_FALLOFF,
        min_edge_weight=MIN_EDGE_WEIGHT,
        blur_sigma=None,
        seed=seed,
        size_scale=size_scale,
    )
    burn_map = burn_map.astype(np.float32)
    active_mask = burn_map > BURN_ACTIVE_THRESHOLD
    burn_map = np.where(active_mask, burn_map, 0.0).astype(np.float32)

    rng = np.random.default_rng(seed + 1)

    temporal_scales = rng.uniform(
        TEMPORAL_SCALE_RANGE[0],
        TEMPORAL_SCALE_RANGE[1],
        size=len(clean_images),
    )

    synthetic_images = []

    for clean, scale in zip(clean_images, temporal_scales):
        synthetic = clean.astype(np.float32)
        synthetic[active_mask] = synthetic[active_mask] + burn_map[active_mask] * scale
        synthetic = np.clip(synthetic, 0, 255).astype(np.uint8)
        synthetic_images.append(synthetic)

    burn_mask = (burn_map >= MASK_THRESHOLD).astype(np.uint8)

    return synthetic_images, burn_map, burn_mask, temporal_scales


def show_group(
    group_dir: Path,
    image_paths: list[Path],
    clean_images: list[np.ndarray],
    synthetic_images: list[np.ndarray],
    burn_map: np.ndarray,
    burn_mask: np.ndarray,
    temporal_scales: np.ndarray,
    seed: int,
):
    plt.clf()
    fig = plt.figure(1, figsize=(10, 8))
    fig._burn_action = None

    first_index = 0
    last_index = EXPECTED_FRAMES - 1

    first_burned = synthetic_images[first_index]
    last_burned = synthetic_images[last_index]
    frame_diff = last_burned.astype(np.float32) - first_burned.astype(np.float32)
    diff_limit = max(float(np.max(np.abs(frame_diff))), 1.0)

    plt.subplot(2, 2, 1)
    plt.imshow(first_burned, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Burned first\n{image_paths[first_index].name}")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(last_burned, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Burned last\n{image_paths[last_index].name}")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(frame_diff, cmap="gray", vmin=-diff_limit, vmax=diff_limit)
    plt.title("Last - first")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(burn_map, cmap="gray", vmin=0)
    plt.title("Burn map")
    plt.axis("off")

    plt.suptitle(f"{group_dir.name} | seed={seed}", fontsize=12)
    plt.tight_layout(rect=(0, 0.14, 1, 0.95))

    reset_ax = plt.axes((0.22, 0.035, 0.23, 0.065))
    save_ax = plt.axes((0.55, 0.035, 0.25, 0.065))
    reset_button = Button(reset_ax, "Reset burn", color="0.88", hovercolor="0.78")
    save_button = Button(save_ax, "Save burn maps", color="0.88", hovercolor="0.78")

    def on_reset(_event):
        fig._burn_action = "reset"

    def on_save(_event):
        fig._burn_action = "accept"

    def on_key(event):
        if event.key == "r":
            fig._burn_action = "reset"
        elif event.key == "s":
            fig._burn_action = "accept"

    def on_click(event):
        return

    reset_button.on_clicked(on_reset)
    save_button.on_clicked(on_save)
    fig._burn_buttons = [reset_button, save_button]

    old_key_cid = getattr(fig, "_burn_key_cid", None)
    if old_key_cid is not None:
        fig.canvas.mpl_disconnect(old_key_cid)
    fig._burn_key_cid = fig.canvas.mpl_connect("key_press_event", on_key)

    old_click_cid = getattr(fig, "_burn_click_cid", None)
    if old_click_cid is not None:
        fig.canvas.mpl_disconnect(old_click_cid)
    fig._burn_click_cid = fig.canvas.mpl_connect("button_press_event", on_click)

    plt.draw()
    plt.pause(0.01)


def save_burn_artifacts_to_sample(
    group_dir: Path,
    burn_map: np.ndarray,
    seed: int,
) -> dict[str, Path]:
    burn_trace_npy = group_dir / "synthetic_burn_trace.npy"
    burn_trace_png = group_dir / "synthetic_burn_trace.png"
    burn_probability_png = group_dir / "synthetic_burn_probability.png"
    metadata_path = group_dir / "synthetic_burn_metadata.txt"

    burn_map_float = burn_map.astype(np.float32)
    np.save(burn_trace_npy, burn_map_float)

    max_value = float(burn_map_float.max())
    if max_value > 0:
        burn_trace_u8 = np.clip(burn_map_float / max_value * 255, 0, 255).astype(np.uint8)
    else:
        burn_trace_u8 = np.zeros_like(burn_map_float, dtype=np.uint8)

    burn_probability = (burn_map_float > 0).astype(np.uint8) * 255

    cv2.imwrite(str(burn_trace_png), burn_trace_u8)
    cv2.imwrite(str(burn_probability_png), burn_probability)

    metadata = [
        f"seed={seed}",
        f"burn_trace_npy={burn_trace_npy.name}",
        f"burn_trace_png={burn_trace_png.name}",
        f"burn_probability_png={burn_probability_png.name}",
        "probability_rule=truncated_burn_intensity_gt_0",
        f"burn_active_threshold={BURN_ACTIVE_THRESHOLD}",
        f"burn_map_min={burn_map_float.min():.6f}",
        f"burn_map_max={burn_map_float.max():.6f}",
    ]
    metadata_path.write_text("\n".join(metadata), encoding="utf-8")

    return {
        "trace_npy": burn_trace_npy,
        "trace_png": burn_trace_png,
        "probability_png": burn_probability_png,
        "metadata": metadata_path,
    }


def _process_one_group_auto_write(
    group_index: int,
    total: int,
    group_dir: Path,
    seed: int,
) -> dict:
    """单个样本组的读取、灼烧生成和保存任务。"""
    _image_paths, clean_images = load_group_images(group_dir)
    _synthetic_images, burn_map, _burn_mask, _temporal_scales = (
        apply_burn_to_group(clean_images, seed)
    )
    saved_paths = save_burn_artifacts_to_sample(
        group_dir=group_dir,
        burn_map=burn_map,
        seed=seed,
    )

    return {
        "group_index": group_index,
        "total": total,
        "group_dir": group_dir,
        "seed": seed,
        "saved_paths": saved_paths,
    }


def _resolve_max_workers() -> int:
    """解析线程数配置。"""
    if MAX_WORKERS is not None:
        workers = int(MAX_WORKERS)
        if workers < 1:
            raise ValueError("MAX_WORKERS must be >= 1 or None.")
        return workers

    cpu_count = os.cpu_count() or 1
    return min(32, cpu_count + 4)


def auto_write_burn_artifacts(
    group_dirs: list[Path],
    rng: np.random.Generator,
    max_groups: int | None = None,
):
    """
    自动生成并保存所有样本的灼烧信息。

    多线程模式下：
    - 主线程预先为每个样本分配唯一随机 seed；
    - 工作线程并行读取图像、生成灼烧和写文件；
    - 每个样本只写入自己的目录，不会发生文件冲突。
    """
    total = len(group_dirs) if max_groups is None else min(len(group_dirs), max_groups)
    selected_groups = group_dirs[:total]

    if total == 0:
        print("No groups selected for auto write.")
        return

    # 预先生成 seed，避免多个线程共享同一个随机数生成器。
    seeds = rng.integers(0, 10**9, size=total, dtype=np.int64)

    written = 0
    failed = 0

    if not ENABLE_MULTITHREADING or total == 1:
        print("Auto write mode: single-threaded.")

        for group_index, (group_dir, seed_value) in enumerate(
            zip(selected_groups, seeds),
            start=1,
        ):
            try:
                result = _process_one_group_auto_write(
                    group_index=group_index,
                    total=total,
                    group_dir=group_dir,
                    seed=int(seed_value),
                )
                saved_paths = result["saved_paths"]
                written += 1
                print(
                    f"[{group_index}/{total}] saved {group_dir.name}: "
                    f"{saved_paths['trace_npy'].name}, "
                    f"{saved_paths['probability_png'].name}"
                )
            except Exception as exc:
                failed += 1
                print(
                    f"[{group_index}/{total}] failed {group_dir}: "
                    f"{type(exc).__name__}: {exc}"
                )

        print(f"Auto write done: written={written}, failed={failed}, total={total}")
        return

    max_workers = min(_resolve_max_workers(), total)
    print(f"Auto write mode: multithreaded, max_workers={max_workers}.")

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="burn-writer",
    ) as executor:
        future_to_info = {}

        for group_index, (group_dir, seed_value) in enumerate(
            zip(selected_groups, seeds),
            start=1,
        ):
            future = executor.submit(
                _process_one_group_auto_write,
                group_index,
                total,
                group_dir,
                int(seed_value),
            )
            future_to_info[future] = (group_index, group_dir)

        for future in as_completed(future_to_info):
            group_index, group_dir = future_to_info[future]

            try:
                result = future.result()
                saved_paths = result["saved_paths"]
                written += 1
                print(
                    f"[{group_index}/{total}] saved {group_dir.name}: "
                    f"{saved_paths['trace_npy'].name}, "
                    f"{saved_paths['probability_png'].name}"
                )
            except Exception as exc:
                failed += 1
                print(
                    f"[{group_index}/{total}] failed {group_dir}: "
                    f"{type(exc).__name__}: {exc}"
                )

    print(f"Auto write done: written={written}, failed={failed}, total={total}")


def save_group(
    output_root: Path,
    group_dir: Path,
    image_paths: list[Path],
    clean_images: list[np.ndarray],
    synthetic_images: list[np.ndarray],
    burn_map: np.ndarray,
    burn_mask: np.ndarray,
    temporal_scales: np.ndarray,
    seed: int,
):
    output_dir = output_root / group_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (path, clean, synthetic) in enumerate(
        zip(image_paths, clean_images, synthetic_images)
    ):
        if SAVE_CLEAN_FRAMES:
            cv2.imwrite(
                str(output_dir / f"clean_{index:02d}_{path.stem}.png"),
                clean,
            )

        cv2.imwrite(
            str(output_dir / f"burn_{index:02d}_{path.stem}.png"),
            synthetic,
        )

    cv2.imwrite(
        str(output_dir / "burn_mask.png"),
        burn_mask * 255,
    )

    # 偏置图保存为 float32，避免 8-bit 精度损失
    np.save(output_dir / "burn_map.npy", burn_map)

    metadata = [
        f"seed={seed}",
        f"burn_base_size_scale={BURN_BASE_SIZE_SCALE}",
        f"burn_size_multiplier={BURN_SIZE_MULTIPLIER}",
        f"burn_size_scale={get_burn_size_scale()}",
        f"burn_disc_radius_multiplier={BURN_DISC_RADIUS_MULTIPLIER}",
        f"burn_disc_overlap_ratio={BURN_DISC_OVERLAP_RATIO}",
        f"burn_horizontal_bias={BURN_HORIZONTAL_BIAS}",
        f"burn_max_stripe_angle_deg={BURN_MAX_STRIPE_ANGLE_DEG}",
        f"burn_disc_roundness_jitter={BURN_DISC_ROUNDNESS_JITTER}",
        f"burn_dwell_intensity_variation={BURN_DWELL_INTENSITY_VARIATION}",
        f"burn_blur_with_size={BURN_BLUR_WITH_SIZE}",
        f"blur_sigma={BLUR_SIGMA}",
        f"burn_cluster_edge_blur_multiplier={BURN_CLUSTER_EDGE_BLUR_MULTIPLIER}",
        f"burn_stripe_edge_blur_multiplier={BURN_STRIPE_EDGE_BLUR_MULTIPLIER}",
        f"burn_stripe_disc_edge_softness={BURN_STRIPE_DISC_EDGE_SOFTNESS}",
        f"mask_threshold={MASK_THRESHOLD}",
        "temporal_scales="
        + ",".join(f"{value:.6f}" for value in temporal_scales),
    ]

    (output_dir / "metadata.txt").write_text(
        "\n".join(metadata),
        encoding="utf-8",
    )


def main():
    if DISABLE_OPENCV_INTERNAL_THREADS and ENABLE_MULTITHREADING:
        cv2.setNumThreads(1)

    if not TRAIN_ROOT:
        raise ValueError("Please set TRAIN_ROOT first.")

    train_root = Path(TRAIN_ROOT)

    if not train_root.is_dir():
        raise FileNotFoundError(f"Training root not found: {train_root}")

    group_dirs = find_group_dirs(train_root)

    if not group_dirs:
        raise ValueError(f"No image group directories found under: {train_root}")

    rng = np.random.default_rng()

    if SHUFFLE_GROUPS:
        rng.shuffle(group_dirs)

    print(f"Found {len(group_dirs)} training groups.")

    if AUTO_WRITE_BURN_ARTIFACTS:
        print("AUTO_WRITE_BURN_ARTIFACTS=True: writing burn artifacts without preview.")
        auto_write_burn_artifacts(
            group_dirs=group_dirs,
            rng=rng,
            max_groups=AUTO_WRITE_MAX_GROUPS,
        )
        return

    print("Use Save burn maps or 's' to save and move on. Use Reset burn or 'r' to regenerate.")

    if SAVE_OUTPUT:
        output_root = Path(OUTPUT_ROOT)
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = None

    plt.ion()
    fig = plt.figure(1, figsize=(10, 8))

    for group_index, group_dir in enumerate(group_dirs, start=1):
        try:
            image_paths, clean_images = load_group_images(group_dir)
        except Exception as exc:
            print(
                f"[{group_index}/{len(group_dirs)}] "
                f"Failed to load: {group_dir}\n"
                f"Error: {type(exc).__name__}: {exc}"
            )
            continue

        seed = int(rng.integers(0, 10**9))
        needs_regenerate = True

        while True:
            if needs_regenerate:
                try:
                    synthetic_images, burn_map, burn_mask, temporal_scales = (
                        apply_burn_to_group(clean_images, seed)
                    )

                    show_group(
                        group_dir=group_dir,
                        image_paths=image_paths,
                        clean_images=clean_images,
                        synthetic_images=synthetic_images,
                        burn_map=burn_map,
                        burn_mask=burn_mask,
                        temporal_scales=temporal_scales,
                        seed=seed,
                    )

                    print(
                        f"[{group_index}/{len(group_dirs)}] "
                        f"{group_dir.name} | seed={seed} | "
                        f"burn max={burn_map.max():.2f}"
                    )

                    if SAVE_OUTPUT and output_root is not None:
                        save_group(
                            output_root=output_root,
                            group_dir=group_dir,
                            image_paths=image_paths,
                            clean_images=clean_images,
                            synthetic_images=synthetic_images,
                            burn_map=burn_map,
                            burn_mask=burn_mask,
                            temporal_scales=temporal_scales,
                            seed=seed,
                        )
                except Exception as exc:
                    print(
                        f"[{group_index}/{len(group_dirs)}] "
                        f"Failed to process: {group_dir}\n"
                        f"Error: {type(exc).__name__}: {exc}"
                    )
                    break

                needs_regenerate = False

            if not plt.fignum_exists(fig.number):
                break

            while (
                plt.fignum_exists(fig.number)
                and getattr(fig, "_burn_action", None) is None
            ):
                plt.pause(0.05)

            if not plt.fignum_exists(fig.number):
                break

            action = getattr(fig, "_burn_action", None)
            fig._burn_action = None

            if action == "reset":
                seed = int(rng.integers(0, 10**9))
                needs_regenerate = True
                continue

            if action == "accept":
                saved_paths = save_burn_artifacts_to_sample(
                    group_dir=group_dir,
                    burn_map=burn_map,
                    seed=seed,
                )
                print(
                    f"Saved burn artifacts to {group_dir}: "
                    f"{saved_paths['trace_npy'].name}, "
                    f"{saved_paths['trace_png'].name}, "
                    f"{saved_paths['probability_png'].name}"
                )
                break

            break

        if not plt.fignum_exists(fig.number):
            break
    plt.ioff()
    plt.show()

    if SAVE_OUTPUT:
        print(f"Synthetic results saved to: {Path(OUTPUT_ROOT).resolve()}")


if __name__ == "__main__":
    main()
