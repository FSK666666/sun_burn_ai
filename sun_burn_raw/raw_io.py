#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


RAW_SHAPES: tuple[tuple[int, int], ...] = (
    (512, 640),
    (1024, 1280),
)

SHAPE_DIR_HINTS: dict[str, tuple[int, int]] = {
    "512": (512, 640),
    "640x512": (512, 640),
    "512x640": (512, 640),
    "1024": (1024, 1280),
    "1280x1024": (1024, 1280),
    "1024x1280": (1024, 1280),
}

DEFAULT_MAX_FOOTER_BYTES = 1024 * 1024


def parse_shape(text: str | None) -> tuple[int, int] | None:
    if text is None or text == "":
        return None
    lowered = text.lower().replace(",", "x")
    parts = [p for p in lowered.split("x") if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid shape: {text}. Use HxW, for example 512x640.")
    return int(parts[0]), int(parts[1])


def _shape_from_path_hint(path: Path) -> tuple[int, int] | None:
    for part in reversed(Path(path).parts):
        key = part.lower()
        if key in SHAPE_DIR_HINTS:
            return SHAPE_DIR_HINTS[key]
    return None


def infer_raw_shape(
    path: Path,
    shapes: tuple[tuple[int, int], ...] = RAW_SHAPES,
    max_footer_bytes: int = DEFAULT_MAX_FOOTER_BYTES,
) -> tuple[int, int]:
    path = Path(path)
    file_size = path.stat().st_size

    hinted = _shape_from_path_hint(path)
    if hinted is not None:
        expected_bytes = hinted[0] * hinted[1] * np.dtype(np.uint16).itemsize
        if file_size >= expected_bytes:
            return hinted

    matches = []
    for shape in shapes:
        expected_bytes = shape[0] * shape[1] * np.dtype(np.uint16).itemsize
        footer_bytes = file_size - expected_bytes
        if footer_bytes == 0:
            matches.append(shape)

    if len(matches) == 1:
        return matches[0]

    footer_matches = []
    for shape in shapes:
        expected_bytes = shape[0] * shape[1] * np.dtype(np.uint16).itemsize
        footer_bytes = file_size - expected_bytes
        if 0 <= footer_bytes <= max_footer_bytes:
            footer_matches.append(shape)

    if len(footer_matches) == 1:
        return footer_matches[0]
    if not footer_matches:
        raise ValueError(
            f"Cannot infer shape for {path}; file size={file_size} bytes. "
            f"Known shapes: {shapes}. Pass shape explicitly if needed."
        )
    raise ValueError(
        f"Ambiguous raw shape for {path}; file size={file_size} bytes fits {footer_matches}. "
        "Pass shape explicitly."
    )


def read_u16_bin(
    path: Path,
    shape: tuple[int, int] | None = None,
    endian: str = "little",
) -> np.ndarray:
    path = Path(path)
    if shape is None:
        shape = infer_raw_shape(path)
    dtype = "<u2" if endian == "little" else ">u2"
    expected = shape[0] * shape[1]
    expected_bytes = expected * np.dtype(np.uint16).itemsize
    file_size = path.stat().st_size
    if file_size < expected_bytes:
        raise ValueError(
            f"{path} is too small: {file_size} bytes, expected at least {expected_bytes} for shape {shape}."
        )
    with path.open("rb") as f:
        payload = f.read(expected_bytes)
    arr = np.frombuffer(payload, dtype=np.dtype(dtype), count=expected)
    return arr.reshape(shape).astype(np.uint16, copy=False)


def write_u16_bin(path: Path, image: np.ndarray, endian: str = "little"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0, 65535).astype(np.uint16)
    dtype = "<u2" if endian == "little" else ">u2"
    image.astype(np.dtype(dtype), copy=False).tofile(path)


def normalize_u16(image: np.ndarray, raw_max: float = 65535.0) -> np.ndarray:
    return np.clip(image.astype(np.float32) / float(raw_max), 0.0, 1.0)


def denormalize_u16(image: np.ndarray, raw_max: float = 65535.0) -> np.ndarray:
    return np.clip(np.round(image.astype(np.float32) * float(raw_max)), 0, 65535).astype(np.uint16)


def resize_float(image: np.ndarray, image_size: tuple[int, int] | None) -> np.ndarray:
    if image_size is None:
        return image
    h, w = image_size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def save_preview_png(path: Path, image: np.ndarray, raw_max: float = 65535.0):
    """Save an 8-bit preview PNG for quick visual inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype == np.uint16:
        preview = np.clip(image.astype(np.float32) / raw_max * 255.0, 0, 255).astype(np.uint8)
    else:
        preview = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), preview)
