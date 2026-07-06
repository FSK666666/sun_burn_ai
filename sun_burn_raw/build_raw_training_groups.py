#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from raw_io import infer_raw_shape, parse_shape


RAW_SUFFIXES = {".bin", ".raw"}


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


def list_raw_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in RAW_SUFFIXES],
        key=natural_key,
    )


def list_sequence_dirs(root: Path) -> list[Path]:
    root = Path(root)
    sequence_dirs = []
    for folder in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        if any(p.is_file() and p.suffix.lower() in RAW_SUFFIXES for p in folder.iterdir()):
            sequence_dirs.append(folder)
    return sorted(sequence_dirs, key=lambda p: str(p).lower())


def build_groups(
    source_root: Path,
    output_root: Path,
    shape: tuple[int, int] | None,
    frame_gap: int,
    stride: int,
    max_groups: int | None,
    copy_mode: str,
):
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "raw_group_manifest.csv"
    total = 0
    rows = []

    for seq_dir in list_sequence_dirs(source_root):
        frames = list_raw_files(seq_dir)
        if not frames:
            continue
        if shape is not None:
            expected_bytes = shape[0] * shape[1] * 2
            frames = [p for p in frames if p.stat().st_size >= expected_bytes]
        else:
            valid = []
            for path in frames:
                try:
                    infer_raw_shape(path)
                    valid.append(path)
                except Exception:
                    pass
            frames = valid
        if len(frames) < 1 + 4 * frame_gap:
            continue
        seq_shape = shape if shape is not None else infer_raw_shape(frames[0])
        shape_text = f"{seq_shape[0]}x{seq_shape[1]}"

        last_start = len(frames) - 1 - 4 * frame_gap
        for start in range(0, last_start + 1, stride):
            selected = [frames[start + k * frame_gap] for k in range(5)]
            sample_dir = output_root / f"sample_{total:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for idx, src in enumerate(selected):
                dst = sample_dir / f"{idx:03d}{src.suffix.lower()}"
                if copy_mode == "hardlink":
                    if dst.exists():
                        dst.unlink()
                    try:
                        dst.hardlink_to(src)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            rows.append(
                {
                    "sample": sample_dir.name,
                    "sequence": str(seq_dir),
                    "shape": shape_text,
                    "start_index": start,
                    "frame_gap": frame_gap,
                    "frames": "|".join(str(p) for p in selected),
                }
            )
            total += 1
            if total == 1 or total % 1000 == 0:
                print(f"built {total} groups...", flush=True)
            if max_groups is not None and total >= max_groups:
                break
        if max_groups is not None and total >= max_groups:
            break

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "sequence", "shape", "start_index", "frame_gap", "frames"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {total} groups")
    print(f"manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build 5-frame raw training groups from continuous uint16 raw sequences.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shape", type=str, default=None, help="HxW, e.g. 512x640 or 1024x1280.")
    parser.add_argument("--frame-gap", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    return parser.parse_args()


def main():
    args = parse_args()
    build_groups(
        source_root=args.source_root,
        output_root=args.output_root,
        shape=parse_shape(args.shape),
        frame_gap=args.frame_gap,
        stride=args.stride,
        max_groups=args.max_groups,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
