#!/usr/bin/env python3
"""Rank low-clutter benchmark images for a supplementary translation subset.

This script scans a YOLO-style image/label split and produces ranked candidate
lists that favor:
1. fewer text instances,
2. larger and clearer text regions,
3. sharper, higher-contrast images,
4. a prominent text region near the image center.

It is intended for creating a small, reviewer-defensible end-to-end translation
evaluation subset from held-out MLT-2019 or ReCTS images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Candidate:
    image_path: str
    label_path: str
    class_id: int
    class_name: str
    text_instances: int
    score: float
    max_area_ratio: float
    mean_area_ratio: float
    total_area_ratio: float
    max_center_distance: float
    sharpness: float
    contrast: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory containing held-out images.")
    parser.add_argument("--labels", required=True, help="Directory containing YOLO label files.")
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names in class-id order. If omitted, class IDs are used.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Number of top candidates to export per class.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=4,
        help="Discard images with more than this many labeled text instances.",
    )
    parser.add_argument(
        "--min-max-area-ratio",
        type=float,
        default=0.015,
        help="Discard images whose largest text region is smaller than this fraction of image area.",
    )
    parser.add_argument(
        "--output-dir",
        default="translation_subset/candidates",
        help="Directory where ranked manifests will be written.",
    )
    return parser.parse_args()


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_label_polygons(label_path: Path, image_w: int, image_h: int) -> list[tuple[int, np.ndarray]]:
    polygons: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return polygons

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]

        if len(coords) == 4:
            x_center, y_center, width, height = coords
            x1 = (x_center - width / 2.0) * image_w
            y1 = (y_center - height / 2.0) * image_h
            x2 = (x_center + width / 2.0) * image_w
            y2 = (y_center + height / 2.0) * image_h
            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        elif len(coords) >= 8:
            polygon = np.array(coords[:8], dtype=np.float32).reshape(4, 2)
            polygon[:, 0] *= image_w
            polygon[:, 1] *= image_h
        else:
            continue

        polygons.append((class_id, polygon))

    return polygons


def compute_image_quality(gray: np.ndarray) -> tuple[float, float]:
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    sharpness = float(laplacian.var())
    contrast = float(gray.std())
    return sharpness, contrast


def prominence_score(area_ratio: float) -> float:
    return clamp01(area_ratio / 0.12)


def sharpness_score(sharpness: float) -> float:
    return clamp01(sharpness / 350.0)


def contrast_score(contrast: float) -> float:
    return clamp01(contrast / 70.0)


def centrality_score(distance: float) -> float:
    return clamp01(1.0 - distance / 0.75)


def instance_count_score(count: int) -> float:
    if count <= 0:
        return 0.0
    if count == 1:
        return 1.0
    if count == 2:
        return 0.78
    if count == 3:
        return 0.55
    if count == 4:
        return 0.3
    return 0.0


def build_candidate(
    image_path: Path,
    label_path: Path,
    class_names: dict[int, str],
    max_instances: int,
    min_max_area_ratio: float,
) -> list[Candidate]:
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            width, height = rgb.size
    except OSError:
        return []

    polygons = load_label_polygons(label_path, width, height)
    if not polygons:
        return []

    sharpness, contrast = compute_image_quality(gray)
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for class_id, polygon in polygons:
        grouped[class_id].append(polygon)

    image_area = float(width * height)
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    image_diag = math.hypot(width, height)

    candidates: list[Candidate] = []
    for class_id, class_polygons in grouped.items():
        instance_count = len(class_polygons)
        if instance_count > max_instances:
            continue

        area_ratios: list[float] = []
        center_distances: list[float] = []
        for polygon in class_polygons:
            area_ratios.append(polygon_area(polygon) / image_area)
            center = polygon.mean(axis=0)
            center_distances.append(float(np.linalg.norm(center - image_center) / image_diag))

        max_area_ratio = max(area_ratios)
        if max_area_ratio < min_max_area_ratio:
            continue

        mean_area_ratio = float(sum(area_ratios) / len(area_ratios))
        total_area_ratio = float(sum(area_ratios))
        max_center_distance = min(center_distances)

        score = 100.0 * (
            0.38 * instance_count_score(instance_count)
            + 0.28 * prominence_score(max_area_ratio)
            + 0.14 * centrality_score(max_center_distance)
            + 0.12 * sharpness_score(sharpness)
            + 0.08 * contrast_score(contrast)
        )

        candidates.append(
            Candidate(
                image_path=str(image_path),
                label_path=str(label_path),
                class_id=class_id,
                class_name=class_names.get(class_id, str(class_id)),
                text_instances=instance_count,
                score=round(score, 4),
                max_area_ratio=round(max_area_ratio, 6),
                mean_area_ratio=round(mean_area_ratio, 6),
                total_area_ratio=round(total_area_ratio, 6),
                max_center_distance=round(max_center_distance, 6),
                sharpness=round(sharpness, 4),
                contrast=round(contrast, 4),
                width=width,
                height=height,
            )
        )

    return candidates


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def write_outputs(output_dir: Path, ranked: dict[int, list[Candidate]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for class_id, items in ranked.items():
        class_name = items[0].class_name if items else str(class_id)
        csv_path = output_dir / f"class_{class_id}_{class_name}.csv"
        json_path = output_dir / f"class_{class_id}_{class_name}.json"

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(items[0]).keys()))
            writer.writeheader()
            for item in items:
                writer.writerow(asdict(item))

        with json_path.open("w") as handle:
            json.dump([asdict(item) for item in items], handle, indent=2)

        summary[class_id] = {
            "class_name": class_name,
            "count": len(items),
            "top_score": items[0].score,
            "csv": str(csv_path),
            "json": str(json_path),
        }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    class_names = {}
    if args.class_names:
        class_names = {index: name for index, name in enumerate(args.class_names)}

    ranked: dict[int, list[Candidate]] = defaultdict(list)
    scanned = 0
    for image_path in iter_images(images_dir):
        scanned += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        for candidate in build_candidate(
            image_path=image_path,
            label_path=label_path,
            class_names=class_names,
            max_instances=args.max_instances,
            min_max_area_ratio=args.min_max_area_ratio,
        ):
            ranked[candidate.class_id].append(candidate)

    if not ranked:
        raise SystemExit(
            "No candidates were found. Check that the image/label split exists and the filters are not too strict."
        )

    trimmed: dict[int, list[Candidate]] = {}
    for class_id, items in ranked.items():
        items.sort(key=lambda item: (-item.score, item.text_instances, -item.max_area_ratio, item.image_path))
        trimmed[class_id] = items[: args.per_class]

    write_outputs(output_dir, trimmed)

    print(f"Scanned {scanned} images.")
    for class_id, items in sorted(trimmed.items()):
        print(f"class {class_id:>2} ({items[0].class_name}): exported {len(items)} candidates")
    print(f"Wrote manifests to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
