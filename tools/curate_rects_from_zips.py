#!/usr/bin/env python3
"""Curate a clean held-out ReCTS subset from detection and recognition zips."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


@dataclass
class Candidate:
    image_id: str
    split: str
    image_member: str
    label_member: str
    gt_member: str
    detection_instances: int
    valid_lines: int
    ignored_lines: int
    mean_text_len: float
    max_text_len: int
    max_area_ratio: float
    mean_area_ratio: float
    min_center_distance: float
    sharpness: float
    contrast: float
    width: int
    height: int
    score: float
    source_text_reference: str
    sample_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-zip", required=True)
    parser.add_argument("--recognition-zip", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-detection-instances", type=int, default=4)
    parser.add_argument("--max-valid-lines", type=int, default=3)
    parser.add_argument("--output-dir", default="translation_subset/rects_from_zip")
    parser.add_argument("--export-images", action="store_true")
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_image_quality(gray: np.ndarray) -> tuple[float, float]:
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(laplacian.var()), float(gray.std())


def parse_yolo_polygon(line: str, width: int, height: int) -> np.ndarray | None:
    parts = line.split()
    if len(parts) < 9:
        return None
    try:
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
    except ValueError:
        return None
    coords[:, 0] *= width
    coords[:, 1] *= height
    return coords


def load_recognition_map(recognition_zip: Path) -> dict[str, dict]:
    recognition = {}
    with ZipFile(recognition_zip) as zf:
        for name in zf.namelist():
            if not name.startswith("gt/train_ReCTS_") or not name.endswith(".json"):
                continue
            image_id = Path(name).stem
            recognition[image_id] = json.loads(zf.read(name))
    return recognition


def candidate_score(
    detection_instances: int,
    valid_lines: int,
    ignored_lines: int,
    max_area_ratio: float,
    min_center_distance: float,
    sharpness: float,
    contrast: float,
    mean_text_len: float,
) -> float:
    det_score = {1: 1.0, 2: 0.8, 3: 0.55, 4: 0.3}.get(detection_instances, 0.0)
    line_score = {1: 1.0, 2: 0.84, 3: 0.62}.get(valid_lines, 0.0)
    ignore_penalty = 1.0 if ignored_lines == 0 else 0.6
    prominence_score = clamp01(max_area_ratio / 0.18)
    centrality_score = clamp01(1.0 - min_center_distance / 0.75)
    sharpness_score = clamp01(sharpness / 350.0)
    contrast_score = clamp01(contrast / 70.0)
    simplicity_score = clamp01(1.0 - mean_text_len / 14.0)

    return 100.0 * (
        0.22 * det_score
        + 0.22 * line_score
        + 0.12 * prominence_score
        + 0.10 * centrality_score
        + 0.08 * sharpness_score
        + 0.06 * contrast_score
        + 0.10 * simplicity_score
        + 0.10 * ignore_penalty
    )


def serializable(item: Candidate) -> dict:
    data = asdict(item)
    for key, value in list(data.items()):
        if isinstance(value, np.generic):
            data[key] = value.item()
    return data


def build_candidates(
    detection_zip: Path,
    recognition_map: dict[str, dict],
    split: str,
    max_detection_instances: int,
    max_valid_lines: int,
) -> list[Candidate]:
    candidates = []
    with ZipFile(detection_zip) as zf:
        names = set(zf.namelist())
        image_prefix = f"rects_yolo_obb_dataset/images/{split}/"
        label_prefix = f"rects_yolo_obb_dataset/labels/{split}/"
        image_members = [name for name in zf.namelist() if name.startswith(image_prefix) and not name.endswith("/")]

        for image_member in image_members:
            image_id = Path(image_member).stem
            label_member = f"{label_prefix}{image_id}.txt"
            if label_member not in names or image_id not in recognition_map:
                continue

            try:
                with Image.open(io.BytesIO(zf.read(image_member))) as image:
                    rgb = image.convert("RGB")
                    gray = np.asarray(image.convert("L"), dtype=np.float32)
                    width, height = rgb.size
            except OSError:
                continue

            label_lines = zf.read(label_member).decode("utf-8-sig", "replace").splitlines()
            polygons = [parse_yolo_polygon(line, width, height) for line in label_lines if line.strip()]
            polygons = [poly for poly in polygons if poly is not None]
            if not polygons or len(polygons) > max_detection_instances:
                continue

            gt_member = f"gt/{image_id}.json"
            gt = recognition_map[image_id]
            lines = gt.get("lines", [])
            valid_lines = [line for line in lines if int(line.get("ignore", 0)) == 0 and line.get("transcription", "").strip() and line.get("transcription") != "###"]
            ignored_lines = [line for line in lines if int(line.get("ignore", 0)) == 1 or line.get("transcription") == "###"]
            if not valid_lines or len(valid_lines) > max_valid_lines:
                continue

            image_area = float(width * height)
            image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            image_diag = math.hypot(width, height)
            area_ratios = [polygon_area(poly) / image_area for poly in polygons]
            center_distances = [float(np.linalg.norm(poly.mean(axis=0) - image_center) / image_diag) for poly in polygons]
            sharpness, contrast = compute_image_quality(gray)

            ordered_lines = []
            for line in valid_lines:
                pts = line["points"]
                xs = pts[0::2]
                ys = pts[1::2]
                ordered_lines.append(
                    {
                        "left": min(xs),
                        "top": min(ys),
                        "text": line["transcription"].strip(),
                    }
                )
            ordered_lines.sort(key=lambda item: (item["top"], item["left"]))
            source_text = " ".join(item["text"] for item in ordered_lines)
            mean_text_len = sum(len(item["text"]) for item in ordered_lines) / len(ordered_lines)
            max_text_len = max(len(item["text"]) for item in ordered_lines)

            candidate = Candidate(
                image_id=image_id,
                split=split,
                image_member=image_member,
                label_member=label_member,
                gt_member=gt_member,
                detection_instances=len(polygons),
                valid_lines=len(valid_lines),
                ignored_lines=len(ignored_lines),
                mean_text_len=round(float(mean_text_len), 4),
                max_text_len=int(max_text_len),
                max_area_ratio=round(float(max(area_ratios)), 6),
                mean_area_ratio=round(float(sum(area_ratios) / len(area_ratios)), 6),
                min_center_distance=round(float(min(center_distances)), 6),
                sharpness=round(float(sharpness), 4),
                contrast=round(float(contrast), 4),
                width=int(width),
                height=int(height),
                score=0.0,
                source_text_reference=source_text,
                sample_text=ordered_lines[0]["text"],
            )
            candidate.score = round(
                candidate_score(
                    detection_instances=candidate.detection_instances,
                    valid_lines=candidate.valid_lines,
                    ignored_lines=candidate.ignored_lines,
                    max_area_ratio=max(area_ratios),
                    min_center_distance=min(center_distances),
                    sharpness=sharpness,
                    contrast=contrast,
                    mean_text_len=mean_text_len,
                ),
                4,
            )
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -item.score,
            item.valid_lines,
            item.ignored_lines,
            item.mean_text_len,
            -item.max_area_ratio,
            item.image_id,
        )
    )
    return candidates


def write_outputs(output_dir: Path, selected: list[Candidate]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not selected:
        raise SystemExit("No ReCTS candidates selected.")

    csv_path = output_dir / "rects_candidates.csv"
    json_path = output_dir / "rects_candidates.json"
    summary_path = output_dir / "summary.json"
    annotation_path = output_dir / "annotation_sheet.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable(selected[0]).keys()))
        writer.writeheader()
        for item in selected:
            writer.writerow(serializable(item))

    with json_path.open("w") as handle:
        json.dump([serializable(item) for item in selected], handle, indent=2)

    with summary_path.open("w") as handle:
        json.dump(
            {
                "count": len(selected),
                "top_score": float(selected[0].score),
                "split": selected[0].split,
            },
            handle,
            indent=2,
        )

    fieldnames = [
        "source_dataset",
        "split",
        "image_id",
        "class_name",
        "image_member",
        "label_member",
        "gt_member",
        "selector_score",
        "source_text_reference",
        "target_language",
        "target_translation_reference",
        "annotator_1",
        "annotator_2",
        "notes",
    ]
    with annotation_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "source_dataset": "ReCTS",
                    "split": item.split,
                    "image_id": item.image_id,
                    "class_name": "text",
                    "image_member": item.image_member,
                    "label_member": item.label_member,
                    "gt_member": item.gt_member,
                    "selector_score": item.score,
                    "source_text_reference": item.source_text_reference,
                    "target_language": "en",
                    "target_translation_reference": "",
                    "annotator_1": "",
                    "annotator_2": "",
                    "notes": item.sample_text,
                }
            )


def export_images(detection_zip: Path, output_dir: Path, selected: list[Candidate]) -> None:
    image_dir = output_dir / "selected_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(detection_zip) as zf:
        for item in selected:
            suffix = Path(item.image_member).suffix.lower()
            (image_dir / f"{item.image_id}{suffix}").write_bytes(zf.read(item.image_member))


def main() -> int:
    args = parse_args()
    detection_zip = Path(args.detection_zip)
    recognition_zip = Path(args.recognition_zip)
    output_dir = Path(args.output_dir)

    recognition_map = load_recognition_map(recognition_zip)
    candidates = build_candidates(
        detection_zip=detection_zip,
        recognition_map=recognition_map,
        split=args.split,
        max_detection_instances=args.max_detection_instances,
        max_valid_lines=args.max_valid_lines,
    )
    selected = candidates[: args.count]
    write_outputs(output_dir, selected)
    if args.export_images:
        export_images(detection_zip, output_dir, selected)

    print(f"Selected {len(selected)} ReCTS candidates from split={args.split}")
    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
