#!/usr/bin/env python3
"""Build a large ICDAR MLT candidate pool for supplementary translation evaluation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


LANGUAGE_TO_CLASS = {
    "Arabic": "Arabic",
    "Latin": "Latin",
    "Chinese": "Chinese",
    "Korean": "Korean",
    "Japanese": "Japanese",
    "Bangla": "Bangla",
    "Hindi": "Hindi",
    "Symbols": "Other",
    "Mixed": "Other",
    "None": "Other",
}

TARGET_CLASSES = ["Arabic", "Bangla", "Chinese", "Hindi", "Japanese", "Korean", "Latin"]


@dataclass
class Candidate:
    class_name: str
    image_member: str
    gt_member: str
    image_id: str
    total_instances: int
    valid_instances: int
    unique_languages: int
    dominant_ratio: float
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
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--per-language", type=int, default=120)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-instances", type=int, default=4)
    parser.add_argument("--min-dominant-ratio", type=float, default=0.8)
    parser.add_argument("--export-images", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
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


def parse_gt_line(line: str) -> tuple[np.ndarray, str, str] | None:
    parts = line.rstrip("\n").split(",", 9)
    if len(parts) < 10:
        return None
    try:
        coords = np.array([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)
    except ValueError:
        return None
    language = parts[8].strip()
    text = parts[9].strip()
    return coords, language, text


def find_image_member(image_id: str, members: set[str]) -> str | None:
    for extension in (".jpg", ".JPG", ".png", ".gif"):
        candidate = f"TrainImages/TrainImages/{image_id}{extension}"
        if candidate in members:
            return candidate
    return None


def class_score(
    valid_instances: int,
    dominant_ratio: float,
    max_area_ratio: float,
    min_center_distance: float,
    sharpness: float,
    contrast: float,
    mean_text_len: float,
) -> float:
    instance_score = {1: 1.0, 2: 0.82, 3: 0.58, 4: 0.32}.get(valid_instances, 0.0)
    prominence_score = clamp01(max_area_ratio / 0.14)
    centrality_score = clamp01(1.0 - min_center_distance / 0.75)
    sharpness_score = clamp01(sharpness / 350.0)
    contrast_score = clamp01(contrast / 70.0)
    simplicity_score = clamp01(1.0 - mean_text_len / 18.0)
    purity_score = clamp01((dominant_ratio - 0.6) / 0.4)

    return 100.0 * (
        0.25 * instance_score
        + 0.18 * prominence_score
        + 0.12 * centrality_score
        + 0.08 * sharpness_score
        + 0.05 * contrast_score
        + 0.12 * simplicity_score
        + 0.20 * purity_score
    )


def load_candidates(
    zip_path: Path,
    max_instances: int,
    min_dominant_ratio: float,
    progress_every: int,
) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)

    with ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        gt_members = [name for name in zf.namelist() if name.startswith("TrainGT/TrainGT/") and name.endswith(".txt")]
        total_gt = len(gt_members)
        print(f"Scanning {total_gt} ICDAR ground-truth files from {zip_path}", flush=True)

        for idx, gt_member in enumerate(gt_members, start=1):
            image_id = Path(gt_member).stem
            image_member = find_image_member(image_id, members)
            if image_member is None:
                continue

            gt_lines = zf.read(gt_member).decode("utf-8-sig", "replace").splitlines()
            parsed = [parse_gt_line(line) for line in gt_lines]
            parsed = [item for item in parsed if item is not None]
            if not parsed:
                continue

            valid = []
            for coords, language, text in parsed:
                mapped = LANGUAGE_TO_CLASS.get(language, "Other")
                if text == "###":
                    continue
                valid.append((coords, language, mapped, text))

            if not valid or len(valid) > max_instances:
                continue

            mapped_counts = Counter(item[2] for item in valid if item[2] in TARGET_CLASSES)
            if not mapped_counts:
                continue

            class_name, dominant_count = mapped_counts.most_common(1)[0]
            dominant_ratio = dominant_count / len(valid)
            if dominant_ratio < min_dominant_ratio:
                continue

            try:
                with Image.open(io.BytesIO(zf.read(image_member))) as image:
                    rgb = image.convert("RGB")
                    gray = np.asarray(image.convert("L"), dtype=np.float32)
                    width, height = rgb.size
            except OSError:
                continue

            image_area = float(width * height)
            image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            image_diag = math.hypot(width, height)
            sharpness, contrast = compute_image_quality(gray)

            area_ratios = []
            center_distances = []
            text_lengths = []
            ordered_text_entries = []
            sample_text = ""
            valid_in_dominant = 0
            languages_in_dominant = set()
            for coords, language, mapped, text in valid:
                if mapped != class_name:
                    continue
                area_ratios.append(polygon_area(coords) / image_area)
                center_distances.append(float(np.linalg.norm(coords.mean(axis=0) - image_center) / image_diag))
                text_lengths.append(len(text))
                ordered_text_entries.append(
                    {
                        "left": float(coords[:, 0].min()),
                        "top": float(coords[:, 1].min()),
                        "text": text,
                    }
                )
                if not sample_text:
                    sample_text = text
                valid_in_dominant += 1
                languages_in_dominant.add(language)

            if not area_ratios:
                continue

            ordered_text_entries.sort(key=lambda item: (item["top"], item["left"]))
            candidate = Candidate(
                class_name=class_name,
                image_member=image_member,
                gt_member=gt_member,
                image_id=image_id,
                total_instances=len(parsed),
                valid_instances=valid_in_dominant,
                unique_languages=len(languages_in_dominant),
                dominant_ratio=round(dominant_ratio, 4),
                mean_text_len=round(sum(text_lengths) / len(text_lengths), 4),
                max_text_len=max(text_lengths),
                max_area_ratio=round(max(area_ratios), 6),
                mean_area_ratio=round(sum(area_ratios) / len(area_ratios), 6),
                min_center_distance=round(min(center_distances), 6),
                sharpness=round(sharpness, 4),
                contrast=round(contrast, 4),
                width=width,
                height=height,
                score=0.0,
                source_text_reference=" ".join(item["text"] for item in ordered_text_entries),
                sample_text=sample_text,
            )
            candidate.score = round(
                class_score(
                    valid_instances=candidate.valid_instances,
                    dominant_ratio=dominant_ratio,
                    max_area_ratio=max(area_ratios),
                    min_center_distance=min(center_distances),
                    sharpness=sharpness,
                    contrast=contrast,
                    mean_text_len=sum(text_lengths) / len(text_lengths),
                ),
                4,
            )
            grouped[class_name].append(candidate)

            if progress_every and idx % progress_every == 0:
                counts = ", ".join(
                    f"{language}={len(grouped.get(language, []))}" for language in TARGET_CLASSES
                )
                print(f"[scan {idx}/{total_gt}] candidate counts: {counts}", flush=True)

        counts = ", ".join(
            f"{language}={len(grouped.get(language, []))}" for language in TARGET_CLASSES
        )
        print(f"Completed scan. Candidate counts: {counts}", flush=True)

    return grouped


def serializable(item: Candidate) -> dict[str, object]:
    data = asdict(item)
    for key, value in list(data.items()):
        if isinstance(value, np.generic):
            data[key] = value.item()
    return data


def export_images(zip_path: Path, output_dir: Path, selected: dict[str, list[Candidate]], overwrite_images: bool) -> dict[str, str]:
    image_paths: dict[str, str] = {}
    preview_dir = output_dir / "selected_images"
    preview_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        for class_name, items in selected.items():
            class_dir = preview_dir / class_name.lower()
            class_dir.mkdir(parents=True, exist_ok=True)
            print(f"[export] {class_name}: {len(items)} images", flush=True)
            for index, item in enumerate(items, start=1):
                suffix = Path(item.image_member).suffix.lower()
                target = class_dir / f"{item.image_id}{suffix}"
                if overwrite_images or not target.exists():
                    target.write_bytes(zf.read(item.image_member))
                image_paths[item.image_id] = str(target.resolve())
                if index % 25 == 0 or index == len(items):
                    print(f"  exported {class_name} {index}/{len(items)}", flush=True)
    return image_paths


def write_outputs(
    output_dir: Path,
    grouped: dict[str, list[Candidate]],
    per_language: int,
    image_paths: dict[str, str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[str, list[Candidate]] = {}
    for class_name in TARGET_CLASSES:
        items = grouped.get(class_name, [])
        items.sort(
            key=lambda item: (
                -item.score,
                item.valid_instances,
                item.unique_languages,
                item.mean_text_len,
                -item.max_area_ratio,
                item.image_id,
            )
        )
        chosen = items[:per_language]
        if chosen:
            selected[class_name] = chosen
        print(
            f"[select] {class_name}: available={len(items)} selected={len(chosen)}",
            flush=True,
        )

    annotation_path = output_dir / "annotation_sheet.csv"
    summary_path = output_dir / "summary.json"

    fieldnames = [
        "source_dataset",
        "split",
        "image_id",
        "class_name",
        "image_member",
        "gt_member",
        "selector_score",
        "source_text_reference",
        "target_language",
        "target_translation_reference",
        "annotator_1",
        "annotator_2",
        "notes",
        "local_image_path",
    ]
    rows = []
    for class_name in TARGET_CLASSES:
        for item in selected.get(class_name, []):
            rows.append(
                {
                    "source_dataset": "ICDAR MLT-2019",
                    "split": "held_out_from_train",
                    "image_id": item.image_id,
                    "class_name": item.class_name,
                    "image_member": item.image_member,
                    "gt_member": item.gt_member,
                    "selector_score": item.score,
                    "source_text_reference": item.source_text_reference,
                    "target_language": "en",
                    "target_translation_reference": "",
                    "annotator_1": "",
                    "annotator_2": "",
                    "notes": item.sample_text,
                    "local_image_path": image_paths.get(item.image_id, ""),
                }
            )

    with annotation_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote annotation sheet with {len(rows)} rows to {annotation_path}", flush=True)

    summary = {
        "per_language": per_language,
        "selected_rows": len(rows),
        "language_counts": {
            class_name: {
                "selected": len(selected.get(class_name, [])),
                "top_selector_score": float(selected[class_name][0].score),
            }
            for class_name in TARGET_CLASSES
            if selected.get(class_name)
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote summary to {summary_path}", flush=True)
    return annotation_path, summary_path


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)

    grouped = load_candidates(
        zip_path=zip_path,
        max_instances=args.max_instances,
        min_dominant_ratio=args.min_dominant_ratio,
        progress_every=args.progress_every,
    )
    if not grouped:
        raise SystemExit("No candidates found in the archive with the current filters.")

    selected_for_export = {
        class_name: sorted(
            grouped.get(class_name, []),
            key=lambda item: (
                -item.score,
                item.valid_instances,
                item.unique_languages,
                item.mean_text_len,
                -item.max_area_ratio,
                item.image_id,
            ),
        )[: args.per_language]
        for class_name in TARGET_CLASSES
    }
    image_paths: dict[str, str] = {}
    if args.export_images:
        image_paths = export_images(
            zip_path=zip_path,
            output_dir=output_dir,
            selected=selected_for_export,
            overwrite_images=args.overwrite_images,
        )

    annotation_path, summary_path = write_outputs(
        output_dir=output_dir,
        grouped=grouped,
        per_language=args.per_language,
        image_paths=image_paths,
    )

    print(f"selected_rows={sum(len(items) for items in selected_for_export.values())}")
    print(f"wrote={annotation_path}")
    print(f"wrote={summary_path}")
    if args.export_images:
        print(f"wrote={output_dir / 'selected_images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
