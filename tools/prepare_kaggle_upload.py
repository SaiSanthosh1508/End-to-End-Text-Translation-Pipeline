#!/usr/bin/env python3
"""Prepare the Kaggle upload folder from the final proportional subset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument(
        "--dataset-id",
        default="",
        help="Optional Kaggle dataset id such as user/dataset-slug. If provided, dataset-metadata.json is written.",
    )
    parser.add_argument(
        "--dataset-title",
        default="",
        help="Optional Kaggle dataset title. Defaults to a title derived from the subset size.",
    )
    parser.add_argument(
        "--license-name",
        default="CC0-1.0",
        help="License name for generated Kaggle dataset metadata.",
    )
    parser.add_argument(
        "--include-score-files",
        action="store_true",
        help="Include final_dataset_scores.csv and selection_summary.json in the Kaggle package.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_image_path(root: Path, row: dict[str, str]) -> Path:
    local_image_path = (row.get("local_image_path") or "").strip()
    if local_image_path:
        candidate = Path(local_image_path)
        if candidate.exists():
            return candidate

    image_id = row["image_id"]
    if row["source_dataset"] == "ICDAR MLT-2019":
        matches = sorted((root / "translation_subset/mlt_from_zip/selected_images").glob(f"**/{image_id}.*"))
    else:
        matches = sorted((root / "translation_subset/rects_from_zips/selected_images").glob(f"{image_id}.*"))
    if not matches:
        raise FileNotFoundError(f"missing local image for {image_id}")
    return matches[0]


def write_readme(path: Path, rows: list[dict[str, str]], summary: dict) -> None:
    counts = Counter(row["selection_language"] for row in rows)
    selected_rows = int(summary["selected_rows"])
    lines = [
        "# Scene Text Translation English Eval Subset",
        "",
        f"This package contains a released {selected_rows}-image multilingual English-reference subset",
        "for supplementary end-to-end OCR-plus-translation evaluation.",
        "",
        "Language distribution:",
    ]
    for language in sorted(counts):
        lines.append(f"- {language}: {counts[language]} samples")
    lines.extend(
        [
            "",
            "Files:",
            f"- `english_only_master_annotation_gold.csv`: {selected_rows}-row gold annotation sheet.",
            f"- `images/`: the {selected_rows} selected evaluation images grouped by language bucket.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_dataset_metadata(
    path: Path,
    dataset_id: str,
    dataset_title: str,
    license_name: str,
) -> None:
    payload = {
        "title": dataset_title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)
    root = args.project_root.resolve()

    gold_csv = subset_dir / "english_only_master_annotation_gold.csv"
    scores_csv = subset_dir / "final_dataset_scores.csv"
    summary_json = subset_dir / "selection_summary.json"

    rows = load_csv(scores_csv)
    summary = json.loads(summary_json.read_text())
    print(f"Loaded {len(rows)} scored rows from {scores_csv}", flush=True)
    selected_rows = int(summary["selected_rows"])

    output_dir.mkdir(parents=True, exist_ok=True)

    for generated_path in [
        output_dir / "english_only_master_annotation_gold.csv",
        output_dir / "final_dataset_scores.csv",
        output_dir / "selection_summary.json",
        output_dir / "README.md",
        output_dir / "dataset-metadata.json",
    ]:
        if generated_path.exists():
            generated_path.unlink()

    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(gold_csv, output_dir / gold_csv.name)
    write_readme(output_dir / "README.md", rows, summary)
    if args.include_score_files:
        shutil.copy2(scores_csv, output_dir / scores_csv.name)
        shutil.copy2(summary_json, output_dir / summary_json.name)
        print("Copied gold sheet, score files, and wrote README", flush=True)
    else:
        print("Copied gold sheet and wrote README", flush=True)

    if args.dataset_id:
        dataset_title = args.dataset_title or "Scene Text Translation English Eval Subset"
        metadata_path = output_dir / "dataset-metadata.json"
        write_dataset_metadata(
            metadata_path,
            dataset_id=args.dataset_id,
            dataset_title=dataset_title,
            license_name=args.license_name,
        )
        print(f"Wrote dataset metadata to {metadata_path}", flush=True)

    total_rows = len(rows)
    for row in rows:
        language_dir = images_dir / row["selection_language"].lower()
        language_dir.mkdir(parents=True, exist_ok=True)
        source_image = resolve_image_path(root, row)
        shutil.copy2(source_image, language_dir / source_image.name)
        copied = sum(1 for _ in images_dir.rglob("*") if _.is_file())
        if copied % 25 == 0 or copied == total_rows:
            print(f"[images] copied {copied}/{total_rows}", flush=True)

    print(f"rows={len(rows)}")
    print(f"images={len(rows)}")
    print(f"wrote={output_dir / gold_csv.name}")
    if args.include_score_files:
        print(f"wrote={output_dir / scores_csv.name}")
        print(f"wrote={output_dir / summary_json.name}")
    print(f"wrote={output_dir / 'README.md'}")
    print(f"wrote={images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
