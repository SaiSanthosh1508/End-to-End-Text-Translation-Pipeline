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
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_image_path(root: Path, row: dict[str, str]) -> Path:
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
    lines = [
        "# Scene Text Translation English Eval Final Top-40 Subset",
        "",
        "This package contains the final 40-sample multilingual evaluation subset",
        "selected from the completed 160-image English-reference evaluation set.",
        "",
        "Selection policy:",
        "- Rank by normalized chrF against the saved end-to-end translation output.",
        "- Allocate samples proportionally by language.",
        "- Merge ReCTS entries into the Chinese language bucket.",
        "",
        f"Selected rows: {summary['selected_rows']}",
        f"Average normalized chrF: {summary['average_normalized_chrf']:.4f}",
        "",
        "Language distribution:",
    ]
    for language in sorted(counts):
        stats = summary["language_counts"][language]
        lines.append(
            f"- {language}: {counts[language]} samples, avg normalized chrF {stats['selected_average_normalized_chrf']:.4f}"
        )
    lines.extend(
        [
            "",
            "Files:",
            "- `english_only_master_annotation_gold.csv`: final 40-row gold annotation sheet.",
            "- `final_dataset_scores.csv`: gold rows with normalized chrF, prediction, and selection ranks.",
            "- `selection_summary.json`: machine-readable selection summary.",
            "- `images/`: the 40 selected evaluation images grouped by language bucket.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)
    root = Path("/Users/saisanthosh/Documents/paper-exp/End-to-End-Text-Translation-Pipeline")

    gold_csv = subset_dir / "english_only_master_annotation_gold.csv"
    scores_csv = subset_dir / "final_dataset_scores.csv"
    summary_json = subset_dir / "selection_summary.json"

    rows = load_csv(scores_csv)
    summary = json.loads(summary_json.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)

    for generated_path in [
        output_dir / "english_only_master_annotation_gold.csv",
        output_dir / "final_dataset_scores.csv",
        output_dir / "selection_summary.json",
        output_dir / "README.md",
    ]:
        if generated_path.exists():
            generated_path.unlink()

    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(gold_csv, output_dir / gold_csv.name)
    shutil.copy2(scores_csv, output_dir / scores_csv.name)
    shutil.copy2(summary_json, output_dir / summary_json.name)
    write_readme(output_dir / "README.md", rows, summary)

    for row in rows:
        language_dir = images_dir / row["selection_language"].lower()
        language_dir.mkdir(parents=True, exist_ok=True)
        source_image = resolve_image_path(root, row)
        shutil.copy2(source_image, language_dir / source_image.name)

    print(f"rows={len(rows)}")
    print(f"images={len(rows)}")
    print(f"wrote={output_dir / gold_csv.name}")
    print(f"wrote={output_dir / scores_csv.name}")
    print(f"wrote={output_dir / summary_json.name}")
    print(f"wrote={output_dir / 'README.md'}")
    print(f"wrote={images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
