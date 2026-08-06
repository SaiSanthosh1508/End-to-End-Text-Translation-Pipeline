#!/usr/bin/env python3
"""Merge dataset-specific annotation sheets into one English-only master sheet."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/Users/saisanthosh/Documents/paper-exp/End-to-End-Text-Translation-Pipeline")
INPUT_SHEETS = [
    ROOT / "translation_subset/mlt_from_zip/annotation_sheet.csv",
    ROOT / "translation_subset/rects_from_zips/annotation_sheet.csv",
]
OUTPUT_SHEET = ROOT / "translation_subset/english_only_master_annotation.csv"


def main() -> int:
    rows = []
    for sheet in INPUT_SHEETS:
        with sheet.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))

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

    OUTPUT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_SHEET.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in fieldnames}
            normalized["target_language"] = "en"
            writer.writerow(normalized)

    print(f"Wrote {len(rows)} rows to {OUTPUT_SHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
