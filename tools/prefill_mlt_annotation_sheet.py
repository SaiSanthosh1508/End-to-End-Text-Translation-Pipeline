#!/usr/bin/env python3
"""Prefill source_text_reference for the MLT held-out annotation sheet."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--annotation-sheet", required=True)
    return parser.parse_args()


def parse_gt_line(line: str):
    parts = line.rstrip("\n").split(",", 9)
    if len(parts) < 10:
        return None
    try:
        coords = [float(x) for x in parts[:8]]
    except ValueError:
        return None
    language = parts[8].strip()
    text = parts[9].strip()
    x_values = coords[0::2]
    y_values = coords[1::2]
    return {
        "left": min(x_values),
        "top": min(y_values),
        "language": language,
        "class_name": LANGUAGE_TO_CLASS.get(language, "Other"),
        "text": text,
    }


def load_grouped_texts(zip_path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith("TrainGT/TrainGT/") or not name.endswith(".txt"):
                continue
            rows = zf.read(name).decode("utf-8-sig", "replace").splitlines()
            for row in rows:
                parsed = parse_gt_line(row)
                if not parsed or parsed["text"] == "###":
                    continue
                grouped[name].append(parsed)
    return grouped


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path)
    sheet_path = Path(args.annotation_sheet)

    grouped = load_grouped_texts(zip_path)

    with sheet_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys() if rows else []

    for row in rows:
        gt_member = row["gt_member"]
        class_name = row["class_name"]
        entries = [
            item
            for item in grouped.get(gt_member, [])
            if item["class_name"] == class_name
        ]
        entries.sort(key=lambda item: (item["top"], item["left"]))
        row["source_text_reference"] = " ".join(item["text"] for item in entries)

    with sheet_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
