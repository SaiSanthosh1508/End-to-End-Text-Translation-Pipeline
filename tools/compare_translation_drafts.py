#!/usr/bin/env python3
"""Compare two translation draft CSVs and flag rows for manual review."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-a", required=True)
    parser.add_argument("--draft-b", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def main() -> int:
    args = parse_args()
    draft_a = Path(args.draft_a)
    draft_b = Path(args.draft_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with draft_a.open(newline="") as handle:
        rows_a = list(csv.DictReader(handle))
    with draft_b.open(newline="") as handle:
        rows_b = list(csv.DictReader(handle))

    by_id_a = {row["image_id"]: row for row in rows_a}
    by_id_b = {row["image_id"]: row for row in rows_b}
    image_ids = sorted(set(by_id_a) & set(by_id_b))

    agreement_rows = []
    flagged_rows = []
    stats = Counter()

    for image_id in image_ids:
        a = by_id_a[image_id]
        b = by_id_b[image_id]

        ta = a.get("target_translation_reference", "")
        tb = b.get("target_translation_reference", "")
        fa = a.get("review_flag", "")
        fb = b.get("review_flag", "")

        same_translation = normalize(ta) == normalize(tb)
        low_low = fa == "low" and fb == "low"

        merged = {
            "image_id": image_id,
            "source_dataset": a.get("source_dataset", ""),
            "split": a.get("split", ""),
            "class_name": a.get("class_name", ""),
            "source_text_reference": a.get("source_text_reference", ""),
            "draft_a_translation": ta,
            "draft_a_flag": fa,
            "draft_b_translation": tb,
            "draft_b_flag": fb,
            "agreement": "yes" if same_translation else "no",
            "manual_review_needed": "no" if same_translation and low_low else "yes",
            "notes_a": a.get("notes", ""),
            "notes_b": b.get("notes", ""),
        }

        if same_translation and low_low:
            agreement_rows.append(merged)
            stats["agreed_low_low"] += 1
        else:
            flagged_rows.append(merged)
            if not same_translation:
                stats["translation_mismatch"] += 1
            if not low_low:
                stats["review_flagged"] += 1

    agreed_path = output_dir / "agreed_rows.csv"
    flagged_path = output_dir / "flagged_rows.csv"
    summary_path = output_dir / "summary.txt"

    if agreement_rows:
        with agreed_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agreement_rows)

    if flagged_rows:
        with flagged_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flagged_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flagged_rows)

    with summary_path.open("w") as handle:
        handle.write(f"total_rows={len(image_ids)}\n")
        handle.write(f"agreed_low_low={stats['agreed_low_low']}\n")
        handle.write(f"translation_mismatch={stats['translation_mismatch']}\n")
        handle.write(f"review_flagged={stats['review_flagged']}\n")
        handle.write(f"flagged_rows={len(flagged_rows)}\n")

    print(summary_path.read_text().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
