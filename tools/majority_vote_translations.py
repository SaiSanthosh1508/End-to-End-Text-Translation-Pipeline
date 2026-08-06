#!/usr/bin/env python3
"""Resolve three translation drafts by majority vote and flag unresolved rows."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-a", required=True)
    parser.add_argument("--draft-b", required=True)
    parser.add_argument("--draft-c", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load(path: str):
        with Path(path).open(newline="") as handle:
            return {row["image_id"]: row for row in csv.DictReader(handle)}

    a = load(args.draft_a)
    b = load(args.draft_b)
    c = load(args.draft_c)

    image_ids = sorted(set(a) & set(b) & set(c))
    accepted = []
    unresolved = []
    stats = Counter()

    for image_id in image_ids:
        ra, rb, rc = a[image_id], b[image_id], c[image_id]
        vals = [
            ("a", ra.get("target_translation_reference", "")),
            ("b", rb.get("target_translation_reference", "")),
            ("c", rc.get("target_translation_reference", "")),
        ]
        normalized = [(name, value, normalize(value)) for name, value in vals]
        counts = Counter(item[2] for item in normalized if item[2])
        top_norm, top_count = ("", 0)
        if counts:
            top_norm, top_count = counts.most_common(1)[0]

        record = {
            "image_id": image_id,
            "source_dataset": ra.get("source_dataset", ""),
            "split": ra.get("split", ""),
            "class_name": ra.get("class_name", ""),
            "source_text_reference": ra.get("source_text_reference", ""),
            "draft_a_translation": ra.get("target_translation_reference", ""),
            "draft_b_translation": rb.get("target_translation_reference", ""),
            "draft_c_translation": rc.get("target_translation_reference", ""),
            "draft_a_flag": ra.get("review_flag", ""),
            "draft_b_flag": rb.get("review_flag", ""),
            "draft_c_flag": rc.get("review_flag", ""),
        }

        if top_count >= 2 and top_norm:
            winner = next(value for _, value, norm in normalized if norm == top_norm)
            accepted.append(
                {
                    **record,
                    "final_translation": winner,
                    "resolution": "majority_vote",
                }
            )
            stats["accepted"] += 1
        else:
            unresolved.append(
                {
                    **record,
                    "final_translation": "",
                    "resolution": "manual_review",
                }
            )
            stats["unresolved"] += 1

    if accepted:
        with (output_dir / "accepted_majority_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(accepted[0].keys()))
            writer.writeheader()
            writer.writerows(accepted)

    if unresolved:
        with (output_dir / "unresolved_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(unresolved[0].keys()))
            writer.writeheader()
            writer.writerows(unresolved)

    with (output_dir / "summary.txt").open("w") as handle:
        handle.write(f"total_rows={len(image_ids)}\n")
        handle.write(f"accepted={stats['accepted']}\n")
        handle.write(f"unresolved={stats['unresolved']}\n")

    print((output_dir / "summary.txt").read_text().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
