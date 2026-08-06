#!/usr/bin/env python3
"""Build a proportional top-scoring final subset across language buckets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-sheet", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-size", type=int, default=40)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def language_bucket(row: dict[str, str]) -> str:
    if row["source_dataset"] == "ReCTS":
        return "Chinese"
    return row["class_name"]


def allocate_counts(language_counts: dict[str, int], target_size: int) -> dict[str, int]:
    total = sum(language_counts.values())
    if total == 0:
        return {}

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0

    for language, count in language_counts.items():
        quota = count * target_size / total
        base = int(quota)
        allocations[language] = base
        assigned += base
        remainders.append((quota - base, language))

    remaining = target_size - assigned
    for _, language in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocations[language] += 1

    return allocations


def selector_score(row: dict[str, str]) -> float:
    try:
        return float(row.get("selector_score", "") or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    args = parse_args()
    gold_sheet = Path(args.gold_sheet)
    eval_csv = Path(args.eval_csv)
    output_dir = Path(args.output_dir)

    gold_rows = load_csv(gold_sheet)
    eval_rows = load_csv(eval_csv)
    if not gold_rows:
        raise SystemExit("gold sheet is empty")
    if not eval_rows:
        raise SystemExit("evaluation csv is empty")

    eval_by_id = {row["image_id"]: row for row in eval_rows}

    scored_by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    language_counts: dict[str, int] = defaultdict(int)

    for gold_row in gold_rows:
        image_id = gold_row["image_id"]
        eval_row = eval_by_id.get(image_id)
        if eval_row is None:
            raise SystemExit(f"missing evaluation row for {image_id}")
        if eval_row.get("status") != "ok":
            raise SystemExit(f"evaluation row is not ok for {image_id}: {eval_row.get('status')}")
        if not (eval_row.get("chrf") or "").strip():
            raise SystemExit(f"missing chrF value for {image_id}")

        language = language_bucket(gold_row)
        combined = dict(gold_row)
        combined["selection_language"] = language
        combined["normalized_chrf"] = eval_row["chrf"]
        combined["predicted_translation"] = eval_row.get("predicted_translation", "")
        combined["normalized_match"] = eval_row.get("normalized_match", "")
        combined["contains_match"] = eval_row.get("contains_match", "")
        scored_by_language[language].append(combined)
        language_counts[language] += 1

    allocations = allocate_counts(language_counts, args.target_size)

    selected_rows: list[dict[str, str]] = []
    for language, rows in scored_by_language.items():
        rows.sort(
            key=lambda row: (
                -float(row["normalized_chrf"]),
                row["normalized_match"] != "true",
                row["contains_match"] != "true",
                -selector_score(row),
                row["image_id"],
            )
        )
        keep = allocations.get(language, 0)
        for rank, row in enumerate(rows[:keep], start=1):
            out = dict(row)
            out["language_rank"] = str(rank)
            selected_rows.append(out)

    selected_rows.sort(
        key=lambda row: (
            row["selection_language"],
            int(row["language_rank"]),
            row["image_id"],
        )
    )
    for global_rank, row in enumerate(selected_rows, start=1):
        row["global_rank"] = str(global_rank)

    output_dir.mkdir(parents=True, exist_ok=True)

    gold_fieldnames = list(gold_rows[0].keys())
    gold_output = output_dir / "english_only_master_annotation_gold.csv"
    with gold_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gold_fieldnames)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in gold_fieldnames} for row in selected_rows])

    scored_fieldnames = gold_fieldnames + [
        "selection_language",
        "normalized_chrf",
        "predicted_translation",
        "normalized_match",
        "contains_match",
        "language_rank",
        "global_rank",
    ]
    scored_output = output_dir / "final_dataset_scores.csv"
    with scored_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scored_fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "target_size": args.target_size,
        "selected_rows": len(selected_rows),
        "selection_policy": {
            "ranking_metric": "normalized_chrf",
            "language_bucketing": "Use MLT class_name directly and merge ReCTS into Chinese.",
            "allocation_rule": "Largest remainder proportional allocation over language counts.",
        },
        "language_counts": {
            language: {
                "available": language_counts[language],
                "selected": allocations.get(language, 0),
                "selected_average_normalized_chrf": round(
                    sum(float(row["normalized_chrf"]) for row in selected_rows if row["selection_language"] == language)
                    / allocations[language],
                    4,
                )
                if allocations.get(language, 0)
                else 0.0,
            }
            for language in sorted(language_counts)
        },
        "source_dataset_counts": {
            dataset: sum(1 for row in selected_rows if row["source_dataset"] == dataset)
            for dataset in sorted({row["source_dataset"] for row in selected_rows})
        },
        "average_normalized_chrf": round(
            sum(float(row["normalized_chrf"]) for row in selected_rows) / len(selected_rows),
            4,
        )
        if selected_rows
        else 0.0,
    }
    summary_output = output_dir / "selection_summary.json"
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"rows={len(selected_rows)}")
    print(f"average_normalized_chrf={summary['average_normalized_chrf']:.4f}")
    print(f"wrote={gold_output}")
    print(f"wrote={scored_output}")
    print(f"wrote={summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
