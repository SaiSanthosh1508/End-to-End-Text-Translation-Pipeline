#!/usr/bin/env python3
"""Evaluate the Hugging Face Space against the gold English sheet."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import httpx
from gradio_client import Client, handle_file


SPACE_URL = "https://sai-santhosh-text-translation-pipeline.hf.space/"
HTTPX_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-sheet", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def resolve_image_path(root: Path, row: dict[str, str]) -> Path | None:
    image_id = row["image_id"]
    dataset = row["source_dataset"]
    if dataset == "ICDAR MLT-2019":
        matches = sorted((root / "translation_subset/mlt_from_zip/selected_images").glob(f"**/{image_id}.*"))
    else:
        matches = sorted((root / "translation_subset/rects_from_zips/selected_images").glob(f"{image_id}.*"))
    return matches[0] if matches else None


def extract_translation_text(result_obj) -> str:
    if isinstance(result_obj, dict):
        data = result_obj.get("data", [])
        rows = []
        for row in data:
            if isinstance(row, list) and len(row) >= 3:
                rows.append(str(row[2]))
        return " ".join(rows).strip()
    if isinstance(result_obj, list):
        rows = []
        for row in result_obj:
            if isinstance(row, list) and len(row) >= 3:
                rows.append(str(row[2]))
        return " ".join(rows).strip()
    return ""


def row_is_completed(row: dict[str, str]) -> bool:
    status = row.get("status", "")
    return status in {"ok", "missing_local_image"}


def ordered_output_rows(
    gold_rows: list[dict[str, str]],
    output_rows_by_id: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return [output_rows_by_id[row["image_id"]] for row in gold_rows if row["image_id"] in output_rows_by_id]


def main() -> int:
    args = parse_args()
    root = Path("/Users/saisanthosh/Documents/paper-exp/End-to-End-Text-Translation-Pipeline")
    gold_sheet = Path(args.gold_sheet)
    output_csv = Path(args.output_csv)

    with gold_sheet.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    client = Client(SPACE_URL, httpx_kwargs={"timeout": HTTPX_TIMEOUT})
    output_rows_by_id: dict[str, dict[str, str]] = {}
    completed_ids = set()
    retry_ids = set()

    if output_csv.exists() and not args.overwrite:
        with output_csv.open(newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        for existing_row in existing_rows:
            image_id = existing_row["image_id"]
            if row_is_completed(existing_row):
                output_rows_by_id[image_id] = existing_row
                completed_ids.add(image_id)
            else:
                retry_ids.add(image_id)
        print(f"Resuming from existing output: {output_csv}", flush=True)
        if retry_ids:
            print(
                f"Retrying {len(retry_ids)} previously incomplete/error rows",
                flush=True,
            )
    else:
        print(f"Starting fresh output: {output_csv}", flush=True)

    pending_rows = [row for row in rows if row["image_id"] not in completed_ids]
    if args.limit:
        pending_rows = pending_rows[: args.limit]

    fieldnames = [
        "image_id",
        "source_dataset",
        "gold_translation",
        "predicted_translation",
        "normalized_match",
        "contains_match",
        "status",
    ]

    def flush() -> None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_output_rows(rows, output_rows_by_id))

    for idx, row in enumerate(pending_rows, start=1):
        image_path = resolve_image_path(root, row)
        if image_path is None:
            output_rows_by_id[row["image_id"]] = {
                "image_id": row["image_id"],
                "source_dataset": row["source_dataset"],
                "gold_translation": row["target_translation_reference"],
                "predicted_translation": "",
                "normalized_match": "",
                "contains_match": "",
                "status": "missing_local_image",
            }
            flush()
            print(f"[{idx}/{len(pending_rows)}] {row['image_id']} missing local image", flush=True)
            continue

        print(f"[{idx}/{len(pending_rows)}] {row['image_id']} -> {image_path.name}", flush=True)
        try:
            result = client.predict(
                handle_file(str(image_path)),
                args.target_language,
                api_name="/pipeline",
            )
            predicted = extract_translation_text(result[2] if isinstance(result, (list, tuple)) and len(result) >= 3 else result)
            gold_norm = normalize(row["target_translation_reference"])
            pred_norm = normalize(predicted)
            match = pred_norm == gold_norm
            contains_match = bool(gold_norm) and (gold_norm in pred_norm or pred_norm in gold_norm)
            status = "ok"
        except Exception as exc:
            predicted = ""
            match = False
            contains_match = False
            status = f"error:{exc}"

        output_rows_by_id[row["image_id"]] = {
            "image_id": row["image_id"],
            "source_dataset": row["source_dataset"],
            "gold_translation": row["target_translation_reference"],
            "predicted_translation": predicted,
            "normalized_match": str(match).lower(),
            "contains_match": str(contains_match).lower(),
            "status": status,
        }
        flush()

    output_rows = ordered_output_rows(rows, output_rows_by_id)
    total = len(output_rows)
    ok = sum(1 for row in output_rows if row["status"] == "ok")
    matches = sum(1 for row in output_rows if row["normalized_match"] == "true")
    contains = sum(1 for row in output_rows if row["contains_match"] == "true")
    print(json.dumps({"total": total, "ok": ok, "exact_matches": matches, "contains_matches": contains, "output_csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
