#!/usr/bin/env python3
"""Evaluate the Hugging Face Space against the gold English sheet."""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from queue import Empty, Queue
from pathlib import Path

import httpx
from gradio_client import Client, handle_file


SPACE_URL = "https://sai-santhosh-text-translation-pipeline.hf.space/"
HTTPX_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-sheet", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def resolve_image_path(root: Path, row: dict[str, str]) -> Path | None:
    local_image_path = (row.get("local_image_path") or "").strip()
    if local_image_path:
        candidate = Path(local_image_path)
        if candidate.exists():
            return candidate

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


def evaluate_row(
    root: Path,
    row: dict[str, str],
    target_language: str,
    model_client: Client,
) -> dict[str, str]:
    image_path = resolve_image_path(root, row)
    if image_path is None:
        return {
            "image_id": row["image_id"],
            "source_dataset": row["source_dataset"],
            "gold_translation": row["target_translation_reference"],
            "predicted_translation": "",
            "normalized_match": "",
            "contains_match": "",
            "status": "missing_local_image",
        }

    result = model_client.predict(
        handle_file(str(image_path)),
        target_language,
        api_name="/pipeline",
    )
    predicted = extract_translation_text(
        result[2] if isinstance(result, (list, tuple)) and len(result) >= 3 else result
    )
    gold_norm = normalize(row["target_translation_reference"])
    pred_norm = normalize(predicted)
    match = pred_norm == gold_norm
    contains_match = bool(gold_norm) and (gold_norm in pred_norm or pred_norm in gold_norm)
    return {
        "image_id": row["image_id"],
        "source_dataset": row["source_dataset"],
        "gold_translation": row["target_translation_reference"],
        "predicted_translation": predicted,
        "normalized_match": str(match).lower(),
        "contains_match": str(contains_match).lower(),
        "status": "ok",
    }


def main() -> int:
    args = parse_args()
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")

    root = args.project_root.resolve()
    gold_sheet = Path(args.gold_sheet)
    output_csv = Path(args.output_csv)

    with gold_sheet.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

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

    print(
        f"Queued {len(pending_rows)} rows with max_concurrency={min(args.max_concurrency, len(pending_rows)) if pending_rows else 0}",
        flush=True,
    )

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

    task_queue: Queue[tuple[int, dict[str, str]]] = Queue()
    result_queue: Queue[tuple[str, int, dict[str, str] | str]] = Queue()
    stop_event = threading.Event()

    for idx, row in enumerate(pending_rows, start=1):
        task_queue.put((idx, row))

    def worker() -> None:
        try:
            client = Client(SPACE_URL, httpx_kwargs={"timeout": HTTPX_TIMEOUT})
        except Exception as exc:  # noqa: BLE001
            while True:
                try:
                    idx, _row = task_queue.get_nowait()
                except Empty:
                    return
                result_queue.put(("error", idx, f"client_init_error:{exc}"))
                task_queue.task_done()
            return

        while not stop_event.is_set():
            try:
                idx, row = task_queue.get_nowait()
            except Empty:
                return

            result_queue.put(("start", idx, row["image_id"]))
            try:
                evaluated = evaluate_row(root, row, args.target_language, client)
                result_queue.put(("ok", idx, evaluated))
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("error", idx, f"error:{exc}"))
            finally:
                task_queue.task_done()

    worker_count = min(args.max_concurrency, len(pending_rows)) if pending_rows else 0
    threads = [
        threading.Thread(target=worker, name=f"hf-eval-worker-{index}", daemon=True)
        for index in range(worker_count)
    ]
    for thread in threads:
        thread.start()

    started = 0
    completed = 0
    try:
        while completed < len(pending_rows):
            try:
                event, idx, payload = result_queue.get(timeout=5.0)
            except Empty:
                in_flight = started - completed
                remaining = len(pending_rows) - completed
                print(
                    f"[progress] completed={completed}/{len(pending_rows)} "
                    f"in_flight={in_flight} remaining={remaining}",
                    flush=True,
                )
                continue

            row = pending_rows[idx - 1]
            if event == "start":
                started += 1
                image_path = resolve_image_path(root, row)
                image_name = image_path.name if image_path is not None else "missing_local_image"
                print(f"[{started}/{len(pending_rows)}] {row['image_id']} -> {image_name}", flush=True)
                continue

            if event == "error":
                completed += 1
                output_rows_by_id[row["image_id"]] = {
                    "image_id": row["image_id"],
                    "source_dataset": row["source_dataset"],
                    "gold_translation": row["target_translation_reference"],
                    "predicted_translation": "",
                    "normalized_match": "false",
                    "contains_match": "false",
                    "status": str(payload),
                }
                flush()
                print(f"  failure on {row['image_id']}: {payload}", flush=True)
                continue

            completed += 1
            output_rows_by_id[row["image_id"]] = payload
            flush()
    except KeyboardInterrupt:
        stop_event.set()
        flush()
        print("Interrupted. Completed rows were saved; rerun the same command to resume.", flush=True)
        raise SystemExit(130)

    output_rows = ordered_output_rows(rows, output_rows_by_id)
    total = len(output_rows)
    ok = sum(1 for row in output_rows if row["status"] == "ok")
    matches = sum(1 for row in output_rows if row["normalized_match"] == "true")
    contains = sum(1 for row in output_rows if row["contains_match"] == "true")
    print(json.dumps({"total": total, "ok": ok, "exact_matches": matches, "contains_matches": contains, "output_csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
