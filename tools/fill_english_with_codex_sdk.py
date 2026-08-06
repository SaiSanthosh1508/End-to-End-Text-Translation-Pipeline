#!/usr/bin/env python3
"""Fill English translation references using the Codex Python SDK, resumably."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
from queue import Empty, Queue
from pathlib import Path

from openai_codex import Codex, Sandbox


SYSTEM_PROMPT = """You translate short scene-text snippets into English.

Return JSON only with this exact shape:
{
  "english_translation": string,
  "review_flag": "low" | "medium" | "high",
  "notes": string
}

Rules:
- Prefer natural short English.
- Preserve brand names where appropriate.
- If the text is primarily a proper noun, transliterate instead of inventing meaning.
- If the text is partial or ambiguous, translate conservatively and explain briefly in notes.
- Never return an empty english_translation when source text is non-empty.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sheet", required=True)
    parser.add_argument("--output-sheet", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("CODEX_MODEL", ""),
        help="Optional Codex model override. Omit to use the account default.",
    )
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_prompt(row: dict[str, str]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Dataset: {row.get('source_dataset', '')}\n"
        f"Split: {row.get('split', '')}\n"
        f"Class: {row.get('class_name', '')}\n"
        f"Source text: {row.get('source_text_reference', '').strip()}\n"
        f"Existing note: {row.get('notes', '').strip()}\n"
    )


def extract_json(text: str) -> dict[str, str]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))

    bare = re.search(r"(\{.*\})", text, re.S)
    if bare:
        return json.loads(bare.group(1))

    raise ValueError("Could not parse JSON from Codex response")


def ensure_fieldnames(fieldnames: list[str]) -> list[str]:
    for field in ["review_flag", "codex_status", "codex_model", "codex_error"]:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def strip_legacy_failure_notes(text: str) -> str:
    if not text:
        return ""
    parts = [part.strip() for part in text.split(" | ") if part.strip()]
    kept = [part for part in parts if not part.startswith("Codex failure:")]
    return " | ".join(kept)


def apply_result(row: dict[str, str], result: dict[str, str], model: str) -> None:
    row["target_translation_reference"] = result["english_translation"].strip()
    row["annotator_1"] = "codex_sdk_draft"
    row["review_flag"] = result["review_flag"].strip()
    row["codex_status"] = "ok"
    row["codex_model"] = model or "account_default"
    row["codex_error"] = ""
    note = result["notes"].strip()
    existing = strip_legacy_failure_notes(row.get("notes", "").strip())
    row["notes"] = (existing + " | " + note).strip(" |") if note else existing


def is_unsupported_model_error(message: str) -> bool:
    return "not supported when using Codex with a ChatGPT account" in message


def should_process(row: dict[str, str], overwrite: bool) -> bool:
    source = row.get("source_text_reference", "").strip()
    existing = row.get("target_translation_reference", "").strip()
    return bool(source) and (overwrite or not existing)


def run_codex_translation(row: dict[str, str], model: str) -> tuple[str, dict[str, str] | str]:
    with Codex() as codex:
        thread = codex.thread_start(model=model or None, sandbox=Sandbox.read_only)
        result = thread.run(build_prompt(row))
    return "ok", extract_json(result.final_response)


def main() -> int:
    args = parse_args()
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")

    input_sheet = Path(args.input_sheet)
    output_sheet = Path(args.output_sheet)

    source_sheet = output_sheet if output_sheet.exists() and not args.overwrite else input_sheet
    if source_sheet == output_sheet and output_sheet.exists():
        print(f"Resuming from existing output: {output_sheet}", flush=True)
    else:
        print(f"Starting from input sheet: {input_sheet}", flush=True)

    with source_sheet.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = ensure_fieldnames(list(rows[0].keys()) if rows else [])

    translated = 0
    failures = 0
    total_candidates = sum(
        1
        for row in rows
        if should_process(row, args.overwrite)
    )

    def flush_rows() -> None:
        output_sheet.parent.mkdir(parents=True, exist_ok=True)
        with output_sheet.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if args.dry_run:
        processed = 0
        for row in rows:
            if not should_process(row, args.overwrite):
                continue
            if args.limit and translated >= args.limit:
                break
            processed += 1
            print(f"[{processed}/{total_candidates}] {row.get('image_id','')} dry-run", flush=True)
            apply_result(
                row,
                {
                    "english_translation": f"DRY_RUN::{source}",
                    "review_flag": "medium",
                    "notes": "Dry run placeholder output.",
                },
                args.model,
            )
            translated += 1
            flush_rows()
    else:
        tasks: list[tuple[int, dict[str, str]]] = []
        for idx, row in enumerate(rows):
            if not should_process(row, args.overwrite):
                continue
            tasks.append((idx, row))
            if args.limit and len(tasks) >= args.limit:
                break

        task_queue: Queue[tuple[int, dict[str, str]]] = Queue()
        result_queue: Queue[tuple[str, int, dict[str, str] | str]] = Queue()
        stop_event = threading.Event()

        for task in tasks:
            task_queue.put(task)

        def worker() -> None:
            while not stop_event.is_set():
                try:
                    idx, row = task_queue.get_nowait()
                except Empty:
                    return

                image_id = row.get("image_id", "")
                result_queue.put(("start", idx, image_id))
                try:
                    status, payload = run_codex_translation(row, args.model)
                    result_queue.put((status, idx, payload))
                except Exception as exc:  # noqa: BLE001
                    result_queue.put(("error", idx, str(exc)))
                finally:
                    task_queue.task_done()

        worker_count = min(args.max_concurrency, len(tasks)) if tasks else 0
        print(
            f"Queued {len(tasks)} rows with max_concurrency={worker_count} "
            f"model={(args.model or 'account_default')}",
            flush=True,
        )

        threads = [
            threading.Thread(target=worker, name=f"codex-worker-{index}", daemon=True)
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()

        started = 0
        completed = 0
        try:
            while completed < len(tasks):
                try:
                    event, idx, payload = result_queue.get(timeout=5.0)
                except Empty:
                    in_flight = started - completed
                    remaining = len(tasks) - completed
                    print(
                        f"[progress] completed={completed}/{len(tasks)} "
                        f"in_flight={in_flight} remaining={remaining} "
                        f"filled={translated} failures={failures}",
                        flush=True,
                    )
                    continue

                row = rows[idx]
                image_id = row.get("image_id", "")

                if event == "start":
                    started += 1
                    print(
                        f"[{started}/{len(tasks)}] processing {image_id} "
                        f"filled={translated} failures={failures}",
                        flush=True,
                    )
                    continue

                if event == "error":
                    error_message = str(payload)
                    failures += 1
                    completed += 1
                    row["review_flag"] = "high"
                    row["codex_status"] = "error"
                    row["codex_model"] = args.model or "account_default"
                    row["codex_error"] = error_message
                    row["annotator_1"] = row.get("annotator_1", "") or "codex_sdk_error"
                    flush_rows()
                    print(f"  failure on {image_id}: {error_message}", flush=True)
                    if is_unsupported_model_error(error_message):
                        stop_event.set()
                        raise SystemExit(
                            "Selected model is unsupported for your Codex account. "
                            "Rerun without --model to use the account default."
                        )
                    continue

                parsed = payload
                apply_result(row, parsed, args.model)
                translated += 1
                completed += 1
                flush_rows()
                print(
                    f"  done {image_id} -> {row.get('target_translation_reference','')[:60]}",
                    flush=True,
                )
        except KeyboardInterrupt:
            stop_event.set()
            flush_rows()
            print("Interrupted. Completed rows were saved; rerun the same command to resume.", flush=True)
            raise SystemExit(130)

    flush_rows()
    print(f"translated={translated}")
    print(f"failures={failures}")
    print(f"wrote={output_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
