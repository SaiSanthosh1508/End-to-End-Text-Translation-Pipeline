#!/usr/bin/env python3
"""Fill English translation references using the Codex Python SDK.

This uses local Codex threads rather than the OpenAI API SDK. It is designed
for short scene-text translation rows in the supplementary evaluation sheet.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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
    parser.add_argument("--model", default="gpt-5.6-terra")
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


def apply_result(row: dict[str, str], result: dict[str, str]) -> None:
    row["target_translation_reference"] = result["english_translation"].strip()
    row["annotator_1"] = "codex_sdk_draft"
    row["review_flag"] = result["review_flag"].strip()
    note = result["notes"].strip()
    existing = row.get("notes", "").strip()
    row["notes"] = (existing + " | " + note).strip(" |") if note else existing


def main() -> int:
    args = parse_args()
    input_sheet = Path(args.input_sheet)
    output_sheet = Path(args.output_sheet)

    source_sheet = output_sheet if output_sheet.exists() and not args.overwrite else input_sheet
    if source_sheet == output_sheet and output_sheet.exists():
        print(f"Resuming from existing output: {output_sheet}", flush=True)
    else:
        print(f"Starting from input sheet: {input_sheet}", flush=True)

    with source_sheet.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    if "review_flag" not in fieldnames:
        fieldnames.append("review_flag")

    translated = 0
    failures = 0

    total_candidates = sum(
        1
        for row in rows
        if row.get("source_text_reference", "").strip()
        and (args.overwrite or not row.get("target_translation_reference", "").strip())
    )
    processed = 0

    def flush_rows() -> None:
        output_sheet.parent.mkdir(parents=True, exist_ok=True)
        with output_sheet.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def log(message: str) -> None:
        print(message, flush=True)

    if args.dry_run:
        for row in rows:
            source = row.get("source_text_reference", "").strip()
            existing = row.get("target_translation_reference", "").strip()
            if not source or (existing and not args.overwrite):
                continue
            if args.limit and translated >= args.limit:
                break
            processed += 1
            log(f"[{processed}/{total_candidates}] {row.get('image_id','')} dry-run")
            apply_result(
                row,
                {
                    "english_translation": f"DRY_RUN::{source}",
                    "review_flag": "medium",
                    "notes": "Dry run placeholder output.",
                },
            )
            translated += 1
    else:
        for row in rows:
            source = row.get("source_text_reference", "").strip()
            existing = row.get("target_translation_reference", "").strip()
            if not source or (existing and not args.overwrite):
                continue
            if args.limit and translated >= args.limit:
                break
            processed += 1
            log(
                f"[{processed}/{total_candidates}] processing {row.get('image_id','')} "
                f"filled={translated} failures={failures}"
            )

            try:
                with Codex() as codex:
                    thread = codex.thread_start(model=args.model, sandbox=Sandbox.read_only)
                    result = thread.run(build_prompt(row))
                parsed = extract_json(result.final_response)
            except Exception as exc:
                failures += 1
                row["review_flag"] = "high"
                row["annotator_1"] = row.get("annotator_1", "") or "codex_sdk_error"
                row["notes"] = (row.get("notes", "").strip() + f" | Codex failure: {exc}").strip(" |")
                flush_rows()
                log(f"  failure on {row.get('image_id','')}: {exc}")
                continue

            apply_result(row, parsed)
            translated += 1
            flush_rows()
            log(
                f"  done {row.get('image_id','')} -> {row.get('target_translation_reference','')[:60]}"
            )

    flush_rows()

    print(f"translated={translated}")
    print(f"failures={failures}")
    print(f"wrote={output_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
