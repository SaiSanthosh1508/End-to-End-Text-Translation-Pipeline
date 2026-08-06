#!/usr/bin/env python3
"""Fill English translation references with the OpenAI Python SDK.

This script is intended for the supplementary scene-text translation evaluation
sheet. It uses the Responses API with a strict JSON schema so each row produces
structured output that is easy to review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """You are translating short scene-text snippets into English.

Rules:
- Return only the structured JSON requested by the schema.
- Preserve brand names when appropriate.
- Prefer natural short English over word-for-word awkward phrasing.
- If the source is mostly a proper noun, transliterate instead of inventing a meaning.
- If the text is ambiguous or partial, translate conservatively and explain briefly in notes.
- Never leave the translation empty unless the source text is empty.
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "english_translation": {
            "type": "string",
            "description": "Best English translation or transliteration for the source text.",
        },
        "review_flag": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How much human review this translation likely needs.",
        },
        "notes": {
            "type": "string",
            "description": "Short note about ambiguity, transliteration, or brand-name handling.",
        },
    },
    "required": ["english_translation", "review_flag", "notes"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sheet", required=True)
    parser.add_argument("--output-sheet", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N translatable rows.")
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; write placeholder outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing target_translation_reference values.")
    return parser.parse_args()


def build_user_prompt(row: dict[str, str]) -> str:
    source_dataset = row.get("source_dataset", "")
    class_name = row.get("class_name", "")
    source_text = row.get("source_text_reference", "").strip()
    notes = row.get("notes", "").strip()
    return (
        f"Dataset: {source_dataset}\n"
        f"Script/Class: {class_name}\n"
        f"Source text: {source_text}\n"
        f"Selector note: {notes}\n\n"
        "Translate the source text into English."
    )


def call_model(client: OpenAI, model: str, prompt: str) -> dict[str, str]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "translation_result",
                "schema": OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(response.output_text)


def fill_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[int, int]:
    client = None if args.dry_run else OpenAI()
    translated = 0
    reviewed = 0

    for row in rows:
        source_text = row.get("source_text_reference", "").strip()
        existing = row.get("target_translation_reference", "").strip()
        if not source_text:
            continue
        if existing and not args.overwrite:
            continue
        if args.limit and translated >= args.limit:
            break

        if args.dry_run:
            result = {
                "english_translation": f"DRY_RUN::{source_text}",
                "review_flag": "medium",
                "notes": "Dry run placeholder output.",
            }
        else:
            result = call_model(client, args.model, build_user_prompt(row))
            time.sleep(args.pause_seconds)

        row["target_translation_reference"] = result["english_translation"].strip()
        row["annotator_1"] = "openai_sdk_draft"
        row["notes"] = (row.get("notes", "").strip() + " | " + result["notes"].strip()).strip(" |")
        row["review_flag"] = result["review_flag"]
        translated += 1
        if result["review_flag"] != "low":
            reviewed += 1

    return translated, reviewed


def main() -> int:
    args = parse_args()
    input_sheet = Path(args.input_sheet)
    output_sheet = Path(args.output_sheet)

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Run with --dry-run or export an API key first.", file=sys.stderr)
        return 2

    with input_sheet.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    if "review_flag" not in fieldnames:
        fieldnames.append("review_flag")

    translated, review_needed = fill_rows(rows, args)

    output_sheet.parent.mkdir(parents=True, exist_ok=True)
    with output_sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"translated={translated}")
    print(f"review_needed={review_needed}")
    print(f"wrote={output_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
