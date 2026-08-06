#!/usr/bin/env python3
"""Generate a first-pass English translation draft for the annotation sheet."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from deep_translator import GoogleTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sheet", required=True)
    parser.add_argument("--output-sheet", required=True)
    parser.add_argument("--pause-seconds", type=float, default=0.4)
    return parser.parse_args()


def translate_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    translator = GoogleTranslator(source="auto", target="en")
    return translator.translate(text)


def main() -> int:
    args = parse_args()
    input_sheet = Path(args.input_sheet)
    output_sheet = Path(args.output_sheet)

    with input_sheet.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    translated = 0
    failures = 0
    for row in rows:
        source = row.get("source_text_reference", "").strip()
        existing = row.get("target_translation_reference", "").strip()
        if existing or not source:
            continue
        try:
            row["target_translation_reference"] = translate_text(source)
            row["annotator_1"] = row.get("annotator_1", "") or "assistant_draft"
            translated += 1
        except Exception:
            row["target_translation_reference"] = ""
            failures += 1
        time.sleep(args.pause_seconds)

    output_sheet.parent.mkdir(parents=True, exist_ok=True)
    with output_sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"translated={translated}")
    print(f"failures={failures}")
    print(f"wrote={output_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
