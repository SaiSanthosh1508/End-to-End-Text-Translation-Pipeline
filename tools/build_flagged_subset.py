#!/usr/bin/env python3
"""Build a source-only subset sheet from flagged comparison rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-sheet", required=True)
    parser.add_argument("--flagged-sheet", required=True)
    parser.add_argument("--output-sheet", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    master_sheet = Path(args.master_sheet)
    flagged_sheet = Path(args.flagged_sheet)
    output_sheet = Path(args.output_sheet)

    with master_sheet.open(newline="") as handle:
        master_rows = list(csv.DictReader(handle))
        fieldnames = list(master_rows[0].keys()) if master_rows else []

    with flagged_sheet.open(newline="") as handle:
        flagged_rows = list(csv.DictReader(handle))

    flagged_ids = {row["image_id"] for row in flagged_rows}
    subset = [row for row in master_rows if row["image_id"] in flagged_ids]

    output_sheet.parent.mkdir(parents=True, exist_ok=True)
    with output_sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset)

    print(f"rows={len(subset)}")
    print(f"wrote={output_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
