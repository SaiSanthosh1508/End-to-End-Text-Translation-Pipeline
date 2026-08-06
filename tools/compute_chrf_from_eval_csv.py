#!/usr/bin/env python3
"""Compute chrF scores from the saved HF Space evaluation CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--normalize-text", action="store_true")
    return parser.parse_args()


def ngrams(text: str, n: int) -> list[str]:
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def f_score(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def chrf_score(reference: str, hypothesis: str, max_order: int = 6, beta: float = 2.0) -> float:
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0

    scores = []
    for n in range(1, max_order + 1):
        ref_ngrams = ngrams(ref, n)
        hyp_ngrams = ngrams(hyp, n)
        if not ref_ngrams or not hyp_ngrams:
            scores.append(0.0)
            continue

        ref_counts = defaultdict(int)
        hyp_counts = defaultdict(int)
        for gram in ref_ngrams:
            ref_counts[gram] += 1
        for gram in hyp_ngrams:
            hyp_counts[gram] += 1

        overlap = 0
        for gram, count in hyp_counts.items():
            overlap += min(count, ref_counts.get(gram, 0))

        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        scores.append(f_score(precision, recall, beta=beta))

    return sum(scores) / len(scores)


def main() -> int:
    args = parse_args()
    eval_csv = Path(args.eval_csv)
    output_csv = Path(args.output_csv)

    with eval_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    if "chrf" not in fieldnames:
        fieldnames.append("chrf")

    for row in rows:
        if row.get("status") != "ok":
            row["chrf"] = ""
            continue
        gold = row.get("gold_translation", "")
        pred = row.get("predicted_translation", "")
        if args.normalize_text:
            gold = normalize(gold)
            pred = normalize(pred)
        row["chrf"] = f"{chrf_score(gold, pred):.4f}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("chrf")]
    avg = sum(float(r["chrf"]) for r in ok_rows) / len(ok_rows) if ok_rows else 0.0
    exact_rows = [r for r in ok_rows if r.get("normalized_match") == "true"]
    exact_avg = sum(float(r["chrf"]) for r in exact_rows) / len(exact_rows) if exact_rows else 0.0
    print(f"rows={len(rows)}")
    print(f"ok_rows={len(ok_rows)}")
    print(f"average_chrf={avg:.4f}")
    print(f"exact_match_rows={len(exact_rows)}")
    print(f"exact_match_average_chrf={exact_avg:.4f}")
    print(f"wrote={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
