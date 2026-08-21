"""Summarise probe runs as mean and standard deviation across seeds.

    python ablation/aggregate.py runs/probe

Each run is reduced by averaging its final epochs rather than reading the last row.
Within the existing MLT run, mAP50 moves 0.0027 and precision 0.022 across epochs
80-100, which is the same order as the entire spread of Table 8; averaging a window
removes that checkpoint lottery from the comparison. The window is reported in the
output so it can be stated in the caption.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = {
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def read_run(results: Path, last_n: int) -> dict[str, float]:
    """Average each metric over the final ``last_n`` epochs of one run."""
    with results.open(encoding="utf-8") as handle:
        rows = [{k.strip(): v for k, v in row.items() if k} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{results} has no rows")

    window = rows[-last_n:]
    return {
        label: statistics.fmean(float(row[column]) for row in window)
        for label, column in METRICS.items()
    }


def collect(root: Path, last_n: int) -> dict[str, dict[int, dict[str, float]]]:
    runs: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for results in sorted(root.glob("*/results.csv")):
        name = results.parent.name
        arm, _, seed = name.rpartition("_seed")
        if not arm or not seed.isdigit():
            print(f"  skipping {name}: expected <arm>_seed<n>", file=sys.stderr)
            continue
        runs[arm][int(seed)] = read_run(results, last_n)
    return runs


def summarise(seeds: dict[int, dict[str, float]]) -> dict[str, tuple[float, float]]:
    return {
        label: (
            statistics.fmean(run[label] for run in seeds.values()),
            statistics.stdev([run[label] for run in seeds.values()]) if len(seeds) > 1 else 0.0,
        )
        for label in METRICS
    }


def render_table(stats: dict[str, dict[str, tuple[float, float]]]) -> str:
    header = f"{'arm':<10}" + "".join(f"{label:>18}" for label in METRICS)
    lines = [header, "-" * len(header)]
    for arm, values in stats.items():
        cells = "".join(f"{mean:>11.4f} ±{sd:<5.4f}" for mean, sd in values.values())
        lines.append(f"{arm:<10}{cells}")
    return "\n".join(lines)


def render_latex(stats: dict[str, dict[str, tuple[float, float]]], labels: dict[str, str]) -> str:
    rows = [
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Configuration} & \\textbf{Prec.} & \\textbf{Rec.} & "
        "\\textbf{mAP50} & \\textbf{mAP50-95} \\\\",
        "\\hline",
    ]
    for arm, values in stats.items():
        cells = " & ".join(f"{mean:.3f} $\\pm$ {sd:.3f}" for mean, sd in values.values())
        rows.append(f"{labels.get(arm, arm)} & {cells} \\\\")
    rows += ["\\hline", "\\end{tabular}"]
    return "\n".join(rows)


def verdict(
    stats: dict[str, dict[str, tuple[float, float]]],
    seed_counts: dict[str, int],
) -> str:
    if set(stats) != {"legacy", "fixed"}:
        return ""
    thin = {arm: n for arm, n in seed_counts.items() if n < 3}
    if thin:
        listed = ", ".join(f"{arm}={n}" for arm, n in sorted(thin.items()))
        return (
            f"\nNo verdict: needs 3 seeds per arm, have {listed}. A standard deviation "
            "over\nfewer than three runs is not a variance estimate worth deciding on."
        )

    lines = ["", "Fixed minus legacy, in units of the pooled standard deviation:"]
    decisive = False
    for label in METRICS:
        (m_fixed, s_fixed), (m_legacy, s_legacy) = stats["fixed"][label], stats["legacy"][label]
        delta = m_fixed - m_legacy
        pooled = ((s_fixed**2 + s_legacy**2) / 2) ** 0.5
        ratio = delta / pooled if pooled else float("inf") if delta else 0.0
        decisive |= abs(ratio) >= 1.0
        lines.append(f"  {label:<10} {delta:+.4f}   {ratio:+.2f} SD")
    lines.append("")
    lines.append(
        "At least one metric moves by 1 SD or more: the bottleneck fix matters, so run\n"
        "the full six-configuration ablation on the fixed modules."
        if decisive
        else "No metric moves by 1 SD: the fix changes nothing measurable. Run the\n"
        "six-configuration ablation on the as-deployed modules and report the\n"
        "displaced reduction ratio as a limitation."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory holding <arm>_seed<n>/ run dirs")
    parser.add_argument("--last-n", type=int, default=5, help="epochs to average per run")
    opts = parser.parse_args()

    runs = collect(opts.root, opts.last_n)
    if not runs:
        return f"no runs found under {opts.root}"

    stats = {arm: summarise(seeds) for arm, seeds in sorted(runs.items())}

    print(f"Averaged over the final {opts.last_n} epochs of each run.")
    for arm, seeds in sorted(runs.items()):
        print(f"  {arm}: seeds {sorted(seeds)}")
    print()
    print(render_table(stats))
    print(verdict(stats, {arm: len(seeds) for arm, seeds in runs.items()}))
    print()
    print(render_latex(stats, {"legacy": "Full model (as deployed)", "fixed": "Full model ($r{=}16$)"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
