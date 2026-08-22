"""Train both probe arms across three seeds on MLT-2019.

    python ablation/run_probe.py --data dataset.yaml --project runs/probe

Six runs of roughly 4.6 h each on two T4s. Each run writes the usual Ultralytics
``results.csv``; aggregate.py turns the set into a mean-and-SD table.

A failing run is contained rather than propagated: a Kaggle batch version that raises
saves no output at all, so one bad run would discard every completed run in the same
session. Failures are recorded and reported, and the exit status is non-zero only when
the session accomplished nothing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO

CONFIGS = Path(__file__).parent / "configs"
ARMS = {
    # Table 8, one component added per row. Row 6 is the deployed model.
    "a1_stock": "a1_stock.yaml",
    "a2_bifpn": "a2_bifpn.yaml",
    "a3_stdcbam": "a3_bifpn_stdcbam.yaml",
    "a4_mscbam": "a4_bifpn_mscbam.yaml",
    "a5_crossattn": "a5_bifpn_crossattn.yaml",
    "legacy": "full_legacy.yaml",
    "fixed": "full_fixed.yaml",
    # The architecture arms, run separately from the bottleneck probe: they answer
    # whether recall is bounded by sampling resolution rather than by feature
    # refinement, which the probe does not address.
    "rearranged": "full_rearranged.yaml",
    "p2": "full_rearranged_p2.yaml",
}
SEEDS = (42, 1337, 2024)
SNAPSHOT_FILES = ("results.csv", "args.yaml")

TRAIN_ARGS = {
    "imgsz": 480,
    "batch": 64,
    "epochs": 100,
    "cos_lr": True,
    "plots": False,
    "deterministic": True,
    # A run that died mid-epoch leaves a directory without results.csv; without this
    # Ultralytics would sidestep it as <name>2 and the skip check would never match.
    "exist_ok": True,
}


def snapshot(run_dir: Path, into: Path) -> None:
    """Copy the few KB that aggregation needs, so persistence stays cheap."""
    target = into / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for filename in SNAPSHOT_FILES:
        source = run_dir / filename
        if source.exists():
            shutil.copyfile(source, target / filename)


def train_one(arm: str, seed: int, opts: argparse.Namespace) -> None:
    YOLO(str(CONFIGS / ARMS[arm])).train(
        data=opts.data,
        project=str(opts.project),
        name=f"{arm}_seed{seed}",
        seed=seed,
        device=opts.device,
        **TRAIN_ARGS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset yaml")
    parser.add_argument("--project", type=Path, default=Path("runs/probe"), help="output root")
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="also copy results.csv and args.yaml here after each run",
    )
    opts = parser.parse_args()

    pending = [(arm, seed) for arm in opts.arms for seed in opts.seeds]
    done: list[str] = []
    failed: list[str] = []
    print(f"{len(pending)} run(s) -> {opts.project}\n")

    for index, (arm, seed) in enumerate(pending, start=1):
        name = f"{arm}_seed{seed}"
        run_dir = opts.project / name

        if (run_dir / "results.csv").exists():
            print(f"[{index}/{len(pending)}] {name}: already complete, skipping")
            done.append(name)
            if opts.snapshot:
                snapshot(run_dir, opts.snapshot)
            continue

        print(f"[{index}/{len(pending)}] {name}")
        try:
            train_one(arm, seed, opts)
        except Exception:  # noqa: BLE001 - one bad run must not discard the others
            traceback.print_exc()
            failed.append(name)
            print(f"  {name} FAILED; continuing with the remaining runs")
        else:
            done.append(name)
        finally:
            if opts.snapshot and run_dir.exists():
                snapshot(run_dir, opts.snapshot)

    print(f"\ncompleted: {done or 'none'}")
    if failed:
        print(f"failed:    {failed}")
        print("Re-run the same command to retry only the failures.")
    print(f"\nNext: python ablation/aggregate.py {opts.project}")

    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
