"""Regenerate the class-wise MLT-2019 table from the trained checkpoint.

    python ablation/revalidate.py --weights best.pt --data dataset.yaml

Table 5 in the manuscript disagrees with Fig. 11(b) on every class, by +1.6 to +15.7
mAP50 points, all in the same direction. Its implied overall mAP50 is 73.8 while the
figure legend says 66.3 and results.csv says 66.1. Precision reconciles (80.3 vs 81.0)
but recall and mAP do not, so those columns did not come from the same evaluation as
the rest of the paper. This runs the evaluation again and prints a table to replace it.

Validation is inference only: roughly two minutes for 1,000 images on one T4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_MAP50 = 0.663  # Fig. 11(b) "all classes" legend
TOLERANCE = 0.01


def register_pickle_aliases() -> None:
    """Expose the custom modules where the checkpoint expects to find them.

    ``best.pt`` pickles class references by their original import path
    (``ultralytics.nn.modules.block``), whereas install_modules.py places them in
    ``...modules.custom``. Without the aliases torch.load raises AttributeError.
    """
    import ultralytics.nn.modules.block as block
    from ultralytics.nn.modules import custom

    for name in custom.__all__:
        if not hasattr(block, name):
            setattr(block, name, getattr(custom, name))


def find(base: Path, pattern: str) -> Path | None:
    return next((p for p in base.glob(pattern)), None)


def per_class_rows(result: object, names: dict[int, str]) -> list[tuple[str, float, float, float, float]]:
    box = result.box
    rows = []
    for i, cls_id in enumerate(box.ap_class_index):
        rows.append(
            (
                names[int(cls_id)],
                float(box.p[i]) * 100,
                float(box.r[i]) * 100,
                float(box.ap50[i]) * 100,
                float(box.ap[i]) * 100,
            )
        )
    return rows


def latex(rows: list[tuple[str, float, float, float, float]]) -> str:
    head = [
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & "
        "\\textbf{mAP0.5} & \\textbf{mAP0.5-0.95} \\\\",
        "\\hline",
    ]
    body = [f"{n:<9}& {p:.1f} & {r:.1f} & {a50:.1f} & {a:.1f} \\\\" for n, p, r, a50, a in rows]
    return "\n".join(head + body + ["\\hline", "\\end{tabular}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="trained MLT checkpoint (best.pt)")
    parser.add_argument("--data", type=Path, help="dataset yaml with a val split")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--device", default="0")
    opts = parser.parse_args()

    weights = opts.weights or find(Path("/kaggle/input"), "**/Text_Translation_Pipeline/best.pt")
    if weights is None or not weights.exists():
        return "could not find best.pt; pass --weights"
    if weights.stat().st_size < 1_000_000:
        return f"{weights} is {weights.stat().st_size} B, which is a git-lfs pointer, not the model"

    register_pickle_aliases()
    from ultralytics import YOLO

    print(f"weights: {weights}\ndata:    {opts.data}\n")
    model = YOLO(str(weights))
    result = model.val(data=str(opts.data), imgsz=opts.imgsz, split="val",
                       device=opts.device, plots=False, verbose=False)

    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    rows = per_class_rows(result, names)

    print(f"{'class':<10}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>11}")
    print("-" * 46)
    for n, p, r, a50, a in rows:
        print(f"{n:<10}{p:>8.1f}{r:>8.1f}{a50:>9.1f}{a:>11.1f}")

    overall = float(result.box.map50)
    print("-" * 46)
    print(f"{'all':<10}{float(result.box.mp)*100:>8.1f}{float(result.box.mr)*100:>8.1f}"
          f"{overall*100:>9.1f}{float(result.box.map)*100:>11.1f}")

    delta = abs(overall - EXPECTED_MAP50)
    print(f"\noverall mAP50 {overall:.4f} vs Fig. 11(b) {EXPECTED_MAP50}: "
          f"{'matches, this is the checkpoint behind the figure' if delta <= TOLERANCE else f'DIFFERS by {delta:.4f} — best.pt is not that run'}")

    print("\n" + latex(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
