"""Generate the Kaggle notebook that regenerates the class-wise MLT table.

    python ablation/build_revalidate.py

Separate from build_notebook.py because the two notebooks answer different questions
and share nothing but the Kaggle boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/SaiSanthosh1508/End-to-End-Text-Translation-Pipeline.git"
BRANCH = "ablation-mscbam-probe"
ULTRALYTICS = "8.3.189"
MLT_DATASET = "rishiksaisanthosh/dataset-test"

INTRO = f"""# Regenerate the class-wise MLT table

Table 5 disagrees with Fig. 11(b) on every class by +1.6 to +15.7 mAP50 points, all in
one direction, and its implied overall mAP50 is 73.8 against 66.3 in the figure legend
and 66.1 in `results.csv`. Precision reconciles (80.3 vs 81.0) but recall and mAP do
not, so those columns did not come from the same evaluation as the rest of the paper.

This runs the evaluation again from `best.pt` and prints a replacement table.
Inference only, roughly two minutes.

Attach **`{MLT_DATASET}`**. It supplies both the validation split and `best.pt`, which
avoids the git-lfs pointer a plain clone would give you.
"""

SETUP = f"""!pip install -q "ultralytics=={ULTRALYTICS}"
!git clone -q --branch {BRANCH} {REPO} /kaggle/working/repo
!cd /kaggle/working/repo && python ablation/install_modules.py"""

LOCATE = '''import pathlib

roots = sorted({
    hit.parent.parent
    for hit in pathlib.Path("/kaggle/input").glob("**/images/val")
    if (hit.parent.parent / "labels/val").is_dir()
})
if not roots:
    raise RuntimeError("attach the MLT dataset before running this")

root = max(roots, key=lambda r: len(list((r / "images/val").glob("*"))))
n_val = len(list((root / "images/val").glob("*")))
print(f"dataset root: {root}  ({n_val} val images)")

weights = next(pathlib.Path("/kaggle/input").glob("**/Text_Translation_Pipeline/best.pt"), None)
if weights is None:
    raise RuntimeError("best.pt not found under /kaggle/input")
size_mb = weights.stat().st_size / 1e6
print(f"weights: {weights}  ({size_mb:.1f} MB)")
if size_mb < 1:
    raise RuntimeError("best.pt is a git-lfs pointer, not the model")

pathlib.Path("/kaggle/working/dataset.yaml").write_text(
    f"""train: {root}/images/train
val: {root}/images/val

nc: 8

names:
  0: Arabic
  1: Latin
  2: Chinese
  3: Korean
  4: Japanese
  5: Bangla
  6: Hindi
  7: Other
"""
)
pathlib.Path("/kaggle/working/weights.txt").write_text(str(weights))'''

RUN = '''import subprocess, torch

# Validation is inference over 1,000 images, so CPU is viable. That matters when the
# probe already holds a GPU session: a CPU session costs no GPU quota and cannot
# collide with it.
device = "0" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

weights = open("/kaggle/working/weights.txt").read().strip()
subprocess.run(
    ["python", "ablation/revalidate.py",
     "--weights", weights,
     "--data", "/kaggle/working/dataset.yaml",
     "--imgsz", "480", "--device", device],
    cwd="/kaggle/working/repo", check=True,
)'''

OUTRO = """If the overall mAP50 lands near **0.663**, `best.pt` is the checkpoint behind
Fig. 11(b) and the LaTeX block above replaces Table 5 verbatim.

If it differs, `best.pt` is from a different run, and the table has to be rebuilt from
whichever checkpoint produced the figure — tell me the number and I will work out which.

One naming note: this prints class 7 as `Other`, which is what the model, `dataset.yaml`
and Fig. 11(b) all call it. The current Table 5 calls it `Symbols`.
"""

CELLS: list[tuple[str, str]] = [
    ("markdown", INTRO),
    ("code", SETUP),
    ("code", LOCATE),
    ("code", RUN),
    ("markdown", OUTRO),
]


def build(cells: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": body.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, body in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("revalidate_kaggle.ipynb")
    out.write_text(json.dumps(build(CELLS), indent=1), encoding="utf-8")
    print(f"wrote {out.name}  ({len(CELLS)} cells)")
