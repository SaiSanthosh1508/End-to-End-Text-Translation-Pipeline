"""Generate the Table 8 seed notebook, one seed per Kaggle account.

    python ablation/build_seed_notebook.py

The reviewer asked for at least three seeds on the ablation. Seed 42 already exists
for both endpoints, so two further seeds close the gap. Each is ~9.2 h, which fits a
single 12 h session, and the two are independent, so they can run concurrently on
two accounts.

Only the endpoints are re-run: row 1 (stock) and row 6 (the deployed model). Those
carry the claim the reviewer disputes. The intermediate rows stay single-seed, which
is defensible now that the text no longer attributes gains to individual components.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/SaiSanthosh1508/End-to-End-Text-Translation-Pipeline.git"
BRANCH = "ablation-mscbam-probe"
ULTRALYTICS = "8.3.189"
MLT_DATASET = "rishiksaisanthosh/dataset-test"

CELLS: list[tuple[str, str]] = [
    ("markdown", f"""# Table 8 seeds: stock vs deployed model

Two arms, one seed, about **9.2 h** on **GPU T4 x2**.

* `a1_stock` is Table 8 row 1, unmodified YOLOv11s-OBB.
* `legacy` is Table 8 row 6, the deployed model, as trained for `best.pt`.

Seed 42 already exists for both. Run this notebook **once per account** with a
different `SEED`, and the three seeds together give the mean and standard deviation
the reviewer asked for.

**Attach the data first:** *Add Input -> Datasets -> `{MLT_DATASET}`*. Both accounts
need it; the run stops in the first minute without it.

**Set the two secrets** under *Add-ons -> Secrets*: `KAGGLE_USERNAME` and
`KAGGLE_KEY`. Results are pushed to a dataset after each run, so a session killed by
timeout cannot take them with it. This is how the recogniser weights were lost.

Then *Save Version -> Save & Run All*."""),

    ("code", '''SEED = 1337        # account A: 1337   |   account B: 2024
ARMS = ("a1_stock", "legacy")

PROJECT = "/kaggle/working/runs/probe"
SNAPSHOT = "/kaggle/working/snapshot"
DEVICE = "0,1"

# Results survive a killed session only if they leave /kaggle/working.
PERSIST_TO_DATASET = True
DATASET_SLUG = "rishiksaisanthosh/mscbam-table8-seeds"'''),

    ("code", f'''!pip install -q "ultralytics=={ULTRALYTICS}"
!git clone -q --branch {BRANCH} {REPO} /kaggle/working/repo
!cd /kaggle/working/repo && python ablation/install_modules.py'''),

    ("markdown", """`verify_install.py` is the gate. If the patch silently failed, the arms would be
built from stock modules and the comparison would be between two identical
networks. This raises instead of letting that reach the GPU."""),

    ("code", '''import subprocess
if subprocess.run(["python", "ablation/verify_install.py"],
                  cwd="/kaggle/working/repo").returncode:
    raise RuntimeError("arms not constructed as intended - do not train")'''),

    ("markdown", "## Data"),

    ("code", f'''import pathlib

def find_roots(base):
    """A YOLO root holds both images/train and labels/train."""
    return sorted(
        {{hit.parent.parent for hit in pathlib.Path(base).glob("**/images/train")
          if (hit.parent.parent / "labels/train").is_dir()}}
    )

roots = find_roots("/kaggle/input")
if not roots:
    raise RuntimeError(
        "No YOLO dataset attached. Add Input -> Datasets -> {MLT_DATASET}"
    )

# dataset-test carries the same tree twice; take the larger, break ties on path length.
root = max(roots, key=lambda r: (len(list((r / "images/train").glob("*"))), -len(str(r))))
counts = {{}}
for split in ("train", "val"):
    imgs = len(list((root / f"images/{{split}}").glob("*")))
    lbls = len(list((root / f"labels/{{split}}").glob("*.txt")))
    counts[split] = imgs
    print(f"{{split}}: {{imgs}} images, {{lbls}} labels")
    if imgs == 0 or imgs != lbls:
        raise RuntimeError(f"{{split}} split is empty or images and labels disagree")

sample = next((root / "labels/train").glob("*.txt"))
cols = {{len(l.split()) for l in sample.read_text().splitlines() if l.strip()}}
if cols != {{9}}:
    raise RuntimeError(f"expected 9-column oriented labels for an OBB run, saw {{cols}}")

pathlib.Path("/kaggle/working/dataset.yaml").write_text(
    f"""train: {{root}}/images/train
val: {{root}}/images/val

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
print("dataset root:", root)'''),

    ("markdown", """## Restore anything already finished

Attach a previous session's output, or the results dataset, and completed runs are
skipped rather than repeated."""),

    ("code", r'''import re, shutil

RUN_DIR = re.compile(r"^(a1_stock|legacy)_seed\d+$")
candidates = {
    hit.parent
    for pattern in ("*/runs/probe/*/results.csv", "*/snapshot/*/results.csv", "*/*/results.csv")
    for hit in pathlib.Path("/kaggle/input").glob(pattern)
}
restored = 0
for prior in sorted(candidates):
    if not RUN_DIR.match(prior.name):
        continue
    target = pathlib.Path(PROJECT) / prior.name
    if not target.exists():
        shutil.copytree(prior, target)
        restored += 1
print(f"restored {restored} completed run(s)")'''),

    ("markdown", """## Train

Each arm is launched separately and failures are contained: a Kaggle version that
raises saves **no output at all**, so one bad run would discard the other."""),

    ("code", '''import os

if PERSIST_TO_DATASET:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")

for arm in ARMS:
    subprocess.run(
        ["python", "ablation/run_probe.py",
         "--data", "/kaggle/working/dataset.yaml",
         "--project", PROJECT, "--snapshot", SNAPSHOT, "--device", DEVICE,
         "--arms", arm, "--seeds", str(SEED)],
        cwd="/kaggle/working/repo",
    )
    if PERSIST_TO_DATASET:
        subprocess.run(
            ["python", "ablation/push_snapshot.py",
             "--dir", SNAPSHOT, "--slug", DATASET_SLUG,
             "--message", f"{arm} seed {SEED}"],
            cwd="/kaggle/working/repo",
        )'''),

    ("code", '''!cd /kaggle/working/repo && python ablation/aggregate.py {PROJECT} --last-n 5'''),

    ("markdown", """The verdict stays withheld until three seeds of both arms are present, so partial
output here is expected until the second account finishes.

Save this version. To combine, attach both accounts' outputs (or the results
dataset) to either notebook and re-run the aggregate cell."""),
]


def notebook() -> dict[str, object]:
    return {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, body in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("seeds_kaggle.ipynb")
    out.write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")
