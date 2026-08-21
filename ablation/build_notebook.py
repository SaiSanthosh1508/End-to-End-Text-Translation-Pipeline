"""Generate the Kaggle probe notebook.

    python ablation/build_notebook.py

Kept as a generator rather than a checked-in .ipynb so the notebook cannot drift from
the package it drives: every cell that matters shells out to ablation/*.py.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/SaiSanthosh1508/End-to-End-Text-Translation-Pipeline.git"
BRANCH = "ablation-mscbam-probe"
ULTRALYTICS = "8.3.189"
MLT_YOLO_DRIVE_ID = "1Ena8CH9H8XxY8yBx15KYmQd82lJQE6Um"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        f"""# MS-CBAM bottleneck probe

Six runs: the full model with the channel-attention reduction ratio displaced
(as deployed) and honoured (`r=16`), three seeds each.

The next cell clones branch `{BRANCH}`. Once that branch is merged to `main`,
drop the `--branch` flag.

**Session plan.** Each run is ~4.6 h on T4 x2, so budget two runs per session and
check your account's GPU session cap first. Edit `THIS_SESSION` below, then
*Save Version -> Save & Run All*. For sessions 2 and 3, attach the previous
session's output as an input dataset so completed runs are skipped.

Do **not** unzip `ultralytics_2.zip` here. This notebook installs a pinned
Ultralytics {ULTRALYTICS} and patches it from the repo, which is what makes the
run reproducible without a Drive link.""",
    ),
    (
        "code",
        '''THIS_SESSION = [
    ("legacy", 42),
    ("fixed", 42),
]
# session 2: [("legacy", 1337), ("fixed", 1337)]
# session 3: [("legacy", 2024), ("fixed", 2024)]

PROJECT = "/kaggle/working/runs/probe"
SNAPSHOT = "/kaggle/working/snapshot"
DEVICE = "0,1"

# Optional second safety net, for a session killed by timeout or OOM rather than by a
# raised exception. Needs KAGGLE_USERNAME and KAGGLE_KEY added as notebook secrets.
PERSIST_TO_DATASET = False
DATASET_SLUG = "your-kaggle-username/mscbam-probe-results"''',
    ),
    (
        "code",
        f'''!pip install -q gdown "ultralytics=={ULTRALYTICS}"
!git clone -q --branch {BRANCH} {REPO} /kaggle/working/repo
!ls /kaggle/working/repo/ablation''',
    ),
    (
        "code",
        f'''import subprocess, pathlib

if not pathlib.Path("/kaggle/working/mlt19_yolo_obb_final.zip").exists():
    subprocess.run(["gdown", "{MLT_YOLO_DRIVE_ID}", "-O",
                    "/kaggle/working/mlt19_yolo_obb_final.zip"], check=True)
    subprocess.run(["unzip", "-q", "-o", "/kaggle/working/mlt19_yolo_obb_final.zip",
                    "-d", "/kaggle/working"], check=True)

roots = [p.parent for p in pathlib.Path("/kaggle/working").glob("*/images/train")]
if len(roots) != 1:
    raise RuntimeError(f"expected one dataset root with images/train, found {{roots}}")
root = roots[0]

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
print("dataset root:", root)
print(" train imgs:", len(list((root / "images/train").glob("*"))))
print("   val imgs:", len(list((root / "images/val").glob("*"))))''',
    ),
    (
        "code",
        '''!cd /kaggle/working/repo && python ablation/install_modules.py''',
    ),
    (
        "markdown",
        """`verify_install.py` is the gate. If the patch silently failed, the fixed arm
rebuilds the one-channel bottleneck and the probe compares a network against
itself. The next cell raises rather than letting that reach the GPU.""",
    ),
    (
        "code",
        '''import subprocess
if subprocess.run(["python", "ablation/verify_install.py"],
                  cwd="/kaggle/working/repo").returncode:
    raise RuntimeError("arms not constructed as intended - do not train")''',
    ),
    (
        "code",
        r'''import pathlib, re, shutil

# Accepts either shape of attached input: a previous session's saved output, or a
# dataset written by push_snapshot.py. Only <arm>_seed<n> directories are taken,
# so an unrelated attached dataset cannot pollute the run set.
RUN_DIR = re.compile(r"^(legacy|fixed)_seed\d+$")

candidates = {
    hit.parent
    for pattern in ("*/runs/probe/*/results.csv", "*/snapshot/*/results.csv")
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
print(f"restored {restored} completed run(s) from attached inputs")''',
    ),
    (
        "markdown",
        """Each run is launched separately and `check=True` is deliberately absent: a
Kaggle version that raises saves **no output at all**, so one failed run would
discard every completed run in the same session. `run_probe.py` contains its own
failures too, and mirrors `results.csv` into `SNAPSHOT` after every run.""",
    ),
    (
        "code",
        '''import os, subprocess

if PERSIST_TO_DATASET:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")

for arm, seed in THIS_SESSION:
    subprocess.run(
        ["python", "ablation/run_probe.py",
         "--data", "/kaggle/working/dataset.yaml",
         "--project", PROJECT,
         "--snapshot", SNAPSHOT,
         "--device", DEVICE,
         "--arms", arm, "--seeds", str(seed)],
        cwd="/kaggle/working/repo",
    )
    if PERSIST_TO_DATASET:
        subprocess.run(
            ["python", "ablation/push_snapshot.py",
             "--dir", SNAPSHOT, "--slug", DATASET_SLUG,
             "--message", f"after {arm} seed {seed}"],
            cwd="/kaggle/working/repo",
        )''',
    ),
    (
        "code",
        '''!cd /kaggle/working/repo && python ablation/aggregate.py {PROJECT} --last-n 5''',
    ),
    (
        "markdown",
        """The verdict stays withheld until all three seeds of both arms are present, so
partial output here is expected until session 3.

Save this version's output, then attach it as an input dataset to the next
session and edit `THIS_SESSION`. If you enabled `PERSIST_TO_DATASET`, attach
`DATASET_SLUG` instead and results survive even a hard-killed session.

Working interactively rather than with *Save & Run All*? Nothing in
`/kaggle/working` is kept unless you hit **Save Version** before the session
ends, so do that after each run.""",
    ),
]


def build() -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": body.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, body in CELLS
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
    out = Path(__file__).with_name("probe_kaggle.ipynb")
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")
