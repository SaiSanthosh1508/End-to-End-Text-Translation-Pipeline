"""Generate the two Kaggle notebooks for the ReCTS detector control.

    python rects_control/build_notebooks.py

Kept as a generator, like ablation/build_notebook.py, so the notebooks cannot drift
from the package they drive. Everything with logic in it lives in rects_control/*.py
and is unit-tested; the cells are setup, orchestration and reporting.

Notebook 1 retrains and *persists* the recogniser. The published one was exported but
never archived before /kaggle/working was cleared, so it no longer exists.
Notebook 2 trains the stock detector and writes RRC submissions for both arms through
that one recogniser.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/SaiSanthosh1508/End-to-End-Text-Translation-Pipeline.git"
BRANCH = "ablation-mscbam-probe"

DRIVE_RECTS = "1orMtLhJt3rQl3pMoLm31eh-SmDG74W1K"
DRIVE_YOLO = "1wWgK4XvoBypaCprHcUFjEizhzPF4M35h"
DRIVE_TEST1 = "1mKqhPBDM-7BgUud69AYvQ7_BYmHqvFJC"
DRIVE_TEST2 = "1E8BlG5kh-JRAGOdYmCO75oi7Jy-UHHoW"


RECOGNIZER_CELLS: list[tuple[str, str]] = [
    ("markdown", """# ReCTS recogniser: retrain, export, **and keep it**

Reviewer item 2 needs stock YOLOv11 and the paper's detector read by *the same*
fine-tuned recogniser. That recogniser no longer exists: it was exported to
`/kaggle/working/PP-OCRv5_server_rec_infer` but the archiving line in the original
notebook was commented out, so it went when the session was cleared.

This notebook rebuilds it and saves it. Nothing to attach — the data comes from Drive.

**Accelerator: GPU T4 x2. Runtime ~3 h.** Use *Save Version -> Save & Run All*, then
add this notebook's output as an input to the control notebook."""),

    ("code", """PADDLE_ROOT = "/kaggle/working/PaddleOCR"
DATA_ROOT = "/kaggle/working/content/train_data/rec"
CONFIG = PADDLE_ROOT + "/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
DICTIONARY = PADDLE_ROOT + "/configs/rec/multi_language/custom_reCTS_dict.txt"
PRETRAINED = PADDLE_ROOT + "/pretrained_models/PP-OCRv5_mobile_rec_pretrained.pdparams"
EXPORT_DIR = "/kaggle/working/PP-OCRv5_rects_rec_infer"

EPOCHS = 35          # as published
BATCH_SIZE = 64
GPUS = "0,1"

# Belt and braces: also publish the export as a standalone Kaggle Dataset.
# Needs KAGGLE_USERNAME and KAGGLE_KEY as notebook secrets.
PUBLISH_DATASET = False
DATASET_SLUG = "rishiksaisanthosh/rects-ppocrv5-finetuned\""""),

    ("code", """!pip install -q gdown
!git clone -q --branch """ + BRANCH + " " + REPO + """ /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),

    ("markdown", "## 1. ReCTS training images and line annotations"),

    ("code", """!gdown -q """ + DRIVE_RECTS + """ -O /kaggle/working/ReCTS.zip
!unzip -q -o /kaggle/working/ReCTS.zip -d /kaggle/working
!ls /kaggle/working | head"""),

    ("code", """from pathlib import Path
from rects_control.crops import build

train_n, val_n = build(Path("/kaggle/working/img"), Path("/kaggle/working/gt"), Path(DATA_ROOT))
print(f"{train_n} train crops, {val_n} val crops")
assert train_n > 10_000, "far fewer crops than expected - check the unzipped layout\""""),

    ("markdown", "## 2. PaddleOCR and the PP-OCRv5 mobile checkpoint"),

    ("code", """!git clone -q https://github.com/PaddlePaddle/PaddleOCR.git {PADDLE_ROOT}
!pip install -q paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
!pip install -q -r {PADDLE_ROOT}/requirements.txt
!pip install -q lmdb rapidfuzz
!wget -q -P {PADDLE_ROOT}/pretrained_models https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"""),

    ("code", """from rects_control.paddle_config import build_dictionary, patch_training_config

n_chars = build_dictionary(Path(DICTIONARY))
patch_training_config(
    Path(CONFIG), data_root=Path(DATA_ROOT), dictionary=Path(DICTIONARY),
    pretrained=Path(PRETRAINED), epochs=EPOCHS, batch_size=BATCH_SIZE,
)
print(f"{n_chars} characters in the label space")"""),

    ("markdown", """## 3. Fine-tune

~2 h on T4 x2. The eval accuracy printed at the end is the sanity check: PP-OCRv5
mobile fine-tuned on ReCTS crops should land well above 0.7 on the held-out 10%."""),

    ("code", """!python3 -m paddle.distributed.launch --gpus '{GPUS}' {PADDLE_ROOT}/tools/train.py \\
    -c {CONFIG} -o Global.pretrained_model={PRETRAINED}"""),

    ("markdown", "## 4. Export for inference"),

    ("code", """!python3 {PADDLE_ROOT}/tools/export_model.py -c {CONFIG} -o \\
    Global.pretrained_model={PADDLE_ROOT}/output/PP-OCRv5_mobile_rec/best_model/model.pdparams \\
    Global.save_inference_dir={EXPORT_DIR}
!ls -la {EXPORT_DIR}"""),

    ("markdown", """## 5. Persist it, then check it

Archiving comes before the quality gate on purpose. A Kaggle version that raises
saves no output at all, so an assertion here would throw away the two hours of
training it was meant to protect. Nothing below this line is allowed to raise."""),

    ("code", """import shutil
archive = shutil.make_archive("/kaggle/working/rects_rec_finetuned", "zip", EXPORT_DIR)
size_kb = Path(archive).stat().st_size // 1024
print(archive, size_kb, "KB")
if size_kb < 1000:
    print("WARNING: archive is much smaller than a PP-OCRv5 mobile export should be")"""),

    ("markdown", """### Optional: publish it as a Kaggle Dataset

The version output above is already permanent, and attaching it to notebook 2 is
enough. A dataset is the stronger option: it is independent of this notebook, so
editing or deleting the notebook cannot take the weights with it, and it is what you
would pull from for the Space.

Needs two notebook secrets - *Add-ons -> Secrets* - named `KAGGLE_USERNAME` and
`KAGGLE_KEY`, then set `PUBLISH_DATASET = True` above. Without them this cell says so
and moves on."""),

    ("code", """import os, shutil, subprocess

if PUBLISH_DATASET:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")

    # Publish the export inside a named folder so the attached path is the same
    # shape as a notebook-output attachment, and notebook 2 finds either.
    stage = Path("/kaggle/working/recognizer_dataset")
    shutil.copytree(EXPORT_DIR, stage / Path(EXPORT_DIR).name, dirs_exist_ok=True)
    subprocess.run(
        ["python", "ablation/push_snapshot.py", "--dir", str(stage),
         "--slug", DATASET_SLUG, "--message", "ReCTS fine-tuned PP-OCRv5 mobile rec"],
        cwd="/kaggle/working/repo",
    )
else:
    print("PUBLISH_DATASET is False; the version output is your only copy")"""),

    ("markdown", """### Does it actually read ReCTS text?

A broken export - wrong dictionary, wrong checkpoint - still loads and still returns
plausible strings. Reading held-out crops whose ground truth we know is the only cheap
way to tell, and it is worth knowing now rather than after the 4.4 h detector run.

This reports; it does not raise. Read the output before starting notebook 2."""),

    ("code", """import cv2
from rects_control.recognizer import PaddleRecognizer

rows = Path(DATA_ROOT, "rec_gt_test.txt").read_text(encoding="utf-8").splitlines()[:12]
paths, truth = zip(*(r.split("\\t") for r in rows))
try:
    predicted = PaddleRecognizer(Path(EXPORT_DIR))([cv2.imread(str(Path(DATA_ROOT, p))) for p in paths])
    hits = sum(p == t for p, t in zip(predicted, truth))
    for p, t in zip(predicted, truth):
        print(f"{'ok ' if p == t else '   '} pred={p!r:20s} gt={t!r}")
    verdict = "looks right" if hits >= 4 else "SUSPECT - do not start notebook 2 yet"
    print(f"\\n{hits}/{len(truth)} exact on a 12-crop sample: {verdict}")
except Exception as error:                 # the weights are already archived above
    print(f"self-test could not run: {error!r}")
    print("The export is still saved; investigate before starting notebook 2.")"""),

    ("markdown", """**Now: Save Version -> Save & Run All.**

When it finishes, open the control notebook and add this notebook's output under
*Add Input -> Notebook Output*. Also worth doing once: publish
`/kaggle/working/PP-OCRv5_rects_rec_infer` as a Kaggle dataset, so a cleared session
can never cost you these weights again."""),
]


CONTROL_CELLS: list[tuple[str, str]] = [
    ("markdown", """# ReCTS detector control

> *"The clean experiment would be stock YOLOv11 plus your same fine-tuned OCR, so
> readers can see what your detector changes contribute to end-to-end 1-NED."*

Two detectors, one recogniser, identical everything else. Both arms are trained on
the ReCTS YOLO split with the published recipe (100 epochs, batch 64, imgsz 480,
cosine LR, `pretrained=True`) and read by the recogniser from notebook 1.

**Attach both inputs first:**
1. *Add Input -> Notebook Output* -> the recogniser notebook's output
2. *Add Input -> Datasets* -> `rishiksaisanthosh/yolo-obb-rects` (the published detector)

**Accelerator: GPU T4 x2. Runtime ~5.5 h** with `TRAIN_PAPER_ARM = False`
(4.4 h stock training, ~1 h submissions)."""),

    ("code", """SEED = 42
PROJECT = "/kaggle/working/runs/rects"

# The published detector already exists at seed 42, so the paper arm is reused rather
# than retrained. Set True to train it here instead - another 4.4 h per seed - which is
# what you need if you want the comparison at more than one seed.
TRAIN_PAPER_ARM = False

TASK3_CONFS = (0.3, 0.4, 0.5)
TASK4_CONFS = (0.4,)
DEVICE = "0,1\""""),

    ("code", """!pip install -q gdown "ultralytics==8.3.189"
!pip install -q paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
!pip install -q paddleocr
!git clone -q --branch """ + BRANCH + " " + REPO + """ /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),

    ("markdown", """## 1. Locate the recogniser and the published detector

Both come from attached inputs. Failing here costs nothing; failing after the
training run costs the session."""),

    ("code", """from pathlib import Path

def locate(what, *patterns):
    # first match across the attached inputs; patterns are tried in order
    for pattern in patterns:
        hits = sorted(Path("/kaggle/input").glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{what} not attached (looked for {' , '.join(patterns)})")

# Accepts the recogniser as a notebook output or as a published dataset, flattened
# or not, so the attachment style you chose in notebook 1 does not matter here.
REC_DIR = locate("recogniser", "**/PP-OCRv5_rects_rec_infer/inference.yml",
                 "**/inference.yml").parent
PAPER_CKPT = locate("published ReCTS detector", "**/runs/obb/train3/weights/best.pt")
print("recogniser:", REC_DIR)
print("paper detector:", PAPER_CKPT)"""),

    ("markdown", "## 2. ReCTS YOLO training split and the Task 3/4 test images"),

    ("code", """!gdown -q """ + DRIVE_YOLO + """ -O /kaggle/working/rects_yolo.zip
!gdown -q """ + DRIVE_TEST1 + """ -O /kaggle/working/test1.zip
!gdown -q """ + DRIVE_TEST2 + """ -O /kaggle/working/test2.zip
!unzip -q -o /kaggle/working/rects_yolo.zip -d /kaggle/working/rects_yolo
!unzip -q -o /kaggle/working/test1.zip -d /kaggle/working
!unzip -q -o /kaggle/working/test2.zip -d /kaggle/working"""),

    ("code", """roots = sorted({h.parent.parent for h in Path("/kaggle/working/rects_yolo").glob("**/images/train")
                if (h.parent.parent / "labels/train").is_dir()})
assert roots, "no YOLO tree in the ReCTS zip"
root = roots[0]

for split in ("train", "val"):
    n_img = len(list((root / f"images/{split}").glob("*")))
    n_lbl = len(list((root / f"labels/{split}").glob("*.txt")))
    print(f"{split}: {n_img} images, {n_lbl} labels")
    assert n_img and n_img == n_lbl, f"{split} split is empty or mismatched"

columns = {len(l.split()) for l in next((root / "labels/train").glob("*.txt")).read_text().split(chr(10)) if l.strip()}
assert columns == {9}, f"expected 9-column oriented labels, saw {columns}"

DATA_YAML = Path("/kaggle/working/rects.yaml")
DATA_YAML.write_text(f"train: {root}/images/train\\nval: {root}/images/val\\n\\nnc: 1\\n\\nnames:\\n  0: text\\n")
print("data:", DATA_YAML.read_text())"""),

    ("code", """from rects_control.submission import image_files
n_test = len(image_files(Path("/kaggle/working")))
print(f"{n_test} Task 3/4 test images")
assert n_test > 1000, "test set looks truncated\""""),

    ("markdown", """## 3. Smoke-test the whole chain before spending GPU hours

Detector to crop to recogniser to submission line, on one image, using the published
checkpoint that already exists. If PaddleOCR cannot load the exported model this is
where it surfaces - not 4.4 h from now, on a version that will save nothing."""),

    ("code", """from PIL import Image
from ultralytics import YOLO
from rects_control.recognizer import PaddleRecognizer
from rects_control.submission import clockwise_points, crop_bgr, detect, image_files

recognizer = PaddleRecognizer(REC_DIR)
probe = image_files(Path("/kaggle/working"))[0]
image = Image.open(probe)
quads = detect(YOLO(str(PAPER_CKPT)), image, 0.4)

crops = [c for c in (crop_bgr(image, q) for q in quads) if c is not None and c.size]
texts = recognizer(crops)
print(f"{probe.name}: {len(quads)} boxes, {sum(bool(t) for t in texts)} transcribed")
for quad, text in list(zip(quads, texts))[:5]:
    print("  ", ",".join(map(str, clockwise_points(quad, *image.size))), text)
assert any(texts), "recogniser returned nothing on a real test image\""""),

    ("markdown", """## 4. Train the arms

`stock` is `a1_stock.yaml` - YOLOv11s-OBB as shipped. `paper` is `full_legacy.yaml`,
the deployed BiFPN + MS-CBAM + cross-attention model. Both get `nc=1`."""),

    ("code", """from rects_control.detectors import adopt_published, train

if TRAIN_PAPER_ARM:
    paper = train("paper", DATA_YAML, Path(PROJECT), SEED, DEVICE)
else:
    paper = adopt_published(PAPER_CKPT, Path(PROJECT), SEED)

stock = train("stock", DATA_YAML, Path(PROJECT), SEED, DEVICE)

# Do not raise here: a failed version saves no output, which would discard the arm
# that succeeded along with the one that did not.
arms = {name: w for name, w in (("paper", paper), ("stock", stock)) if w}
for name in ("paper", "stock"):
    print(f"{name}: {arms.get(name, 'FAILED - see the traceback above')}")"""),

    ("markdown", """## 5. Submissions

One `PaddleRecognizer` instance serves both arms, so any 1-NED difference is
attributable to detection. Task 3 is written at three confidences because the
detection operating point is a free parameter; Task 4 at the published 0.4."""),

    ("code", """from rects_control.submission import write_task3, write_task4

out = Path("/kaggle/working/submissions"); out.mkdir(exist_ok=True)
summary = []

for arm, weights in arms.items():
    model = YOLO(str(weights))
    for conf in TASK3_CONFS:
        f = out / f"task3_{arm}_seed{SEED}_conf{conf:.1f}.txt"
        summary.append((arm, "task3", conf, write_task3(model, Path("/kaggle/working"), f, conf)))
        print(summary[-1])
    for conf in TASK4_CONFS:
        f = out / f"task4_{arm}_seed{SEED}_conf{conf:.1f}.txt"
        summary.append((arm, "task4", conf, write_task4(model, recognizer, Path("/kaggle/working"), f, conf)))
        print(summary[-1])"""),

    ("code", """import shutil
for f in sorted(out.glob("*.txt")):
    shutil.make_archive(str(f.with_suffix("")), "zip", f.parent, f.name)
    print(f.name, f.stat().st_size // 1024, "KB")

print("\\nboxes written")
for arm, task, conf, n in summary:
    print(f"  {arm:6s} {task} conf={conf} -> {n}")"""),

    ("markdown", """## 6. Upload

Save Version, download `/kaggle/working/submissions/*.zip`, and submit each to the
RRC ReCTS server — Task 3 for H-mean, Task 4 for 1-NED. The number the reviewer
asked for is `task4_stock` vs `task4_paper` 1-NED.

Report both, whichever way they fall. If the gap is small, the honest framing the
reviewer already offered is a deployment and systems contribution, which is still
publishable; the seed-to-seed spread in the MLT ablation table is the right yardstick
for deciding whether a given gap means anything."""),
]


RECOVERY_CELLS: list[tuple[str, str]] = [
    ("markdown", """# Recover the recogniser from the cancelled run

The 35-epoch fine-tune reached epoch 34 and was cut off at Kaggle's 12 h session
limit, 32 minutes from the end. Nothing needs retraining: PaddleOCR had already
written `best_model` at epoch 32 (acc 0.760, norm-edit-dis 0.874), and epochs 33-35
were not going to move that. This notebook exports that checkpoint and publishes it.

**Attach one input:** *Add Input -> Notebook Output* -> the cancelled `rects` run.

**No accelerator needed.** Set it to None; export is a CPU operation and this takes
about ten minutes."""),

    ("code", """EXPORT_DIR = "/kaggle/working/PP-OCRv5_rects_rec_infer"
RECOVERED = "/kaggle/working/recovered"

PUBLISH_DATASET = True
DATASET_SLUG = "rishiksaisanthosh/rects-ppocrv5-finetuned\""""),

    ("code", f"""!pip install -q paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
!git clone -q https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR
!pip install -q -r /kaggle/working/PaddleOCR/requirements.txt
!git clone -q --branch {BRANCH} {REPO} /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),

    ("markdown", """## 1. What is actually attached?

The previous attempt failed here with nothing under `/kaggle/input`, and could not
say whether the input was missing or the checkpoint was. Inventory first, so the
answer is in the log either way."""),

    ("code", """from pathlib import Path

INPUT = Path("/kaggle/input")
roots = sorted(p for p in INPUT.glob("*")) if INPUT.exists() else []
print(f"{len(roots)} input(s) attached:", [p.name for p in roots])

for root in roots:
    entries = sorted(root.rglob("*"))
    print(f"\\n{root.name}: {len(entries)} entries")
    for entry in entries[:25]:
        marker = "/" if entry.is_dir() else f"  {entry.stat().st_size // 1024} KB"
        print("   ", entry.relative_to(root), marker)
    if len(entries) > 25:
        print(f"    ... and {len(entries) - 25} more")

if not roots:
    raise RuntimeError(
        "Nothing is attached. Add Input -> Notebook Output -> the cancelled 'rects' "
        "run, wait for it to finish mounting, then re-run."
    )"""),

    ("markdown", """## 2. Unpack whatever shape it arrived in

Kaggle stores a session as a single `_output_.zip` but usually presents an attached
notebook output extracted. Handle both."""),

    ("code", """import zipfile

WANTED = ("/best_model/", "/latest", "custom_reCTS_dict.txt", "rec_gt_test.txt")
RECOVERED_DIR = Path(RECOVERED)

archives = sorted(INPUT.glob("**/*.zip"))
print("archives found:", [a.name for a in archives])

for archive in archives:
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.namelist()
                   if any(w in m for w in WANTED) or m.endswith("PP-OCRv5_mobile_rec.yml")]
        print(f"{archive.name}: {len(bundle.namelist())} members, {len(members)} wanted")
        RECOVERED_DIR.mkdir(parents=True, exist_ok=True)
        bundle.extractall(RECOVERED_DIR, members=members)

search_roots = ([RECOVERED_DIR] if RECOVERED_DIR.exists() else []) + roots
print("searching:", [str(r) for r in search_roots])"""),

    ("code", """CHECKPOINT_PATTERNS = (
    "**/best_model/model.pdparams",     # what PaddleOCR writes on a new best epoch
    "**/best_accuracy.pdparams",
    "**/latest.pdparams",               # end-of-epoch snapshot, equivalent here
    "**/*.pdparams",
)

def find(what, *patterns, required=True):
    for root in search_roots:
        for pattern in patterns:
            hits = sorted(root.glob(pattern))
            if hits:
                return hits[0]
    if not required:
        return None
    listing = "\\n".join(f"    {p}" for root in search_roots
                         for p in sorted(root.rglob("*"))[:60])
    raise FileNotFoundError(f"{what} not found. Patterns: {patterns}\\nSaw:\\n{listing}")

CHECKPOINT = find("training checkpoint", *CHECKPOINT_PATTERNS)
CONFIG_SRC = find("training config", "**/PP-OCRv5_mobile_rec.yml", "**/*mobile_rec.yml")
DICT_SRC = find("character dictionary", "**/custom_reCTS_dict.txt", required=False)

for label, path in (("checkpoint", CHECKPOINT), ("config", CONFIG_SRC), ("dictionary", DICT_SRC)):
    print(f"{label:11s} {path}" + (f"  ({path.stat().st_size // 1024} KB)" if path else ""))
if DICT_SRC:
    print("dictionary entries:", len(DICT_SRC.read_text(encoding="utf-8").splitlines()))
else:
    print("dictionary not recovered; it will be rebuilt from the same sources")"""),

    ("markdown", """## 3. Export

The recovered config still names the old session's paths, and the only one export
actually reads is the dictionary - get it wrong and the model's output indices mean
nothing. `Global.pretrained_model` takes the checkpoint stem: PaddleOCR appends
`.pdparams` itself."""),

    ("code", """import shutil
from rects_control.paddle_config import build_dictionary, retarget_for_export

work = Path("/kaggle/working/export"); work.mkdir(exist_ok=True)
config = work / "PP-OCRv5_mobile_rec.yml"
dictionary = work / "custom_reCTS_dict.txt"
shutil.copyfile(CONFIG_SRC, config)

if DICT_SRC:
    shutil.copyfile(DICT_SRC, dictionary)
else:
    # Deterministic given the same sources, so a rebuild reproduces the label space
    # the checkpoint was trained against. The count must match the training log.
    print(build_dictionary(dictionary), "characters rebuilt (training log said 6711)")
retarget_for_export(config, dictionary)

stem = str(CHECKPOINT.with_suffix(""))
print(subprocess.run(
    ["python3", "/kaggle/working/PaddleOCR/tools/export_model.py", "-c", str(config),
     "-o", f"Global.pretrained_model={stem}", f"Global.save_inference_dir={EXPORT_DIR}"],
    capture_output=True, text=True).stdout[-2500:])
print(sorted(p.name for p in Path(EXPORT_DIR).iterdir()))"""),

    ("markdown", """## 4. Persist before checking

Same rule as before: a raising cell saves no output, so the weights are secured
first and nothing below is allowed to raise."""),

    ("code", """archive = shutil.make_archive("/kaggle/working/rects_rec_finetuned", "zip", EXPORT_DIR)
size_kb = Path(archive).stat().st_size // 1024
print(archive, size_kb, "KB")
if size_kb < 1000:
    print("WARNING: much smaller than a PP-OCRv5 mobile export should be")"""),

    ("code", """import os

if PUBLISH_DATASET:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")

    stage = Path("/kaggle/working/recognizer_dataset")
    shutil.copytree(EXPORT_DIR, stage / Path(EXPORT_DIR).name, dirs_exist_ok=True)
    subprocess.run(
        ["python", "ablation/push_snapshot.py", "--dir", str(stage),
         "--slug", DATASET_SLUG, "--message", "ReCTS fine-tuned PP-OCRv5 mobile rec, epoch 32"],
        cwd="/kaggle/working/repo",
    )
else:
    print("PUBLISH_DATASET is False; the version output is your only copy")"""),

    ("markdown", """### Does it read ReCTS text?

Held-out crops with known ground truth, recovered from the same run. Reports; does
not raise."""),

    ("code", """try:
    import cv2
    from rects_control.recognizer import PaddleRecognizer

    labels = find("**/rec_gt_test.txt", "held-out labels")
    rows = labels.read_text(encoding="utf-8").splitlines()
    crops_root = labels.parent
    sample = [r.split(chr(9)) for r in rows[:200]]
    usable = [(crops_root / rel, text) for rel, text in sample if (crops_root / rel).exists()][:12]

    predicted = PaddleRecognizer(Path(EXPORT_DIR))([cv2.imread(str(p)) for p, _ in usable])
    hits = sum(p == t for p, (_, t) in zip(predicted, usable))
    for p, (_, t) in zip(predicted, usable):
        print(f"{'ok ' if p == t else '   '} pred={p!r:20s} gt={t!r}")
    print(f"\\n{hits}/{len(usable)} exact - expect roughly 8/12 at acc 0.76")
except Exception as error:
    print(f"self-test could not run: {error!r}")
    print("The export is already archived above; this does not affect it.")"""),

    ("markdown", """**Save Version -> Save & Run All**, then go straight to the control
notebook and attach either this notebook's output or the published dataset."""),
]


def notebook(cells: list[tuple[str, str]]) -> dict[str, object]:
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
    here = Path(__file__).parent
    for name, cells in (
        ("rects_recognizer_kaggle.ipynb", RECOGNIZER_CELLS),
        ("rects_control_kaggle.ipynb", CONTROL_CELLS),
        ("rects_export_kaggle.ipynb", RECOVERY_CELLS),
    ):
        (here / name).write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {name}  ({len(cells)} cells)")
