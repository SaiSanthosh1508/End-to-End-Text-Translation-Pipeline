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
SAVE_MODEL_DIR = "/kaggle/working/rec_output"

# 98,253 crops at ~21 min/epoch on T4 x2. 35 epochs is ~12.2 h and does not fit
# Kaggle's 12 h cap - that is what killed the first attempt at epoch 34. The first
# run's log shows accuracy essentially flat from epoch 29 (0.748) to its best at 32
# (0.760), so 25 epochs costs about a point and finishes with hours to spare.
EPOCHS = 25
BATCH_SIZE = 64
GPUS = "0,1"

# Belt and braces: also publish the export as a standalone Kaggle Dataset.
# Needs KAGGLE_USERNAME and KAGGLE_KEY as notebook secrets.
PUBLISH_DATASET = True
DATASET_SLUG = "rishiksaisanthosh/rects-ppocrv5-finetuned\""""),

    ("markdown", """## 0. Install first, and prove it worked

The paddle wheel comes from a CDN that intermittently times out. Last attempt it did:
training never started, and the failure only surfaced eleven minutes later as
"training saved nothing". Installing and importing it up front makes a bad CDN day
cost two minutes and say so plainly."""),

    ("code", f"""!pip install -q gdown
!git clone -q --branch {BRANCH} {REPO} /kaggle/working/repo
!pip install -q --timeout 180 --retries 10 paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ || pip install -q --timeout 180 --retries 10 paddlepaddle-gpu
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),

    ("code", """import paddle

print("paddle", paddle.__version__, "| GPUs visible:", paddle.device.cuda.device_count())
assert paddle.device.cuda.device_count() >= 2, (
    "need two GPUs - training launches with --gpus '0,1'. "
    "Set Accelerator to GPU T4 x2 and re-run."
)"""),

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
!pip install -q -r {PADDLE_ROOT}/requirements.txt
!pip install -q lmdb rapidfuzz
!wget -q -P {PADDLE_ROOT}/pretrained_models https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"""),

    ("code", """from rects_control.paddle_config import build_dictionary, patch_training_config

n_chars = build_dictionary(Path(DICTIONARY))
patch_training_config(
    Path(CONFIG), data_root=Path(DATA_ROOT), dictionary=Path(DICTIONARY),
    pretrained=Path(PRETRAINED), epochs=EPOCHS, batch_size=BATCH_SIZE,
    save_model_dir=Path(SAVE_MODEL_DIR),
)
print(f"{n_chars} characters in the label space")"""),

    ("markdown", """## 3. Fine-tune

**~9 h on T4 x2.** Start it with a clear 12 h ahead of you.

Checkpoints go to an absolute `SAVE_MODEL_DIR`. The stock config uses a relative
`./output/...`, and that is what lost the first attempt: it trained for twelve hours,
logged `save model in ./output/PP-OCRv5_mobile_rec/latest` every epoch, and none of it
appeared in the session snapshot. Expect accuracy around 0.75 by epoch 25."""),

    ("code", """!python3 -m paddle.distributed.launch --gpus '{GPUS}' {PADDLE_ROOT}/tools/train.py \\
    -c {CONFIG} -o Global.pretrained_model={PRETRAINED}"""),

    ("markdown", """## 4. Export for inference

Check the checkpoint is on disk before exporting. If this is empty the run produced
nothing, and no amount of exporting will conjure it back."""),

    ("code", """checkpoints = sorted(Path(SAVE_MODEL_DIR).rglob("*.pdparams"))
for path in checkpoints:
    print(f"{path}  {path.stat().st_size // 1024} KB")
assert checkpoints, f"no checkpoint under {SAVE_MODEL_DIR} - training saved nothing"

best = Path(SAVE_MODEL_DIR) / "best_model" / "model.pdparams"
chosen = best if best.exists() else Path(SAVE_MODEL_DIR) / "latest.pdparams"
CHOSEN_STEM = str(chosen.with_suffix(""))   # PaddleOCR appends .pdparams itself
print("exporting from", CHOSEN_STEM)"""),

    ("code", """!python3 {PADDLE_ROOT}/tools/export_model.py -c {CONFIG} -o \\
    Global.pretrained_model={CHOSEN_STEM} \\
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
DEVICE = "0,1"
REC_DEVICE = "cpu"     # see the install cell: paddle GPU breaks torch here"""),

    ("code", """!pip install -q gdown "ultralytics==8.3.189"
# CPU paddle on purpose. paddlepaddle-gpu installs its own nvidia-* wheels, which
# replace the NCCL that Kaggle's torch was built against, so `import ultralytics`
# then dies with "undefined symbol: ncclCommShrink". Recognition is the cheap half
# of this notebook; ~1 h on CPU beats fighting two CUDA stacks in one environment.
!pip install -q --timeout 180 --retries 10 paddlepaddle==3.2.0
!pip install -q paddleocr
!git clone -q --branch """ + BRANCH + " " + REPO + """ /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),

    ("markdown", """### Register the custom attention modules

`best.pt` pickles `CrossAttentionBlock` and `MultiScaleCBAM` under
`ultralytics.nn.modules.block`, so stock Ultralytics cannot unpickle it and the
paper arm's config cannot be built. `install_modules.py` patches a clean 8.3.189
install; it has to run before anything imports ultralytics, because patching
site-packages does not affect a module already loaded in this process."""),

    ("code", """!cd /kaggle/working/repo && python ablation/install_modules.py"""),

    ("code", """import paddle
import torch
from ultralytics import YOLO
from paddleocr import TextRecognition

# install_modules.py puts the custom classes in ultralytics.nn.modules.custom, but
# best.pt pickles them under ...modules.block. Aliasing is what makes the checkpoint
# loadable; importing from block afterwards proves both halves worked.
from rects_control.detectors import register_pickle_aliases

register_pickle_aliases()
from ultralytics.nn.modules.block import CrossAttentionBlock, MultiScaleCBAM

print("torch", torch.__version__, "| CUDA:", torch.cuda.device_count(), "GPU(s)")
print("paddle", paddle.__version__, "| running recognition on", REC_DEVICE)
print("custom modules aliased:", CrossAttentionBlock.__name__, MultiScaleCBAM.__name__)
assert torch.cuda.device_count() >= 2, "need GPU T4 x2 for the detector arms\""""),

    ("markdown", """## 1. Locate the recogniser and the published detector

Both come from attached inputs. Failing here costs nothing; failing after the
training run costs the session."""),

    ("code", """from pathlib import Path

INPUT = Path("/kaggle/input")
roots = sorted(INPUT.glob("*")) if INPUT.exists() else []
print(f"{len(roots)} input(s):", [p.name for p in roots])
for root in roots:
    entries = sorted(root.rglob("*"))[:12]
    print(f"  {root.name}:", [str(e.relative_to(root)) for e in entries] or "EMPTY")
if not roots:
    raise RuntimeError("Nothing attached. Add the recogniser notebook output and yolo-obb-rects.")

# A large notebook output arrives as _output_.zip rather than extracted files, which is
# how the previous attempt found nothing. Pull just the recogniser out of it; random
# access means the 9 GB around it is never read.
import zipfile

STAGED = Path("/kaggle/working/staged_recognizer")
for archive in sorted(INPUT.glob("**/*.zip")):
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.namelist()
                   if "PP-OCRv5_rects_rec_infer/" in m and not m.endswith("/")]
        if members:
            print(f"extracting {len(members)} file(s) from {archive.name}")
            bundle.extractall(STAGED, members=members)
            break

SEARCH = ([STAGED] if STAGED.exists() else []) + roots

def locate(what, *patterns):
    for root in SEARCH:
        for pattern in patterns:
            hits = sorted(root.glob(pattern))
            if hits:
                return hits[0]
    seen = chr(10).join(f"    {p}" for root in SEARCH for p in sorted(root.rglob("*"))[:50])
    raise FileNotFoundError(f"{what} not found. Tried {patterns}, saw:{chr(10)}{seen}")

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

recognizer = PaddleRecognizer(REC_DIR, device=REC_DEVICE)
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

    ("markdown", """### Confirm both arms are the deployed size

Ultralytics picks the scale from the file stem. A stem without `yolo11s` falls back
to nano, which is how an earlier run compared a 10.5M checkpoint against a 2.7M
baseline. This builds both arms and checks the parameter count before training."""),

    ("code", """from rects_control.detectors import ARMS, arm_config, assert_scale

for arm in sorted(ARMS):
    cfg = arm_config(arm, Path(PROJECT) / "configs")
    print(f"{arm:6s} {cfg.name:26s} {assert_scale(cfg):>11,} parameters")
print("deployed best.pt: 10,504,551 parameters")"""),

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


SWEEP_CELLS: list[tuple[str, str]] = [
    ("markdown", """# ReCTS Task 4 confidence sweep

The control ran Task 4 at conf 0.4 and scored 1-NED 66.06 (paper) against 64.11
(stock). The paper reports 0.8566, and the Task 3 sweep shows why the two are not
comparable: published precision/recall of 92.69/81.7 sits just beyond conf 0.5 on
the measured curve, so the original submission was made at a higher threshold.

    conf 0.3   R 89.29  P 80.55  H 84.70
    conf 0.4   R 87.61  P 85.73  H 86.66
    conf 0.5   R 85.22  P 89.54  H 87.32
    paper      R 81.70  P 92.69  H 86.85

This notebook re-runs Task 4 at 0.5 and 0.6 for both arms. Nothing is retrained:
both detectors and the recogniser come from the control run's saved output, so it
is detection plus CPU recognition only, about 20 minutes per pass.

**Attach one input:** *Add Input -> Notebook Output* -> the finished control run.

**Accelerator: GPU T4 x2.** Detection uses the GPU; recognition stays on CPU for the
same reason as before."""),

    ("code", """SEED = 42
CONFS = (0.5, 0.6)
REC_DEVICE = "cpu"
OUT = "/kaggle/working/submissions\""""),

    ("code", """!pip install -q "ultralytics==8.3.189"
!pip install -q --timeout 180 --retries 10 paddlepaddle==3.2.0
!pip install -q paddleocr
!git clone -q --branch """ + BRANCH + " " + REPO + """ /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),

    ("code", """!cd /kaggle/working/repo && python ablation/install_modules.py"""),

    ("code", """import torch
from ultralytics import YOLO
from rects_control.detectors import register_pickle_aliases

register_pickle_aliases()
from ultralytics.nn.modules.block import CrossAttentionBlock, MultiScaleCBAM
print("torch", torch.__version__, "| CUDA:", torch.cuda.device_count(), "GPU(s)")"""),

    ("markdown", """## 1. Recover both detectors, the recogniser and the test images

All four come out of the control run's output. The test images are needed too - the
submissions are regenerated from them, not from cached detections."""),

    ("code", """import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
roots = sorted(INPUT.glob("*"))
print(f"{len(roots)} input(s):", [p.name for p in roots])

STAGED = Path("/kaggle/working/staged")
WANTED = ("PP-OCRv5_rects_rec_infer/", "paper_seed42/weights/best.pt",
          "stock_seed42/weights/best.pt", "Task3_and_Task4/img/", "task3_and_task4/img/")
for archive in sorted(INPUT.glob("**/*.zip")):
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.namelist()
                   if any(w in m for w in WANTED) and not m.endswith("/")]
        if members:
            print(f"extracting {len(members)} file(s) from {archive.name}")
            bundle.extractall(STAGED, members=members)

SEARCH = ([STAGED] if STAGED.exists() else []) + roots

def locate(what, *patterns):
    for root in SEARCH:
        for pattern in patterns:
            hits = sorted(root.glob(pattern))
            if hits:
                return hits[0]
    seen = chr(10).join(f"    {p}" for r in SEARCH for p in sorted(r.rglob("*"))[:40])
    raise FileNotFoundError(f"{what} not found. Tried {patterns}, saw:{chr(10)}{seen}")

REC_DIR = locate("recogniser", "**/PP-OCRv5_rects_rec_infer/inference.yml").parent
ARMS = {arm: locate(f"{arm} detector", f"**/{arm}_seed{SEED}/weights/best.pt")
        for arm in ("paper", "stock")}
# image_files() joins "ReCTS_test_part<n>/..." onto the root, so the root is three
# levels above the image, not two.
TEST_ROOT = locate("ReCTS test images", "**/Task3_and_Task4/img/*.jpg",
                   "**/task3_and_task4/img/*.jpg").parents[3]
print("recogniser:", REC_DIR)
for arm, path in ARMS.items():
    print(f"{arm}:", path)
print("test root:", TEST_ROOT)"""),

    ("code", """from rects_control.submission import image_files
n = len(image_files(TEST_ROOT))
print(f"{n} test images")
assert n == 5000, f"expected the full 5000-image ReCTS test set, found {n}\""""),

    ("markdown", """## 2. Regenerate Task 4

Both arms are read by one recogniser instance at each threshold, so the comparison
stays paired. Task 3 is written alongside at the same thresholds, which gives the
detection-only numbers at the operating point the paper reports."""),

    ("code", """from rects_control.recognizer import PaddleRecognizer
from rects_control.submission import write_task3, write_task4

recognizer = PaddleRecognizer(REC_DIR, device=REC_DEVICE)
out = Path(OUT); out.mkdir(exist_ok=True)
summary = []

for arm, weights in ARMS.items():
    model = YOLO(str(weights))
    for conf in CONFS:
        f3 = out / f"task3_{arm}_seed{SEED}_conf{conf:.1f}.txt"
        summary.append((arm, "task3", conf, write_task3(model, TEST_ROOT, f3, conf)))
        print(summary[-1], flush=True)
        f4 = out / f"task4_{arm}_seed{SEED}_conf{conf:.1f}.txt"
        summary.append((arm, "task4", conf, write_task4(model, recognizer, TEST_ROOT, f4, conf)))
        print(summary[-1], flush=True)"""),

    ("code", """for f in sorted(out.glob("*.txt")):
    print(f"{f.name}  {f.stat().st_size // 1024} KB")
print()
for arm, task, conf, boxes in summary:
    print(f"  {arm:6s} {task} conf={conf} -> {boxes} boxes")"""),

    ("markdown", """## 3. Upload

The `.txt` files upload directly - RRC rejects zips for these tasks. Submit each
`task4_*` to Task 4 and each `task3_*` to Task 3, naming them by arm and threshold.

The two numbers that matter: `task4_paper` against `task4_stock` at the same
threshold, which is the control, and whether `task4_paper` approaches the published
0.8566 at conf 0.6, which is the reproducibility question."""),
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
        ("rects_sweep_kaggle.ipynb", SWEEP_CELLS),
    ):
        (here / name).write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {name}  ({len(cells)} cells)")
