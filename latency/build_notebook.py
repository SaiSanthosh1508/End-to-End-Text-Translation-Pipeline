"""Generate the deployment-latency notebook.

    python latency/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/SaiSanthosh1508/End-to-End-Text-Translation-Pipeline.git"
BRANCH = "ablation-mscbam-probe"
SPACE_WEIGHTS = (
    "https://huggingface.co/spaces/sai-santhosh/Text_Translation_Pipeline/"
    "resolve/main/best.pt"
)
MLT_DATASET = "rishiksaisanthosh/dataset-test"

CELLS: list[tuple[str, str]] = [
    ("markdown", f"""# Deployment latency, stage by stage

Table 6 reports 193 ms/image on CPU and 72 ms on a T4, with no measurement code in
the repository. A reviewer timed the deployed application at 1.2-2.5 s per image.

This notebook times each stage separately on the same weights the Space serves, so
the two figures can be reconciled instead of defended. The deployed pipeline is
detection, axis-aligned cropping, script-specific PaddleOCR recognition, then a
`GoogleTranslator` call per detected line. That last stage is a network request to
an external service, not model inference.

**Attach one dataset:** *Add Input -> Datasets -> `{MLT_DATASET}`* for the MLT
images.

**Accelerator: GPU T4 x1.** GPU and CPU are both measured; one T4 is enough.

Kaggle gives 4 vCPUs against the Space's 2, so the CPU figures here are optimistic
for the deployment and should be reported as measured on this machine."""),

    ("code", """N_IMAGES = 30          # timed images; the spread matters as much as the mean
WARMUP = 5             # excluded - first calls pay lazy init and cuDNN autotune
CONF = 0.4
TARGET_LANG = "en"
MEASURE_TRANSLATION = True   # needs Internet on; it is a network call, not compute"""),

    ("code", f"""!pip install -q "ultralytics==8.3.189" deep-translator
!pip install -q --timeout 180 --retries 10 paddlepaddle==3.2.0
!pip install -q paddleocr
!git clone -q --branch {BRANCH} {REPO} /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),

    ("code", """!cd /kaggle/working/repo && python ablation/install_modules.py"""),

    ("markdown", """## 1. The weights the Space actually serves

`best.pt` is stored in Git LFS, so a plain clone yields a 133-byte pointer. Pulling
it from the Space guarantees these timings describe the deployed model."""),

    ("code", f"""from pathlib import Path

!wget -q -O /kaggle/working/best.pt {SPACE_WEIGHTS}
weights = Path("/kaggle/working/best.pt")
print(weights, weights.stat().st_size, "bytes")
assert weights.stat().st_size == 21363137, "not the deployed checkpoint"

import torch
from rects_control.detectors import register_pickle_aliases
register_pickle_aliases()
from ultralytics import YOLO

detector = YOLO(str(weights))
params = sum(p.numel() for p in detector.model.parameters())
print(f"{{params/1e6:.2f}}M parameters")"""),

    ("markdown", """### Is this GPU usable?

Kaggle offers P100 as well as T4, and its PyTorch build supports sm_70 and above.
A P100 is sm_60, so selecting it makes every CUDA call fail with "no kernel image
is available". Checking here means a wrong accelerator costs the GPU column
rather than the whole run."""),

    ("code", """arches = torch.cuda.get_arch_list()
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    usable = f"sm_{major}{minor}" in arches
    print(f"{name}  sm_{major}{minor}  usable: {usable}")
else:
    name, usable = None, False
    print("no GPU visible")

DEVICES = ["cuda:0", "cpu"] if usable else ["cpu"]
if not usable:
    print(f"This PyTorch supports {' '.join(arches)}.")
    print("Timing CPU only. For GPU numbers set Accelerator to GPU T4 and re-run.")"""),

    ("markdown", "## 2. Images"),

    ("code", """import itertools

candidates = sorted(itertools.islice(
    (p for p in Path("/kaggle/input").rglob("*.jpg") if "images/val" in str(p)),
    N_IMAGES + WARMUP,
))
if not candidates:
    candidates = sorted(itertools.islice(Path("/kaggle/input").rglob("*.jpg"),
                                         N_IMAGES + WARMUP))
assert len(candidates) >= N_IMAGES + WARMUP, f"only {len(candidates)} images found"
print(f"{len(candidates)} images")"""),

    ("markdown", """## 3. The pipeline stages

Recognition mirrors the deployed application: one PaddleOCR mobile model per script
class, selected by the detector's predicted label, loaded once and reused."""),

    ("code", """import cv2
import numpy as np
from paddleocr import TextRecognition
from deep_translator import GoogleTranslator

CLASS_MODELS = {
    0: "arabic_PP-OCRv3_mobile_rec", 1: "en_PP-OCRv3_mobile_rec",
    2: "ch_PP-OCRv4_mobile_rec",     3: "korean_PP-OCRv3_mobile_rec",
    4: "japan_PP-OCRv3_mobile_rec",  5: "bangla_PP-OCRv3_mobile_rec",
    6: "devanagari_PP-OCRv3_mobile_rec", 7: "en_PP-OCRv3_mobile_rec",
}
engines = {}

def recogniser_for(cls):
    name = CLASS_MODELS[int(cls)]
    if name not in engines:
        engines[name] = TextRecognition(model_name=name, device="cpu")
    return engines[name]

def crop(image, quad):
    pts = np.asarray(quad, dtype=int)
    h, w = image.shape[:2]
    y0, y1 = max(0, pts[:, 1].min() - 2), min(h, pts[:, 1].max() + 2)
    x0, x1 = max(0, pts[:, 0].min() - 2), min(w, pts[:, 0].max() + 2)
    return image[y0:y1, x0:x1]

def translate(texts):
    if not (MEASURE_TRANSLATION and texts):
        return texts
    return [GoogleTranslator(source="auto", target=TARGET_LANG).translate(t) for t in texts]"""),

    ("markdown", """## 4. Measure

Warm-up runs are discarded: the first inference pays lazy model initialisation and
cuDNN autotuning, which is not what a deployed request costs."""),

    ("code", """from latency.measure import as_table, summarise, time_image

def run(device):
    detector.to(device)
    classes_seen = []

    def detect(image):
        result = detector.predict(image, conf=CONF, device=device, verbose=False)[0]
        if result.obb is None or len(result.obb) == 0:
            classes_seen.clear()
            return []
        classes_seen[:] = result.obb.cls.cpu().numpy().tolist()
        return list(result.obb.xyxyxyxy.cpu().numpy())

    def recognise(crops):
        out = []
        for c, cls in zip(crops, classes_seen):
            prediction = recogniser_for(cls).predict([c], batch_size=1)
            out.append(prediction[0]["rec_text"].strip())
        return out

    runs = []
    for n, path in enumerate(candidates):
        image = cv2.imread(str(path))
        times = time_image(image, detect, crop, recognise, translate)
        if n >= WARMUP:
            runs.append(times)
    return summarise(runs)

results = {}
for device in DEVICES:
    print(f"\\n===== {device} =====", flush=True)
    results[device] = run(device)
    print(as_table(results[device]))"""),

    ("markdown", "## 5. The table for the paper"),

    ("code", """header = "".join(f"{d:>20s}" for d in DEVICES)
print(f"{'stage':12s}{header}")
print("-" * (12 + 20 * len(DEVICES)))
for stage in ("detect", "crop", "recognise", "translate", "total"):
    cells = "".join(f"{results[d][stage][0]:11.1f} +/-{results[d][stage][1]:5.1f}" for d in DEVICES)
    print(f"{stage:12s}{cells}")
print()
for d in DEVICES:
    print(f"detection only, {d:7s} {results[d]['detect'][0]:7.0f} ms")
    print(f"end-to-end,     {d:7s} {results[d]['total'][0]:7.0f} ms")"""),

    ("markdown", """## How to read this

Compare `detect` against the paper's 72 ms (GPU) and 193 ms (CPU). If they match,
Table 6 was reporting detection only and simply needs relabelling, with an
end-to-end row added.

The reviewer's 1.2-2.5 s should be close to `total`, and `translate` is expected to
dominate it: one network request per detected line, which is neither model
inference nor something the architecture affects. Reporting it separately is the
honest way to answer that comment."""),
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
    out = Path(__file__).with_name("latency_kaggle.ipynb")
    out.write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")
