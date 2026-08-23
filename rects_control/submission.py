"""RRC ReCTS Task 3 and Task 4 submission writers.

Point order follows the released scripts: Ultralytics returns the quadrilateral
counter-clockwise and the evaluation server expects clockwise, so corners are
emitted as (P1, P4, P3, P2). Changing it would silently halve the reported H-mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .recognizer import Recognizer

TEST_DIRS = (
    "ReCTS_test_part1/Task3_and_Task4/img",
    "ReCTS_test_part2/task3_and_task4/img",
)
SUBMITTED_PREFIX = "test_ReCTS_task3_and_task_4_"
CLOCKWISE = (0, 3, 2, 1)


def submission_name(path: Path) -> str:
    """The server keys on the short form of the test image name."""
    name = path.name
    return name.replace(SUBMITTED_PREFIX, "test_") if name.startswith(SUBMITTED_PREFIX) else name


def clockwise_points(quad: np.ndarray, width: int, height: int) -> list[int]:
    """Clip a counter-clockwise (4, 2) quad to the frame and reorder it clockwise."""
    x = np.clip(quad[:, 0], 0, width - 1).astype(int)
    y = np.clip(quad[:, 1], 0, height - 1).astype(int)
    return [int(v) for i in CLOCKWISE for v in (x[i], y[i])]


def crop_bgr(image: Image.Image, quad: np.ndarray) -> np.ndarray | None:
    """The axis-aligned crop the recogniser was trained on, as BGR for Paddle."""
    width, height = image.size
    x1, x2 = (int(np.clip(f(quad[:, 0]), 0, width - 1)) for f in (np.min, np.max))
    y1, y2 = (int(np.clip(f(quad[:, 1]), 0, height - 1)) for f in (np.min, np.max))
    if x1 >= x2 or y1 >= y2:
        return None
    return np.asarray(image.crop((x1, y1, x2, y2)))[:, :, ::-1]


def detect(model, image: Image.Image, conf: float) -> np.ndarray:
    """Oriented quadrilaterals as (N, 4, 2), empty when nothing is found."""
    result = model.predict(image, conf=conf, verbose=False)[0]
    if result.obb is None or len(result.obb) == 0:
        return np.empty((0, 4, 2), dtype=np.float32)
    return result.obb.xyxyxyxy.cpu().numpy()


def image_files(root: Path) -> list[Path]:
    present = [root / d for d in TEST_DIRS if (root / d).is_dir()]
    if not present:
        raise FileNotFoundError(f"no ReCTS test image directory under {root}")
    return [p for d in present for p in sorted(d.glob("*.jpg"))]


def write_task3(model, root: Path, out: Path, conf: float) -> int:
    """Detection-only submission. Returns the number of boxes written."""
    boxes = 0
    with out.open("w", encoding="utf-8") as handle:
        for path in image_files(root):
            image = Image.open(path)
            width, height = image.size
            handle.write(f"{submission_name(path)}\n")
            for quad in detect(model, image, conf):
                handle.write(",".join(map(str, clockwise_points(quad, width, height))) + "\n")
                boxes += 1
    return boxes


def write_task4(
    model, recognizer: Recognizer, root: Path, out: Path, conf: float
) -> int:
    """End-to-end submission. Returns the number of transcribed boxes written."""
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for path in image_files(root):
            image = Image.open(path)
            width, height = image.size

            kept: list[np.ndarray] = []
            crops: list[np.ndarray] = []
            for quad in detect(model, image, conf):
                crop = crop_bgr(image, quad)
                if crop is not None and crop.size:
                    kept.append(quad)
                    crops.append(crop)

            texts = recognizer(crops)
            handle.write(f"{submission_name(path)}\n")
            for quad, text in zip(kept, texts):
                if not text:
                    continue
                points = ",".join(map(str, clockwise_points(quad, width, height)))
                handle.write(f"{points},{text}\n")
                written += 1
    return written
