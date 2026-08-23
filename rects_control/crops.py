"""Build the PP-OCRv5 recognition training set from ReCTS line annotations.

Reproduces the preparation used for the published recogniser: an axis-aligned crop
per annotated line, a 90/10 file-level split at ``random_state=42``, and PaddleOCR
label files. The recogniser trained here is shared by both arms of the detector
control, so any deviation from the original preparation would confound it.

One departure is deliberate: annotation files are sorted before the split. The original
passed ``glob.glob`` order, which is filesystem order, so its exact split cannot be
reproduced by anyone including its author. Sorting makes the split reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

VAL_SPLIT = 0.1
SPLIT_SEED = 42
IMAGE_SUFFIXES = (".jpg", ".png", ".jpeg")
QUAD_COORDS = 8


@dataclass(frozen=True)
class Crop:
    """One recognition sample: the file written to disk and its transcription."""

    filename: str
    transcription: str


def axis_aligned_crop(image: np.ndarray, points: list[float]) -> np.ndarray | None:
    """Return the horizontal box enclosing a quadrilateral, or None if degenerate."""
    corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
    height, width = image.shape[:2]
    x_min = max(0, int(corners[:, 0].min()))
    y_min = max(0, int(corners[:, 1].min()))
    x_max = min(width, int(corners[:, 0].max()))
    y_max = min(height, int(corners[:, 1].max()))
    if x_min >= x_max or y_min >= y_max:
        return None
    return image[y_min:y_max, x_min:x_max]


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def extract(
    label_files: list[Path], image_dir: Path, out_dir: Path, start_index: int
) -> tuple[list[Crop], int]:
    """Write one crop per usable line; return the samples and the next free index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Crop] = []
    index = start_index

    for label_file in label_files:
        image_path = find_image(image_dir, label_file.stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        lines = json.loads(label_file.read_text(encoding="utf-8")).get("lines", [])
        for line in lines:
            if line.get("ignore", 0) != 0 or not line.get("transcription"):
                continue
            points = line.get("points")
            if not points or len(points) != QUAD_COORDS:
                continue
            crop = axis_aligned_crop(image, points)
            if crop is None or crop.size == 0:
                continue

            filename = f"word_{index:07d}.jpg"
            cv2.imwrite(str(out_dir / filename), crop)
            samples.append(Crop(filename, line["transcription"]))
            index += 1

    return samples, index


def write_labels(samples: list[Crop], path: Path, subfolder: str) -> None:
    path.write_text(
        "".join(f"{subfolder}/{s.filename}\t{s.transcription}\n" for s in samples),
        encoding="utf-8",
    )


def build(image_dir: Path, label_dir: Path, out_root: Path) -> tuple[int, int]:
    """Create ``<out_root>/{train,test}`` plus the two label files."""
    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        raise RuntimeError(f"no ReCTS json annotations under {label_dir}")

    train_files, val_files = train_test_split(
        label_files, test_size=VAL_SPLIT, random_state=SPLIT_SEED
    )
    train, index = extract(train_files, image_dir, out_root / "train", 1)
    val, _ = extract(val_files, image_dir, out_root / "test", index)

    write_labels(train, out_root / "rec_gt_train.txt", "train")
    write_labels(val, out_root / "rec_gt_test.txt", "test")
    return len(train), len(val)
