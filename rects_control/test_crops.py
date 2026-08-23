"""Crop preparation is checked on a synthetic ReCTS tree.

The recogniser is shared by both arms of the control, so a preparation bug would
move both 1-NED numbers together and stay invisible in the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from rects_control.crops import axis_aligned_crop, build

BOX = [10, 10, 60, 10, 60, 40, 10, 40]


def line(text: str, *, ignore: int = 0, points: list[int] | None = None) -> dict:
    return {"transcription": text, "ignore": ignore, "points": BOX if points is None else points}


@pytest.fixture
def rects_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    images, labels = tmp_path / "img", tmp_path / "gt"
    images.mkdir(), labels.mkdir()

    for index in range(20):
        cv2.imwrite(str(images / f"img_{index}.jpg"), np.full((80, 100, 3), 128, np.uint8))
        (labels / f"img_{index}.json").write_text(
            json.dumps({"lines": [
                line("店"),
                line("skipped", ignore=1),
                line(""),
                line("bad", points=[0, 0, 5, 5]),
            ]}),
            encoding="utf-8",
        )
    return images, labels, tmp_path / "rec"


def test_only_usable_lines_become_crops(rects_tree: tuple[Path, Path, Path]) -> None:
    images, labels, out = rects_tree
    train, val = build(images, labels, out)

    assert (train, val) == (18, 2)                        # one usable line per image, 90/10
    assert len(list((out / "train").glob("*.jpg"))) == 18
    assert len(list((out / "test").glob("*.jpg"))) == 2


def test_indices_are_unique_across_splits(rects_tree: tuple[Path, Path, Path]) -> None:
    images, labels, out = rects_tree
    build(images, labels, out)

    names = [p.name for p in (out / "train").glob("*.jpg")] + [
        p.name for p in (out / "test").glob("*.jpg")
    ]
    assert len(set(names)) == len(names)
    assert min(names) == "word_0000001.jpg"


def test_label_file_format(rects_tree: tuple[Path, Path, Path]) -> None:
    images, labels, out = rects_tree
    build(images, labels, out)

    rows = (out / "rec_gt_train.txt").read_text(encoding="utf-8").splitlines()
    path, transcription = rows[0].split("\t")
    assert path.startswith("train/word_")
    assert transcription == "店"
    assert (out / "rec_gt_test.txt").read_text(encoding="utf-8").startswith("test/word_")


def test_missing_image_is_skipped(tmp_path: Path) -> None:
    images, labels = tmp_path / "img", tmp_path / "gt"
    images.mkdir(), labels.mkdir()
    for index in range(10):
        (labels / f"img_{index}.json").write_text(json.dumps({"lines": [line("x")]}))
    cv2.imwrite(str(images / "img_0.jpg"), np.full((80, 100, 3), 128, np.uint8))

    assert sum(build(images, labels, tmp_path / "rec")) <= 1


def test_crop_clips_to_image_bounds() -> None:
    image = np.zeros((30, 40, 3), np.uint8)
    crop = axis_aligned_crop(image, [-20, -20, 90, -20, 90, 90, -20, 90])
    assert crop is not None and crop.shape[:2] == (30, 40)


def test_degenerate_quad_returns_none() -> None:
    assert axis_aligned_crop(np.zeros((30, 40, 3), np.uint8), [5, 5, 5, 5, 5, 5, 5, 5]) is None


def test_empty_annotation_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "gt").mkdir()
    with pytest.raises(RuntimeError, match="no ReCTS json"):
        build(tmp_path, tmp_path / "gt", tmp_path / "rec")
