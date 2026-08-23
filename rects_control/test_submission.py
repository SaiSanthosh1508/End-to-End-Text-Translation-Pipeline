"""The submission format is verified against the released scripts, not against intent.

An error here is invisible on Kaggle and only shows up as a bad score from the
evaluation server days later, so the point ordering and the name rewriting are
checked byte-for-byte against a transcription of the original code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rects_control.submission import (
    clockwise_points,
    crop_bgr,
    submission_name,
    write_task3,
    write_task4,
)


def original_points(quad: np.ndarray, width: int, height: int) -> list[int]:
    """Verbatim transcription of the released generate_rrc_task3 point handling."""
    flat = quad.reshape(8).tolist()
    x1 = int(np.clip(flat[0], 0, width - 1)); y1 = int(np.clip(flat[1], 0, height - 1))
    x2 = int(np.clip(flat[2], 0, width - 1)); y2 = int(np.clip(flat[3], 0, height - 1))
    x3 = int(np.clip(flat[4], 0, width - 1)); y3 = int(np.clip(flat[5], 0, height - 1))
    x4 = int(np.clip(flat[6], 0, width - 1)); y4 = int(np.clip(flat[7], 0, height - 1))
    return [x1, y1, x4, y4, x3, y3, x2, y2]


QUADS = [
    np.array([[10, 20], [90, 25], [88, 60], [8, 55]], dtype=np.float32),
    np.array([[-5, -5], [700, 3], [699, 480], [0, 470]], dtype=np.float32),
    np.array([[0.4, 0.6], [10.9, 1.2], [11.5, 9.9], [0.1, 9.4]], dtype=np.float32),
]


@pytest.mark.parametrize("quad", QUADS)
def test_matches_released_point_order(quad: np.ndarray) -> None:
    assert clockwise_points(quad, 640, 480) == original_points(quad, 640, 480)


def test_clipped_into_frame() -> None:
    points = clockwise_points(QUADS[1], 640, 480)
    assert all(0 <= v <= 639 for v in points[0::2])
    assert all(0 <= v <= 479 for v in points[1::2])


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("test_ReCTS_task3_and_task_4_000123.jpg", "test_000123.jpg"),
        ("test_000123.jpg", "test_000123.jpg"),
    ],
)
def test_submission_name(given: str, expected: str, tmp_path: Path) -> None:
    assert submission_name(tmp_path / given) == expected


def test_crop_is_bgr_and_axis_aligned() -> None:
    rgb = np.zeros((50, 60, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200                                   # pure red in RGB
    crop = crop_bgr(Image.fromarray(rgb), QUADS[0])
    assert crop is not None
    assert crop[0, 0].tolist() == [0, 0, 200]            # blue channel first
    assert crop.shape[:2] == (49 - 20, 59 - 8)           # quad clipped to the 60x50 frame


def test_degenerate_crop_is_dropped() -> None:
    flat = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)
    assert crop_bgr(Image.new("RGB", (20, 20)), flat) is None


class FakeOBB:
    def __init__(self, quads: np.ndarray) -> None:
        self.xyxyxyxy = self
        self._quads = quads

    def __len__(self) -> int:
        return len(self._quads)

    def cpu(self) -> "FakeOBB":
        return self

    def numpy(self) -> np.ndarray:
        return self._quads


class FakeResult:
    def __init__(self, quads: np.ndarray | None) -> None:
        self.obb = None if quads is None else FakeOBB(quads)


class FakeModel:
    """Returns two boxes for the first image and nothing for the second."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image, conf: float, verbose: bool = False) -> list[FakeResult]:
        self.calls += 1
        return [FakeResult(np.stack(QUADS[:2]) if self.calls == 1 else None)]


@pytest.fixture
def test_root(tmp_path: Path) -> Path:
    images = tmp_path / "ReCTS_test_part1/Task3_and_Task4/img"
    images.mkdir(parents=True)
    for name in ("test_ReCTS_task3_and_task_4_000002.jpg", "test_ReCTS_task3_and_task_4_000001.jpg"):
        Image.new("RGB", (640, 480), (30, 60, 90)).save(images / name)
    return tmp_path


def test_task3_lists_every_image_even_when_empty(test_root: Path) -> None:
    out = test_root / "task3.txt"
    boxes = write_task3(FakeModel(), test_root, out, conf=0.4)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert boxes == 2
    assert lines[0] == "test_000001.jpg"                 # sorted, prefix rewritten
    assert lines[3] == "test_000002.jpg"                 # present with no boxes under it
    assert len(lines[1].split(",")) == 8


def test_task4_drops_empty_transcriptions(test_root: Path) -> None:
    out = test_root / "task4.txt"
    written = write_task4(
        FakeModel(), lambda crops: ["店", ""], test_root, out, conf=0.4
    )

    lines = out.read_text(encoding="utf-8").splitlines()
    assert written == 1
    assert lines[1].endswith(",店")
    assert len(lines[1].split(",")) == 9
    assert lines[2] == "test_000002.jpg"
