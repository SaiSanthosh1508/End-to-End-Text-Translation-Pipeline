"""Timing code that mis-attributes a stage is worse than no timing at all."""

from __future__ import annotations

import time

import numpy as np
import pytest

from latency.measure import StageTimes, as_table, summarise, time_image

IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)
QUAD = np.array([[0, 0], [30, 0], [30, 20], [0, 20]], dtype=np.float32)


def sleeper(ms: float):
    def work(*args, **kwargs):
        time.sleep(ms / 1000)
        return []
    return work


def test_each_stage_is_attributed_to_itself() -> None:
    times = time_image(
        IMAGE,
        detect=lambda img: (time.sleep(0.030), [QUAD, QUAD])[1],
        crop=lambda img, q: (time.sleep(0.005), np.zeros((10, 10, 3), np.uint8))[1],
        recognise=lambda crops: (time.sleep(0.020), ["a"] * len(crops))[1],
        translate=lambda texts: (time.sleep(0.040), texts)[1],
    )

    assert times.detect == pytest.approx(30, abs=25)
    assert times.crop == pytest.approx(10, abs=25)        # two crops at 5 ms
    assert times.recognise == pytest.approx(20, abs=25)
    assert times.translate == pytest.approx(40, abs=25)
    assert times.boxes == 2


def test_total_is_the_sum_of_stages() -> None:
    times = StageTimes(detect=10.0, crop=2.0, recognise=30.0, translate=800.0, boxes=5)
    assert times.total == pytest.approx(842.0)


def test_empty_crops_are_not_recognised() -> None:
    """A degenerate box must not be sent to the recogniser as a zero-size array."""
    seen: list[int] = []
    time_image(
        IMAGE,
        detect=lambda img: [QUAD, QUAD],
        crop=lambda img, q: np.zeros((0, 0, 3), np.uint8),
        recognise=lambda crops: seen.append(len(crops)) or [],
        translate=lambda texts: texts,
    )
    assert seen == [0]


def test_summary_reports_mean_and_spread() -> None:
    runs = [
        StageTimes(10.0, 1.0, 20.0, 100.0, 3),
        StageTimes(20.0, 1.0, 40.0, 300.0, 5),
    ]
    summary = summarise(runs)

    assert summary["detect"] == pytest.approx((15.0, 7.071), abs=1e-3)
    assert summary["translate"][0] == pytest.approx(200.0)
    assert summary["total"][0] == pytest.approx(246.0)
    assert summary["boxes"][0] == pytest.approx(4.0)


def test_single_run_has_zero_spread() -> None:
    assert summarise([StageTimes(1.0, 1.0, 1.0, 1.0, 1)])["detect"] == (1.0, 0.0)


def test_no_runs_is_an_error() -> None:
    with pytest.raises(ValueError, match="no timing runs"):
        summarise([])


def test_table_lists_every_stage() -> None:
    table = as_table(summarise([StageTimes(10.0, 1.0, 20.0, 100.0, 3)]))
    for stage in ("detect", "crop", "recognise", "translate", "total", "boxes/img"):
        assert stage in table
