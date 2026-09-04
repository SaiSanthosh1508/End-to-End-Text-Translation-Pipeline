"""Per-stage latency of the deployed detect-recognise-translate pipeline.

Table 6 of the paper reports 193 ms/image on CPU with no accompanying measurement
code, and a reviewer timed the deployed application at 1.2-2.5 s. Splitting the
pipeline into stages shows which figure describes what: translation is a network
call to an external service and is not model inference at all.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

STAGES = ("detect", "crop", "recognise", "translate")


class Detector(Protocol):
    def __call__(self, image: np.ndarray) -> Sequence[np.ndarray]: ...


class Recogniser(Protocol):
    def __call__(self, crops: list[np.ndarray]) -> list[str]: ...


class Translator(Protocol):
    def __call__(self, texts: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class StageTimes:
    """Milliseconds spent in each stage of one image, and the boxes found."""

    detect: float
    crop: float
    recognise: float
    translate: float
    boxes: int

    @property
    def total(self) -> float:
        return self.detect + self.crop + self.recognise + self.translate


def _elapsed_ms(work: Callable[[], object]) -> tuple[float, object]:
    start = perf_counter()
    value = work()
    return (perf_counter() - start) * 1000, value


def time_image(
    image: np.ndarray, detect: Detector, crop: Callable[[np.ndarray, np.ndarray], np.ndarray],
    recognise: Recogniser, translate: Translator,
) -> StageTimes:
    """Time one image through the pipeline, stage by stage."""
    detect_ms, quads = _elapsed_ms(lambda: list(detect(image)))
    crop_ms, crops = _elapsed_ms(lambda: [crop(image, q) for q in quads])
    usable = [c for c in crops if c is not None and c.size]
    recognise_ms, texts = _elapsed_ms(lambda: recognise(usable))
    translate_ms, _ = _elapsed_ms(lambda: translate([t for t in texts if t]))

    return StageTimes(detect_ms, crop_ms, recognise_ms, translate_ms, len(usable))


def summarise(runs: Sequence[StageTimes]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation per stage, plus the total and box count.

    A single-image timing is dominated by scheduling noise, so the spread matters
    as much as the mean; reporting one without the other is what makes a latency
    claim unfalsifiable.
    """
    if not runs:
        raise ValueError("no timing runs to summarise")

    def spread(values: Sequence[float]) -> tuple[float, float]:
        return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    summary = {stage: spread([getattr(r, stage) for r in runs]) for stage in STAGES}
    summary["total"] = spread([r.total for r in runs])
    summary["boxes"] = spread([float(r.boxes) for r in runs])
    return summary


def as_table(summary: dict[str, tuple[float, float]]) -> str:
    rows = [f"{'stage':12s} {'mean (ms)':>12s} {'sd':>9s}", "-" * 34]
    for stage in (*STAGES, "total"):
        mean, sd = summary[stage]
        rows.append(f"{stage:12s} {mean:12.1f} {sd:9.1f}")
    rows.append(f"{'boxes/img':12s} {summary['boxes'][0]:12.1f} {summary['boxes'][1]:9.1f}")
    return "\n".join(rows)
