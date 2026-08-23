"""Recognition back-ends for the end-to-end submissions.

Both detector arms are read by the same recogniser instance, which is what makes the
1-NED difference attributable to detection alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

REC_BATCH = 64
PPOCR_V5_MOBILE = "PP-OCRv5_mobile_rec"


class Recognizer(Protocol):
    """Maps a batch of BGR crops to one transcription each, in order."""

    def __call__(self, crops: list[np.ndarray]) -> list[str]: ...


class PaddleRecognizer:
    """The ReCTS fine-tuned PP-OCRv5 mobile model, exported for inference."""

    def __init__(
        self, model_dir: Path, *, device: str = "gpu:0",
        model_name: str = PPOCR_V5_MOBILE, batch_size: int = REC_BATCH,
    ) -> None:
        from paddleocr import TextRecognition

        if not (model_dir / "inference.yml").exists():
            raise FileNotFoundError(
                f"{model_dir} has no inference.yml; export it with "
                "PaddleOCR/tools/export_model.py before running the control"
            )
        self._model = TextRecognition(
            model_name=model_name, model_dir=str(model_dir), device=device
        )
        self._batch_size = batch_size

    def __call__(self, crops: list[np.ndarray]) -> list[str]:
        if not crops:
            return []
        predictions = self._model.predict(crops, batch_size=self._batch_size)
        # PaddleX drops inputs it cannot read with only a warning; a short result list
        # would shift every transcription onto the wrong box from that point on.
        if len(predictions) != len(crops):
            raise RuntimeError(
                f"recogniser returned {len(predictions)} results for {len(crops)} crops"
            )
        return [p["rec_text"].strip() for p in predictions]


class EasyOCRRecognizer:
    """The recogniser the released submission script actually used.

    Kept because the published 1-NED cannot currently be attributed to either
    back-end; running both resolves which one produced it.
    """

    def __init__(self, languages: tuple[str, ...] = ("ch_sim", "en")) -> None:
        import easyocr

        self._reader = easyocr.Reader(list(languages))

    def __call__(self, crops: list[np.ndarray]) -> list[str]:
        return [" ".join(self._reader.readtext(c, detail=0)).strip() for c in crops]
