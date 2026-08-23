"""Train the two ReCTS detector arms under one recipe.

The reviewer's question is what the architecture contributes once the recogniser is
held fixed, so the arms differ only in the model yaml. Every training argument here
is taken from the published run's ``args.yaml`` (runs/obb/train3): 100 epochs,
batch 64, imgsz 480, cosine schedule, ``pretrained=True``, seed 42.
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

import yaml

ABLATION_CONFIGS = Path(__file__).resolve().parent.parent / "ablation" / "configs"
ARMS = {
    "stock": "a1_stock.yaml",       # baseline the reviewer asked to see
    "paper": "full_legacy.yaml",    # BiFPN + MS-CBAM + cross-attention, as deployed
}
RECTS_CLASSES = 1

TRAIN_ARGS = {
    "epochs": 100,
    "batch": 64,
    "imgsz": 480,
    "cos_lr": True,
    "deterministic": True,
    "pretrained": True,
    "plots": False,
    "exist_ok": True,
}


def arm_config(arm: str, out_dir: Path) -> Path:
    """Copy an ablation config with ``nc`` set for ReCTS's single text class."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    spec = yaml.safe_load((ABLATION_CONFIGS / ARMS[arm]).read_text())
    spec["nc"] = RECTS_CLASSES

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{arm}_rects.yaml"
    target.write_text(yaml.dump(spec, default_flow_style=False, sort_keys=False))
    return target


def train(arm: str, data: Path, project: Path, seed: int, device: str = "0,1") -> Path | None:
    """Train one arm and return its best checkpoint, or None if the run failed.

    A raised exception would cost every completed run in the same Kaggle session,
    which saves no output at all when a version errors, so failures are contained.
    """
    from ultralytics import YOLO

    name = f"{arm}_seed{seed}"
    best = project / name / "weights" / "best.pt"
    if best.exists():
        print(f"{name}: already trained, skipping")
        return best

    try:
        YOLO(str(arm_config(arm, project / "configs"))).train(
            data=str(data), project=str(project), name=name,
            seed=seed, device=device, **TRAIN_ARGS,
        )
    except Exception:  # noqa: BLE001 - one bad arm must not discard the other
        traceback.print_exc()
        print(f"{name} FAILED; continuing")
        return None
    return best if best.exists() else None


def adopt_published(checkpoint: Path, project: Path, seed: int = 42) -> Path:
    """Place the released ReCTS checkpoint where the control expects the paper arm.

    Reusing it rather than retraining keeps the comparison against the detector the
    paper actually reports, and saves 4.4 h.
    """
    target = project / f"paper_seed{seed}" / "weights" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(checkpoint, target)
    return target
