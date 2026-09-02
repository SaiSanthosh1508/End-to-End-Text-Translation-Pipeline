"""Checkpoint loading depends on where the custom classes can be found.

`best.pt` pickles class references by import path. Three runs were lost to getting
that path wrong, so the aliasing is checked here rather than on Kaggle.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from rects_control.detectors import ARMS, arm_config, register_pickle_aliases


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, types.ModuleType]:
    """Stand in for a patched install: classes live in .custom, absent from .block."""
    block = types.ModuleType("ultralytics.nn.modules.block")
    custom = types.ModuleType("ultralytics.nn.modules.custom")
    custom.__all__ = ["CrossAttentionBlock", "MultiScaleCBAM"]
    for name in custom.__all__:
        setattr(custom, name, type(name, (), {}))

    modules = types.ModuleType("ultralytics.nn.modules")
    modules.custom = custom
    for name, module in {
        "ultralytics": types.ModuleType("ultralytics"),
        "ultralytics.nn": types.ModuleType("ultralytics.nn"),
        "ultralytics.nn.modules": modules,
        "ultralytics.nn.modules.block": block,
        "ultralytics.nn.modules.custom": custom,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return block, custom


def test_aliases_land_where_the_checkpoint_looks(fake_ultralytics) -> None:
    block, custom = fake_ultralytics
    assert not hasattr(block, "CrossAttentionBlock")

    register_pickle_aliases()

    for name in custom.__all__:
        assert getattr(block, name) is getattr(custom, name)


def test_existing_attributes_are_not_overwritten(fake_ultralytics) -> None:
    block, _ = fake_ultralytics
    sentinel = type("Original", (), {})
    block.MultiScaleCBAM = sentinel

    register_pickle_aliases()

    assert block.MultiScaleCBAM is sentinel


def test_registration_is_idempotent(fake_ultralytics) -> None:
    block, custom = fake_ultralytics
    register_pickle_aliases()
    first = block.CrossAttentionBlock
    register_pickle_aliases()
    assert block.CrossAttentionBlock is first


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_configs_differ_and_target_one_class(arm: str, tmp_path: Path) -> None:
    spec = yaml.safe_load(arm_config(arm, tmp_path).read_text())
    assert spec["nc"] == 1
    assert spec["backbone"] and spec["head"]


def test_arms_are_not_the_same_network(tmp_path: Path) -> None:
    heads = {arm: len(yaml.safe_load(arm_config(arm, tmp_path).read_text())["head"])
             for arm in ARMS}
    assert len(set(heads.values())) == len(heads), f"arms collapsed to one network: {heads}"


def test_unknown_arm_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown arm"):
        arm_config("nonexistent", tmp_path)
