"""Checks on the source surgery in install_modules, which rewrites site-packages."""

from __future__ import annotations

import ast

import pytest

from install_modules import (
    MARKER,
    PatchError,
    patch_modules_init_source,
    patch_tasks_source,
)

STOCK_TASKS = '''\
from ultralytics.utils import LOGGER


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml into a PyTorch model."""
    max_channels = 1024
    width = 0.5
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = globals()[m]
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        else:
            c2 = ch[f]
    return model
'''


def test_patched_tasks_is_valid_python() -> None:
    ast.parse(patch_tasks_source(STOCK_TASKS))


def test_branch_chains_onto_base_modules() -> None:
    out = patch_tasks_source(STOCK_TASKS)
    assert "        elif m in base_modules:" in out
    assert "        if m is MultiScaleCBAM:" in out
    assert out.count("if m in base_modules:") == 1


def test_import_precedes_parse_model() -> None:
    out = patch_tasks_source(STOCK_TASKS)
    assert out.index("from ultralytics.nn.modules.custom import") < out.index("def parse_model(")


def test_legacy_branch_reproduces_the_displaced_reduction() -> None:
    """The as-deployed arm must still bind r to the scaled channel count."""
    out = patch_tasks_source(STOCK_TASKS)
    branch = out[out.index("if m is MultiScaleCBAM:") : out.index("elif m in {MSCBAMFixed")]
    assert "args = [c1, c2, 1, *args[1:]]" in branch


def test_fixed_branch_binds_reduction_directly() -> None:
    out = patch_tasks_source(STOCK_TASKS)
    branch = out[out.index("elif m in {MSCBAMFixed") : out.index("elif m in {SimpleChannelAttention")]
    assert "args = [c1, args[1] if len(args) > 1 else 16]" in branch


def test_patch_is_idempotent() -> None:
    once = patch_tasks_source(STOCK_TASKS)
    assert patch_tasks_source(once) == once


def test_refuses_hand_modified_copy() -> None:
    modified = STOCK_TASKS.replace("elif m is Concat:", "elif m is CrossAttentionBlock:")
    with pytest.raises(PatchError, match="hand-modified"):
        patch_tasks_source(modified)


def test_refuses_missing_anchor() -> None:
    with pytest.raises(PatchError, match="parse_model"):
        patch_tasks_source(STOCK_TASKS.replace("        if m in base_modules:", "        if m in x:"))


def test_modules_init_appends_once() -> None:
    once = patch_modules_init_source("from .conv import Conv\n")
    assert once.count(MARKER) == 1
    assert patch_modules_init_source(once) == once
