"""Assert that each probe arm was built with the channel bottleneck it claims.

Run after install_modules.py and before training. Without this check a failed patch is
invisible: the fixed arm would silently rebuild the 1-channel bottleneck and the probe
would compare a network against itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch import nn
from ultralytics.nn.tasks import OBBModel

CONFIGS = Path(__file__).parent / "configs"
REDUCTION = 16


def bottlenecks(model: nn.Module, class_name: str) -> list[tuple[int, int]]:
    """Return (input_channels, bottleneck_channels) for every CBAM of the given class."""
    found = []
    for module in model.modules():
        if type(module).__name__ != class_name:
            continue
        squeeze, _, expand = module.ca.fc
        found.append((squeeze.in_channels, squeeze.out_channels))
    return found


def check(config: str, class_name: str, expect_ratio: int | None) -> bool:
    model = OBBModel(str(CONFIGS / config), ch=3, nc=8, verbose=False)
    sites = bottlenecks(model, class_name)
    params = sum(p.numel() for p in model.parameters())

    if not sites:
        print(f"FAIL {config}: no {class_name} instances found")
        return False

    ok = True
    print(f"\n{config}  ({class_name}, {params/1e6:.2f}M params)")
    for c1, hidden in sites:
        want = max(1, c1 // expect_ratio) if expect_ratio else 1
        status = "ok " if hidden == want else "BAD"
        ok &= hidden == want
        print(f"  {status} c1={c1:<5} bottleneck={hidden:<4} expected={want}")
    return ok


def main() -> int:
    legacy = check("full_legacy.yaml", "MultiScaleCBAM", expect_ratio=None)
    fixed = check("full_fixed.yaml", "MSCBAMFixed", expect_ratio=REDUCTION)

    print()
    if legacy and fixed:
        print("Both arms constructed as intended. Safe to train.")
        return 0
    print("Arms are not as intended. Re-run install_modules.py; do not train.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
