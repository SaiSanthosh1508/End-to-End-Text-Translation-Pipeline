"""Register the ablation's attention modules in the installed Ultralytics package.

Run once per environment, before training:

    python ablation/install_modules.py
    python ablation/verify_install.py

The patch adds one explicit branch to ``parse_model`` that handles every custom module
by name, rather than relying on the ``base_modules`` / ``repeat_modules`` sets. Those
sets rewrite args on the assumption that a module's second positional is an
output-channel count, which is what silently turns the ``MultiScaleCBAM`` reduction
ratio into a channel count. The branch reproduces that rewrite verbatim for
``MultiScaleCBAM``, so the as-deployed arm of the probe stays identical to the network
behind ``best.pt``, and binds ``(c1, r)`` correctly for the fixed variants.

Targets a stock Ultralytics install. It refuses to touch a copy that already carries
hand-added custom modules, such as the vendored tree under ``Text_Translation_Pipeline``,
which serves the Hugging Face Space and is not a training environment.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

MARKER = "# --- ablation modules registered ---"
IMPORT_LINE = "from ultralytics.nn.modules.custom import *  # noqa: F403,E402\n"
ANCHOR_BRANCH = "        if m in base_modules:"
ANCHOR_IMPORT = "def parse_model("

BRANCH = f"""        {MARKER}
        if m is MultiScaleCBAM:
            # Reproduces the base_modules + repeat_modules double rewrite that produced
            # best.pt: the YAML reduction ratio is displaced into *args and r binds to
            # the scaled channel count, collapsing the bottleneck to a single channel.
            c1 = ch[f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, 1, *args[1:]]
        elif m in {{MSCBAMFixed, StandardCBAM}}:
            c1 = ch[f]
            args = [c1, args[1] if len(args) > 1 else 16]
            c2 = c1
        elif m in {{SimpleChannelAttention, MultiScaleSpatialAttention, SingleScaleSpatialAttention}}:
            c1 = ch[f]
            args = [c1, *args[1:]]
            c2 = c1
        elif m is CrossAttentionBlock:
            args[0] = make_divisible(min(args[0], max_channels) * width, 8)
            args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            c2 = args[0]
        el"""


class PatchError(RuntimeError):
    """Raised when the target package is not in a state the patch can handle."""


def patch_tasks_source(text: str) -> str:
    """Return ``tasks.py`` source with the custom-module branch and import added."""
    if MARKER in text:
        return text
    if "CrossAttentionBlock" in text:
        raise PatchError(
            "tasks.py already references CrossAttentionBlock; this looks like a "
            "hand-modified copy. Point the patch at a clean Ultralytics install."
        )
    if text.count(ANCHOR_BRANCH) != 1:
        raise PatchError(
            f"expected exactly one {ANCHOR_BRANCH.strip()!r} in parse_model, "
            f"found {text.count(ANCHOR_BRANCH)}"
        )
    if ANCHOR_IMPORT not in text:
        raise PatchError("could not find parse_model in tasks.py")

    text = text.replace(ANCHOR_BRANCH, BRANCH + ANCHOR_BRANCH.strip(), 1)
    text = text.replace(ANCHOR_IMPORT, f"{IMPORT_LINE}\n\n{ANCHOR_IMPORT}", 1)

    ast.parse(text)
    return text


def patch_modules_init_source(text: str) -> str:
    if MARKER in text:
        return text
    return f"{text}\n{MARKER}\nfrom .custom import *  # noqa: F403,E402\n"


def locate_ultralytics() -> Path:
    try:
        import ultralytics
    except ImportError as exc:
        raise PatchError("ultralytics is not installed in this environment") from exc
    return Path(ultralytics.__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change without writing to the package",
    )
    opts = parser.parse_args()

    root = locate_ultralytics()
    tasks = root / "nn" / "tasks.py"
    init = root / "nn" / "modules" / "__init__.py"

    patched_tasks = patch_tasks_source(tasks.read_text(encoding="utf-8"))
    patched_init = patch_modules_init_source(init.read_text(encoding="utf-8"))

    print(f"ultralytics: {root}")
    if opts.check:
        for path, new in ((tasks, patched_tasks), (init, patched_init)):
            state = "up to date" if new == path.read_text(encoding="utf-8") else "would patch"
            print(f"  {state}  {path.relative_to(root)}")
        return 0

    target = root / "nn" / "modules" / "custom.py"
    shutil.copyfile(Path(__file__).with_name("custom_modules.py"), target)
    print(f"  wrote      {target.relative_to(root)}")

    for path, new in ((init, patched_init), (tasks, patched_tasks)):
        changed = new != path.read_text(encoding="utf-8")
        if changed:
            path.write_text(new, encoding="utf-8")
        print(f"  {'patched   ' if changed else 'up to date '}{path.relative_to(root)}")

    print("\nNext: python ablation/verify_install.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PatchError as exc:
        sys.exit(f"error: {exc}")
