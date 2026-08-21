"""Publish probe snapshots to a Kaggle Dataset so results survive a killed session.

    python ablation/push_snapshot.py --dir runs/snapshot --slug <user>/mscbam-probe

Kaggle saves a batch version's output only when the notebook finishes. Catching a
failed run (see run_probe.py) covers the common case, but a session killed by timeout
or OOM still loses everything. Pushing after each run puts the few kilobytes that
matter somewhere the kernel cannot take down with it.

Credentials come from KAGGLE_USERNAME and KAGGLE_KEY. Inside a Kaggle notebook, set
them from an attached secret:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["KAGGLE_USERNAME"] = UserSecretsClient().get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = UserSecretsClient().get_secret("KAGGLE_KEY")
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def write_metadata(directory: Path, slug: str) -> None:
    title = slug.split("/")[-1].replace("-", " ").title()
    (directory / "dataset-metadata.json").write_text(
        json.dumps({"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2),
        encoding="utf-8",
    )


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def push(directory: Path, slug: str, message: str) -> int:
    write_metadata(directory, slug)

    version = run(["kaggle", "datasets", "version", "-p", str(directory), "-m", message, "-r", "zip"])
    if version.returncode == 0:
        print(version.stdout.strip() or f"pushed new version of {slug}")
        return 0

    combined = f"{version.stdout}\n{version.stderr}"
    if "404" not in combined and "not found" not in combined.lower():
        print(combined.strip(), file=sys.stderr)
        return version.returncode

    print(f"{slug} does not exist yet; creating it")
    created = run(["kaggle", "datasets", "create", "-p", str(directory), "-r", "zip"])
    print((created.stdout or created.stderr).strip())
    return created.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="snapshot directory to publish")
    parser.add_argument("--slug", required=True, help="owner/dataset-name")
    parser.add_argument("--message", default="probe snapshot", help="version note")
    opts = parser.parse_args()

    if not all(os.environ.get(key) for key in ("KAGGLE_USERNAME", "KAGGLE_KEY")):
        print("KAGGLE_USERNAME and KAGGLE_KEY are not set; skipping push", file=sys.stderr)
        return 0
    if not opts.dir.is_dir() or not any(opts.dir.iterdir()):
        print(f"{opts.dir} is empty; nothing to push", file=sys.stderr)
        return 0

    return push(opts.dir, opts.slug, opts.message)


if __name__ == "__main__":
    sys.exit(main())
