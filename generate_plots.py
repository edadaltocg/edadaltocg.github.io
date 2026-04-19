#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# ///
"""Find and run all plots.py scripts under content/blog/."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for script in sorted((ROOT / "content" / "blog").rglob("plots.py")):
    print(f"Running {script.relative_to(ROOT)} ...")
    r = subprocess.run(["uv", "run", script.name], cwd=script.parent)
    if r.returncode != 0:
        sys.exit(r.returncode)
