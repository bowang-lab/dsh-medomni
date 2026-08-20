"""
Shared venv bootstrap for every skill script. Import this from a script's own
directory (`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`)
before importing anything third-party — this module itself is stdlib-only so
it can run under the ambient interpreter, before the venv exists.

ensure_venv_and_reexec() re-execs the calling script into skills/.venv on
first call, installing BASE_PACKAGES (the deps every script needs) pinned
against the ambient torch build via a constraints file. ensure_extra_packages()
installs additional deps into that same venv, once, gated by a marker file —
for a dependency group only some scripts need (e.g. BiomedParse's detectron2),
so scripts that don't need it never pay its install cost.
"""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
VENV_DIR = SKILLS_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
CONSTRAINTS_FILE = SKILLS_DIR / ".uv-constraints.txt"

BASE_PACKAGES = [
    "transformers>=4.50.0,<4.52", "accelerate", "pillow", "protobuf",
    "sentencepiece", "numpy", "scikit-image", "nibabel", "simpleitk",
    "TotalSegmentator",
]


def ensure_venv_and_reexec(tag):
    if sys.executable == str(VENV_PYTHON):
        return

    if not VENV_PYTHON.exists():
        print(f"[{tag}] Creating venv ...", file=sys.stderr)
        subprocess.check_call([
            "uv", "venv", "--system-site-packages",
            "--python", sys.executable, str(VENV_DIR),
        ])

        try:
            torch_pin = f"torch=={importlib.metadata.version('torch')}"
        except importlib.metadata.PackageNotFoundError:
            torch_pin = None
        CONSTRAINTS_FILE.write_text(f"{torch_pin}\n" if torch_pin else "")

        subprocess.check_call([
            "uv", "pip", "install", "--python", str(VENV_PYTHON),
            "--constraint", str(CONSTRAINTS_FILE), *BASE_PACKAGES,
        ])
        print(f"[{tag}] Venv ready.", file=sys.stderr)

    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)


def ensure_extra_packages(tag, marker_name, packages, extra_install_args=None):
    """Install `packages` into the already-active venv, once, tracked by a
    marker file — call only after ensure_venv_and_reexec() has re-exec'd."""
    marker = VENV_DIR / f".installed-{marker_name}"
    if marker.exists():
        return
    print(f"[{tag}] Installing {marker_name} dependencies ...", file=sys.stderr)
    cmd = ["uv", "pip", "install", "--python", sys.executable,
           "--constraint", str(CONSTRAINTS_FILE)]
    if extra_install_args:
        cmd += extra_install_args
    cmd += packages
    subprocess.check_call(cmd)
    marker.write_text("ok\n")
    print(f"[{tag}] {marker_name} ready.", file=sys.stderr)
