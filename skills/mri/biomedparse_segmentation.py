#!/usr/bin/env python3
"""
BiomedParse (microsoft/BiomedParse) text-prompted segmentation for MRI volumes.
Percentile normalization is applied automatically (no --site needed). Single-slice
by default; --all_slices processes every slice and reconstructs a 3D binary NIfTI
mask per prompt.

Usage:
  python biomedparse_segmentation.py --input brain.nii.gz --prompts "tumor core" --slice_idx 89 --channel_idx 2
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("biomedparse-mri")

import argparse
import json

from biomedparse import _core

_core.ensure_ready("biomedparse-mri")


def main():
    parser = argparse.ArgumentParser(description="BiomedParse segmentation for MRI volumes")
    parser.add_argument("--input", required=True, help="NIfTI volume (.nii or .nii.gz) — DICOM is not accepted")
    parser.add_argument("--prompts", required=True, nargs="+", help="One or more structures/findings, e.g. --prompts \"tumor core\" \"enhancing tumor\"")
    parser.add_argument("--slice_idx", type=int, default=None, help="Slice index (defaults to the middle slice)")
    parser.add_argument("--all_slices", action="store_true", default=False,
                         help="Process every slice and reconstruct a 3D NIfTI mask per prompt (slow — one model call per slice per prompt)")
    parser.add_argument("--channel_idx", type=int, default=None, help="Channel index for multi-channel MRI (e.g. BRATS-style: 0-3)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    try:
        result = _core.run_3d(args.input, args.prompts, False, None, args.slice_idx, args.all_slices,
                               args.channel_idx, args.output_dir, args.gpu, args.threshold,
                               tag="biomedparse-mri", modality="mri")
    except Exception as e:
        result = {"status": "error", "error": str(e)}
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
