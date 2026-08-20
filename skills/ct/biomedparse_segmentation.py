#!/usr/bin/env python3
"""
BiomedParse (microsoft/BiomedParse) text-prompted segmentation for CT volumes.
Applies CT-specific windowing per --site. Single-slice by default; --all_slices
processes every slice and reconstructs a 3D binary NIfTI mask per prompt.

Usage:
  python biomedparse_segmentation.py --input ct.nii.gz --prompts liver kidney --site abdomen --slice_idx 68
  python biomedparse_segmentation.py --input ct.nii.gz --prompts liver --site abdomen --all_slices
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("biomedparse-ct")

import argparse
import json

from biomedparse import _core

_core.ensure_ready("biomedparse-ct")


def main():
    parser = argparse.ArgumentParser(description="BiomedParse segmentation for CT volumes")
    parser.add_argument("--input", required=True, help="NIfTI volume (.nii or .nii.gz) — DICOM is not accepted")
    parser.add_argument("--prompts", required=True, nargs="+", help="One or more structures/findings, e.g. --prompts liver kidney")
    parser.add_argument("--site", required=True, choices=["abdomen", "lung", "pelvis", "liver", "colon", "pancreas"],
                         help="Anatomical site, for CT windowing")
    parser.add_argument("--slice_idx", type=int, default=None, help="Slice index (defaults to the middle slice)")
    parser.add_argument("--all_slices", action="store_true", default=False,
                         help="Process every slice and reconstruct a 3D NIfTI mask per prompt (slow — one model call per slice per prompt)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    try:
        result = _core.run_3d(args.input, args.prompts, True, args.site, args.slice_idx, args.all_slices,
                               None, args.output_dir, args.gpu, args.threshold,
                               tag="biomedparse-ct", modality="ct")
    except Exception as e:
        result = {"status": "error", "error": str(e)}
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
