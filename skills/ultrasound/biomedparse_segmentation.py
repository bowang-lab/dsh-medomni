#!/usr/bin/env python3
"""
BiomedParse (microsoft/BiomedParse) text-prompted segmentation for ultrasound.

Usage:
  python biomedparse_segmentation.py --input scan.png --prompts gallstone "gallbladder wall"
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("biomedparse-ultrasound")

import argparse
import json

from biomedparse import _core

_core.ensure_ready("biomedparse-ultrasound")


def main():
    parser = argparse.ArgumentParser(description="BiomedParse segmentation for ultrasound")
    parser.add_argument("--input", required=True, help="Path to the ultrasound image")
    parser.add_argument("--prompts", required=True, nargs="+", help="One or more structures/findings, e.g. --prompts gallstone \"gallbladder wall\"")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    try:
        result = _core.run_2d(args.input, args.prompts, args.output_dir, args.gpu,
                               tag="biomedparse-ultrasound", modality="ultrasound")
    except Exception as e:
        result = {"status": "error", "error": str(e)}
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
