#!/usr/bin/env python3
"""
Meng Wei on 15th Aug 2026

TotalSegmentator organ segmentation for CT.

Usage:
  python totalseg_segmentation.py --input ct.nii.gz --task total
  python totalseg_segmentation.py --input ct_dicom_dir --task lung_vessels --fast
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("totalseg-ct")
_VENV_DIR = _bootstrap.VENV_DIR

VALID_TASKS = ["total", "lung_vessels"]


def build_command(args) -> list[str]:
    cmd = [
        str(_VENV_DIR / "bin" / "TotalSegmentator"),
        "-i", args.input,
        "-o", args.output,
        "--task", args.task,
    ]
    if args.fast:
        cmd.append("--fast")
    if args.ml:
        cmd.append("--ml")
    if args.statistics:
        cmd.append("--statistics")
    if args.preview:
        cmd.append("--preview")
    if args.roi_subset:
        cmd += ["--roi_subset"] + args.roi_subset
    if args.output_type:
        cmd += ["--output_type", args.output_type]
    return cmd


def collect_outputs(output_dir: Path, task: str) -> dict:
    seg_files = sorted(
        str(f) for f in output_dir.glob("*.nii.gz") if f.name != "multilabel.nii.gz"
    )
    structures_found = [Path(f).stem.replace(".nii", "") for f in seg_files]

    result = {
        "status": "success",
        "task": task,
        "modality": "CT",
        "output_dir": str(output_dir),
        "segmentation_files": seg_files,
        "structures_found": structures_found,
        "n_structures": len(structures_found),
        "multilabel_file": None,
        "statistics_file": None,
        "preview_image_path": None,
    }

    multilabel = output_dir / "multilabel.nii.gz"
    if multilabel.exists():
        result["multilabel_file"] = str(multilabel)

    stats = output_dir / "statistics.json"
    if stats.exists():
        result["statistics_file"] = str(stats)
        result["statistics"] = json.loads(stats.read_text())

    preview = output_dir / "preview.png"
    if preview.exists():
        result["preview_image_path"] = str(preview)

    return result


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator CT organ segmentation")
    parser.add_argument("--input", "-i", required=True,
                        help="CT NIfTI file or DICOM folder/zip")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: a fresh temp directory)")
    parser.add_argument("--task", "-ta", default="total", choices=VALID_TASKS)
    parser.add_argument("--fast", action="store_true",
                        help="Use fast 3mm resolution model")
    parser.add_argument("--ml", action="store_true",
                        help="Save multilabel NIfTI (all classes in one file)")
    parser.add_argument("--statistics", action="store_true",
                        help="Compute volume (mm3) and mean intensity per structure")
    parser.add_argument("--preview", action="store_true",
                        help="Generate PNG preview of segmentation")
    parser.add_argument("--roi_subset", nargs="+",
                        help="Only segment specific structures, e.g. liver kidney_right")
    parser.add_argument("--output_type", default="nifti", choices=["nifti", "dicom"])
    parser.add_argument("--gpu", default=None, help="GPU index, e.g. 0 (default: auto)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(json.dumps({"status": "error", "error": f"Input not found: {args.input}"}))
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="totalseg_ct_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output = str(output_dir)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = build_command(args)
    print(f"[totalseg-ct] Running: {' '.join(cmd)}", file=sys.stderr)

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            print(json.dumps({
                "status": "error",
                "error": proc.stderr or "TotalSegmentator failed with non-zero exit",
            }))
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(json.dumps({"status": "error", "error": "TotalSegmentator timed out after 3600 seconds"}))
        sys.exit(1)
    except FileNotFoundError:
        print(json.dumps({"status": "error", "error": "TotalSegmentator executable not found in the venv"}))
        sys.exit(1)

    result = collect_outputs(output_dir, args.task)

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result))


if __name__ == "__main__":
    main()
