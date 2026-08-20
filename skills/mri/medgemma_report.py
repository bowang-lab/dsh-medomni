#!/usr/bin/env python3
"""
Meng Wei

MRI report-generation candidate with MedGemma 1.5 4B.

UNOFFICIAL / INFERRED PREPROCESSING: Google's own reference notebook for
MedGemma 1.5's per-slice-sequence 3D technique
(Google-Health/medgemma, notebooks/high_dimensional_ct_hugging_face.ipynb)
covers CT only — there is no equivalent published MRI notebook. MRI has no
fixed absolute intensity scale like CT's Hounsfield units, so a Hounsfield-
style fixed window cannot transfer. This script instead normalizes each
slice by the 0.5-99.5 percentile intensity range of its own nonzero voxels,
by analogy with the same percentile scheme BiomedParse's own vendored MRI
reader uses (skills/biomedparse/BiomedParse/inference_utils/processing_utils.py).
Treat this as an unvalidated candidate, more so than ct/medgemma_report.py.

Shared per-slice-sequence plumbing lives in skills/medgemma_3d/_core.py; this
file only supplies the MRI-specific normalization and prompt text.

Usage:
  python medgemma_report.py --input mri_volume.nii.gz
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("medgemma-mri")

import argparse
import json
import tempfile
import time

from medgemma_3d import _core

_INTRO = (
    "You are reviewing a contiguous sequence of MRI slices, sampled "
    "uniformly through the volume and shown in order. Each slice is "
    "normalized by its own intensity range, not a fixed physical scale."
)


def percentile_slice_rgb(slice_2d):
    """One MRI slice -> 3-channel uint8 RGB via 0.5-99.5 percentile
    intensity normalization over nonzero voxels, broadcast across all three
    channels (grayscale, unlike CT's three distinct HU windows — MRI has no
    equivalent multi-window convention). Matches BiomedParse's own MRI
    normalization (see module docstring)."""
    import numpy as np

    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size == 0:
        low, high = 0.0, 1.0
    else:
        low, high = np.percentile(nonzero, 0.5), np.percentile(nonzero, 99.5)
        if high <= low:
            high = low + 1.0
    clipped = np.clip(slice_2d, low, high).astype("float32")
    normalized = (clipped - low) / (high - low) * 255.0
    return np.round(np.stack([normalized] * 3, axis=-1)).astype("uint8")


def build_query(indication):
    query = (
        "Based on the slices above, generate the findings section of an MRI "
        "radiology report. Be concise, organize by organ system, and avoid "
        "findings that are not visible in the provided slices."
    )
    if indication:
        query += f" Clinical indication: {indication}"
    return query


def main() -> None:
    parser = argparse.ArgumentParser(description="MedGemma 1.5 MRI per-slice-sequence report candidate")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--study_id", default=None)
    parser.add_argument("--indication", default=None)
    parser.add_argument("--model", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--n_slices", type=int, default=32,
                        help="Slices uniformly sampled through the volume and sent to the model, one per SLICE block (default 32 — more slices costs more inference time per call, proportional to n_slices additional images through the vision encoder).")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--montage_output", default=None,
                        help="Path to save the contact-sheet preview PNG (optional, default: a temp file). Display only — not what the model sees; the model sees each slice as a separate full-resolution image.")
    args = parser.parse_args()

    t0 = time.time()
    try:
        nifti_path = _core.resolve_nifti(args.input)
        rgb_slices, indices = _core.sample_slices(nifti_path, args.n_slices, percentile_slice_rgb)
        contact_sheet = _core.make_contact_sheet(rgb_slices, indices)
        montage_path = (
            Path(args.montage_output) if args.montage_output
            else Path(tempfile.mkdtemp(prefix="medgemma_mri_montage_")) / "preview.png"
        )
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet.save(str(montage_path))
        pipe = _core.load_pipeline("medgemma-mri", args.gpu, args.model)
        report_text = _core.run_report(pipe, rgb_slices, _INTRO, build_query(args.indication), args.max_new_tokens)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        sys.exit(1)

    result = {
        "status": "success",
        "study_id": args.study_id,
        "model": args.model,
        "mode": "mri_per_slice_sequence_report",
        "image_path": args.input,
        "resolved_nifti_path": nifti_path,
        "n_slices_sent": len(rgb_slices),
        "slice_indices": indices,
        "report_text": report_text,
        "preview_image_path": str(montage_path),
        "elapsed_seconds": round(time.time() - t0, 2),
        "research_only": True,
        "preprocessing_note": "MRI per-slice normalization (0.5-99.5 percentile) is an inferred, unofficial analogue of CT's published HU-windowing technique — no equivalent MRI reference exists.",
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
