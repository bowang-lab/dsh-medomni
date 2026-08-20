#!/usr/bin/env python3
"""
Meng Wei

CT report-generation candidate with MedGemma 1.5 4B.

MedGemma 1.5 4B interprets a 3D CT volume as a sequence of per-slice image
blocks, each windowed into a 3-channel RGB encoding: R = wide window (-1024
to 1024 HU), G = soft-tissue window (-135 to 215 HU), B = brain window (0 to
80 HU). This is the technique documented in Google's own reference notebook
(Google-Health/medgemma, notebooks/high_dimensional_ct_hugging_face.ipynb).
Shared per-slice-sequence plumbing lives in skills/medgemma_3d/_core.py; this
file only supplies the CT-specific windowing and prompt text.

Treat this as a complementary candidate for a selector, not as a replacement
for a native 3D CT model.

Usage:
  python medgemma_report.py --input ct_volume.nii.gz
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("medgemma-ct")

import argparse
import json
import tempfile
import time

from medgemma_3d import _core

# Per Google's reference notebook: wide, soft-tissue, and brain HU windows,
# stacked as the R/G/B channels of one image per slice.
_WINDOW_CLIPS = [(-1024, 1024), (-135, 215), (0, 80)]

_INTRO = (
    "You are reviewing a contiguous sequence of axial CT slices, sampled "
    "uniformly through the volume and shown in order from superior to "
    "inferior. Each slice is a three-channel image combining a wide window, "
    "a soft-tissue window, and a brain window."
)


def _norm(slice_2d, low, high):
    import numpy as np

    clipped = np.clip(slice_2d, low, high).astype("float32")
    return (clipped - low) / (high - low) * 255.0


def window_slice_rgb(slice_2d):
    """One CT slice -> 3-channel uint8 RGB, per _WINDOW_CLIPS."""
    import numpy as np

    channels = [_norm(slice_2d, lo, hi) for lo, hi in _WINDOW_CLIPS]
    return np.round(np.stack(channels, axis=-1)).astype("uint8")


def build_query(indication):
    query = (
        "Based on the slices above, generate the findings section of a CT "
        "radiology report. Be concise, organize by organ system, and avoid "
        "findings that are not visible in the provided slices."
    )
    if indication:
        query += f" Clinical indication: {indication}"
    return query


def main() -> None:
    parser = argparse.ArgumentParser(description="MedGemma 1.5 CT per-slice-sequence report candidate")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--study_id", default=None)
    parser.add_argument("--indication", default=None)
    parser.add_argument("--model", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--n_slices", type=int, default=32,
                        help="Slices uniformly sampled through the volume and sent to the model, one per SLICE block (default 32; Google's own notebook demo uses up to 85 — more slices costs more inference time per call, proportional to n_slices additional images through the vision encoder).")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--montage_output", default=None,
                        help="Path to save the contact-sheet preview PNG (optional, default: a temp file). Display only — not what the model sees; the model sees each slice as a separate full-resolution image.")
    args = parser.parse_args()

    t0 = time.time()
    try:
        nifti_path = _core.resolve_nifti(args.input)
        rgb_slices, indices = _core.sample_slices(nifti_path, args.n_slices, window_slice_rgb)
        contact_sheet = _core.make_contact_sheet(rgb_slices, indices)
        montage_path = (
            Path(args.montage_output) if args.montage_output
            else Path(tempfile.mkdtemp(prefix="medgemma_ct_montage_")) / "preview.png"
        )
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet.save(str(montage_path))
        pipe = _core.load_pipeline("medgemma-ct", args.gpu, args.model)
        report_text = _core.run_report(pipe, rgb_slices, _INTRO, build_query(args.indication), args.max_new_tokens)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        sys.exit(1)

    result = {
        "status": "success",
        "study_id": args.study_id,
        "model": args.model,
        "mode": "ct_per_slice_sequence_report",
        "image_path": args.input,
        "resolved_nifti_path": nifti_path,
        "n_slices_sent": len(rgb_slices),
        "slice_indices": indices,
        "report_text": report_text,
        "preview_image_path": str(montage_path),
        "elapsed_seconds": round(time.time() - t0, 2),
        "research_only": True,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
