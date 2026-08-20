#!/usr/bin/env python3
"""
Author: Meng Wei

Chest X-Ray Report Generation — google/medgemma-4b-it

MedGemma 4B (instruction-tuned) used for radiology findings generation from a
single frontal chest X-ray.  The MedGemma technical report (arXiv:2507.05201)
evaluated this checkpoint on MIMIC-CXR with input = image + indication →
output = findings section.  After fine-tuning, 81 % of generated reports were
rated as clinically equivalent by a US board-certified radiologist.

No official fine-tuned report-generation checkpoint is released on HuggingFace;
this script prompts the base instruction-tuned model using the task format
described in the technical report (Appendix A6 / A7).

Usage:
  python medgemma_report.py --input xray.png

  python medgemma_report.py --input xray.png \\
      --indication "Shortness of breath." --gpu 1

  python medgemma_report.py --input xray.png --output result.json
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("medgemma-report")

import argparse
import json
import re
import tempfile
import time


# ── Prompt ────────────────────────────────────────────────────────────────────
# Based on MedGemma technical report (arXiv:2507.05201, Appendix A6/A7):
#   Input:  chest X-ray image  [+ clinical indication if available]
#   Output: findings section of the radiology report
#
# The system prompt "You are an expert radiologist." is recommended in the
# official Google Health documentation and community notebooks.

_SYSTEM_PROMPT = "You are an expert radiologist."

def build_prompt(indication: str = None) -> str:
    return (
        f"Generate the findings section of a radiology report for this "
        f"chest X-ray. Clinical indication: {indication}"
        if indication else
        "Generate the findings section of a radiology report for this chest X-ray."
    )


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(image_path: str):
    """Load chest X-ray (PNG, JPG, DICOM) → RGB PIL Image."""
    from PIL import Image
    import numpy as np

    path = Path(image_path)
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
            dcm = pydicom.dcmread(str(path))
            arr = dcm.pixel_array.astype(float)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            img = Image.fromarray(arr.astype("uint8"))
        except ImportError:
            raise RuntimeError("pydicom required for DICOM: pip install pydicom")
    else:
        img = Image.open(str(path))

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline(gpu: int, model_id: str = "google/medgemma-4b-it"):
    """Load MedGemma image-text-to-text pipeline."""
    import torch
    from transformers import pipeline

    print(f"[medgemma-report] Loading {model_id} ...", file=sys.stderr, flush=True)

    # A bare "cuda:N" string isn't a real device_map (HF expects "auto", a
    # dict, or an int/device) -- from_pretrained doesn't register it as
    # already-placed, so Pipeline.__init__ still does its own self.model.to(
    # self.device) afterward, and self.device defaults to cuda:0 since no
    # explicit `device=` was passed. On a shared GPU box where cuda:0 is
    # someone else's full model, that redundant .to(cuda:0) OOMs even though
    # the model already loaded fine on the GPU we actually asked for. The
    # {"": "cuda:N"} dict form is the real single-device device_map and
    # short-circuits that second .to() call.
    device_map = {"": f"cuda:{gpu}"} if gpu >= 0 else {"": "cpu"}
    print(f"[medgemma-report] device_map={device_map}", file=sys.stderr, flush=True)

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    print(f"[medgemma-report] Pipeline loaded.", file=sys.stderr, flush=True)
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────

def run_report(pipe, image, indication: str = None, max_new_tokens: int = 512) -> str:
    """
    Generate findings text for one frontal X-ray.

    Returns plain text (stripped of any thinking trace).
    """
    prompt = build_prompt(indication)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        },
    ]

    output  = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    content = output[0]["generated_text"][-1]["content"]

    if isinstance(content, list):
        response = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    else:
        response = str(content)

    # Strip thinking trace emitted between <unused94> … <unused95> (MedGemma 1.5)
    if "<unused95>" in response:
        response = response.split("<unused95>", 1)[1].lstrip()

    return response.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MedGemma chest X-ray report generation"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Frontal chest X-ray (PNG, JPG, or DICOM .dcm)",
    )
    parser.add_argument("--indication", default=None,
                        help="Clinical indication e.g. 'Shortness of breath.'")
    parser.add_argument("--model", default="google/medgemma-4b-it",
                        help="HuggingFace model ID (default: google/medgemma-4b-it)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu", "-g", type=int, default=0,
                        help="GPU index (-1 for CPU)")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to write result JSON (optional)")
    parser.add_argument("--preview_output", default=None,
                        help="Path to save the preview PNG (optional, default: a temp file)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(json.dumps({"status": "error", "error": f"Input not found: {args.input}"}))
        sys.exit(1)

    print(f"[medgemma-report] frontal={args.input}", file=sys.stderr)

    try:
        image = load_image(args.input)
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Image loading failed: {e}"}))
        sys.exit(1)

    t0 = time.time()
    try:
        pipe = load_pipeline(args.gpu, args.model)
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Model loading failed: {e}"}))
        sys.exit(1)

    print("[medgemma-report] Running inference ...", file=sys.stderr)
    try:
        report_text = run_report(
            pipe, image,
            indication=args.indication,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Inference failed: {e}"}))
        sys.exit(1)

    elapsed = round(time.time() - t0, 2)

    result = {
        "status":      "success",
        "image_path":  str(args.input),
        "model":       args.model,
        "indication":  args.indication,
        "report_text": report_text,
        "elapsed_s":   elapsed,
    }

    preview_path = (
        Path(args.preview_output) if args.preview_output
        else Path(tempfile.mkdtemp(prefix="medgemma_report_preview_")) / "preview.png"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(preview_path))
    result["preview_image_path"] = str(preview_path)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[medgemma-report] Results saved to {args.output}", file=sys.stderr)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Report: {report_text[:300]}", file=sys.stderr)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
