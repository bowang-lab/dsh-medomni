#!/usr/bin/env python3
"""
Author: Meng Wei

Retinal (Fundus) Report Generation — google/medgemma-4b-it

MedGemma 4B's image encoder is pre-trained on ophthalmology (fundus) images
alongside chest X-ray, dermatology, and histopathology (per the model card:
https://developers.google.com/health-ai-developer-foundations/medgemma/model-card).
No official fine-tuned report-generation checkpoint is released on HuggingFace
for fundus images; this prompts the base instruction-tuned model directly,
the same approach skills/xray/medgemma_report.py uses for chest X-ray.

Usage:
  python medgemma_report.py --input fundus.png

  python medgemma_report.py --input fundus.png \\
      --indication "Diabetic retinopathy screening." --gpu 1
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("medgemma-retinal")

import argparse
import json
import tempfile
import time


_SYSTEM_PROMPT = "You are an expert ophthalmologist."

def build_prompt(indication: str = None) -> str:
    return (
        f"Generate the findings section of an ophthalmology report for this "
        f"retinal (fundus) image. Clinical indication: {indication}"
        if indication else
        "Generate the findings section of an ophthalmology report for this retinal (fundus) image."
    )


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(image_path: str):
    """Load a retinal fundus photograph (PNG, JPG) → RGB PIL Image."""
    from PIL import Image

    img = Image.open(str(Path(image_path)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline(gpu: int, model_id: str = "google/medgemma-4b-it"):
    import torch
    from transformers import pipeline

    print(f"[medgemma-retinal] Loading {model_id} ...", file=sys.stderr, flush=True)

    # See skills/xray/medgemma_report.py for why device_map must be the
    # {"": "cuda:N"} dict form rather than a bare "cuda:N" string.
    device_map = {"": f"cuda:{gpu}"} if gpu >= 0 else {"": "cpu"}
    print(f"[medgemma-retinal] device_map={device_map}", file=sys.stderr, flush=True)

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    print(f"[medgemma-retinal] Pipeline loaded.", file=sys.stderr, flush=True)
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────

def run_report(pipe, image, indication: str = None, max_new_tokens: int = 512) -> str:
    prompt = build_prompt(indication)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
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
    parser = argparse.ArgumentParser(description="MedGemma retinal (fundus) report generation")
    parser.add_argument("--input", "-i", required=True, help="Retinal fundus photograph (PNG or JPG)")
    parser.add_argument("--indication", default=None,
                        help="Clinical indication e.g. 'Diabetic retinopathy screening.'")
    parser.add_argument("--model", default="google/medgemma-4b-it",
                        help="HuggingFace model ID (default: google/medgemma-4b-it)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU index (-1 for CPU)")
    parser.add_argument("--output", "-o", default=None, help="Path to write result JSON (optional)")
    parser.add_argument("--preview_output", default=None,
                        help="Path to save the preview PNG (optional, default: a temp file)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(json.dumps({"status": "error", "error": f"Input not found: {args.input}"}))
        sys.exit(1)

    print(f"[medgemma-retinal] fundus={args.input}", file=sys.stderr)

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

    print("[medgemma-retinal] Running inference ...", file=sys.stderr)
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
        else Path(tempfile.mkdtemp(prefix="medgemma_retinal_preview_")) / "preview.png"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(preview_path))
    result["preview_image_path"] = str(preview_path)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[medgemma-retinal] Results saved to {args.output}", file=sys.stderr)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Report: {report_text[:300]}", file=sys.stderr)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
