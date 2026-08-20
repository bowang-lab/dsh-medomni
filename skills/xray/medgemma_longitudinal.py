#!/usr/bin/env python3
"""
Author: Meng Wei

Chest X-Ray Longitudinal Comparison — google/medgemma-1.5-4b-it

MedGemma 1.5 4B's documented "Longitudinal & Temporal Radiology Analysis"
capability: evaluates a PAIR of chest X-rays (prior vs. current study) to
describe interval change. Benchmarked on MS-CXR-T (65.7% macro-accuracy)
across Improved/Stable/Worsened categories for Consolidation, Edema,
Pleural Effusion, Pneumonia, and Pneumothorax.

Message/prompt pattern taken directly from Google's own official notebook
(github.com/Google-Health/medgemma/blob/main/notebooks/
cxr_longitudinal_comparison_with_hugging_face.ipynb) -- two images in ONE
user-turn content list ([prior_image, current_image, text]), not two
separate tool calls. Google's documentation gives no exact structured
output format for this task, so the prompt here steers toward the
documented categories/pathologies without claiming a validated
structured-output format -- output is freeform comparison text, matching
what the official notebook actually demonstrates, not an invented rigid
parse.

Differences from the single-image report script (medgemma_report.py):
  - Requires TWO images: --input (current) and --prior (earlier study)
  - Model is medgemma-1.5-4b-it specifically (not the plain 4b-it) --
    this capability is documented for 1.5, and 1.5 is already the model
    this skill's own anatomy-localization script uses successfully.
  - Uses the device_map={"": f"cuda:{gpu}"} dict form (not a bare
    "cuda:N" string) -- see medgemma_report.py's own comment for the
    confirmed OOM bug this avoids on a shared GPU box.

Usage:
  python medgemma_longitudinal.py --input current.png --prior prior.png

  python medgemma_longitudinal.py --input current.png --prior prior.png \\
      --indication "Follow-up for pneumonia." --gpu 1
"""

import sys
import tempfile
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("medgemma-longitudinal")

import argparse
import json
import time

_SYSTEM_PROMPT = "You are a helpful radiology assistant."

_COMPARISON_PATHOLOGIES = (
    "consolidation, pulmonary edema, pleural effusion, pneumonia, and pneumothorax"
)


def side_by_side(prior, current, gap: int = 8):
    from PIL import Image, ImageDraw, ImageFont

    h = max(prior.height, current.height)
    w = prior.width + current.width + gap
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(prior, (0, 0))
    canvas.paste(current, (prior.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, 90, 24], fill=(0, 0, 0))
    draw.text((4, 3), "PRIOR", fill=(255, 255, 255), font=font)
    draw.rectangle([prior.width + gap, 0, prior.width + gap + 110, 24], fill=(0, 0, 0))
    draw.text((prior.width + gap + 4, 3), "CURRENT", fill=(255, 255, 255), font=font)

    return canvas


def build_prompt(indication: str = None) -> str:
    base = (
        "You will be shown two chest X-rays in order: the FIRST image is the "
        "PRIOR (earlier) study, and the SECOND image is the CURRENT (most "
        "recent) study. Compare the current study to the prior study for "
        "interval change. "
        f"For each of the following findings that is present in either image -- "
        f"{_COMPARISON_PATHOLOGIES} -- state explicitly whether it has improved, "
        "remained stable, or worsened since the prior study. Also note any new "
        "finding not present on the prior study, and any finding present on the "
        "prior study that has since resolved."
    )
    if indication:
        base += f" Clinical indication: {indication}"
    return base


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(image_path: str):
    """Load chest X-ray (PNG, JPG, DICOM) → RGB PIL Image."""
    from PIL import Image

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

def load_pipeline(gpu: int, model_id: str = "google/medgemma-1.5-4b-it"):
    """Load MedGemma 1.5 image-text-to-text pipeline."""
    import torch
    from transformers import pipeline

    print(f"[medgemma-longitudinal] Loading {model_id} ...", file=sys.stderr, flush=True)

    # {"": "cuda:N"} dict form, not a bare "cuda:N" string -- see
    # medgemma_report.py's own comment for the confirmed OOM bug this avoids
    # (Pipeline.__init__ redundantly re-placing the model onto cuda:0 when
    # device_map isn't a real HF-recognized device_map value).
    device_map = {"": f"cuda:{gpu}"} if gpu >= 0 else {"": "cpu"}
    print(f"[medgemma-longitudinal] device_map={device_map}", file=sys.stderr, flush=True)

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    # Matches the official notebook: do_sample is a generation_config
    # attribute here, NOT a valid pipe(...) call kwarg for this processor
    # (passing it to the call is silently ignored with a warning).
    pipe.model.generation_config.do_sample = False
    print(f"[medgemma-longitudinal] Pipeline loaded.", file=sys.stderr, flush=True)
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────

def run_comparison(pipe, prior_image, current_image, indication: str = None,
                   max_new_tokens: int = 600) -> str:
    """
    Compare a prior and current chest X-ray for interval change.

    Message structure matches Google's official notebook exactly: ONE
    user turn, content = [prior_image, current_image, text] -- NOT two
    separate tool calls or turns.

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
                {"type": "image", "image": prior_image},
                {"type": "image", "image": current_image},
                {"type": "text",  "text": prompt},
            ],
        },
    ]

    output  = pipe(text=messages, max_new_tokens=max_new_tokens)
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
        description="MedGemma 1.5 chest X-ray longitudinal (prior vs. current) comparison"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Current (most recent) frontal chest X-ray (PNG, JPG, or DICOM .dcm)",
    )
    parser.add_argument(
        "--prior", "-p", required=True,
        help="Prior (earlier) frontal chest X-ray for comparison (PNG, JPG, or DICOM .dcm)",
    )
    parser.add_argument("--indication", default=None,
                        help="Clinical indication e.g. 'Follow-up for pneumonia.'")
    parser.add_argument("--model", default="google/medgemma-1.5-4b-it",
                        help="HuggingFace model ID (default: google/medgemma-1.5-4b-it)")
    parser.add_argument("--max_new_tokens", type=int, default=600)
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
    if not Path(args.prior).exists():
        print(json.dumps({"status": "error", "error": f"Prior image not found: {args.prior}"}))
        sys.exit(1)

    print(f"[medgemma-longitudinal] current={args.input}  prior={args.prior}", file=sys.stderr)

    try:
        current_image = load_image(args.input)
        prior_image = load_image(args.prior)
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Image loading failed: {e}"}))
        sys.exit(1)

    t0 = time.time()
    try:
        pipe = load_pipeline(args.gpu, args.model)
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Model loading failed: {e}"}))
        sys.exit(1)

    print("[medgemma-longitudinal] Running inference ...", file=sys.stderr)
    try:
        comparison_text = run_comparison(
            pipe, prior_image, current_image,
            indication=args.indication,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Inference failed: {e}"}))
        sys.exit(1)

    elapsed = round(time.time() - t0, 2)

    result = {
        "status":           "success",
        "image_path":       str(args.input),
        "prior_image_path": str(args.prior),
        "model":             args.model,
        "mode":              "longitudinal_comparison",
        "indication":        args.indication,
        "comparison_text":   comparison_text,
        "elapsed_s":         elapsed,
    }

    preview_path = (
        Path(args.preview_output) if args.preview_output
        else Path(tempfile.mkdtemp(prefix="medgemma_longitudinal_preview_")) / "preview.png"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    side_by_side(prior_image, current_image).save(str(preview_path))
    result["preview_image_path"] = str(preview_path)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[medgemma-longitudinal] Results saved to {args.output}", file=sys.stderr)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Comparison: {comparison_text[:300]}", file=sys.stderr)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
