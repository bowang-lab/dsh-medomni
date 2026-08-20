#!/usr/bin/env python3
"""
Author: Meng Wei

Ultrasound Zero-Shot Classification — microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

BiomedCLIP (Zhang et al., arXiv 2303.00915) is a CLIP-style vision-language
model pre-trained on 15M biomedical image-text pairs from PubMed Central
(~200M params, ViT-Base/16 + PubMedBERT). Given an image and a list of
candidate text labels, it scores each label by image-text cosine similarity
and returns a softmax distribution over exactly those labels — a forced
choice among the labels given, not open-ended diagnosis. A close spread
across all candidates (rather than one dominant probability) is the signal
that none of them fit well.

Four built-in label panels (--task anatomy/breast/thyroid/cardiac/general),
plus free-form zero-shot via --task cls --labels "label1,label2,...".

Usage:
  python biomedclip_classify.py --task anatomy --input us_image.png
  python biomedclip_classify.py --task breast --input breast_us.png
  python biomedclip_classify.py --task cls --input us.png \\
      --labels "normal kidney,kidney cyst,kidney stone,renal mass"
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("biomedclip-us")
_bootstrap.ensure_extra_packages("biomedclip-us", "biomedclip-deps", ["open_clip_torch==2.23.0"])

import argparse
import json
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_CONTEXT_LENGTH = 256
_TEMPLATE = "this is a photo of "

# ── Pre-defined label panels ────────────────────────────────────────────────

_PANELS = {
    "anatomy": [
        "breast ultrasound",
        "thyroid ultrasound",
        "cardiac echocardiography",
        "abdominal ultrasound",
        "fetal ultrasound",
        "musculoskeletal ultrasound",
        "vascular doppler ultrasound",
        "renal ultrasound",
        "pelvic ultrasound",
    ],
    "breast": [
        "benign breast lesion ultrasound",
        "malignant breast lesion ultrasound",
        "normal breast tissue ultrasound",
        "breast cyst ultrasound",
        "breast fibroadenoma ultrasound",
    ],
    "busi": [
        "normal breast ultrasound",
        "benign breast tumor ultrasound",
        "malignant breast tumor ultrasound",
    ],
    "thyroid": [
        "normal thyroid ultrasound",
        "benign thyroid nodule ultrasound",
        "malignant thyroid nodule ultrasound",
        "thyroid cyst ultrasound",
        "multinodular goiter ultrasound",
    ],
    "cardiac": [
        "normal cardiac echocardiography",
        "reduced ejection fraction echocardiography",
        "left ventricular hypertrophy echocardiography",
        "pericardial effusion echocardiography",
        "mitral valve disease echocardiography",
    ],
    "general": [
        "normal ultrasound",
        "lesion or mass in ultrasound",
        "cyst in ultrasound",
        "calcification in ultrasound",
        "fluid collection in ultrasound",
        "vascular abnormality in ultrasound",
    ],
}


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(gpu: int):
    from open_clip import create_model_from_pretrained, get_tokenizer

    print(f"[biomedclip-us] Loading {_MODEL_ID} ...", file=sys.stderr, flush=True)
    model, preprocess = create_model_from_pretrained(_MODEL_ID)
    tokenizer = get_tokenizer(_MODEL_ID)

    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    model = model.to(device).eval()
    print(f"[biomedclip-us] Model loaded on {device}.", file=sys.stderr, flush=True)
    return model, preprocess, tokenizer, device


# ── Inference ────────────────────────────────────────────────────────────────

def classify(image, labels: list[str], gpu: int) -> dict:
    """Score one PIL image against `labels`. Returns the softmax distribution
    plus the argmax prediction — always picks a winner from `labels`, never
    a category outside the given list."""
    import torch

    model, preprocess, tokenizer, device = load_model(gpu)

    image_tensor = preprocess(image).unsqueeze(0).to(device)
    texts = tokenizer(
        [_TEMPLATE + lbl for lbl in labels],
        context_length=_CONTEXT_LENGTH,
    ).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_tensor, texts)
        probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        probs = probs.cpu().float().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        "prediction":    labels[pred_idx],
        "confidence":    round(float(probs[pred_idx]), 4),
        "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BiomedCLIP zero-shot ultrasound classification")
    parser.add_argument("--task", "-t", required=True, choices=list(_PANELS.keys()) + ["cls"],
                        help="Built-in panels: anatomy | breast | busi | thyroid | cardiac | general. Free-form: cls (requires --labels)")
    parser.add_argument("--input", "-i", required=True, help="Ultrasound image (PNG or JPG)")
    parser.add_argument("--labels", default=None, help="Comma-separated label list for --task cls")
    parser.add_argument("--output", "-o", default=None, help="Path to write result JSON (optional)")
    parser.add_argument("--preview_output", default=None,
                        help="Path to save the preview PNG (optional, default: a temp file)")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU index (-1 for CPU)")
    args = parser.parse_args()

    if args.task == "cls":
        if not args.labels:
            print(json.dumps({"status": "error", "error": "--labels required for --task cls"}))
            sys.exit(1)
        labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    else:
        labels = _PANELS[args.task]

    if not Path(args.input).exists():
        print(json.dumps({"status": "error", "error": f"Input not found: {args.input}"}))
        sys.exit(1)

    try:
        image = Image.open(args.input)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Image loading failed: {e}"}))
        sys.exit(1)

    print(f"[biomedclip-us] task={args.task} labels={len(labels)}", file=sys.stderr)

    t0 = time.time()
    try:
        result = classify(image, labels, args.gpu)
    except Exception as e:
        import traceback
        print(json.dumps({"status": "error", "error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)

    output = {
        "status":     "success",
        "task":       args.task,
        "model":      "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "image_path": str(args.input),
        "labels":     labels,
        "elapsed_s":  round(time.time() - t0, 2),
        **result,
    }

    preview_path = (
        Path(args.preview_output) if args.preview_output
        else Path(tempfile.mkdtemp(prefix="biomedclip_us_preview_")) / "preview.png"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(preview_path))
    output["preview_image_path"] = str(preview_path)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[biomedclip-us] Saved to {args.output}", file=sys.stderr)

    print(json.dumps(output))


if __name__ == "__main__":
    main()
