#!/usr/bin/env python3
"""
Author: Meng Wei

Chest X-Ray Grounded Report Generation — MAIRA-2

MAIRA-2 is a multimodal transformer (RAD-DINO image encoder + Vicuna-7B LLM)
for generating radiology findings from chest X-rays, with optional spatial grounding.

Three use-cases (controlled by --mode):

  report          Generate findings section as plain narrative text
                  → fast, no bounding boxes

  grounded_report Generate findings with bounding boxes per sentence
                  → each finding gets (x1,y1,x2,y2) coords on the frontal image

  phrase_grounding Given a phrase (e.g. "Pleural effusion"), locate it on the image
                  → returns the phrase + bounding box

Bounding box format: (x_topleft, y_topleft, x_bottomright, y_bottomright)
Coordinates are relative to the CROPPED image MAIRA-2 sees internally.
Use processor.adjust_box_for_original_image_size() to map back to original pixels.

Inputs:
  --input         Frontal chest X-ray (required) — PNG, JPG, DICOM .dcm
  --lateral       Lateral view from same study (optional but recommended)
  --prior         Prior frontal X-ray (optional)
  --prior_report  Prior radiology report text (optional, used with --prior)
  --indication    Clinical indication / reason for exam (optional)
  --technique     Technique / protocol description (optional)
  --comparison    Comparison string e.g. "Compared to 01/01/2024" (optional)
  --phrase        Phrase to ground — required for phrase_grounding mode


Usage:
  # Plain report
  python maira2_report.py --input xray.png --mode report

  # Grounded report (findings + bounding boxes)
  python maira2_report.py --input xray.png --lateral lat.png --mode grounded_report

  # Phrase grounding
  python maira2_report.py --input xray.png --mode phrase_grounding \
      --phrase "Pleural effusion"

  # Full context
  python maira2_report.py --input xray.png --lateral lat.png \
      --indication "Dyspnea." --technique "PA and lateral." --comparison "None." \
      --mode grounded_report --gpu 1
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("maira-2")

import argparse
import json
import tempfile
import traceback


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(image_path: str):
    """Load any chest X-ray image (PNG, JPG, DICOM) → PIL Image in RGB."""
    from PIL import Image

    path = Path(image_path)

    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
            import numpy as np
            dcm = pydicom.dcmread(str(path))
            arr = dcm.pixel_array.astype(float)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            arr = arr.astype("uint8")
            img = Image.fromarray(arr)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except ImportError:
            raise RuntimeError("pydicom required for DICOM: pip install pydicom")
    else:
        img = Image.open(str(path))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(gpu: int):
    """
    Load MAIRA-2 model + processor from HuggingFace.
    Requires HF_TOKEN env var or prior `huggingface-cli login` (gated model).
    Model is ~14GB on disk. First run downloads automatically.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    print("[maira-2] Loading model microsoft/maira-2 ...", file=sys.stderr)
    print("[maira-2] Note: first run downloads ~14GB weights", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/maira-2",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "microsoft/maira-2",
        trust_remote_code=True,
    )

    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    model = model.eval().to(device)
    print(f"[maira-2] Model loaded on {device}", file=sys.stderr)
    return model, processor, device


# ── Inference ─────────────────────────────────────────────────────────────────

def run_report(model, processor, device, frontal, lateral, prior, prior_report,
               indication, technique, comparison, get_grounding: bool):
    """Use-case 1 & 2: findings generation with or without grounding."""
    import torch

    processed = processor.format_and_preprocess_reporting_input(
        current_frontal = frontal,
        current_lateral = lateral,
        prior_frontal   = prior,
        indication      = indication,
        technique       = technique,
        comparison      = comparison,
        prior_report    = prior_report,
        return_tensors  = "pt",
        get_grounding   = get_grounding,
    )
    processed = processed.to(device)

    # Grounded output includes extra markup around phrases and boxes. A short
    # limit can truncate a delimiter and make MAIRA's processor assert while
    # parsing the grounded sequence.
    max_new_tokens = 900 if get_grounding else 300

    with torch.no_grad():
        output = model.generate(
            **processed,
            max_new_tokens = max_new_tokens,
            use_cache      = True,
        )

    prompt_length = processed["input_ids"].shape[-1]
    decoded = processor.decode(output[0][prompt_length:], skip_special_tokens=True)
    decoded = decoded.lstrip()  # findings completions have a single leading space
    try:
        prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
    except AssertionError:
        print(
            "[maira-2] Grounding parse failed; returning ungrounded decoded text.",
            file=sys.stderr,
            flush=True,
        )
        prediction = (
            decoded
            + "\n\n(MAIRA grounding parse warning: generated grounding markup was incomplete, "
            + "so no boxes were parsed.)"
        )
    return prediction


def run_phrase_grounding(model, processor, device, frontal, phrase: str):
    """Use-case 3: locate a specific phrase on the frontal image."""
    import torch

    processed = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image  = frontal,
        phrase         = phrase,
        return_tensors = "pt",
    )
    processed = processed.to(device)

    with torch.no_grad():
        output = model.generate(
            **processed,
            max_new_tokens = 150,
            use_cache      = True,
        )

    prompt_length = processed["input_ids"].shape[-1]
    decoded = processor.decode(output[0][prompt_length:], skip_special_tokens=True)
    try:
        prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
    except AssertionError:
        print(
            "[maira-2] Phrase grounding parse failed; returning decoded text.",
            file=sys.stderr,
            flush=True,
        )
        prediction = (
            decoded
            + "\n\n(MAIRA phrase-grounding parse warning: generated grounding markup was incomplete, "
            + "so no box was parsed.)"
        )
    return prediction


# ── Preview image ──────────────────────────────────────────────────────────────

def draw_report_boxes(image, findings: list[dict]):
    from PIL import ImageDraw, ImageFont

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    palette = [
        (255,  80,  80), ( 80, 200,  80), ( 80, 130, 255), (255, 200,  50),
        (200,  80, 255), ( 50, 220, 220), (255, 140,   0), (180, 255,  80),
        (255,  80, 200), ( 80, 255, 180), (160, 120, 255), (255, 255,  80),
        ( 80, 180, 255), (255, 120, 120), (120, 255, 120), (200, 200,  50),
        ( 50, 200, 200), (200,  50, 200), (255, 180,  80), (100, 200, 255),
    ]

    idx = 0
    box_idx = 0
    for finding in findings:
        boxes = finding.get("boxes_original")
        if not boxes:
            continue
        for box in boxes:
            color = palette[box_idx % len(palette)]
            x1, y1, x2, y2 = box_to_pixels(box, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            inset = 8
            ix1 = min(max(x1 + inset, 0), width - 1)
            iy1 = min(max(y1 + inset, 0), height - 1)
            ix2 = max(min(x2 - inset, width - 1), ix1 + 1)
            iy2 = max(min(y2 - inset, height - 1), iy1 + 1)
            draw.rectangle([ix1, iy1, ix2, iy2], fill=color + (55,))
            draw.rectangle([ix1, iy1, ix2, iy2], outline=color + (255,), width=5)
            handle = 14
            for hx, hy in ((ix1, iy1), (ix2, iy1), (ix1, iy2), (ix2, iy2)):
                draw.rectangle(
                    [hx - handle // 2, hy - handle // 2, hx + handle // 2, hy + handle // 2],
                    fill=color + (255,),
                )
            cx = (ix1 + ix2) // 2
            cy = (iy1 + iy2) // 2
            draw.line([cx - 10, cy, cx + 10, cy], fill=color + (255,), width=3)
            draw.line([cx, cy - 10, cx, cy + 10], fill=color + (255,), width=3)
            label = str(idx + 1)
            label_bbox = draw.textbbox((ix1, iy1), label, font=font)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]
            label_y = max(iy1 - label_h - 10, 0)
            draw.rectangle([ix1, label_y, ix1 + label_w + 12, label_y + label_h + 10], fill=color + (220,))
            draw.text((ix1 + 6, label_y + 4), label, fill=(255, 255, 255, 255), font=font)
            box_idx += 1
        idx += 1

    return img


def box_to_pixels(box, width: int, height: int):
    """Convert MAIRA boxes to pixel coords.

    MAIRA processor versions have returned adjusted boxes as either absolute
    pixels or normalized floats in [0, 1]. Normalize both forms here.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    x1, x2 = sorted((max(0, min(width - 1, int(round(x1)))), max(0, min(width - 1, int(round(x2))))))
    y1, y2 = sorted((max(0, min(height - 1, int(round(y1)))), max(0, min(height - 1, int(round(y2))))))
    return x1, y1, x2, y2


def draw_box_map(image, findings: list[dict]):
    from PIL import Image, ImageDraw, ImageFont

    width, height = image.size
    img = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    palette = [
        (255,  80,  80), ( 80, 200,  80), ( 80, 130, 255), (255, 200,  50),
        (200,  80, 255), ( 50, 220, 220), (255, 140,   0), (180, 255,  80),
        (255,  80, 200), ( 80, 255, 180), (160, 120, 255), (255, 255,  80),
        ( 80, 180, 255), (255, 120, 120), (120, 255, 120), (200, 200,  50),
        ( 50, 200, 200), (200,  50, 200), (255, 180,  80), (100, 200, 255),
    ]

    idx = 0
    box_idx = 0
    for finding in findings:
        boxes = finding.get("boxes_original")
        if not boxes:
            continue
        for box in boxes:
            color = palette[box_idx % len(palette)]
            x1, y1, x2, y2 = box_to_pixels(box, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([x1, y1, x2, y2], fill=color + (60,), outline=color + (255,), width=6)
            draw.line([x1, y1, x2, y2], fill=color + (255,), width=3)
            draw.line([x1, y2, x2, y1], fill=color + (255,), width=3)
            label = str(idx + 1)
            label_bbox = draw.textbbox((x1, y1), label, font=font)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]
            draw.rectangle([x1, y1, x1 + label_w + 16, y1 + label_h + 14], fill=color + (220,))
            draw.text((x1 + 8, y1 + 6), label, fill=(255, 255, 255, 255), font=font)
            box_idx += 1
        idx += 1

    return img


# ── Output serialisation ──────────────────────────────────────────────────────

def serialise_prediction(prediction, frontal_size: tuple, processor) -> dict:
    """
    Convert MAIRA-2 prediction to a JSON-serialisable dict.
 
    Plain text (non-grounded):   prediction is a str
    Grounded output:             prediction is list of (sentence, boxes | None)
                                 boxes are (x1,y1,x2,y2) relative to cropped image
    frontal_size: PIL .size → (W, H)
    """
    if isinstance(prediction, str):
        return {
            "type":     "report",
            "text":     prediction,
            "findings": None,
        }
 
    # PIL .size is (W, H) — adjust_box_for_original_image_size takes (box, width, height)
    width, height = frontal_size
 
    findings = []
    for sentence, boxes in prediction:
        entry = {"text": sentence, "boxes_cropped": None, "boxes_original": None}
        if boxes:
            entry["boxes_cropped"]  = [list(b) for b in boxes]
            # Map from cropped-image coords → original image coords
            adjusted = [
                processor.adjust_box_for_original_image_size(b, width, height)
                for b in boxes
            ]
            entry["boxes_original"] = [list(b) for b in adjusted]
        findings.append(entry)
 
    plain_text = " ".join(f["text"] for f in findings)
 
    return {
        "type":     "grounded",
        "text":     plain_text,
        "findings": findings,   # list of {text, boxes_cropped, boxes_original}
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAIRA-2 grounded radiology report generation for chest X-rays"
    )

    # Images
    parser.add_argument("--input", "-i", required=True,
                        help="Frontal chest X-ray (PNG, JPG, or DICOM .dcm)")
    parser.add_argument("--lateral",     default=None,
                        help="Lateral view from same study (optional)")
    parser.add_argument("--prior",       default=None,
                        help="Prior frontal X-ray (optional)")
    parser.add_argument("--prior_report",default=None,
                        help="Prior radiology report text (optional)")

    # Clinical context
    parser.add_argument("--indication",  default=None,
                        help="Clinical indication e.g. 'Dyspnea.'")
    parser.add_argument("--technique",   default=None,
                        help="Technique e.g. 'PA and lateral views.'")
    parser.add_argument("--comparison",  default=None,
                        help="Comparison e.g. 'None.' or 'Compared to 01/01/2024.'")

    # Mode
    parser.add_argument("--mode", default="grounded_report",
                        choices=["report", "grounded_report", "phrase_grounding"],
                        help=(
                            "report            → plain narrative findings text  "
                            "grounded_report   → findings with bounding boxes (default)  "
                            "phrase_grounding  → locate a phrase on the image"
                        ))
    parser.add_argument("--phrase", default=None,
                        help="Phrase to ground (required for phrase_grounding mode)") #default=None

    # Runtime
    parser.add_argument("--gpu",    "-g", type=int, default=2,
                        help="GPU index (-1 for CPU)")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to write result JSON (optional)")
    parser.add_argument("--preview_output", default=None,
                        help="Path to save the preview PNG (optional, default: a temp file)")
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if not Path(args.input).exists():
        print(json.dumps({"status": "error", "error": f"Input not found: {args.input}"}))
        sys.exit(1)

    if args.mode == "phrase_grounding" and not args.phrase:
        print(json.dumps({"status": "error",
                          "error": "--phrase is required for phrase_grounding mode"}))
        sys.exit(1)

    print(f"[maira-2] mode={args.mode}  frontal={args.input}", file=sys.stderr)

    # ── Load images ───────────────────────────────────────────────────────────
    try:
        frontal = load_image(args.input)
        lateral = load_image(args.lateral) if args.lateral else None
        prior   = load_image(args.prior)   if args.prior   else None
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Image loading failed: {e}"}))
        sys.exit(1)

    frontal_size = frontal.size  # (W, H) — needed for box coord adjustment

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, processor, device = load_model(args.gpu)
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Model loading failed: {e}"}))
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    print("[maira-2] Running inference ...", file=sys.stderr)
    try:
        if args.mode == "phrase_grounding":
            prediction = run_phrase_grounding(
                model, processor, device, frontal, args.phrase
            )
        else:
            get_grounding = (args.mode == "grounded_report")
            prediction = run_report(
                model, processor, device,
                frontal      = frontal,
                lateral      = lateral,
                prior        = prior,
                prior_report = args.prior_report,
                indication   = args.indication,
                technique    = args.technique,
                comparison   = args.comparison,
                get_grounding= get_grounding,
            )
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": f"Inference failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)

    # ── Serialise ─────────────────────────────────────────────────────────────
    try:
        pred_dict = serialise_prediction(prediction, frontal_size, processor)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": f"Serialisation failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)

    result = {
        "status":     "success",
        "image_path": str(args.input),
        "mode":       args.mode,
        "phrase":     args.phrase,
        "prediction": pred_dict,
        # Convenience top-level fields
        "report_text":    pred_dict["text"],
        "findings":       pred_dict["findings"],   # None for plain report
        "n_findings":     len(pred_dict["findings"]) if pred_dict["findings"] else 0,
        "n_with_boxes":   len([f for f in pred_dict["findings"] if f["boxes_original"]])
                          if pred_dict["findings"] else 0,
    }

    # ── Preview image ─────────────────────────────────────────────────────────
    preview_image = frontal
    if result["n_with_boxes"] > 0:
        preview_image = draw_report_boxes(frontal, pred_dict["findings"])
    preview_path = (
        Path(args.preview_output) if args.preview_output
        else Path(tempfile.mkdtemp(prefix="maira2_preview_")) / "preview.png"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_image.save(str(preview_path))
    result["preview_image_path"] = str(preview_path)
    if result["n_with_boxes"] > 0:
        box_map_path = preview_path.with_name("boxes_only.png")
        draw_box_map(frontal, pred_dict["findings"]).save(str(box_map_path))
        result["box_map_image_path"] = str(box_map_path)
        result["preview_image_paths"] = [str(preview_path), str(box_map_path)]

    # ── Optional: save to file ────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[maira-2] Results saved to {args.output}", file=sys.stderr)

    # ── Summary to stderr ─────────────────────────────────────────────────────
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Mode:  {args.mode}", file=sys.stderr)
    if args.mode == "report":
        print(f"Text:  {pred_dict['text'][:300]}", file=sys.stderr)
    elif args.mode == "grounded_report":
        print(f"Findings: {result['n_findings']} sentences | "
              f"{result['n_with_boxes']} with bounding boxes", file=sys.stderr)
        for f in pred_dict["findings"]:
            box_str = str(f["boxes_original"]) if f["boxes_original"] else "—"
            print(f"  {box_str}  {f['text']}", file=sys.stderr)
    elif args.mode == "phrase_grounding":
        print(f"Phrase: {args.phrase}", file=sys.stderr)
        if pred_dict["findings"]:
            print(f"Box:    {pred_dict['findings'][0]['boxes_original']}", file=sys.stderr)

    # ── stdout → agent reads this ─────────────────────────────────────────────
    print(json.dumps(result))


if __name__ == "__main__":
    main()
