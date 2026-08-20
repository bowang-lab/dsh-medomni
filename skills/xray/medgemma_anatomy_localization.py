#!/usr/bin/env python3
"""
Author: Meng Wei

CXR Anatomy Localization — google/medgemma-1.5-4b-it

MedGemma 1.5 (4B instruction-tuned) fine-tuned on the Chest ImaGenome
bounding-box dataset.  Given a chest X-ray, returns [y0, x0, y1, x1]
bounding boxes (coordinates normalized to [0, 1000]) for each requested
anatomical structure.

Default anatomy set (Chest ImaGenome):
  right lung, left lung, cardiac silhouette, trachea,
  right clavicle, left clavicle, right hemidiaphragm, left hemidiaphragm,
  right costophrenic angle, left costophrenic angle, aortic arch,
  mediastinum, right hilar structures, left hilar structures,
  right upper lung zone, right mid lung zone, right lower lung zone,
  left upper lung zone, left mid lung zone, left lower lung zone

Usage:
  python medgemma_anatomy_localization.py --input chest.png
  python medgemma_anatomy_localization.py --input chest.png --anatomy "right lung" "left lung"
  python medgemma_anatomy_localization.py --input chest.dcm --output ./out/result.json --gpu 1
  python medgemma_anatomy_localization.py --input chest.png --draw annotated.png
"""

import subprocess
import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap
_bootstrap.ensure_venv_and_reexec("medgemma-loc")

import argparse
import json
import re
import tempfile
import time
import traceback


# ── Default anatomy set (Chest ImaGenome labels) ──────────────────────────────

DEFAULT_ANATOMIES = [
    "right lung",
    "left lung",
    "cardiac silhouette",
    "trachea",
    "right clavicle",
    "left clavicle",
    "right hemidiaphragm",
    "left hemidiaphragm",
    "right costophrenic angle",
    "left costophrenic angle",
    "aortic arch",
    "mediastinum",
    "right hilar structures",
    "left hilar structures",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
]


# ── GPU auto-selection ────────────────────────────────────────────────────────

def get_free_gpu(exclude: list = None) -> int:
    """Return index of GPU with most free memory, skipping excluded indices."""
    import subprocess
    import torch

    if not torch.cuda.is_available():
        return -1

    exclude = exclude or []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        free_mem = []
        for line in out.strip().splitlines():
            idx, free = line.split(", ")
            idx = int(idx)
            if idx not in exclude:
                free_mem.append((int(free), idx))
        return max(free_mem)[1] if free_mem else -1
    except Exception:
        import torch
        for i in range(torch.cuda.device_count()):
            if i not in exclude:
                return i
        return -1


# ── Image loading & preprocessing ────────────────────────────────────────────

def load_image(image_path: str):
    """Load any chest X-ray (PNG, JPG, DICOM) → square-padded PIL Image RGB.

    MedGemma was trained on square-padded images; padding before inference
    avoids aspect-ratio distortion that degrades localization accuracy.
    """
    import numpy as np
    import skimage
    import skimage.color
    import skimage.util
    from PIL import Image

    path = Path(image_path)
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
            dcm = pydicom.dcmread(str(path))
            arr = dcm.pixel_array.astype(float)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            img = Image.fromarray((arr * 255).astype("uint8"))
        except ImportError:
            raise RuntimeError("pydicom required for DICOM: pip install pydicom")
    else:
        img = Image.open(str(path))

    arr = np.array(img)
    arr = skimage.util.img_as_ubyte(arr) if arr.dtype != "uint8" else arr

    # Ensure 3-channel
    if arr.ndim == 2:
        arr = skimage.color.gray2rgb(arr)
    elif arr.shape[2] == 4:
        arr = skimage.color.rgba2rgb(arr)
        arr = (arr * 255).astype("uint8")

    # Pad to square (model was trained this way)
    h, w = arr.shape[:2]
    if h < w:
        dh = w - h
        arr = np.pad(arr, ((dh // 2, dh - dh // 2), (0, 0), (0, 0)))
    elif w < h:
        dw = h - w
        arr = np.pad(arr, ((0, 0), (dw // 2, dw - dw // 2), (0, 0)))

    return Image.fromarray(arr.astype("uint8"))


# ── Prompt construction ───────────────────────────────────────────────────────

_BBOX_INSTRUCTIONS = (
    "The following user query will require outputting bounding boxes. "
    "The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be "
    "the top-left corner and (y1, x1) the bottom-right corner. "
    "Always normalize the x and y coordinates to the range [0, 1000]. "
    "meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. "
    "You MUST output a single parseable json list of objects enclosed into json... brackets, "
    'for instance json[{"box_2d": [800, 3, 840, 471], "label": "car"}, '
    '{"box_2d": [400, 22, 600, 73], "label": "dog"}] is a valid output. '
    "Now answer to the user query.\n\n "
)


def build_prompt(anatomies: list[str]) -> str:
    anatomy_list = ", ".join(f'"{a}"' for a in anatomies)
    return (
        _BBOX_INSTRUCTIONS
        + f"Localize the following anatomical structures in this chest X-ray: "
        + f"{anatomy_list}. "
        + 'Don\'t give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:'
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline(gpu: int, attn_implementation: str = None):
    """Load MedGemma 1.5-4b-it image-text pipeline."""
    import torch
    from transformers import pipeline

    model_id = "google/medgemma-1.5-4b-it"
    print(f"[medgemma-loc] Loading {model_id} ...", file=sys.stderr, flush=True)

    if gpu < 0:
        gpu = get_free_gpu(exclude=[0])    # keep GPU 0 for LLM

    # Use the dict form to avoid Pipeline doing an extra .to(cuda:0). This
    # matches the other MedGemma tools in this plugin.
    device_map = {"": f"cuda:{gpu}"} if gpu >= 0 else {"": "cpu"}
    print(f"[medgemma-loc] device_map={device_map}", file=sys.stderr, flush=True)
    model_kwargs = {}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        model_kwargs=model_kwargs,
    )
    print(f"[medgemma-loc] Pipeline loaded on {device_map}", file=sys.stderr, flush=True)
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(pipe, image, anatomies: list[str], max_new_tokens: int = 2048) -> dict:
    """Run MedGemma anatomy localization; return parsed bounding boxes."""
    prompt = build_prompt(anatomies)
    print(f"[medgemma-loc] Prompt (first 200 chars): {prompt[:200]}", file=sys.stderr, flush=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    print(f"[medgemma-loc] Running inference (max_new_tokens={max_new_tokens}) ...", file=sys.stderr, flush=True)
    output  = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    print(f"[medgemma-loc] Inference done. Output type: {type(output)}, len={len(output)}", file=sys.stderr, flush=True)

    content = output[0]["generated_text"][-1]["content"]
    print(f"[medgemma-loc] content type: {type(content)}", file=sys.stderr, flush=True)

    # content may be a plain string or a list of typed content blocks
    if isinstance(content, list):
        print(f"[medgemma-loc] content is list, len={len(content)}", file=sys.stderr, flush=True)
        response = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    else:
        response = content

    print(f"[medgemma-loc] response length: {len(response)}", file=sys.stderr, flush=True)
    print(f"[medgemma-loc] response (first 500 chars):\n{response[:500]}", file=sys.stderr, flush=True)

    # Strip thinking trace emitted between <unused94> … <unused95>
    if "<unused95>" in response:
        print(f"[medgemma-loc] Stripping thinking trace (<unused95> found)", file=sys.stderr, flush=True)
        response = response.split("<unused95>", 1)[1].lstrip()
        print(f"[medgemma-loc] After strip (first 300 chars):\n{response[:300]}", file=sys.stderr, flush=True)

    boxes = _parse_boxes(response)
    print(f"[medgemma-loc] Parsed {len(boxes)} box(es)", file=sys.stderr, flush=True)
    return {"raw_response": response, "boxes": boxes}


def _extract_array_blob(text: str, start: int) -> str:
    """From a `[` at *start*, return the array up to as far as it was completed.

    max_new_tokens can cut the model's response off mid-array. Bracket-depth
    tracking alone then never finds a closing `]`, and without this recovery
    the whole blob would be discarded even though most elements are intact.
    Instead, keep every complete top-level `{...}` element seen before the
    cut and close the array manually — only the one dangling partial element
    is lost.
    """
    depth = 0
    end = start + 1
    closed = False
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                closed = True
                break
        elif ch == "}" and depth == 1:
            end = i + 1  # end of one complete array element
    blob = text[start:end]
    if not closed:
        blob = blob.rstrip()
        if blob.endswith(","):
            blob = blob[:-1]
        blob += "]"
    return blob


def _parse_boxes(text: str) -> list[dict]:
    """Extract bounding-box JSON from model output and parse it."""
    blob = None

    # 1. Fenced JSON block (```json ... ```). Tolerate a missing closing
    #    fence — truncated output cuts it off before ``` ever appears.
    m = re.search(r"```json\s*", text)
    if m:
        start = text.find("[", m.end())
        if start != -1:
            blob = _extract_array_blob(text, start)
            print(f"[medgemma-loc][parse] fenced-block match, blob length={len(blob)}", file=sys.stderr, flush=True)
            print(f"[medgemma-loc][parse] blob[:150]: {blob[:150]}", file=sys.stderr, flush=True)

    if blob is None:
        # 2. json[...] anywhere in the text (model's native output format).
        mj = re.search(r"json\s*\[", text)
        if mj:
            print(f"[medgemma-loc][parse] json[ match at pos {mj.start()}", file=sys.stderr, flush=True)
            blob = _extract_array_blob(text, mj.end() - 1)  # mj.end() - 1 = position of the opening [
        else:
            # 3. "Final Answer: [...]" — model followed the explicit prompt format.
            print(f"[medgemma-loc][parse] no fenced block or json[; trying Final Answer", file=sys.stderr, flush=True)
            fa = re.search(r"Final Answer\s*:\s*", text, re.IGNORECASE)
            search_text = text[fa.end():] if fa else text

            start = search_text.find("[")
            if start == -1:
                print(f"[medgemma-loc][parse] no [ ] found — returning empty", file=sys.stderr, flush=True)
                return []
            blob = _extract_array_blob(search_text, start)

    if blob is None:
        print(f"[medgemma-loc][parse] blob is None — returning empty", file=sys.stderr, flush=True)
        return []

    # Strip trailing commas (LLMs commonly emit them; json.loads rejects them)
    blob_clean = re.sub(r",\s*([}\]])", r"\1", blob)

    try:
        data = json.loads(blob_clean)
        print(f"[medgemma-loc][parse] JSON parsed OK — {len(data)} items", file=sys.stderr, flush=True)
    except json.JSONDecodeError as e:
        print(f"[medgemma-loc][parse] JSONDecodeError: {e}", file=sys.stderr, flush=True)
        print(f"[medgemma-loc][parse] blob_clean[:300]: {blob_clean[:300]}", file=sys.stderr, flush=True)
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        box = item.get("box_2d") or item.get("bbox") or item.get("bounding_box")
        label = item.get("label") or item.get("anatomy") or ""
        if box and len(box) == 4:
            y0, x0, y1, x1 = [float(v) for v in box]
            results.append({
                "label":  label,
                "box_2d": [y0, x0, y1, x1],   # [y0, x0, y1, x1] in [0, 1000]
            })
    return results


# ── Visualization ────────────────────────────────────────────────────────────

# Distinct colours for up to 20 anatomy labels (RGB tuples)
_PALETTE = [
    (255,  80,  80), ( 80, 200,  80), ( 80, 130, 255), (255, 200,  50),
    (200,  80, 255), ( 50, 220, 220), (255, 140,   0), (180, 255,  80),
    (255,  80, 200), ( 80, 255, 180), (160, 120, 255), (255, 255,  80),
    ( 80, 180, 255), (255, 120, 120), (120, 255, 120), (200, 200,  50),
    ( 50, 200, 200), (200,  50, 200), (255, 180,  80), (100, 200, 255),
]


def draw_bounding_boxes(image, boxes: list[dict], font_size: int = 18):
    """Overlay anatomy bounding boxes on *image* and return an annotated PIL Image.

    Coordinates in *boxes* are [y0, x0, y1, x1] normalised to [0, 1000]
    (the format MedGemma returns).  This function scales them to pixel space
    before drawing.

    Args:
        image:     Square-padded PIL Image (output of load_image()).
        boxes:     List of {"label": str, "box_2d": [y0, x0, y1, x1]} dicts.
        font_size: Approximate label font size in pixels.

    Returns:
        PIL Image with coloured rectangles and labels drawn on it.
    """
    from PIL import ImageDraw, ImageFont

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # Try a truetype font, fall back to PIL default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    label_color_map: dict[str, tuple] = {}

    for box_entry in boxes:
        label = box_entry.get("label", "")
        box   = box_entry.get("box_2d", [])
        if len(box) != 4:
            continue

        y0_n, x0_n, y1_n, x1_n = box

        # Scale from [0, 1000] → pixel coords
        x0 = int(x0_n / 1000 * W)
        y0 = int(y0_n / 1000 * H)
        x1 = int(x1_n / 1000 * W)
        y1 = int(y1_n / 1000 * H)

        # Assign a stable colour per label
        if label not in label_color_map:
            idx = len(label_color_map) % len(_PALETTE)
            label_color_map[label] = _PALETTE[idx]
        color = label_color_map[label]

        # Semi-transparent fill + solid border
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=2,
                        fill=color + (30,))

        # Label background + text
        text_bbox = draw.textbbox((x0, y0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        ty = max(y0 - th - 4, 0)
        draw.rectangle([x0, ty, x0 + tw + 4, ty + th + 4], fill=color + (200,))
        draw.text((x0 + 2, ty + 2), label, fill=(255, 255, 255, 255), font=font)

    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CXR anatomy localization using MedGemma 1.5-4b-it"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to chest X-ray (PNG, JPG, or DICOM .dcm)")
    parser.add_argument("--anatomy", "-a", nargs="+", default=None,
                        metavar="ANATOMY",
                        help="Anatomy structures to localize (default: full Chest ImaGenome set)")
    parser.add_argument("--gpu",    "-g", type=int, default=-1,
                        help="GPU index (-1 = auto-select free GPU)")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to write result JSON (optional)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max new tokens for generation")
    parser.add_argument("--draw", "-d", default=None, metavar="PATH",
                        help="Save annotated image with bounding boxes to this path (PNG); default: a temp file")
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if not Path(args.input).exists():
        print(json.dumps({"status": "error", "error": f"Input not found: {args.input}"}))
        sys.exit(1)

    anatomies = args.anatomy or DEFAULT_ANATOMIES
    print(f"[medgemma-loc] Input:    {args.input}", file=sys.stderr, flush=True)
    print(f"[medgemma-loc] Anatomies ({len(anatomies)}): {anatomies}", file=sys.stderr, flush=True)
    print(f"[medgemma-loc] GPU arg: {args.gpu}", file=sys.stderr, flush=True)

    # ── Load image ────────────────────────────────────────────────────────────
    print(f"[medgemma-loc] Loading image ...", file=sys.stderr, flush=True)
    try:
        image = load_image(args.input)
        print(f"[medgemma-loc] Image loaded: size={image.size} mode={image.mode}", file=sys.stderr, flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Image loading failed: {e}"}))
        sys.exit(1)

    # ── Load model + run inference ────────────────────────────────────────────
    t0 = time.time()
    try:
        try:
            pipe = load_pipeline(args.gpu)
            pred = run_inference(pipe, image, anatomies, max_new_tokens=args.max_tokens)
        except RuntimeError as e:
            if "attn_bias_ptr is not correctly aligned" not in str(e):
                raise
            print(
                "[medgemma-loc] Attention kernel alignment failed; retrying with eager attention.",
                file=sys.stderr,
                flush=True,
            )
            del pipe
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe = load_pipeline(args.gpu, attn_implementation="eager")
            pred = run_inference(pipe, image, anatomies, max_new_tokens=args.max_tokens)
    except Exception as e:
        print(f"[medgemma-loc] Inference exception: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({
            "status": "error",
            "error": f"Inference failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)
    elapsed = round(time.time() - t0, 2)

    # ── Build output ──────────────────────────────────────────────────────────
    result = {
        "status":       "success",
        "image_path":   str(args.input),
        "model":        "medgemma-1.5-4b-it",
        "anatomies":    anatomies,
        "boxes":        pred["boxes"],       # list of {"label", "box_2d": [y0,x0,y1,x1]}
        "elapsed_s":    elapsed,
        "raw_response": pred["raw_response"],
    }

    # ── Optional: save JSON ───────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[medgemma-loc] Saved JSON to {args.output}", file=sys.stderr)

    # ── Optional: draw annotated image ────────────────────────────────────────
    draw_path = (
        Path(args.draw) if args.draw
        else Path(tempfile.mkdtemp(prefix="medgemma_loc_preview_")) / "preview.png"
    )
    draw_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_bounding_boxes(image, pred["boxes"])
    annotated.save(str(draw_path))
    print(f"[medgemma-loc] Saved annotated image to {draw_path}", file=sys.stderr)
    result["preview_image_path"] = str(draw_path)

    # ── Summary to stderr ─────────────────────────────────────────────────────
    print(f"\n{'='*55}", file=sys.stderr)
    print(f"Detected {len(pred['boxes'])} structure(s)  ({elapsed}s)", file=sys.stderr)
    for b in pred["boxes"]:
        coords = b["box_2d"]
        print(f"  {b['label']:<35s}  {coords}", file=sys.stderr)

    # stdout → agent reads this
    print(json.dumps(result))


if __name__ == "__main__":
    main()
