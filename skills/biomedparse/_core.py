"""
Shared BiomedParse (microsoft/BiomedParse) implementation used by every
modality's thin biomedparse_segmentation.py wrapper (xray, ultrasound,
retinal for 2D images; ct, mri for 3D NIfTI volumes). One text-prompted
segmentation model, no modality-specific model code — the wrappers only
differ in which of run_2d/run_3d they call and which CLI flags they expose.

Top-level imports here must stay stdlib-only: this module is imported by a
wrapper both before and after the shared-venv re-exec (see _bootstrap.py),
and third-party packages (numpy, PIL, matplotlib, torch) aren't installed
yet on the first pass. Those imports are deferred into the functions that
need them.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import _bootstrap

_BIOMEDPARSE_DIR = Path(__file__).resolve().parent
REPO_DIR = _BIOMEDPARSE_DIR / "BiomedParse"
WEIGHTS_DIR = _BIOMEDPARSE_DIR / "weights"

HF_REPO = "microsoft/BiomedParse"
HF_FILENAME = "biomedparse_v1.pt"

# Packages BiomedParse needs beyond _bootstrap.BASE_PACKAGES (which already
# covers transformers/accelerate/pillow/numpy/scikit-image/nibabel/simpleitk).
EXTRA_PACKAGES = [
    "huggingface_hub", "matplotlib", "scipy", "einops", "timm", "omegaconf",
    "pycocotools", "opencv-python-headless", "fvcore", "hydra-core", "nltk",
    "kornia", "pydicom",
]

_COLORS = [
    (0.20, 0.60, 0.86),  # blue
    (0.86, 0.37, 0.22),  # orange-red
    (0.20, 0.73, 0.44),  # green
    (0.76, 0.35, 0.73),  # purple
    (0.93, 0.78, 0.20),  # yellow
]


def _ensure_repo(tag):
    # The 'inference' branch has modeling/BaseModel.py and
    # inference_utils/processing_utils.py (NIfTI slice extraction); the
    # main branch is a restructured v2 (hydra/npz) that lacks both.
    needs_clone = not REPO_DIR.exists()
    if not needs_clone:
        missing = (
            not (REPO_DIR / "modeling" / "BaseModel.py").exists()
            or not (REPO_DIR / "inference_utils" / "processing_utils.py").exists()
        )
        if missing:
            import shutil
            print(f"[{tag}] Re-cloning BiomedParse inference branch ...", file=sys.stderr)
            shutil.rmtree(REPO_DIR)
            needs_clone = True
    if needs_clone:
        print(f"[{tag}] Cloning BiomedParse inference branch ...", file=sys.stderr)
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--branch", "inference",
            "https://github.com/microsoft/BiomedParse.git", str(REPO_DIR),
        ])
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))


def ensure_ready(tag):
    """Clone the BiomedParse repo and install its deps into the shared venv,
    once. Call after _bootstrap.ensure_venv_and_reexec() has re-exec'd —
    scripts that never call this never pay BiomedParse's install cost."""
    _ensure_repo(tag)
    _bootstrap.ensure_extra_packages(tag, "biomedparse-deps", EXTRA_PACKAGES)
    _bootstrap.ensure_extra_packages(
        tag, "biomedparse-detectron2",
        ["git+https://github.com/facebookresearch/detectron2.git"],
        extra_install_args=["--no-build-isolation"],
    )


def ensure_weights(tag):
    """Download BiomedParse's weights from HuggingFace if not already present. No token required (ungated)."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = WEIGHTS_DIR / HF_FILENAME
    if weights_path.exists():
        return weights_path
    print(f"[{tag}] Downloading {HF_FILENAME} from {HF_REPO} ...", file=sys.stderr)
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        dl = Path(hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME,
                                   local_dir=str(WEIGHTS_DIR), token=token))
        if dl.resolve() != weights_path.resolve():
            dl.rename(weights_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {HF_FILENAME} from {HF_REPO}: {e}\n"
            f"Manual: huggingface-cli download {HF_REPO} {HF_FILENAME} --local-dir {WEIGHTS_DIR}"
        )
    print(f"[{tag}] Weights saved -> {weights_path}", file=sys.stderr)
    return weights_path


def load_model(weights_path, device, tag):
    """Load the BiomedParse model via the inference-branch repo API."""
    print(f"[{tag}] Loading model ...", file=sys.stderr)
    # mpi4py is needed only for distributed training; stub it for inference.
    if "mpi4py" not in sys.modules:
        _mpi = types.ModuleType("mpi4py")

        class _MPI:
            class COMM_WORLD:
                @staticmethod
                def bcast(v, root=0):
                    return v
        _mpi.MPI = _MPI
        sys.modules["mpi4py"] = _mpi

    from utilities.arguments import load_opt_from_config_files
    from utilities.distributed import init_distributed
    from utilities.constants import BIOMED_CLASSES
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    import torch

    # BiomedParse's own opt/config loading reads paths relative to the repo.
    orig_cwd = os.getcwd()
    os.chdir(str(REPO_DIR))
    try:
        config_path = REPO_DIR / "configs" / "biomedparse_inference.yaml"
        opt = load_opt_from_config_files([str(config_path)])
        opt = init_distributed(opt)
        model = BaseModel(opt, build_model(opt)).from_pretrained(str(weights_path)).eval()
        model = model.to(device)
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                BIOMED_CLASSES + ["background"], is_eval=True,
            )
    finally:
        os.chdir(orig_cwd)

    print(f"[{tag}] Model loaded.", file=sys.stderr)
    return model


def infer_2d(model, image, prompts):
    """Run inference on a single PIL image. Returns [{"prompt","mask","score"}, ...]."""
    orig_cwd = os.getcwd()
    os.chdir(str(REPO_DIR))
    try:
        from inference_utils.inference import interactive_infer_image
    finally:
        os.chdir(orig_cwd)

    import numpy as np
    results = []
    for prompt in prompts:
        pred_mask_prob, _texts = interactive_infer_image(model, image, [prompt])
        prob = pred_mask_prob[0] if pred_mask_prob.ndim == 3 else pred_mask_prob
        mask_np = (prob > 0.5).astype(np.uint8)
        results.append({"prompt": prompt, "mask": mask_np, "score": float(prob.max())})
    return results


def infer_slice(model, image_arr, prompts):
    """Run inference on one (H,W,3) uint8 NIfTI slice. Returns one (H,W) float mask per prompt."""
    orig_cwd = os.getcwd()
    os.chdir(str(REPO_DIR))
    try:
        from inference_utils.inference import interactive_infer_image
    finally:
        os.chdir(orig_cwd)

    from PIL import Image
    image_pil = Image.fromarray(image_arr)
    pred_mask, _pred_text = interactive_infer_image(model, image_pil, prompts)
    if pred_mask.ndim == 3:
        return [pred_mask[i] for i in range(pred_mask.shape[0])]
    return [pred_mask]


def load_nifti_slice(nifti_path, is_ct, slice_idx, site, channel_idx):
    """Extract one 2D slice from a NIfTI volume -> (H,W,3) uint8 numpy array."""
    from inference_utils.processing_utils import read_nifti
    return read_nifti(nifti_path, is_ct, slice_idx, site=site,
                       HW_index=(0, 1), channel_idx=channel_idx)


def get_n_slices(nifti_path):
    import nibabel as nib
    shape = nib.load(nifti_path).get_fdata().shape
    depth_axis = 2  # after moveaxis to (H,W,depth[,channel])
    return shape[depth_axis] if len(shape) > depth_axis else 1


def save_mask_overlay(image, mask, prompt, out_path, color_idx=0):
    """Save a two-panel (input | segmentation) PNG for a single 2D image."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    from skimage.transform import resize

    img_arr = np.array(image.convert("RGB")).astype(float) / 255.0
    h, w = img_arr.shape[:2]
    if mask.shape != (h, w):
        mask = resize(mask.astype(float), (h, w), order=0) > 0.5

    color = np.array(_COLORS[color_idx % len(_COLORS)])
    overlay = img_arr.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.45 + color * 0.55
    contour = mask.astype(bool) & ~binary_erosion(mask.astype(bool))
    overlay[contour] = color

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    axes[0].imshow(img_arr)
    axes[0].set_title("Input", fontsize=9, color="#222")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Segmentation: {prompt}", fontsize=9, color="#222")
    axes[1].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mask_png(mask, out_path):
    import numpy as np
    from PIL import Image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(str(out_path))


def save_slice_overlay(image_arr, mask, out_path, threshold=0.5):
    """Save a two-panel (slice | segmentation) PNG for one NIfTI slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_arr[:, :, 0], cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    overlay = image_arr[:, :, :3].copy().astype(float) / 255.0
    overlay[mask > threshold] = [1.0, 0.0, 0.0]
    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_2d(input_path, prompts, output_dir, gpu, tag, modality):
    """Segment `prompts` in a single 2D image. Returns the tool-facing result dict."""
    import time
    from PIL import Image

    if not prompts:
        return {"status": "error", "error": "No prompts provided"}
    if not Path(input_path).exists():
        return {"status": "error", "error": f"Input not found: {input_path}"}
    if str(input_path).lower().endswith(".dcm"):
        return {"status": "error", "error": f"BiomedParse does not accept DICOM (.dcm) input: {input_path}. Convert to PNG/JPG first."}

    import torch
    device = f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu"

    out_dir = Path(output_dir) if output_dir else Path(
        __import__("tempfile").mkdtemp(prefix=f"biomedparse_{modality}_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    weights_path = ensure_weights(tag)
    model = load_model(weights_path, device, tag)
    image = Image.open(input_path).convert("RGB")
    seg_results = infer_2d(model, image, prompts)
    elapsed = round(time.time() - t0, 2)

    outputs = []
    for idx, seg in enumerate(seg_results):
        prompt, mask = seg["prompt"], seg["mask"]
        safe_name = prompt.replace(" ", "_").replace("/", "-")
        overlay_path = out_dir / f"overlay_{safe_name}.png"
        mask_path = out_dir / f"mask_{safe_name}.png"
        save_mask_overlay(image, mask, prompt, overlay_path, color_idx=idx)
        save_mask_png(mask, mask_path)

        pixel_count = int(mask.sum())
        outputs.append({
            "prompt": prompt,
            "overlay": str(overlay_path),
            "mask": str(mask_path),
            "score": round(seg["score"], 4),
            "score_kind": "mask_peak_probability",
            "pixel_count": pixel_count,
            "coverage_pct": round(100.0 * pixel_count / mask.size, 2),
        })
        print(f"[{tag}] {prompt}: coverage={100.0 * pixel_count / mask.size:.1f}%", file=sys.stderr)

    return {
        "status": "success",
        "model": f"BiomedParse ({HF_REPO})",
        "modality": modality,
        "input": str(input_path),
        "prompts": prompts,
        "score_kind": "mask_peak_probability",
        "device": device,
        "outputs": outputs,
        # Attach both the overlay composite (original | segmented) and the
        # binary mask per prompt — index.js shows up to MAX_ATTACHED_PREVIEWS
        # (4) of these as images in the conversation.
        "preview_image_paths": [p for o in outputs for p in (o["overlay"], o["mask"])],
        "elapsed_s": elapsed,
    }


def run_3d(input_path, prompts, is_ct, site, slice_idx, all_slices, channel_idx,
           output_dir, gpu, threshold, tag, modality):
    """Segment `prompts` in a NIfTI volume, one slice or every slice. Returns the tool-facing result dict."""
    import numpy as np

    if is_ct and not site:
        return {"status": "error", "error": "site is required when is_ct is true (abdomen|lung|pelvis|liver|colon|pancreas)"}
    if not prompts:
        return {"status": "error", "error": "No prompts provided"}
    if not Path(input_path).exists():
        return {"status": "error", "error": f"Input not found: {input_path}"}
    if not str(input_path).endswith((".nii", ".nii.gz")):
        return {"status": "error", "error": f"BiomedParse only accepts a NIfTI volume (.nii/.nii.gz), not: {input_path}. For a DICOM series, convert it first or use the _totalseg tool for this modality."}

    out_dir = Path(output_dir) if output_dir else Path(
        __import__("tempfile").mkdtemp(prefix=f"biomedparse_{modality}_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu"

    weights_path = ensure_weights(tag)
    model = load_model(weights_path, device, tag)

    n_slices = get_n_slices(input_path)
    if all_slices:
        slice_indices = list(range(n_slices))
        print(f"[{tag}] Processing all {n_slices} slices ...", file=sys.stderr)
    else:
        idx = slice_idx if slice_idx is not None else n_slices // 2
        slice_indices = [idx]
        print(f"[{tag}] Processing slice {idx} of {n_slices} ...", file=sys.stderr)

    vol_masks = {p: [] for p in prompts}
    slice_results = []

    for sl in slice_indices:
        try:
            img_arr = load_nifti_slice(input_path, is_ct, sl, site, channel_idx)
        except Exception as e:
            print(f"[{tag}] Skipping slice {sl}: {e}", file=sys.stderr)
            for p in prompts:
                vol_masks[p].append(None)
            continue

        try:
            masks = infer_slice(model, img_arr, prompts)
        except Exception as e:
            print(f"[{tag}] Inference failed on slice {sl}: {e}", file=sys.stderr)
            for p in prompts:
                vol_masks[p].append(None)
            continue

        slice_info = {"slice_idx": sl, "prompts": []}
        for i, prompt in enumerate(prompts):
            mask = masks[i] if i < len(masks) else np.zeros(img_arr.shape[:2])
            vol_masks[prompt].append((mask > threshold).astype(np.uint8))

            slug = prompt.replace(" ", "_").replace("/", "-")
            overlay_path = out_dir / f"overlay_{slug}_slice{sl:04d}.png"
            save_slice_overlay(img_arr, mask, overlay_path, threshold)

            coverage = float((mask > threshold).mean())
            slice_info["prompts"].append({
                "prompt": prompt,
                "coverage": coverage,
                "score_max": float(mask.max()),
                "overlay_path": str(overlay_path),
            })
            print(f"[{tag}] slice={sl:4d}  {prompt:30s}  coverage={coverage:.1%}", file=sys.stderr)

        slice_results.append(slice_info)

    nifti_mask_paths = {}
    if all_slices:
        import nibabel as nib
        ref = nib.load(input_path)
        for prompt, slices in vol_masks.items():
            valid = [s for s in slices if s is not None]
            if not valid:
                continue
            filled = [s if s is not None else np.zeros_like(valid[0]) for s in slices]
            vol = np.stack(filled, axis=2).astype(np.uint8)
            slug = prompt.replace(" ", "_").replace("/", "-")
            mask_path = out_dir / f"mask_{slug}.nii.gz"
            nib.save(nib.Nifti1Image(vol, ref.affine, ref.header), str(mask_path))
            nifti_mask_paths[prompt] = str(mask_path)
            print(f"[{tag}] 3D mask saved -> {mask_path}", file=sys.stderr)

    preview_paths = [sp["overlay_path"] for s in slice_results for sp in s["prompts"]]

    return {
        "status": "success",
        "model": HF_REPO,
        "modality": modality,
        "input": str(input_path),
        "is_ct": is_ct,
        "site": site,
        "n_slices_total": n_slices,
        "n_slices_processed": len(slice_results),
        "prompts": prompts,
        "slices": slice_results,
        "nifti_masks": nifti_mask_paths,
        "preview_image_paths": preview_paths,
    }
