"""
Shared MedGemma 1.5 4B 3D-volume report logic, used by both ct/medgemma_report.py
and mri/medgemma_report.py. MedGemma 1.5 is trained to interpret a 3D volume as
a sequence of per-slice image blocks (each labeled "SLICE N" in the chat
template) — NOT a single flattened montage image, which the base 2D MedGemma
4B was never trained on. The two wrappers differ only in per-slice windowing
(CT: fixed Hounsfield-unit windows; MRI: percentile intensity normalization,
since MRI has no fixed absolute intensity scale) and their intro/query text.

Top-level imports here must stay stdlib-only: this module is imported by a
wrapper both before and after the shared-venv re-exec (see _bootstrap.py).
"""

import sys
import tempfile
from pathlib import Path

SYSTEM_PROMPT = "You are an expert radiologist."


def dicom_to_nifti(dicom_dir: Path) -> str:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(names)
    image = reader.Execute()
    tmp_dir = tempfile.mkdtemp(prefix="medgemma_3d_dicom_")
    out = str(Path(tmp_dir) / "volume.nii.gz")
    sitk.WriteImage(image, out)
    return out


def resolve_nifti(path_text: str) -> str:
    path = Path(path_text).expanduser().resolve()
    if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return str(path)
    if path.is_dir():
        return dicom_to_nifti(path)
    raise ValueError("Input must be a .nii/.nii.gz file or a DICOM series directory")


def sample_slices(nifti_path: str, n_slices: int, window_fn):
    """Load the volume and uniformly sample up to n_slices slices from the
    central 84% of the depth axis (skipping the outermost edges, usually
    air/table), each converted to a 3-channel uint8 RGB image by `window_fn`.
    Returns (rgb_slices, source_indices)."""
    import numpy as np
    import nibabel as nib

    img = nib.load(nifti_path)
    volume = np.asanyarray(img.dataobj)
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")

    z_count = volume.shape[2]
    start = max(0, int(z_count * 0.08))
    stop = min(z_count - 1, int(z_count * 0.92))
    indices = np.linspace(start, stop, min(n_slices, z_count)).astype(int)

    rgb_slices = [window_fn(np.rot90(volume[:, :, idx])) for idx in indices]
    return rgb_slices, indices.tolist()


def make_contact_sheet(rgb_slices, indices, tile_size: int = 160):
    """Downsampled grid of the same windowed slices sent to the model — for
    human display only; the model sees the full-resolution per-slice
    sequence via run_report(), not this composite."""
    import numpy as np
    from PIL import Image, ImageDraw

    tiles = []
    for arr, idx in zip(rgb_slices, indices):
        tile = Image.fromarray(arr).resize((tile_size, tile_size), Image.BILINEAR)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 58, 18), fill=(0, 0, 0))
        draw.text((4, 3), f"z={idx}", fill=(255, 255, 255))
        tiles.append(tile)

    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_size, rows * tile_size), (0, 0, 0))
    for i, tile in enumerate(tiles):
        x = (i % cols) * tile_size
        y = (i // cols) * tile_size
        sheet.paste(tile, (x, y))
    return sheet


def load_pipeline(tag: str, gpu: int, model_id: str):
    import torch
    from transformers import pipeline

    # {"": "cuda:N"} rather than a bare "cuda:N" string or device_map="auto"
    # — see skills/xray/medgemma_report.py's load_pipeline() for why a bare
    # string leaves Pipeline.__init__ doing a redundant .to("cuda:0") that
    # can OOM on a shared GPU box; "auto" (which Google's own reference
    # notebook uses, with offload_buffers=True for disk/CPU spillover) would
    # change this to a multi-device placement, inconsistent with every other
    # tool here taking one explicit --gpu index.
    device_map = {"": f"cuda:{gpu}"} if gpu >= 0 else {"": "cpu"}
    print(f"[{tag}] Loading {model_id} on {device_map}", file=sys.stderr)
    return pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


def build_messages(rgb_slices, intro_text: str, query_text: str):
    from PIL import Image

    content = [{"type": "text", "text": intro_text}]
    for slice_number, arr in enumerate(rgb_slices, 1):
        content.append({"type": "image", "image": Image.fromarray(arr)})
        content.append({"type": "text", "text": f"SLICE {slice_number}"})
    content.append({"type": "text", "text": query_text})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def run_report(pipe, rgb_slices, intro_text: str, query_text: str, max_new_tokens: int) -> str:
    messages = build_messages(rgb_slices, intro_text, query_text)
    output = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    content = output[0]["generated_text"][-1]["content"]
    if isinstance(content, list):
        response = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    else:
        response = str(content)
    # Strip thinking trace emitted between <unused94> … <unused95> (MedGemma 1.5)
    if "<unused95>" in response:
        response = response.split("<unused95>", 1)[1].lstrip()
    return response.strip()
