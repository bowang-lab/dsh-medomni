---
name: medomni-mri
description: >
  Analyze MRI volumes with report generation and anatomical or free-text
  segmentation. Select tools according to whether the request needs a report,
  fixed-label anatomy, or a prompted finding mask. Treat generated reports and
  masks as research outputs and state uncertainty when the study or sequence is
  incomplete.
license: MIT
category: medical-imaging
requirements: [gpu, python3, uv]
metadata:
  display-name: MedOmni MRI
  modality: mri
---

# MRI specialist

## Tool selection

- Report candidate: `mri_report_medgemma`.
- Fixed-label anatomical segmentation: `mri_segmentation_totalseg`.
- Free-text finding or structure segmentation: `mri_segmentation_biomedparse`.

## Workflow

1. Require a filesystem path to the MRI volume and preserve the clinical indication.
2. Use report generation for clinical-question or abnormality-summary requests.
3. Use fixed-label segmentation for structures in the supported anatomical label set.
4. Use BiomedParse for a specific free-text prompt or pathology/anatomy mask.
5. For multi-channel studies, provide the appropriate channel when required.
6. Treat segmentation as localization evidence, not diagnostic confirmation.

## Clinical trigger checklist

- **Neurologic symptoms:** preserve the symptom and relevant time course in `indication`; use report generation and state when the available volume or sequence information is incomplete.
- **Spine or joint pain:** use report generation for the clinical question and segment a named structure or finding only when localization is requested.
- **Tumor or treatment follow-up:** include the history and comparison context in the indication; do not infer treatment response from a mask alone.
- **Request for a next test or sequence:** identify the limitation in the current volume and defer protocol, contrast, and follow-up decisions to the responsible clinician or radiologist.

## Running tools

These are DSH tools, not shell commands. Pass a NIfTI path or supported DICOM directory as `input`.

- `mri_report_medgemma`: `input`; add `indication` and adjust `n_slices` when appropriate.
- `mri_segmentation_totalseg`: `input`; optionally set `roi_subset` and `preview`.
- `mri_segmentation_biomedparse`: NIfTI `input` and required `prompts`; set `channel_idx` for multi-channel studies when needed.

## Errors worth recognizing

- BiomedParse does not accept a DICOM directory; convert the study to NIfTI or use a compatible MRI tool.
- CUDA driver errors indicate a host NVIDIA/PyTorch mismatch.
- Hugging Face authorization errors require accepted model terms and a valid token.
- A mask localizes the prompt and does not confirm pathology.

## Input and limitations

NIfTI volumes are supported by all MRI tools. Report generation and fixed-label segmentation can also accept a DICOM series directory. BiomedParse requires NIfTI input. MRI report generation uses a research preprocessing strategy and is not a substitute for a native 3D MRI clinical model.
