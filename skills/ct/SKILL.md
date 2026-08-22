---
name: medomni-ct
description: >
  Analyze CT volumes with report generation and anatomical or free-text
  segmentation. Select tools according to whether the request needs a report,
  fixed-label anatomy, or a prompted finding mask. Treat generated reports and
  masks as research outputs rather than definitive clinical interpretation.
license: MIT
category: medical-imaging
requirements: [gpu, python3, uv]
metadata:
  display-name: MedOmni CT
  modality: ct
---

# CT specialist

## Tool selection

- Report candidate: `ct_report_medgemma`.
- Fixed-label anatomical segmentation: `ct_segmentation_totalseg`.
- Free-text finding or structure segmentation: `ct_segmentation_biomedparse`.

## Workflow

1. Require a filesystem path to the CT volume and preserve the clinical indication.
2. Use report generation for clinical-question or abnormality-summary requests.
3. Use fixed-label segmentation for structures in the supported anatomical label set.
4. Use BiomedParse for a specific free-text prompt or pathology/anatomy mask.
5. Explain that a segmentation mask localizes a prompt and does not independently confirm disease.

## Clinical trigger checklist

- **Trauma:** report the relevant body region and inspect for fractures, hemorrhage, organ injury, and free air or fluid as appropriate to the study; do not use fixed-label organ segmentation as a fracture detector.
- **Abdominal pain:** preserve the location and clinical question in `indication`; use report generation first, then segment a named organ or finding when localization is requested.
- **Known lesion or cancer follow-up:** use the report tool with the clinical history and comparison information when available; use segmentation only to localize a named structure or lesion.
- **Request for a next test:** describe the image evidence and missing information, but do not select contrast, protocol, biopsy, or treatment automatically.

## Running tools

These are DSH tools, not shell commands. Pass a NIfTI path or supported DICOM directory as `input`.

- `ct_report_medgemma`: `input`; add `indication` and adjust `n_slices` when appropriate.
- `ct_segmentation_totalseg`: `input`; optionally set `task`, `roi_subset`, `preview`, or `statistics`.
- `ct_segmentation_biomedparse`: NIfTI `input`, required `prompts`, and required CT `site`.

## Errors worth recognizing

- BiomedParse does not accept a DICOM directory; convert the study to NIfTI or use a compatible CT tool.
- CUDA driver errors indicate a host NVIDIA/PyTorch mismatch.
- Hugging Face authorization errors require accepted model terms and a valid token.
- A mask localizes the prompt and does not confirm pathology.

## Input and limitations

NIfTI volumes are supported by all CT tools. Report generation and fixed-label segmentation can also accept a DICOM series directory. BiomedParse requires NIfTI input. CT report generation samples slices and is a research candidate, not a substitute for a validated native 3D clinical model.
