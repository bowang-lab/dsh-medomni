---
name: medomni-retinal
description: >
  Analyze retinal fundus photographs with report generation and prompted
  localization. Use reporting for a narrative assessment and segmentation only
  when the user explicitly requests a named finding, structure, mask, or overlay.
  Treat localization as evidence rather than diagnostic confirmation.
license: MIT
category: medical-imaging
requirements: [gpu, python3, uv]
metadata:
  display-name: MedOmni Retinal Imaging
  modality: retinal
---

# Retinal imaging specialist

## Tool selection

- Narrative assessment: `retinal_report_medgemma`.
- Explicit mask, overlay, or named-finding localization: `retinal_segmentation_biomedparse`.

## Workflow

1. Preserve the clinical indication and identify the image as a retinal fundus photograph.
2. Use report generation for broad assessment requests.
3. Use segmentation only for an explicitly named finding or structure.
4. State image-quality limitations and uncertainty.

## Clinical trigger checklist

- **Diabetes or screening:** use report generation for a broad assessment and state image-quality limitations.
- **Vision loss, floaters, or a named lesion:** use report generation first, then use segmentation only to localize the explicitly named finding or structure.
- **Follow-up request:** preserve the prior diagnosis and treatment context in the indication; do not infer progression or treatment response from mask area alone.
- **Request for the next test or treatment:** describe the image evidence and recommend clinician or ophthalmology review rather than prescribing management.

## Running tools

These are DSH tools, not shell commands. Pass the fundus photograph as `input`.

- `retinal_report_medgemma`: `input`; add the clinical `indication` when available.
- `retinal_segmentation_biomedparse`: `input` and required `prompts` list; use only for explicit localization or masks.

## Errors worth recognizing

- A segmentation mask localizes the prompt and does not confirm disease or severity.
- Poor illumination, focus, field of view, or image artifacts can limit interpretation.
- CUDA driver errors indicate a host NVIDIA/PyTorch mismatch.
- Hugging Face authorization errors require accepted model terms and a valid token.

## Input and limitations

Accept a retinal photograph by path or supported pasted-image attachment. A mask localizes the prompted region and does not confirm disease or severity.
