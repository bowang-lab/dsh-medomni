---
name: medomni-ultrasound
description: >
  Analyze ultrasound images with classification-first reasoning and prompted
  localization. Classify when the anatomy or category is uncertain, then use
  segmentation only to localize the selected finding or structure. Do not infer
  diagnosis from mask coverage or prompt agreement.
license: MIT
category: medical-imaging
requirements: [gpu, python3, uv]
metadata:
  display-name: MedOmni Ultrasound
  modality: ultrasound
---

# Ultrasound specialist

## Tool selection

- Classification followed by localization: `ultrasound_classify_then_segment`.
- Classification only: `ultrasound_classify_biomedclip`.
- Explicit mask or overlay request: `ultrasound_segmentation_biomedparse`.

## Workflow

1. Use classification first when the anatomy, domain, or finding is uncertain.
2. Use the composite workflow when the user needs both category assessment and localization.
3. Use explicit segmentation only when the user names a finding or asks for a mask or overlay.
4. Keep classification and localization separate in the final explanation.
5. Do not infer benign/malignant status, cyst-versus-solid status, or diagnostic certainty from a mask, peak probability, or prompt agreement.

## Clinical context and next evidence

- **Breast lump or breast ultrasound request:** use the breast or BUSI classification domain when appropriate, then localize a named lesion with segmentation. Do not infer malignancy or recommend biopsy from the model output alone.
- **Uncertain anatomy or symptom description:** classify the ultrasound anatomy first, then choose a domain or candidate-label panel before localization.
- **Request for the next test:** describe the observed category and localized region, then defer decisions about diagnostic mammography, biopsy, follow-up interval, or other testing to the clinician and applicable age- and risk-specific guidance.

## Running tools

These are DSH tools, not shell commands. Pass the ultrasound image as `input`.

- `ultrasound_classify_then_segment`: `input`; set `domain`, optional `labels`, and optional `segment_prompts`.
- `ultrasound_classify_biomedclip`: `input` and required `task`; provide `labels` when `task="cls"`.
- `ultrasound_segmentation_biomedparse`: `input` and required `prompts` list.

## Errors worth recognizing

- Classification selects among candidate labels; it does not provide calibrated diagnostic certainty.
- A segmentation mask does not establish benign/malignant status or cyst-versus-solid status.
- CUDA driver errors indicate a host NVIDIA/PyTorch mismatch.
- Hugging Face authorization errors require accepted model terms and a valid token.

## References

- [ACR Appropriateness Criteria: Palpable Breast Masses](https://acsearch.acr.org/docs/69495/Narrative/)

## Input and limitations

Accept a 2D ultrasound image by path or supported pasted-image attachment. BiomedCLIP selects among the provided labels; it does not invent categories or provide calibrated diagnostic certainty.
