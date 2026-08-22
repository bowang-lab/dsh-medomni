---
name: medomni-xray
description: >
  Analyze chest X-rays with report generation, grounded finding localization,
  normal-anatomy localization, segmentation overlays, and prior-versus-current
  comparison. Use this skill for chest radiograph questions and route each task
  to the narrowest appropriate tool. Grounded boxes localize model evidence;
  they do not establish diagnostic certainty.
license: MIT
category: medical-imaging
requirements: [gpu, python3, uv]
metadata:
  display-name: MedOmni Chest X-ray
  modality: chest-xray
---

# Chest X-ray specialist

## Tool selection

- Broad report: run `xray_grounded_report_maira` plus `xray_report_medgemma` when the indication includes trauma or suspected abnormality; otherwise run `xray_report_maira` and/or `xray_report_medgemma`.
- Evidence-focused report with finding boxes: `xray_grounded_report_maira` in its default `grounded_report` mode.
- Locate or adjudicate a named finding: `xray_phrase_grounding_maira`.
- Locate a normal anatomical structure: `xray_anatomy_localization` only. It is not an abnormality detector.
- Compare current and prior studies: `xray_longitudinal_comparison` with both images.
- Explicit mask or segmentation request: `xray_segmentation_biomedparse`.

## Workflow

1. Preserve the user's clinical indication and focused question.
2. For trauma or suspected abnormalities, run `xray_grounded_report_maira` in `grounded_report` mode first, preserving the full clinical indication, then run `xray_report_medgemma` as an independent narrative cross-check.
3. For a broad non-trauma request, run the normal MAIRA report and MedGemma report when practical. Do not treat either narrative as definitive by itself.
4. Translate the indication into an image-assessable differential before interpreting the reports. Include high-priority complications suggested by the symptom, mechanism, risk factor, or clinical question even when the reports do not mention them explicitly.
5. Use phrase grounding after the reports for (a) findings on which the reports disagree and (b) the highest-priority findings implied by the indication. Use the shortest exact phrase for each candidate and run separate calls when needed.
6. Do not present phrase-grounding boxes as confirmation. They localize the prompted phrase and require interpretation alongside both reports, image quality, and clinical context.
7. Do not substitute anatomy localization for pathology localization. Describe disagreement and uncertainty explicitly rather than merging contradictory findings into a definitive claim.

## Clinical context and next evidence

- **Cough, fever, or dyspnea:** assess air-space opacity/consolidation, edema, pleural fluid, pneumothorax, atelectatic volume loss, and cardiomediastinal size. Use report generation with the symptoms in `indication`.
- **Chest pain without trauma:** assess pneumothorax, pleural fluid, focal opacity, pulmonary edema, and acute osseous abnormality. Use grounded reporting when the question is finding-specific.
- **Clinical indication:** use symptoms, mechanism, and risk factors to prioritize image-assessable findings and select phrase-grounding targets. An empty or failed model output is indeterminate, not evidence of absence. Treat boxes as localization evidence, not confirmation. Do not use `xray_anatomy_localization` for pathology.
- **Chronic cough, weight loss, or cancer history:** assess focal mass or nodule, hilar/mediastinal enlargement, pleural abnormality, and interval change. Use a prior image with longitudinal comparison when available.
- **Recent procedure or line/tube placement:** check device position and procedure-related pneumothorax or pleural complication.
- **Follow-up or worsening symptoms:** use longitudinal comparison when a genuine prior image is available. Do not treat a single negative image as proof that disease is absent when clinical concern remains high.
- **Request for the next test:** report what the image shows and its limitations; do not automatically prescribe CT, repeat radiographs, or treatment. Further imaging depends on examination, vital signs, risk factors, prior studies, and clinician-directed appropriateness criteria.

## Running tools

These are DSH tools, not shell commands. Pass the image as `input`; include `indication` for report-generation tools.

- `xray_grounded_report_maira`: `input`, `indication`; optional `lateral`, `prior`, and `prior_report`.
- `xray_phrase_grounding_maira`: `input`, exact `phrase`.
- `xray_report_maira`: `input`, `indication`; use `mode="phrase_grounding"` only with `phrase`.
- `xray_report_medgemma`: `input`, `indication`.
- `xray_anatomy_localization`: `input`, optional `anatomy` list.
- `xray_longitudinal_comparison`: current `input` and required earlier `prior` image.
- `xray_segmentation_biomedparse`: image `input` and required `prompts` list; use only for explicit masks or overlays.

## Errors worth recognizing

- CUDA driver errors indicate a host NVIDIA/PyTorch mismatch; fix the driver or use a compatible Torch build before retrying.
- Hugging Face authorization errors require accepted model terms and a valid `HF_TOKEN` or `hf auth login`.
- An empty MAIRA grounded result or grounding-parse warning is indeterminate, not a normal study; run the MedGemma cross-check before answering.
- A missing grounded box is not proof that the finding is absent.
- Anatomy localization cannot detect fractures or other abnormalities.

## References

- [RadiologyInfo: Chest X-ray](https://www.radiologyinfo.org/en/info/chestrad)
- [ACR Appropriateness Criteria: Acute Respiratory Illness](https://acsearch.acr.org/docs/69446/narrative/)
- [ACR Appropriateness Criteria: Rib Fractures](https://acsearch.acr.org/docs/69450/Narrative/)
- [ACR Appropriateness Criteria: Minor Blunt Trauma](https://acsearch.acr.org/list/TopicNarrativePdf?topicId=334)

## Input and limitations

Accept a frontal chest X-ray by path or supported pasted-image attachment. Lateral and prior images are optional when the selected tool supports them. Do not use anatomy localization for pathology, and do not infer a diagnosis from a BiomedParse mask alone.
