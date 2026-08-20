<h1 align="center">DeepSeek Harness × MedOmni: A Composable Agentic Framework for Biomedical Image Analysis</h1>

<p align="center"><strong>Plugin a part of MedOmni to DeepSeek Harness</strong></p>

<p align="center">
  <img src="assets/preview_workflow.svg" width="100%" alt="dsh-medomni workflow: a user request is routed by the DeepSeek Harness agent to a modality-specific tool (X-ray, CT, MRI, ultrasound, retinal), backed by MAIRA-2 / MedGemma / TotalSegmentator / BiomedParse / BiomedCLIP running in one shared Python venv, returning a structured result plus a preview image." />
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EA44F?style=flat-square" alt="License: MIT" /></a>
  <a href="package.json"><img src="https://img.shields.io/badge/Node.js-%3E%3D22-339933?style=flat-square&logo=nodedotjs&logoColor=white" alt="Node.js >=22" /></a>
  <a href="cordis.patch.yml"><img src="https://img.shields.io/badge/DSH-plugin-5B4CF0?style=flat-square" alt="DSH plugin" /></a>
  <img src="https://img.shields.io/badge/GPU-recommended-orange?style=flat-square" alt="GPU recommended" />
</p>

## Contents

- [What this plugin does](#what-this-plugin-does)
- [Demo](#demo)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Enable the vision route (paste an image)](#enable-the-vision-route-paste-an-image)
- [Usage examples](#usage-examples)
- [Tools](#tools)
- [Adding a new tool](#adding-a-new-tool)
- [Disable / re-enable](#disable--re-enable)
- [Configure](#configure)
- [Troubleshooting](#troubleshooting)
- [Scope](#scope)
- [License](#license)

## Demo

https://github.com/user-attachments/assets/8a098d95-8d24-44bd-8435-67e52450c524


## What this plugin does

Each tool shells out to a Python script that self-manages its own shared virtual environment and downloads its own model weights on first use — no manual setup beyond what's listed in [Requirements](#requirements). One tool call in, one structured JSON result plus a preview image out. See the diagram at the top of this page for the full path from a chat message to a report/segmentation/classification result.

## Requirements

- A DeepSeek Harness `dsh` installation.
- **An NVIDIA GPU with CUDA.** These are multi-GB vision-language and segmentation models — CPU (`gpu: -1`, where a tool exposes it) works but is slow enough to be impractical for anything beyond a quick smoke test.
- [`uv`](https://docs.astral.sh/uv/), `git`, and Python 3 on `PATH` — every script bootstraps its own shared venv and, for BiomedParse, clones a repo and builds `detectron2` from source on first use.
- Hugging Face access: accept the gated-model terms for [MAIRA-2 (`microsoft/maira-2`)](https://huggingface.co/microsoft/maira-2), [BiomedParse (`microsoft/BiomedParse`, including `biomedparse_v1.pt`)](https://huggingface.co/microsoft/BiomedParse), and [MedGemma (`google/medgemma-4b-it` / `google/medgemma-1.5-4b-it`)](https://huggingface.co/google/medgemma-1.5-4b-it), then set `HF_TOKEN` or run `hf auth login`. [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) is public and needs no token.
- `HF_TOKEN` is a Hugging Face User Access/API token, not a model checkpoint. Create a token with read access at [Hugging Face settings](https://huggingface.co/settings/tokens), then export it with `export HF_TOKEN=hf_...` or run `hf auth login`.
- Checkpoints download automatically on the first call to the corresponding tool; no manual model download is required.
- Optional prefetch: after authenticating, run `hf download microsoft/maira-2`, `hf download microsoft/BiomedParse biomedparse_v1.pt`, `hf download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, or the relevant `google/medgemma-*` command to download a checkpoint before inference.

## Quick start

### 1. Install the plugin

```sh
dsh plugin --profile web add github:medfm-flare/dsh-medomni
```


This adds `dsh-medomni` to your profile's `package.json` and installs `cordis.patch.yml` (mounting the plugin under id `dsh-medomni`) automatically. Restart `dsh` (or your DSH Desktop/web session) afterward so the new bundle loads.

### 2. Just ask, in plain text

Nothing needs to be pre-built or downloaded first. Ask your agent something like:

> "Generate a radiology report for this chest X-ray: `/path/to/chest_xray.png`"

and it picks the matching tool itself. The first call for any given model/dependency group is slower — it creates the shared venv, downloads weights, and (for BiomedParse) clones a repo and builds `detectron2` — every call after that skips straight to inference. Run `dsh-medomni doctor` any time to see exactly how far setup has progressed (see [Troubleshooting](#troubleshooting)).

## Enable the vision route (paste an image)

Tools that take a 2D image (X-ray, ultrasound, retinal, and the classification/report tools) also accept a **pasted image** instead of a filesystem path — but only on the right model route.

> [!IMPORTANT]
> **Before pasting an image, open the model selector in the lower-right corner of the chat composer and choose the entry marked "+ dsh-medomni Vision".**
>
> DSH rejects a pasted image on a text-only route (e.g. plain DeepSeek chat-completions) before any plugin sees it, because that route's catalog declares text-only input. dsh-medomni registers a `<provider>-dsh-medomni` twin route for every live provider that declares image input (e.g. `deepseek-official-dsh-medomni`, shown as "DeepSeek + dsh-medomni Vision" in the picker) — pick the twin, then paste normally. The twin follows the live model catalog; no restart needed when models change in Settings.

What happens under the hood: the twin rewrites the pasted image into a compact attachment-id marker before handing the turn to your normal text model (the chat UI still shows the real image); the model passes that id straight to a dsh-medomni tool as `input`; the tool resolves it back to real bytes and hands a temp file path to the Python script. CT/MRI tools take NIfTI volumes or DICOM directories, so they always accept paths only — pasting doesn't apply there.

Set `wrapProviders: false` in the plugin config to disable the twin routes (see [Configure](#configure)).

## Usage examples

**Chest X-ray report:**
> "Generate a radiology report for this chest X-ray: `/path/to/chest_xray.png`"

calls `xray_report_medgemma`, `xray_report_maira`, or `xray_grounded_report_maira` when finding evidence/bounding boxes are useful.

**Find something specific, by name, on any of the five modalities:**
> "Segment the gallstone in this ultrasound image."
> "Are there any microaneurysms in this fundus photo?"

calls the matching `_segmentation_biomedparse` tool only when you ask for segmentation, masks, overlays, or localization. BiomedParse takes any free-text finding or anatomical structure, but its mask is localization, not diagnosis.

**Disambiguate a vague ultrasound request first:**
> "What's in this ultrasound before you segment anything?"

calls `ultrasound_classify_biomedclip` to narrow down anatomy/pathology, then a segmentation tool with the winning label as the prompt.

**Whole-body organ segmentation on a CT or MRI volume:**
> "Segment the liver and kidneys in this CT scan: `/path/to/scan.nii.gz`"

calls `ct_segmentation_totalseg` for the fixed anatomical-structure list, or `ct_segmentation_biomedparse` if you'd rather name a pathology than an organ.

**Prior-vs-current comparison:**
> "Compare this current chest X-ray to the prior one and describe interval change."

calls `xray_longitudinal_comparison` with both images.

## Tools

<p align="center">
  <img src="assets/tools_overview.svg" width="100%" alt="Overview of dsh-medomni's 15 tools across X-ray, CT, MRI, ultrasound, and retinal (fundus), grouped by report generation, segmentation, and classification, each naming its backing model." />
</p>

`_totalseg` and `_biomedparse` on the same modality are complementary: TotalSegmentator segments a fixed list of named structures with no text prompt; BiomedParse segments whatever free-text prompt you give it, at the cost of needing you to name what you're looking for.

Every tool also attaches one or more preview PNGs to its result — inline in the chat on an image-capable route, or saved under `dsh-medomni/previews/` in the session workspace on a text-only route.

## Adding a new tool

New tools should follow the existing pattern: a Python script under `skills/<modality>/`, a `SCRIPTS` entry plus `defineTool` registration in `index.js`, explicit agent-facing instructions in the tool `description`, optional preview attachment support, and package/test updates.

See [Adding a New Tool](ADDING_TOOLS.md) for the full step-by-step checklist and examples.

## Disable / re-enable

```yaml
- id: dsh-medomni
  disabled: true
```

Set it back to `false` (or remove the line) to re-enable. Unloading removes the tools, the vision twin routes, and the settings surface; anything already written to the session workspace remains.

## Configure

Nothing is required — the plugin defaults to the `skills/` directory shipped inside this package. Override only if you want to point at a different copy of these scripts, in your profile's `cordis.patch.yml`:

```yaml
- upsert:
    - id: dsh-medomni
      config:
        skillsDir: /path/to/other/skills   # optional, default: this package's own skills/
        # pythonBin: python3               # optional, default "python3"
        # timeoutMs: 1800000                # optional, default 30 minutes
        # wrapProviders: true              # optional, default true — image-capable
        #                                  #   "<provider>-dsh-medomni" twin routes
        # excludedProviders: []            # optional — provider ids never wrapped
```

## Troubleshooting

`dsh-medomni doctor` checks this machine's setup without calling any model — whether `uv`/`git`/`python3` are on `PATH`, whether the shared venv exists, and how far BiomedParse's one-time setup has progressed:

```sh
cd ~/.dsh/profiles/web
./node_modules/.bin/dsh-medomni doctor
```

or, against a local checkout of this repo directly:

```sh
node /path/to/dsh-medomni/lib/doctor-cli.js
```

```
skills/: /path/to/dsh-medomni/skills
✓ uv — uv 0.10.7
✓ git — git version 2.34.1
✓ python3 — Python 3.12.12

✓ shared venv
    torch pin: torch==2.10.0
    torch 2.10.0+cu128, CUDA available: true

✓ BiomedParse repo cloned
✓ BiomedParse extra dependencies installed
✓ detectron2 built
✓ BiomedParse weights downloaded — 1.7GB
```

A `✗` line names exactly what's missing and why it matters. An unchecked BiomedParse line is not itself an error — that stage only runs on a `_biomedparse` tool's first call — but is the first place to look if such a call fails. Pass `--skills-dir <path>` if your profile overrides the plugin's default `skillsDir`.

## Scope

- This is the preview release of MedOmni and its tool strategy, not a general registry of every medical-imaging model family. Only tools that follow the MedOmni tool protocol are included: each tool declares its contract in `index.js`, uses the standard execution path, returns structured JSON, and bootstraps an isolated Python environment before importing model dependencies.
- Other model families and tools that do not follow this protocol are intentionally excluded from this preview. For example, some tube/line detection and bone-fracture tools have backing scripts with no isolated-venv bootstrap of their own. They import bare `torch`/`transformers` from whatever ambient Python environment happens to be active, which can silently conflict with a pinned range such as MAIRA-2's `transformers>=4.48,<4.52`.

If you want to add tools back in this style, follow [Adding a New Tool](ADDING_TOOLS.md): a `SCRIPTS` entry pointing at a script under `skills/<modality>/`, a matching `defineTool` registration whose `execute` shells out to it and parses its JSON stdout, explicit agent-facing tool instructions, and `skills/_bootstrap.py`'s bootstrap at the top of the script.

## License

[MIT](LICENSE)
