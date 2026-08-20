# Adding a New Tool

This guide covers the standard dsh-medomni pattern for adding a new specialist tool:

1. A Python script under `skills/<modality>/` does the model work.
2. `index.js` registers the script as a DeepSeek Harness tool.
3. Tool descriptions and route hints tell the agent when to use it.
4. Optional preview images are attached back into the chat.

## 1. Add the Python skill script

Create a script under the matching modality folder:

```text
skills/xray/my_new_tool.py
skills/ultrasound/my_new_tool.py
skills/ct/my_new_tool.py
skills/mri/my_new_tool.py
skills/retinal/my_new_tool.py
```

Requirements:

- Print progress logs to `stderr`.
- Print one final JSON object to `stdout`.
- Return `{"status": "success", ...}` on success.
- Return `{"status": "error", "error": "..."}` for tool-level failures.
- Include `preview_image_path` or `preview_image_paths` if the tool generates preview PNGs.

Minimal script shape:

```python
#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(json.dumps({"status": "error", "error": f"Input not found: {args.input}"}))
        sys.exit(1)

    print(json.dumps({
        "status": "success",
        "input": args.input,
        "prediction": "example",
        "confidence": 0.91
    }))

if __name__ == "__main__":
    main()
```

For model dependencies, use the shared bootstrap pattern:

```python
import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR.parent))
import _bootstrap

_bootstrap.ensure_venv_and_reexec("my-tool-env")
_bootstrap.ensure_extra_packages("my-tool-env", "my-tool-deps", [
    "some_package==1.2.3",
])
```

Use a separate environment name when dependency ranges may conflict with existing tools.

## 2. Add the script to `index.js`

Add a `SCRIPTS` entry:

```js
const SCRIPTS = {
  // existing entries...
  ultrasoundMyNewTool: 'ultrasound/my_new_tool.py',
}
```

Register the tool inside `apply(ctx, config = {})`:

```js
ctx.tools.register(defineTool({
  name: 'ultrasound_my_new_tool',
  description: 'Classify an ultrasound image for a specific use case. Say exactly when to use this tool and when not to use it.',
  parameters: {
    input: {
      type: 'string',
      required: true,
      description: 'Absolute path to the ultrasound image (PNG or JPG).' + IMAGE_INPUT_NOTE,
    },
    gpu: {
      type: 'integer',
      description: 'GPU index (-1 for CPU). Default 0.',
    },
  },
  output: {
    schema: outputSchema('success'),
    render: (_args, value) => {
      const text = renderResult(value, (v) =>
        `Prediction: ${v.prediction} (confidence ${v.confidence})`
      )
      return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
    },
  },
  isConcurrencySafe: () => false,
  async execute(args, exec) {
    const input = await resolveImageInput(ctx, exec, args.input)
    const cliArgs = ['--input', input, '--gpu', String(args.gpu ?? 0)]
    const value = await run(SCRIPTS.ultrasoundMyNewTool, cliArgs, exec.signal)
    return attachPreview(ctx, exec, value)
  },
}))
```

Use `resolveImageInput(ctx, exec, args.input)` for 2D image tools so pasted image attachment ids like `sha256:...` work. Use `attachPreview(ctx, exec, value)` when the Python output may include preview paths.

## 3. Write agent instructions in the tool description

The `description` field is the main instruction the agent sees during tool selection. Make it explicit and operational.

Weak:

```js
description: 'Run my ultrasound model.'
```

Better:

```js
description: 'Classify a thyroid ultrasound image for nodule category. Use this only for thyroid ultrasound triage questions. Do not use it for breast ultrasound, chest X-ray, CT, MRI, segmentation, or report generation. This is forced-choice classification over the provided labels, not calibrated diagnostic certainty.'
```

Include:

- What the tool does.
- Which modality and input type it accepts.
- When to use it.
- When not to use it.
- Whether the output is a report, classification, segmentation/localization, comparison, or candidate result.
- Which tool should run before or after it, if the workflow depends on ordering.

For example, CXR report tools should say they answer "what abnormalities/findings are present?" A CXR segmentation/localization tool should say it must not be used as a first-line report tool.

## 4. Update the vision route marker only for first-line tools

`lib/vision-route.js` controls the text marker that pasted images become on text-only routes. Add the new tool there only if it should be commonly selected from a generic image request.

Good candidates:

- Report tools for "read this image" requests.
- Composite workflow tools that choose the right substeps.
- Classifiers that are intended as first-line triage.

Avoid adding niche localization/segmentation tools to the generic list unless the user normally asks for them directly. Otherwise the model may call them for broad interpretation questions.

## 5. Add preview-card support

If the tool can return preview images, add its name to `TOOL_NAMES` in `client.js`:

```js
const TOOL_NAMES = [
  // existing entries...
  'ultrasound_my_new_tool',
]
```

The existing renderer will load attachments from the session and show clickable preview thumbnails in the tool response.

## 6. Include files in the package

If you add a Python file, include it in `package.json` `files`:

```json
"skills/ultrasound/my_new_tool.py"
```

If you add docs or client files that must be available in packaged installs, include those too.

## 7. Validate

Run focused checks before reinstalling:

```sh
node --check index.js
node --check client.js
node --check lib/vision-route.js
python3 -m py_compile skills/ultrasound/my_new_tool.py
node --test tests/vision-route.test.js tests/attachment-input.test.js
npm pack --dry-run --cache /tmp/dsh-medomni-npm-cache
```

Then reinstall/reload the plugin and restart Harness so the new tool schema, route hints, and client bundle are picked up.
