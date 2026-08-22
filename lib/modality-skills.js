import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const MODALITY_SKILLS = [
  {
    name: 'medomni-xray',
    modality: 'chest-xray',
    file: 'xray/SKILL.md',
    description: 'Choose and sequence dsh-medomni tools for chest X-ray reporting, grounding, localization, segmentation, and longitudinal comparison.',
  },
  {
    name: 'medomni-ct',
    modality: 'ct',
    file: 'ct/SKILL.md',
    description: 'Choose dsh-medomni report and segmentation tools for CT volumes, including NIfTI and supported DICOM inputs.',
  },
  {
    name: 'medomni-mri',
    modality: 'mri',
    file: 'mri/SKILL.md',
    description: 'Choose dsh-medomni report and segmentation tools for MRI volumes and state their input and interpretation limitations.',
  },
  {
    name: 'medomni-ultrasound',
    modality: 'ultrasound',
    file: 'ultrasound/SKILL.md',
    description: 'Apply classification-first reasoning to ultrasound requests, then use dsh-medomni localization tools when appropriate.',
  },
  {
    name: 'medomni-retinal',
    modality: 'retinal',
    file: 'retinal/SKILL.md',
    description: 'Choose dsh-medomni report and prompted localization tools for retinal fundus photographs.',
  },
]

function skillBody(markdown) {
  return markdown.replace(/^---\s*\n[\s\S]*?\n---\s*\n/, '').trim()
}

/** Register modality workflows as model-invocable DSH skills. */
export function registerModalitySkills(ctx, skillsDir) {
  if (!ctx.skills || typeof ctx.skills.register !== 'function') return []

  const disposers = []
  for (const skill of MODALITY_SKILLS) {
    try {
      const content = skillBody(readFileSync(join(skillsDir, skill.file), 'utf8'))
      const dispose = ctx.skills.register({
        name: skill.name,
        description: skill.description,
        source: 'runtime',
        content,
        metadata: { modality: skill.modality, package: 'dsh-medomni' },
      })
      if (typeof dispose === 'function') disposers.push(dispose)
    } catch (error) {
      ctx.logger?.warn?.(
        'dsh-medomni: modality skill %s was not registered: %s',
        skill.name,
        error && error.message ? error.message : String(error),
      )
    }
  }

  if (disposers.length > 0 && typeof ctx.effect === 'function') {
    ctx.effect(() => () => {
      for (const dispose of disposers.reverse()) dispose()
    }, 'dsh-medomni: modality skills')
  }
  return disposers
}
