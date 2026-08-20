import { existsSync, readFileSync, statSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const PACKAGE_DIR = fileURLToPath(new URL('..', import.meta.url))

/** Whether `bin` is on PATH and runnable, and its first output line if so. Never throws. */
function commandVersion(bin, args = ['--version'], spawn = spawnSync) {
  let result
  try {
    result = spawn(bin, args, { encoding: 'utf8' })
  } catch {
    return { found: false }
  }
  if (!result || result.error || result.status !== 0) return { found: false }
  const line = (result.stdout || result.stderr || '').trim().split('\n')[0]
  return { found: true, output: line }
}

/** torch build + CUDA availability inside the shared venv, or undefined if the venv doesn't exist or the check itself fails. */
function venvTorchInfo(venvPython, spawn) {
  const result = spawn(venvPython, [
    '-c',
    'import json,torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.cuda.is_available()}))',
  ], { encoding: 'utf8' })
  if (!result || result.error || result.status !== 0) return undefined
  try {
    return JSON.parse((result.stdout || '').trim())
  } catch {
    return undefined
  }
}

/**
 * Inspect this machine's dsh-medomni setup: host prerequisites (uv/git/python3
 * on PATH), the shared venv's state, and how far BiomedParse's one-time
 * setup (repo clone, extra deps, detectron2, weights) has progressed. Every
 * check is a PATH lookup or filesystem read — nothing here runs a model or
 * makes a network call.
 *
 * @param {{ skillsDir?: string, spawn?: typeof spawnSync }} [options]
 */
export function doctorSkills({ skillsDir = `${PACKAGE_DIR}skills`, spawn = spawnSync } = {}) {
  const venvDir = path.join(skillsDir, '.venv')
  const venvPython = path.join(venvDir, 'bin', 'python')
  const constraintsFile = path.join(skillsDir, '.uv-constraints.txt')
  const biomedparseRepo = path.join(skillsDir, 'biomedparse', 'BiomedParse')
  const biomedparseWeights = path.join(skillsDir, 'biomedparse', 'weights', 'biomedparse_v1.pt')

  const uv = commandVersion('uv', ['--version'], spawn)
  const git = commandVersion('git', ['--version'], spawn)
  const python3 = commandVersion('python3', ['--version'], spawn)

  const venvExists = existsSync(venvPython)
  const constraintsPin = existsSync(constraintsFile) ? readFileSync(constraintsFile, 'utf8').trim() || undefined : undefined
  const torch = venvExists ? venvTorchInfo(venvPython, spawn) : undefined

  const biomedparseRepoCloned = existsSync(path.join(biomedparseRepo, 'modeling', 'BaseModel.py'))
    && existsSync(path.join(biomedparseRepo, 'inference_utils', 'processing_utils.py'))
  const biomedparseDepsInstalled = existsSync(path.join(venvDir, '.installed-biomedparse-deps'))
  const biomedparseDetectron2Installed = existsSync(path.join(venvDir, '.installed-biomedparse-detectron2'))
  let biomedparseWeightsBytes
  if (existsSync(biomedparseWeights)) {
    try {
      biomedparseWeightsBytes = statSync(biomedparseWeights).size
    } catch {
      // Race with a concurrent download: leave it undefined and report "not found".
    }
  }

  return {
    skillsDir,
    venvPython,
    uv,
    git,
    python3,
    venvExists,
    constraintsPin,
    torch,
    biomedparseRepoCloned,
    biomedparseDepsInstalled,
    biomedparseDetectron2Installed,
    biomedparseWeightsBytes,
    ok: uv.found && git.found && python3.found,
  }
}
