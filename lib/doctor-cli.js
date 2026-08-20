#!/usr/bin/env node

import { realpathSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { doctorSkills } from './doctor.js'

function usage() {
  return `dsh-medomni doctor [--skills-dir <path>]\n\nChecks this machine's dsh-medomni setup: whether uv/git/python3 are on PATH,\nwhether the shared venv exists, and how far BiomedParse's one-time setup\n(repo clone, extra dependencies, detectron2, weights) has progressed.\n\nOptions:\n  --skills-dir <path>  Check a skills/ directory other than this package's\n                       own (match your plugin config's skillsDir if you set one).\n  --help               Show this help.\n`
}

function parseArgs(argv) {
  const args = [...argv]
  if (args[0] && !args[0].startsWith('-') && args[0] !== 'doctor') {
    throw new Error(`unknown command: ${args[0]}`)
  }
  if (args[0] === 'doctor') args.shift()

  let skillsDir
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index]
    if (value === '--help' || value === '-h') return { help: true }
    if (value === '--skills-dir') {
      skillsDir = args[index + 1]
      index += 1
      if (!skillsDir) throw new Error('--skills-dir requires a path')
      continue
    }
    if (value.startsWith('--skills-dir=')) {
      skillsDir = value.slice('--skills-dir='.length)
      if (!skillsDir) throw new Error('--skills-dir requires a path')
      continue
    }
    throw new Error(`unknown argument: ${value}`)
  }
  return { skillsDir }
}

function formatBytes(bytes) {
  const gb = bytes / 1024 ** 3
  return gb >= 0.1 ? `${gb.toFixed(1)}GB` : `${(bytes / 1024 ** 2).toFixed(0)}MB`
}

function mark(ok) {
  return ok ? '✓' : '✗'
}

export function run(argv = process.argv.slice(2), io = console) {
  let options
  try {
    options = parseArgs(argv)
  } catch (error) {
    io.error(`dsh-medomni: ${error instanceof Error ? error.message : String(error)}`)
    io.error(usage().trimEnd())
    return 2
  }
  if (options.help) {
    io.log(usage().trimEnd())
    return 0
  }

  const report = doctorSkills(options.skillsDir ? { skillsDir: options.skillsDir } : {})

  io.log(`skills/: ${report.skillsDir}`)
  io.log(`${mark(report.uv.found)} uv${report.uv.found ? ` — ${report.uv.output}` : ' not found on PATH (required to bootstrap the shared venv)'}`)
  io.log(`${mark(report.git.found)} git${report.git.found ? ` — ${report.git.output}` : ' not found on PATH (required to clone the BiomedParse model repo)'}`)
  io.log(`${mark(report.python3.found)} python3${report.python3.found ? ` — ${report.python3.output}` : ' not found on PATH'}`)
  io.log('')
  io.log(`${mark(report.venvExists)} shared venv${report.venvExists ? '' : ' — not created yet (created automatically on the first tool call)'}`)
  if (report.constraintsPin) io.log(`    torch pin: ${report.constraintsPin}`)
  if (report.torch) io.log(`    torch ${report.torch.torch}, CUDA available: ${report.torch.cuda}`)
  io.log('')
  io.log(`${mark(report.biomedparseRepoCloned)} BiomedParse repo cloned`)
  io.log(`${mark(report.biomedparseDepsInstalled)} BiomedParse extra dependencies installed`)
  io.log(`${mark(report.biomedparseDetectron2Installed)} detectron2 built`)
  io.log(`${mark(report.biomedparseWeightsBytes !== undefined)} BiomedParse weights downloaded${report.biomedparseWeightsBytes !== undefined ? ` — ${formatBytes(report.biomedparseWeightsBytes)}` : ''}`)
  if (!report.biomedparseRepoCloned) {
    io.log('  (BiomedParse setup runs automatically the first time a _segmentation_biomedparse tool is called — nothing above being unchecked is an error by itself.)')
  }

  return report.ok ? 0 : 1
}

// Compare real paths, not raw ones: `bin` entries are typically reached
// through a symlink (npm's node_modules/.bin, or this package's own
// `link:`-installed dev checkout), and Node's ESM loader resolves
// import.meta.url to the symlink's real target — a raw pathToFileURL(argv[1])
// comparison would never match through one and the CLI would silently no-op.
const entry = process.argv[1]
if (entry) {
  try {
    if (import.meta.url === pathToFileURL(realpathSync(entry)).href) {
      process.exitCode = run()
    }
  } catch {
    // entry doesn't resolve on disk: not a real self-invocation, skip.
  }
}
