#!/usr/bin/env node

import { existsSync, realpathSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { doctorSkills } from './doctor.js'

const PACKAGE_DIR = fileURLToPath(new URL('..', import.meta.url))

function usage() {
  return `dsh-medomni setup [--skills-dir <path>]

Creates the shared Python environment used by dsh-medomni and installs the
common dependencies. Model checkpoints and BiomedParse extras stay lazy and
are downloaded only when their corresponding tool is first used.

Options:
  --skills-dir <path>  Prepare a custom skills/ directory.
  --help               Show this help.
`
}

function parseArgs(argv) {
  const args = [...argv]
  if (args[0] === 'setup') args.shift()
  let skillsDir
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index]
    if (value === '--help' || value === '-h') return { help: true }
    if (value === '--skills-dir') {
      skillsDir = args[++index]
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

export function run(argv = process.argv.slice(2), io = console, spawn = spawnSync) {
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

  const skillsDir = options.skillsDir || path.join(PACKAGE_DIR, 'skills')
  const report = doctorSkills({ skillsDir, spawn })
  const missing = ['uv', 'python3'].filter((name) => !report[name].found)
  if (missing.length) {
    io.error(`Missing required command(s): ${missing.join(', ')}`)
    io.error('Install uv from https://docs.astral.sh/uv/getting-started/installation/ and Python 3, then run this command again.')
    return 1
  }

  io.log(`Preparing dsh-medomni skills: ${skillsDir}`)
  if (!report.git.found) {
    io.log('Warning: git is not on PATH. BiomedParse will need it later for its first segmentation call.')
  }
  if (process.env.HF_TOKEN) {
    io.log('HF_TOKEN detected. Gated model access will be checked when a model is first used.')
  } else {
    io.log('HF_TOKEN not set. For MAIRA-2, MedGemma, or BiomedParse, run `hf auth login` or export a read token before inference.')
  }

  const bootstrap = path.join(skillsDir, '_bootstrap.py')
  if (!existsSync(bootstrap)) {
    io.error(`Cannot find bootstrap script: ${bootstrap}`)
    return 1
  }
  const result = spawn('python3', [bootstrap, '--setup'], { stdio: 'inherit' })
  if (!result || result.error || result.status !== 0) {
    io.error('Base environment setup failed.')
    return result?.status || 1
  }
  io.log('Setup complete. Run `dsh-medomni doctor` to inspect the environment.')
  return 0
}

const entry = process.argv[1]
if (entry) {
  try {
    if (import.meta.url === pathToFileURL(realpathSync(entry)).href) process.exitCode = run()
  } catch {
    // Ignore an unresolved entry path when this module is imported by tests.
  }
}
