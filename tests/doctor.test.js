import { mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'
import test from 'node:test'
import assert from 'node:assert/strict'
import { doctorSkills } from '../lib/doctor.js'
import { run as runDoctorCli } from '../lib/doctor-cli.js'

const DOCTOR_CLI_PATH = fileURLToPath(new URL('../lib/doctor-cli.js', import.meta.url))

function fakeSpawn({ commandVersions = {}, torch } = {}) {
  return (bin, args) => {
    if (args[0] === '--version') {
      const output = commandVersions[bin]
      return output === undefined ? { status: 1 } : { status: 0, stdout: output }
    }
    if (typeof args[1] === 'string' && args[1].includes('import json,torch')) {
      return torch === undefined ? { status: 1 } : { status: 0, stdout: JSON.stringify(torch) }
    }
    return { status: 1 }
  }
}

test('doctorSkills: reports missing tools and an unbootstrapped skills dir as not ok', () => {
  const skillsDir = mkdtempSync(path.join(tmpdir(), 'dsh-medomni-doctor-'))
  try {
    const report = doctorSkills({ skillsDir, spawn: fakeSpawn() })
    assert.equal(report.ok, false)
    assert.equal(report.uv.found, false)
    assert.equal(report.venvExists, false)
    assert.equal(report.biomedparseRepoCloned, false)
    assert.equal(report.biomedparseWeightsBytes, undefined)
  } finally {
    rmSync(skillsDir, { recursive: true, force: true })
  }
})

test('doctorSkills: detects a bootstrapped venv, pinned torch, and completed BiomedParse setup', () => {
  const skillsDir = mkdtempSync(path.join(tmpdir(), 'dsh-medomni-doctor-'))
  try {
    const venvDir = path.join(skillsDir, '.venv')
    mkdirSync(path.join(venvDir, 'bin'), { recursive: true })
    writeFileSync(path.join(venvDir, 'bin', 'python'), '')
    writeFileSync(path.join(skillsDir, '.uv-constraints.txt'), 'torch==2.10.0\n')
    writeFileSync(path.join(venvDir, '.installed-biomedparse-deps'), 'ok\n')
    writeFileSync(path.join(venvDir, '.installed-biomedparse-detectron2'), 'ok\n')

    const repoDir = path.join(skillsDir, 'biomedparse', 'BiomedParse')
    mkdirSync(path.join(repoDir, 'modeling'), { recursive: true })
    mkdirSync(path.join(repoDir, 'inference_utils'), { recursive: true })
    writeFileSync(path.join(repoDir, 'modeling', 'BaseModel.py'), '')
    writeFileSync(path.join(repoDir, 'inference_utils', 'processing_utils.py'), '')

    const weightsDir = path.join(skillsDir, 'biomedparse', 'weights')
    mkdirSync(weightsDir, { recursive: true })
    writeFileSync(path.join(weightsDir, 'biomedparse_v1.pt'), Buffer.alloc(1024))

    const report = doctorSkills({
      skillsDir,
      spawn: fakeSpawn({
        commandVersions: { uv: 'uv 0.10.7', git: 'git version 2.34.1', python3: 'Python 3.12.12' },
        torch: { torch: '2.10.0+cu128', cuda: true },
      }),
    })

    assert.equal(report.ok, true)
    assert.equal(report.venvExists, true)
    assert.equal(report.constraintsPin, 'torch==2.10.0')
    assert.deepEqual(report.torch, { torch: '2.10.0+cu128', cuda: true })
    assert.equal(report.biomedparseRepoCloned, true)
    assert.equal(report.biomedparseDepsInstalled, true)
    assert.equal(report.biomedparseDetectron2Installed, true)
    assert.equal(report.biomedparseWeightsBytes, 1024)
  } finally {
    rmSync(skillsDir, { recursive: true, force: true })
  }
})

test('doctor-cli: --help prints usage and exits 0 without touching the filesystem', () => {
  const logs = []
  const exitCode = runDoctorCli(['--help'], { log: (line) => logs.push(line), error: () => {} })
  assert.equal(exitCode, 0)
  assert.match(logs.join('\n'), /dsh-medomni doctor/)
})

test('doctor-cli: an unknown argument exits 2 with an error line', () => {
  const errors = []
  const exitCode = runDoctorCli(['--bogus'], { log: () => {}, error: (line) => errors.push(line) })
  assert.equal(exitCode, 2)
  assert.match(errors.join('\n'), /unknown argument: --bogus/)
})

test('doctor-cli: still runs when invoked through a symlink (e.g. a pnpm bin link)', () => {
  // Regression test: the self-invocation guard compared process.argv[1]'s
  // raw path against import.meta.url, which Node resolves through symlinks —
  // a bin symlink (exactly how `dsh plugin add` / pnpm install this package)
  // made the guard's paths never match, so the CLI silently produced no
  // output and exited 0. This spawns the real file through a symlink, the
  // same way a package manager's node_modules/.bin entry would.
  const dir = mkdtempSync(path.join(tmpdir(), 'dsh-medomni-doctor-symlink-'))
  try {
    const linkPath = path.join(dir, 'doctor-cli-link.js')
    symlinkSync(DOCTOR_CLI_PATH, linkPath)
    const output = execFileSync(process.execPath, [linkPath, '--help'], { encoding: 'utf8' })
    assert.match(output, /dsh-medomni doctor/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
