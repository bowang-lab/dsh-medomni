#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { spawn, spawnSync } from 'node:child_process'
import process from 'node:process'

const DEFAULT_SESSIONS_ROOT = '/home/mwei26/.dsh/sessions'
const DEFAULT_OUTPUT_DIR = '/tmp'
const INTERNAL_USER_SOURCE_KINDS = new Set(['plugin', 'skill-catalog'])
const INTERNAL_ASSISTANT_BLOCK_TYPES = new Set(['reasoning'])
const AUTO_COPY_EXTENSIONS = new Set([
  '.bmp',
  '.csv',
  '.gif',
  '.html',
  '.jpeg',
  '.jpg',
  '.json',
  '.md',
  '.nii',
  '.pdf',
  '.png',
  '.svg',
  '.tif',
  '.tiff',
  '.txt',
  '.webp',
])

function usage() {
  return `Usage:
  node scripts/export-dsh-pantheon-replay.mjs --session <session.jsonl.zstd|session-dir> [options]
  node scripts/export-dsh-pantheon-replay.mjs --latest /home/mwei26/codebase/dsh-medomni [options]

Options:
  --name <title>          Replay title. Default: DSH session title or cwd basename.
  --out <dir>             Directory for the ZIP. Default: ${DEFAULT_OUTPUT_DIR}
  --workdir <dir>         Workspace root used to resolve relative artifact paths.
  --include <glob>        Copy matching workspace files into files/. Repeatable.
                          Supported: exact path, prefix/**, *.ext, **/*.ext.
  --keep-staging          Keep the unzipped staging directory next to the ZIP.
  --help                  Show this help.

Examples:
  node scripts/export-dsh-pantheon-replay.mjs --latest /home/mwei26/codebase/dsh-medomni --include 'dsh-medomni/previews/**'
  node scripts/export-dsh-pantheon-replay.mjs --session ~/.dsh/sessions/--home-mwei26-codebase-dsh-medomni--/session-bac.../session.jsonl.zstd
`
}

function parseArgs(argv) {
  const opts = { includes: [] }
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg === '--help' || arg === '-h') opts.help = true
    else if (arg === '--session') opts.session = argv[++i]
    else if (arg === '--latest') opts.latest = argv[++i]
    else if (arg === '--name') opts.name = argv[++i]
    else if (arg === '--out') opts.out = argv[++i]
    else if (arg === '--workdir') opts.workdir = argv[++i]
    else if (arg === '--include') opts.includes.push(argv[++i])
    else if (arg === '--keep-staging') opts.keepStaging = true
    else throw new Error(`Unknown argument: ${arg}`)
  }
  if (opts.help) return opts
  if ((opts.session === undefined) === (opts.latest === undefined)) {
    throw new Error('Pass exactly one of --session or --latest.')
  }
  return opts
}

function expandHome(path) {
  if (path === undefined) return undefined
  if (path === '~') return process.env.HOME ?? path
  return path.startsWith('~/') ? join(process.env.HOME ?? '~', path.slice(2)) : path
}

function encodeWorkspace(workspace) {
  return `--${workspace.replace(/^\/+|\/+$/g, '').replaceAll('/', '-')}--`
}

async function findLatestSession(workspace) {
  const encoded = encodeWorkspace(resolve(workspace))
  const dir = join(DEFAULT_SESSIONS_ROOT, encoded)
  const entries = await readdir(dir, { withFileTypes: true })
  const candidates = []
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const sessionPath = join(dir, entry.name, 'session.jsonl.zstd')
    try {
      const info = await stat(sessionPath)
      candidates.push({ sessionPath, mtimeMs: info.mtimeMs })
    } catch {
      // A partial session directory is ignored.
    }
  }
  candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)
  if (candidates.length === 0) throw new Error(`No DSH sessions found under ${dir}`)
  return candidates[0].sessionPath
}

async function resolveSessionPath(input) {
  const path = resolve(expandHome(input))
  const info = await stat(path)
  if (info.isDirectory()) return join(path, 'session.jsonl.zstd')
  return path
}

async function readSessionEvents(sessionPath) {
  if (sessionPath.endsWith('.zstd') || sessionPath.endsWith('.zst')) {
    const result = spawnSync('zstd', ['-dc', sessionPath], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 1024 })
    if (result.status !== 0) throw new Error(result.stderr || `zstd exited ${result.status}`)
    return result.stdout.split('\n').filter(Boolean).map(line => JSON.parse(line))
  }
  const text = await readFile(sessionPath, 'utf8')
  return text.split('\n').filter(Boolean).map(line => JSON.parse(line))
}

function textFromContent(content) {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .filter(block => block?.type === 'text' && typeof block.text === 'string')
    .map(block => block.text)
    .join('\n\n')
}

function visibleAssistantText(content) {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .filter(block => !INTERNAL_ASSISTANT_BLOCK_TYPES.has(block?.type))
    .filter(block => block?.type === 'text' && typeof block.text === 'string')
    .map(block => block.text)
    .join('\n\n')
}

function assistantToolCalls(content) {
  if (!Array.isArray(content)) return []
  return content
    .filter(block => block?.type === 'tool-call')
    .map(block => ({
      id: block.id,
      type: 'function',
      function: {
        name: block.name,
        arguments: typeof block.arguments === 'string' ? block.arguments : JSON.stringify(block.arguments ?? {}),
      },
    }))
}

function attachmentPath(attachmentId) {
  if (typeof attachmentId !== 'string' || !attachmentId.startsWith('sha256:')) return undefined
  const digest = attachmentId.slice('sha256:'.length)
  return join(process.env.HOME ?? '/home/mwei26', '.dsh/attachments/v1/objects', digest.slice(0, 2), digest)
}

function pantheonContent(content) {
  if (!Array.isArray(content)) return content
  return content
    .map(block => {
      if (block?.type !== 'image') return block
      const path = attachmentPath(block.attachment?.attachmentId)
      if (path === undefined) return undefined
      return {
        type: 'image_url',
        image_url: { url: path },
      }
    })
    .filter(block => block !== undefined)
}

function toolResultsFromMessage(message) {
  if (!Array.isArray(message?.content)) return []
  return message.content
    .filter(block => block?.type === 'tool-result')
    .map(block => {
      const parts = Array.isArray(block.content) ? block.content : []
      const text = parts
        .filter(part => part?.type === 'text' && typeof part.text === 'string')
        .map(part => part.text)
        .join('\n\n')
      let rawContent = text
      try {
        rawContent = JSON.parse(text)
      } catch {
        // Plain text tool output is valid replay content.
      }
      return {
        role: 'tool',
        tool_name: undefined,
        name: undefined,
        tool_call_id: block.toolCallId,
        raw_content: rawContent,
        content: text,
        is_error: Boolean(block.isError),
      }
    })
}

function seconds(ms) {
  return Math.round(ms / 1000) / 1000
}

function projectEvents(events, chatId) {
  const chat = []
  let session = {}
  let title
  let requestModel
  const toolNames = new Map()

  for (const event of events) {
    if (event.type === 'session') {
      session = event
      continue
    }
    if (event.type === 'session/title' && typeof event.data?.title === 'string') {
      title = event.data.title
      continue
    }
    if (event.type === 'request/context') {
      requestModel = event.data
      continue
    }
    if (event.type === 'tool/call') {
      toolNames.set(event.data?.callId, event.data?.name)
      continue
    }
    if (event.type === 'user/message') {
      const sourceKind = event.data?.source?.kind
      if (sourceKind !== 'user') continue
      if (INTERNAL_USER_SOURCE_KINDS.has(sourceKind)) continue
      const text = textFromContent(event.data?.content)
      if (text.trim() === '') continue
      chat.push({
        role: 'user',
        content: pantheonContent(event.data.content),
        chat_id: '',
        id: event.data.id ?? randomUUID(),
      })
      continue
    }
    if (event.type === 'assistant/message') {
      const message = event.data?.message
      if (message?.role !== 'assistant') continue
      const content = visibleAssistantText(message.content)
      const toolCalls = assistantToolCalls(message.content)
      if (content.trim() === '' && toolCalls.length === 0) continue
      const record = {
        role: 'assistant',
        content: content.trim() === '' ? null : content,
        id: message.id ?? randomUUID(),
        timestamp: seconds(event.time),
        agent_name: 'Leader',
      }
      if (toolCalls.length > 0) record.tool_calls = toolCalls
      if (event.data.usage !== undefined) {
        record._metadata = {
          total_tokens: event.data.usage.inputTokens + event.data.usage.outputTokens,
          input_tokens: event.data.usage.inputTokens,
          output_tokens: event.data.usage.outputTokens,
        }
      }
      chat.push(record)
      continue
    }
    if (event.type === 'tool/result') {
      const message = event.data?.message
      for (const result of toolResultsFromMessage(message)) {
        const name = toolNames.get(result.tool_call_id) ?? result.tool_call_id ?? 'tool'
        result.tool_name = name
        result.name = name
        result.id = randomUUID()
        result.timestamp = seconds(event.time)
        result.agent_name = 'Leader'
        chat.push(result)
      }
    }
  }

  return {
    chat,
    meta: {
      id: chatId,
      name: title ?? basename(session.cwd ?? 'dsh-medomni replay'),
      extra_data: {
        last_activity_date: new Date(Math.max(...events.map(event => event.time ?? session.createdAt ?? 0))).toISOString(),
        project: {
          workspace_mode: 'project',
          workspace_path: session.cwd,
        },
        team_template: {
          id: 'dsh',
          name: 'DeepSeek Harness',
          description: 'Imported DeepSeek Harness session',
          icon: 'DSH',
          category: 'imported',
          version: '1.0.0',
          agents: [
            {
              id: 'leader',
              name: 'Leader',
              model: requestModel?.model ?? 'unknown',
              description: 'Imported DSH assistant',
              icon: 'DSH',
              instructions: '',
              toolsets: [],
              tags: [],
              source_path: '',
              scope: 'project',
            },
          ],
        },
        source: {
          kind: 'deepseek-harness',
          session_id: session.id,
          cwd: session.cwd,
          agent_preset: session.agentPreset,
          request_model: requestModel,
        },
      },
    },
    session,
  }
}

function sanitizeSlug(name) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60) || 'dsh-replay'
}

function toLocalArchivePath(original) {
  const normalized = original.replaceAll('\\', '/').replace(/^\/+/, '')
  return `files/${normalized}`
}

function extensionOf(path) {
  const name = basename(path).toLowerCase()
  if (name.endsWith('.nii.gz')) return '.nii'
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot)
}

function shouldAutoCopyPath(path) {
  const normalized = path.replaceAll('\\', '/')
  if (normalized.includes('/.dsh/attachments/v1/objects/')) return true
  if (normalized.includes('/.dsh/sessions/')) return false
  if (normalized.includes('/node_modules/')) return false
  if (normalized.includes('/.git/')) return false
  if (normalized.includes('/.venv/')) return false
  if (normalized.includes('/skills/.venv/')) return false
  if (normalized.includes('/dsh-medomni/previews/')) return true
  if (normalized.includes('/outputs/')) return true
  if (normalized.includes('/artifacts/')) return true
  return AUTO_COPY_EXTENSIONS.has(extensionOf(normalized))
}

function shouldAutoCopyToolPath(path) {
  const normalized = path.replaceAll('\\', '/')
  return normalized.includes('/.dsh/attachments/v1/objects/')
    || normalized.includes('/dsh-medomni/previews/')
    || normalized.includes('/outputs/')
    || normalized.includes('/artifacts/')
}

async function copyArtifact(rootDir, original, bundleRoot, manifestFiles, copied) {
  const absolute = isAbsolute(original) ? original : resolve(rootDir, original)
  if (copied.has(absolute)) return
  let info
  try {
    info = await stat(absolute)
  } catch {
    return
  }
  if (!info.isFile()) return
  const local = toLocalArchivePath(absolute)
  const dest = join(bundleRoot, local)
  await mkdir(dirname(dest), { recursive: true })
  await cp(absolute, dest)
  manifestFiles.push({ original: absolute, local, size: info.size })
  copied.add(absolute)
}

function matchInclude(rel, pattern) {
  const normalized = rel.split(sep).join('/')
  if (pattern.endsWith('/**')) return normalized.startsWith(pattern.slice(0, -3).replace(/\/$/, '') + '/')
  if (pattern.startsWith('**/*.')) return normalized.endsWith(pattern.slice(4))
  if (pattern.startsWith('*.')) return basename(normalized).endsWith(pattern.slice(1))
  return normalized === pattern || normalized.startsWith(pattern.replace(/\/$/, '') + '/')
}

async function walkFiles(dir) {
  const out = []
  async function walk(current) {
    let entries
    try {
      entries = await readdir(current, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      if (entry.name === 'node_modules' || entry.name === '.git' || entry.name === '.venv') continue
      const path = join(current, entry.name)
      if (entry.isDirectory()) await walk(path)
      else if (entry.isFile()) out.push(path)
    }
  }
  await walk(dir)
  return out
}

async function collectIncludedArtifacts(workdir, includes, bundleRoot, manifestFiles, copied) {
  if (includes.length === 0) return
  const files = await walkFiles(workdir)
  for (const file of files) {
    const rel = relative(workdir, file)
    if (includes.some(pattern => matchInclude(rel, pattern))) {
      await copyArtifact(workdir, file, bundleRoot, manifestFiles, copied)
    }
  }
}

function collectReferencedPaths(chat, workdir) {
  const paths = new Set()
  const absolutePattern = /(?:^|[\s"'`(])((?:\/home\/|\/tmp\/)[^\s"'`)]+)/g
  const relativePattern = /(?:^|[\s"'`(])((?:\.?dsh-medomni\/previews\/|\.?outputs\/|\.?assets\/)[^\s"'`)]+)/g
  const addCandidate = (path, record) => {
    if (record.role === 'tool') {
      if (shouldAutoCopyToolPath(path)) paths.add(path)
      return
    }
    if (shouldAutoCopyPath(path)) paths.add(path)
  }
  for (const record of chat) {
    if (Array.isArray(record.content)) {
      for (const block of record.content) {
        const attachmentId = block?.attachment?.attachmentId
        if (typeof attachmentId === 'string' && attachmentId.startsWith('sha256:')) {
          const digest = attachmentId.slice('sha256:'.length)
          paths.add(join(process.env.HOME ?? '/home/mwei26', '.dsh/attachments/v1/objects', digest.slice(0, 2), digest))
        }
      }
    }
    const text = typeof record.content === 'string' ? record.content : JSON.stringify(record.content ?? '')
    for (const match of text.matchAll(absolutePattern)) addCandidate(match[1], record)
    for (const match of text.matchAll(relativePattern)) addCandidate(resolve(workdir, match[1].replace(/^\.\//, '')), record)
    const raw = JSON.stringify(record.raw_content ?? '')
    for (const match of raw.matchAll(absolutePattern)) addCandidate(match[1], record)
    for (const match of raw.matchAll(relativePattern)) addCandidate(resolve(workdir, match[1].replace(/^\.\//, '')), record)
  }
  return [...paths]
}

function jsonlText(records) {
  return records.map(record => JSON.stringify(record)).join('\n') + '\n'
}

function rewriteBundledPaths(text, manifestFiles) {
  const rewrites = manifestFiles
    .map(file => [file.original, `./${file.local}`])
    .sort((a, b) => b[0].length - a[0].length)
  let rewritten = text
  for (const [original, local] of rewrites) {
    rewritten = rewritten.split(original).join(local)
    rewritten = rewritten.split(`file://${original}`).join(local)
  }
  return rewritten
}

async function sha256(path) {
  const hash = createHash('sha256')
  await pipeline(createReadStream(path), hash)
  return hash.digest('hex')
}

async function zipDirectory(sourceDir, zipPath) {
  await rm(zipPath, { force: true })
  const child = spawn('zip', ['-qr', zipPath, basename(sourceDir)], { cwd: dirname(sourceDir), stdio: 'inherit' })
  const code = await new Promise((resolveCode, reject) => {
    child.on('error', reject)
    child.on('close', resolveCode)
  })
  if (code !== 0) throw new Error(`zip exited ${code}`)
}

async function main() {
  const opts = parseArgs(process.argv.slice(2))
  if (opts.help) {
    process.stdout.write(usage())
    return
  }

  const sessionPath = opts.latest !== undefined
    ? await findLatestSession(expandHome(opts.latest))
    : await resolveSessionPath(opts.session)
  const events = await readSessionEvents(sessionPath)
  const chatId = randomUUID()
  const projected = projectEvents(events, chatId)
  const name = opts.name ?? projected.meta.name
  projected.meta.name = name
  const workdir = resolve(expandHome(opts.workdir) ?? projected.session.cwd ?? process.cwd())
  const outDir = resolve(expandHome(opts.out) ?? DEFAULT_OUTPUT_DIR)
  const slug = sanitizeSlug(name)
  const stagingRoot = join(outDir, `${slug}-${chatId}`)
  const bundleRoot = join(stagingRoot, chatId)
  const zipPath = join(outDir, `${slug}.zip`)

  await rm(stagingRoot, { recursive: true, force: true })
  await mkdir(bundleRoot, { recursive: true })
  await mkdir(join(bundleRoot, 'files'), { recursive: true })

  const manifestFiles = []
  const copied = new Set()
  for (const path of collectReferencedPaths(projected.chat, workdir)) {
    await copyArtifact(workdir, path, bundleRoot, manifestFiles, copied)
  }
  await collectIncludedArtifacts(workdir, opts.includes, bundleRoot, manifestFiles, copied)

  const chatText = rewriteBundledPaths(jsonlText(projected.chat), manifestFiles)
  const metaText = rewriteBundledPaths(JSON.stringify(projected.meta, null, 2) + '\n', manifestFiles)
  await writeFile(join(bundleRoot, 'chat.jsonl'), chatText)
  await writeFile(join(bundleRoot, 'chat.meta.json'), metaText)
  await writeFile(join(bundleRoot, 'manifest.json'), JSON.stringify({
    version: '1.0',
    chat_id: chatId,
    chat_name: name,
    exported_at: new Date().toISOString(),
    files: manifestFiles.sort((a, b) => a.local.localeCompare(b.local)),
  }, null, 2) + '\n')

  await zipDirectory(bundleRoot, zipPath)
  const digest = await sha256(zipPath)
  if (!opts.keepStaging) await rm(stagingRoot, { recursive: true, force: true })

  process.stdout.write(JSON.stringify({
    zip: zipPath,
    sha256: digest,
    chat_id: chatId,
    messages: projected.chat.length,
    files: manifestFiles.length,
    session: sessionPath,
    upload_hint: `huggingface-cli upload <you>/<dataset> ${zipPath}`,
    replay_url_hint: `https://pantheon-ui.aristoteleo.com/#/replay?url=${encodeURIComponent('https://huggingface.co/datasets/<you>/<dataset>/resolve/main/' + basename(zipPath))}`,
  }, null, 2) + '\n')
}

main().catch(error => {
  process.stderr.write(`${error.stack ?? error.message}\n`)
  process.exitCode = 1
})
