import { readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  collectEventAttachmentRefs,
  isAttachmentIdInput,
  lookupAttachmentRef,
  resolveImageInput,
} from '../lib/attachment-input.js'

const ID_A = 'sha256:' + 'a'.repeat(64)
const ID_B = 'sha256:' + 'b'.repeat(64)
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3])

/** Minimal durable attachment ref shaped like the session log records. */
function ref(id, mediaType = 'image/png', bytes = 42) {
  return { attachmentId: id, mediaType, bytes, width: 10, height: 10 }
}

test('isAttachmentIdInput: recognizes durable attachment ids, rejects paths', () => {
  assert.equal(isAttachmentIdInput('sha256:' + 'a'.repeat(64)), true)
  assert.equal(isAttachmentIdInput('  sha256:' + 'b'.repeat(64) + '  '), true)
  assert.equal(isAttachmentIdInput('/path/to/xray.png'), false)
  assert.equal(isAttachmentIdInput('xray.png'), false)
  assert.equal(isAttachmentIdInput(''), false)
  assert.equal(isAttachmentIdInput(undefined), false)
  assert.equal(isAttachmentIdInput(42), false)
})

test('collectEventAttachmentRefs: collects refs across event shapes and dedupes', () => {
  const events = [
    { type: 'turn/start', data: {} },
    { type: 'user/message', data: { content: [{ type: 'image', attachment: ref(ID_A) }] } },
    { type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'hi' }] } } },
    {
      type: 'tool/result',
      data: {
        message: {
          content: [{ type: 'tool-result', content: [{ type: 'image', attachment: ref(ID_B) }] }],
        },
      },
    },
    // Same id again — must be deduped.
    { type: 'user/message', data: { content: [{ type: 'image', attachment: ref(ID_A) }] } },
  ]
  const refs = collectEventAttachmentRefs(events)
  assert.deepEqual(refs.map(r => r.attachmentId), [ID_A, ID_B])
  assert.equal(collectEventAttachmentRefs(undefined).length, 0)
  assert.equal(collectEventAttachmentRefs([]).length, 0)
})

test('lookupAttachmentRef: finds a recorded id, misses unknown ids and unusable sessions', () => {
  const session = {
    events: [{ type: 'user/message', data: { content: [{ type: 'image', attachment: ref(ID_A, 'image/jpeg') }] } }],
  }
  const hit = lookupAttachmentRef(session, ID_A)
  assert.ok(hit)
  assert.equal(hit.mediaType, 'image/jpeg')
  assert.equal(lookupAttachmentRef(session, 'sha256:' + 'c'.repeat(64)), undefined)
  assert.equal(lookupAttachmentRef(undefined, ID_A), undefined)
  assert.equal(lookupAttachmentRef({}, ID_A), undefined)
  // A session whose events getter throws is treated as unresolvable, never a crash.
  const throwing = { get events() { throw new Error('nope') } }
  assert.equal(lookupAttachmentRef(throwing, ID_A), undefined)
})

test('resolveImageInput: plain paths pass through unchanged', async () => {
  const ctx = { get: () => undefined }
  assert.equal(await resolveImageInput(ctx, undefined, '/data/img.png'), '/data/img.png')
  assert.equal(await resolveImageInput(ctx, undefined, ''), '')
})

test('resolveImageInput: materializes a pasted image to a temp file', async () => {
  const exec = {
    agent: { session: { events: [{ type: 'user/message', data: { content: [{ type: 'image', attachment: ref(ID_A) }] } }] } },
    signal: undefined,
  }
  const attachments = { readImage: async (r) => ({ ref: r, data: PNG }) }
  const ctx = { get: (name) => (name === 'attachments' ? attachments : undefined) }
  const target = await resolveImageInput(ctx, exec, ID_A)
  assert.ok(target.startsWith(path.join(tmpdir(), 'dsh-medomni-pasted-')))
  assert.ok(target.endsWith('.png'))
  const bytes = new Uint8Array(await readFile(target))
  assert.deepEqual(bytes, PNG)
})

test('resolveImageInput: unknown attachment id throws a descriptive error', async () => {
  const ctx = { get: () => ({ readImage: async () => { throw new Error('must not be called') } }) }
  const exec = { agent: { session: { events: [] } } }
  await assert.rejects(
    resolveImageInput(ctx, exec, 'sha256:' + 'd'.repeat(64)),
    /unknown attachment id/,
  )
})

test('resolveImageInput: missing attachment service throws', async () => {
  const ctx = { get: () => undefined }
  const exec = { agent: { session: { events: [{ type: 'user/message', data: { content: [{ type: 'image', attachment: ref(ID_A) }] } }] } } }
  await assert.rejects(resolveImageInput(ctx, exec, ID_A), /no attachment service/)
})

test('resolveImageInput: non-raster attachment throws', async () => {
  const id = ID_A
  const exec = {
    agent: { session: { events: [{ type: 'user/message', data: { content: [{ type: 'image', attachment: ref(id, 'image/avif') }] } }] } },
  }
  const attachments = { readImage: async (r) => ({ ref: { ...r, mediaType: 'image/avif' }, data: PNG }) }
  const ctx = { get: (name) => (name === 'attachments' ? attachments : undefined) }
  await assert.rejects(resolveImageInput(ctx, exec, id), /not a raster image/)
})

test('resolveImageInput: storage read failure is wrapped with the id', async () => {
  const id = ID_A
  const exec = { agent: { session: { events: [{ type: 'user/message', data: { content: [{ type: 'image', attachment: ref(id) }] } }] } } }
  const attachments = { readImage: async () => { throw new Error('disk on fire') } }
  const ctx = { get: (name) => (name === 'attachments' ? attachments : undefined) }
  await assert.rejects(resolveImageInput(ctx, exec, id), /failed to read attachment .*disk on fire/)
})
