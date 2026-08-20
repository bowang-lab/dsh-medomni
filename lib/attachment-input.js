/**
 * Resolve a tool's image argument to a filesystem path, whether it was given
 * as a plain path or as the durable attachment id of an image pasted into the
 * conversation. Mirrors how dsh-vision-router's pixel tools accept both
 * ("sha256:..." ids resolve through the session's recorded upload index), so
 * the model can feed a pasted image straight to a dsh-medomni script without
 * hunting for the content-addressed file on disk.
 * @module dsh-medomni/lib/attachment-input
 */

import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

/**
 * True when the string is a durable attachment id such as "sha256:<hex>" —
 * the form the harness uses for uploaded images and that image-aware routes
 * (e.g. a vision-router auto-vision group) announce in the model prompt.
 */
export function isAttachmentIdInput(input) {
  return typeof input === 'string' && /^[a-z0-9]+:[0-9a-f]{32,}$/i.test(input.trim())
}

/** Extension for the raster media types the attachment service stores. */
const MEDIA_TYPE_EXTENSION = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/webp': '.webp',
  'image/gif': '.gif',
}

/**
 * Collect every distinct image attachment ref from a session event log,
 * descending into nested tool-result content (e.g. the built-in read_image
 * tool's re-uploads) exactly like the harness's own image walk. Event shapes:
 * `user/message` carries the message directly; `assistant/message` and
 * `tool/result` nest it under `data.message`.
 *
 * @param {readonly unknown[] | undefined} events - the session event log
 *   (`session.events`), or any array shaped like it.
 * @returns {Array<{ attachmentId: unknown, mediaType?: unknown }>} distinct
 *   refs in first-seen order.
 */
export function collectEventAttachmentRefs(events) {
  const refs = []
  const seen = new Set()
  const walk = (content) => {
    if (!Array.isArray(content)) return
    for (const block of content) {
      if (!block || typeof block !== 'object') continue
      const attachment = block && block.attachment
      if (attachment && attachment.attachmentId && !seen.has(String(attachment.attachmentId))) {
        seen.add(String(attachment.attachmentId))
        refs.push(attachment)
      }
      if (Array.isArray(block.content)) walk(block.content)
    }
  }
  for (const event of events ?? []) {
    if (!event || !event.data) continue
    let message
    if (event.type === 'user/message') {
      message = event.data
    } else if (event.type === 'assistant/message' || event.type === 'tool/result') {
      message = event.data.message
    } else {
      continue
    }
    if (message && Array.isArray(message.content)) walk(message.content)
  }
  return refs
}

/**
 * Resolve an attachment id to the full reference recorded in the session
 * event log. The id alone is not enough for `attachments.readImage`, which
 * verifies stored bytes against the ref's recorded metadata (mediaType,
 * bytes, width, height), so the full ref must come from the conversation's
 * own history.
 *
 * @returns the recorded ref, or undefined when the id is unknown or the
 *   session/event log is unavailable.
 */
export function lookupAttachmentRef(session, id) {
  if (!session) return undefined
  let events
  try {
    events = session.events
  } catch {
    return undefined // not a host Session (or the getter is unavailable)
  }
  if (!Array.isArray(events) || events.length === 0) return undefined
  const target = String(id).trim()
  return collectEventAttachmentRefs(events).find(ref => String(ref.attachmentId) === target)
}

/**
 * Turn a tool's image argument into a filesystem path the python scripts can
 * read. Plain paths pass through unchanged; a durable attachment id (the form
 * a pasted image reaches the model as) is resolved through the session event
 * log and the attachment service, then materialized to a temp file so the
 * child script can open it like any other input path.
 *
 * @param {import('@deepseek-ai/cordis').Context} ctx - the plugin context
 *   (used for the `attachments` service).
 * @param {{ agent?: { session?: unknown }, signal?: AbortSignal } | undefined} exec - the tool-execution context.
 * @param {string} input - the raw `input`/`lateral`/`prior` argument.
 * @returns {Promise<string>} a filesystem path.
 * @throws a descriptive error for an unknown id, a missing attachment
 *   service, a non-raster attachment, or a storage read failure.
 */
export async function resolveImageInput(ctx, exec, input) {
  const value = String(input ?? '').trim()
  if (!isAttachmentIdInput(value)) return input

  const attachments = ctx.get('attachments')
  if (attachments === undefined) {
    throw new Error('cannot use attachment "' + value + '" as input: no attachment service is mounted in this profile')
  }
  const session = exec && exec.agent && exec.agent.session
  const ref = lookupAttachmentRef(session, value)
  if (ref === undefined) {
    throw new Error(
      'unknown attachment id "' + value + '": it must be an image pasted or uploaded in this conversation. '
      + 'Pass a filesystem path instead if the image lives on disk.',
    )
  }
  let stored
  try {
    stored = await attachments.readImage(ref, exec && exec.signal)
  } catch (error) {
    throw new Error('failed to read attachment "' + value + '" (' + (error && error.message ? error.message : String(error)) + ')')
  }
  const mediaType = stored && stored.ref && stored.ref.mediaType
  const extension = MEDIA_TYPE_EXTENSION[mediaType]
  if (extension === undefined) {
    throw new Error('attachment "' + value + '" is not a raster image (' + (mediaType ?? 'unknown') + '); only PNG/JPG/WebP/GIF can be used as a model input')
  }
  const dir = await mkdtemp(join(tmpdir(), 'dsh-medomni-pasted-'))
  const target = join(dir, 'pasted-image' + extension)
  await writeFile(target, stored.data)
  return target
}
