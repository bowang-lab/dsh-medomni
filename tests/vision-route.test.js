import test from 'node:test'
import assert from 'node:assert/strict'
import {
  applyVisionRoutes,
  createImageCapableTwin,
  imageMarkerFor,
  rewriteImagesDeep,
} from '../lib/vision-route.js'

/** Source text/multimodal adapter shaped like a real LlmAdapter. */
function makeSourceAdapter({ name = 'Source', image = false } = {}) {
  return {
    providerInfo: (p) => ({ id: p, name }),
    providerRetryPolicy: () => undefined,
    listModels: async (p) => [
      { provider: p, id: 'model-a', name: 'Model A', inputModalities: image ? ['text', 'image'] : ['text'] },
      { provider: p, id: 'model-b', name: 'Model B' },
    ],
    resolveModel: async (p, m) => ({
      provider: p,
      id: m,
      name: m,
      ...(image ? { inputModalities: ['text', 'image'] } : {}),
    }),
    stream: async function* () { yield { type: 'finish', reason: { kind: 'stop' } } },
  }
}

/** Fake llm service with a live route registry, mirroring the real contracts. */
function makeLlm(initial) {
  const adapters = new Map()
  const regs = new Map()
  const streams = []
  for (const [name, adapter] of Object.entries(initial)) {
    adapters.set(name, adapter)
    regs.set(name, { adapter })
  }
  return {
    adapters,
    regs,
    streams,
    listProviders() {
      return [...adapters.keys()].map((id) => ({ id, name: id }))
    },
    registration(provider) {
      const reg = regs.get(provider)
      if (!reg) throw new Error('no adapter registered for provider "' + provider + '"')
      return reg
    },
    registerAdapter(providers, adapter) {
      for (const p of providers) {
        if (adapters.has(p)) throw new Error('an adapter for provider "' + p + '" is already registered')
      }
      for (const p of providers) {
        adapters.set(p, adapter)
        regs.set(p, { adapter })
      }
      const handle = () => {
        for (const p of providers) {
          if (adapters.get(p) === adapter) adapters.delete(p)
          if (regs.get(p) && regs.get(p).adapter === adapter) regs.delete(p)
        }
      }
      return handle
    },
    async listModels(provider) {
      const reg = regs.get(provider)
      return reg && typeof reg.adapter.listModels === 'function' ? reg.adapter.listModels(provider) : []
    },
    async resolveModelInfo(provider, model) {
      const reg = regs.get(provider)
      if (!reg || typeof reg.adapter.resolveModel !== 'function') return { provider, id: model, name: model }
      return reg.adapter.resolveModel(provider, model)
    },
    async *stream(options) {
      streams.push(options)
      yield { type: 'finish', reason: { kind: 'stop' } }
    },
  }
}

function makeCtx({ providers = {} } = {}) {
  const llm = makeLlm(providers)
  const events = new Map()
  const effects = []
  const ctx = {
    llm,
    logger: { warn() {} },
    on(event, handler) { events.set(event, handler) },
    effect(fn) { effects.push(fn()) },
  }
  return { ctx, llm, events, effects }
}

test('rewriteImagesDeep: rewrites top-level and nested image blocks, leaves text untouched', () => {
  const content = [
    { type: 'text', text: 'hi' },
    { type: 'image', attachment: { attachmentId: 'sha256:aaa' } },
    { type: 'tool-result', content: [{ type: 'image', attachment: { attachmentId: 'sha256:bbb' } }] },
  ]
  const result = rewriteImagesDeep(content, (block) =>
    [{ type: 'text', text: 'marker:' + block.attachment.attachmentId }],
  )
  assert.equal(result.changed, true)
  assert.deepEqual(result.content, [
    { type: 'text', text: 'hi' },
    { type: 'text', text: 'marker:sha256:aaa' },
    { type: 'tool-result', content: [{ type: 'text', text: 'marker:sha256:bbb' }] },
  ])
})

test('rewriteImagesDeep: unchanged input returns the original array and changed=false', () => {
  const content = [{ type: 'text', text: 'hi' }]
  const result = rewriteImagesDeep(content, () => undefined)
  assert.equal(result.changed, false)
  assert.equal(result.content, content)
  assert.equal(rewriteImagesDeep(undefined, () => undefined).changed, false)
})

test('rewriteImagesDeep: replace returning undefined drops the block', () => {
  const result = rewriteImagesDeep([{ type: 'image', attachment: {} }], () => undefined)
  assert.equal(result.changed, true)
  assert.deepEqual(result.content, [])
})

test('imageMarkerFor: carries the attachment id and display name, no vision-describe nudge', () => {
  const marker = imageMarkerFor('sha256:abc', 'us.png')
  assert.match(marker, /attachment id "sha256:abc"/)
  assert.match(marker, /us\.png/)
  assert.match(marker, /what abnormalities\/findings are present/)
  assert.match(marker, /call xray_grounded_report_maira/)
  assert.match(marker, /specific suspected finding, trauma\/fall/)
  assert.match(marker, /prefer xray_grounded_report_maira/)
  assert.match(marker, /always runs MAIRA-2 in grounded_report mode/)
  assert.match(marker, /copy the user clinical context\/question into the indication argument/)
  assert.match(marker, /xray_phrase_grounding_maira with phrase/)
  assert.match(marker, /xray_anatomy_localization only when the user explicitly asks/)
  assert.match(marker, /not as the only source for fracture-specific questions/)
  assert.match(marker, /Do not call xray_segmentation_biomedparse unless the user explicitly asks/)
  assert.doesNotMatch(marker, /\(xray_report_maira, xray_report_medgemma, xray_anatomy_localization/)
  assert.doesNotMatch(marker, /\(xray_report_maira, xray_report_medgemma, xray_longitudinal_comparison, xray_segmentation_biomedparse/)
  assert.match(marker, /prefer ultrasound_classify_then_segment/)
  assert.match(marker, /domain="auto"/)
  assert.match(marker, /Do not infer cyst vs solid or benign vs malignant status/)
  // the marker must not steer the model toward separate vision-describe/verify tools
  assert.doesNotMatch(marker, /vision_describe/)
  assert.doesNotMatch(marker, /verify any image/)
  assert.doesNotMatch(marker, /preview_image_paths/)
  assert.match(imageMarkerFor('sha256:abc', ''), /Image "image"/)
})

test('createImageCapableTwin: mirrors catalog and declares image input', async () => {
  const { ctx, llm } = makeCtx({ providers: { 'deepseek-official': makeSourceAdapter({ name: 'DeepSeek' }) } })
  const twin = createImageCapableTwin(ctx, 'deepseek-official', 'deepseek-official-dsh-medomni')
  const info = twin.providerInfo('deepseek-official-dsh-medomni')
  assert.equal(info.id, 'deepseek-official-dsh-medomni')
  assert.match(info.name, /DeepSeek \+ dsh-medomni Vision/)
  const models = await twin.listModels('deepseek-official-dsh-medomni')
  assert.equal(models[0].provider, 'deepseek-official-dsh-medomni')
  assert.deepEqual(models[0].inputModalities, ['text', 'image'])
  const resolved = await twin.resolveModel('deepseek-official-dsh-medomni', 'model-a')
  assert.equal(resolved.provider, 'deepseek-official-dsh-medomni')
  assert.equal(resolved.id, 'model-a')
  assert.deepEqual(resolved.inputModalities, ['text', 'image'])
})

test('createImageCapableTwin: stream rewrites images to markers and delegates to the source provider', async () => {
  const { ctx, llm } = makeCtx({ providers: { 'deepseek-official': makeSourceAdapter() } })
  const twin = createImageCapableTwin(ctx, 'deepseek-official', 'deepseek-official-dsh-medomni')
  const messages = [
    { role: 'user', content: [
      { type: 'text', text: 'hi' },
      { type: 'image', attachment: { attachmentId: 'sha256:aaa', name: 'us.png' } },
    ] },
    { role: 'assistant', content: [
      { type: 'tool-result', content: [{ type: 'image', attachment: { attachmentId: 'sha256:bbb' } }] },
    ] },
  ]
  const chunks = []
  for await (const chunk of twin.stream({ provider: 'deepseek-official-dsh-medomni', model: 'model-a', messages })) chunks.push(chunk)
  assert.equal(chunks.length, 1)
  assert.equal(llm.streams.length, 1)
  const delegated = llm.streams[0]
  assert.equal(delegated.provider, 'deepseek-official')
  assert.equal(delegated.model, 'model-a')
  const userContent = delegated.messages[0].content
  assert.equal(userContent[0].type, 'text')
  assert.equal(userContent[1].type, 'text')
  assert.match(userContent[1].text, /attachment id "sha256:aaa"/)
  assert.match(userContent[1].text, /us\.png/)
  const asstContent = delegated.messages[1].content
  assert.equal(asstContent[0].type, 'tool-result')
  assert.equal(asstContent[0].content[0].type, 'text')
  assert.match(asstContent[0].content[0].text, /sha256:bbb/)
})

test('createImageCapableTwin: stream keeps original images when the source model accepts them', async () => {
  const { ctx, llm } = makeCtx({ providers: { 'vision-source': makeSourceAdapter({ image: true }) } })
  const twin = createImageCapableTwin(ctx, 'vision-source', 'vision-source-dsh-medomni')
  const messages = [{ role: 'user', content: [{ type: 'image', attachment: { attachmentId: 'sha256:aaa' } }] }]
  for await (const _chunk of twin.stream({ provider: 'vision-source-dsh-medomni', model: 'model-a', messages })) {}
  const delegated = llm.streams[0]
  assert.equal(delegated.provider, 'vision-source')
  assert.equal(delegated.messages[0].content[0].type, 'image')
})

test('applyVisionRoutes: registers image-capable twins for live providers', () => {
  const { ctx, llm } = makeCtx({
    providers: {
      'deepseek-official': makeSourceAdapter({ name: 'DeepSeek' }),
      'opencode-go': makeSourceAdapter({ name: 'OpenCode Go' }),
    },
  })
  applyVisionRoutes(ctx)
  assert.ok(llm.adapters.has('deepseek-official-dsh-medomni'))
  assert.ok(llm.adapters.has('opencode-go-dsh-medomni'))
  // the source provider stays untouched — the twin is an addition, not a takeover
  assert.ok(llm.adapters.has('deepseek-official'))
})

test('applyVisionRoutes: never wraps own routes, -vision twins, or excluded providers', () => {
  const { ctx, llm } = makeCtx({
    providers: {
      'deepseek-official': makeSourceAdapter(),
      'already-dsh-medomni': makeSourceAdapter(),
      'router-vision': makeSourceAdapter(),
    },
  })
  applyVisionRoutes(ctx, { excludedProviders: ['deepseek-official'] })
  assert.ok(!llm.adapters.has('deepseek-official-dsh-medomni'))
  assert.ok(!llm.adapters.has('already-dsh-medomni-dsh-medomni'))
  assert.ok(!llm.adapters.has('router-vision-dsh-medomni'))
})

test('applyVisionRoutes: re-syncs on llm/adapters-updated (new providers gain twins, vanished ones lose them)', () => {
  const { ctx, llm, events } = makeCtx({ providers: { 'deepseek-official': makeSourceAdapter() } })
  applyVisionRoutes(ctx)
  assert.ok(llm.adapters.has('deepseek-official-dsh-medomni'))
  // a provider appears later (settings-backed registration)
  llm.adapters.set('opencode-go', makeSourceAdapter())
  llm.regs.set('opencode-go', { adapter: llm.adapters.get('opencode-go') })
  events.get('llm/adapters-updated')()
  assert.ok(llm.adapters.has('opencode-go-dsh-medomni'))
  // a provider disappears
  llm.adapters.delete('deepseek-official')
  llm.regs.delete('deepseek-official')
  events.get('llm/adapters-updated')()
  assert.ok(!llm.adapters.has('deepseek-official-dsh-medomni'))
})

test('applyVisionRoutes: cleanup disposer unregisters every twin', () => {
  const { ctx, llm, effects } = makeCtx({ providers: { 'deepseek-official': makeSourceAdapter() } })
  applyVisionRoutes(ctx)
  assert.ok(llm.adapters.has('deepseek-official-dsh-medomni'))
  assert.equal(effects.length, 1)
  effects[0]()
  assert.ok(!llm.adapters.has('deepseek-official-dsh-medomni'))
})

test('applyVisionRoutes: wrapProviders false registers nothing', () => {
  const { ctx, llm } = makeCtx({ providers: { 'deepseek-official': makeSourceAdapter() } })
  applyVisionRoutes(ctx, { wrapProviders: false })
  assert.ok(!llm.adapters.has('deepseek-official-dsh-medomni'))
})

test('applyVisionRoutes: missing llm service is a silent no-op', () => {
  applyVisionRoutes({ on() {}, effect() {} }, { wrapProviders: true })
  applyVisionRoutes(undefined, { wrapProviders: true })
})
