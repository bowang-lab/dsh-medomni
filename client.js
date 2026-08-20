window.__ModuleLoader__.load({
  id: 'dsh-medomni',
  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports
    Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' })

    const { createElement, useEffect, useMemo, useState } = require('react')

    const TOOL_NAMES = [
      'xray_report_maira',
      'xray_grounded_report_maira',
      'xray_phrase_grounding_maira',
      'xray_report_medgemma',
      'xray_anatomy_localization',
      'xray_longitudinal_comparison',
      'ct_report_medgemma',
      'mri_report_medgemma',
      'retinal_report_medgemma',
      'ultrasound_classify_biomedclip',
      'ultrasound_classify_then_segment',
      'ct_segmentation_totalseg',
      'mri_segmentation_totalseg',
      'xray_segmentation_biomedparse',
      'ultrasound_segmentation_biomedparse',
      'retinal_segmentation_biomedparse',
      'ct_segmentation_biomedparse',
      'mri_segmentation_biomedparse',
    ]

    const inject = ['slots', 'sessions']

    function apply(ctx) {
      const sessions = ctx.get('sessions')
      for (const key of TOOL_NAMES) {
        ctx.slots.inject('tool.call.toolview', () => ctx.slots.register({
          name: 'tool.call.toolview',
          key,
          inject: (sessionId) => ({
            loadImage: async (attachment) => {
              const session = sessions.binding(sessionId)?.session
              if (session === undefined) throw new Error(`dsh-MedOmni preview: unknown session "${sessionId}"`)
              const result = await session.readAttachment(attachment.attachmentId)
              if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`)
              const bytes = Uint8Array.from(result.value.data)
              if (typeof URL.createObjectURL !== 'function') {
                return `data:${result.value.attachment.mediaType};base64,${bytesToBase64(bytes)}`
              }
              return URL.createObjectURL(new Blob([bytes.buffer], { type: result.value.attachment.mediaType }))
            },
          }),
        }, DshMedomniToolRow))
      }
    }

    function DshMedomniToolRow({ toolName, block, loadImage }) {
      const settled = Object.prototype.hasOwnProperty.call(block, 'kind')
      const status = !settled ? 'running' : block.isError ? 'error' : block.error?.code === 'interrupted' ? 'stopped' : 'ok'
      const text = settled ? resultText(block) : ''
      const images = settled ? resultImages(block) : []
      const title = titleFor(toolName)
      const summary = settled ? firstLine(text) || block.callId : callSummary(block)
      return createElement('section', {
        style: rowStyle(status),
        'data-dsh-medomni-tool-row': toolName,
      },
        createElement('div', { style: headerStyle },
          createElement('span', { style: dotStyle(status), 'aria-hidden': true }),
          createElement('strong', { style: titleStyle }, title),
          summary !== '' && createElement('span', { style: summaryStyle }, summary),
        ),
        text !== '' && createElement('pre', { style: textStyle }, text),
        images.length > 0 && createElement('div', { style: galleryStyle },
          images.map((attachment, index) => createElement(PreviewImage, {
            key: `${attachment.attachmentId}:${index}`,
            attachment,
            loadImage,
          })),
        ),
      )
    }

    function PreviewImage({ attachment, loadImage }) {
      const [src, setSrc] = useState(null)
      const [error, setError] = useState(false)
      const [expanded, setExpanded] = useState(false)
      useEffect(() => {
        let live = true
        let objectUrl
        setSrc(null)
        setError(false)
        loadImage(attachment).then((url) => {
          objectUrl = url.startsWith('blob:') ? url : undefined
          if (live) setSrc(url)
        }).catch(() => {
          if (live) setError(true)
        })
        return () => {
          live = false
          if (objectUrl !== undefined) URL.revokeObjectURL(objectUrl)
        }
      }, [attachment, loadImage])
      useEffect(() => {
        if (!expanded) return undefined
        const onKeyDown = (event) => {
          if (event.key === 'Escape') setExpanded(false)
        }
        document.addEventListener('keydown', onKeyDown)
        return () => document.removeEventListener('keydown', onKeyDown)
      }, [expanded])

      const label = attachment.name ?? 'dsh-MedOmni preview'
      const fit = useMemo(() => previewFit(attachment), [attachment])
      if (error) return createElement('div', { style: imageFallbackStyle }, 'Preview failed to load')
      if (src === null) return createElement('div', { style: imageFallbackStyle }, 'Loading preview...')
      return createElement('div', null,
        createElement('button', {
          type: 'button',
          title: 'Open preview image',
          onClick: () => setExpanded(true),
          style: imageButtonStyle,
        }, createElement('img', {
          src,
          alt: label,
          style: { ...imageStyle, width: fit.width, height: fit.height },
        })),
        expanded && createElement('div', {
          role: 'dialog',
          'aria-modal': true,
          'aria-label': label,
          style: lightboxBackdropStyle,
          onClick: () => setExpanded(false),
        },
          createElement('button', {
            type: 'button',
            title: 'Close preview',
            onClick: () => setExpanded(false),
            style: lightboxCloseStyle,
          }, 'Close'),
          createElement('img', {
            src,
            alt: label,
            onClick: (event) => event.stopPropagation(),
            style: lightboxImageStyle,
          }),
        ),
      )
    }

    function resultText(node) {
      const parts = []
      for (const block of node.content ?? []) {
        if (block?.type === 'text' && typeof block.text === 'string') parts.push(block.text)
      }
      if (parts.length === 0 && node.error !== undefined) parts.push(`${node.error.name}: ${node.error.code}`)
      return parts.join('\n')
    }

    function resultImages(node) {
      return (node.content ?? [])
        .filter(block => block?.type === 'image' && block.attachment !== undefined)
        .map(block => block.attachment)
    }

    function callSummary(block) {
      const raw = block.argsRaw ?? block.call?.argsRaw ?? ''
      if (raw === '') return block.callId ?? ''
      try {
        const parsed = JSON.parse(raw)
        if (parsed && typeof parsed === 'object') {
          const input = parsed.input ?? parsed.lateral ?? parsed.current ?? parsed.output ?? parsed.output_dir
          if (typeof input === 'string' && input !== '') return input
        }
      } catch {
        // fall through to raw first line
      }
      return firstLine(raw)
    }

    function titleFor(toolName) {
      if (toolName.includes('segmentation')) return 'dsh-MedOmni Segmentation'
      if (toolName.includes('classify')) return 'dsh-MedOmni Classification'
      if (toolName.includes('report')) return 'dsh-MedOmni Report'
      return 'dsh-MedOmni'
    }

    function firstLine(text) {
      const value = String(text ?? '')
      const index = value.indexOf('\n')
      return index === -1 ? value : value.slice(0, index)
    }

    function previewFit(attachment) {
      const width = Number(attachment.width) || 240
      const height = Number(attachment.height) || 180
      const scale = Math.min(1, 260 / width, 180 / height)
      return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) }
    }

    function bytesToBase64(bytes) {
      let binary = ''
      for (const byte of bytes) binary += String.fromCharCode(byte)
      return btoa(binary)
    }

    const headerStyle = {
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      minWidth: 0,
    }

    const titleStyle = {
      fontSize: 13,
      fontWeight: 600,
      whiteSpace: 'nowrap',
    }

    const summaryStyle = {
      color: 'var(--dsw-alias-fg-muted, #6b7280)',
      fontSize: 13,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    }

    const textStyle = {
      margin: '8px 0 0',
      whiteSpace: 'pre-wrap',
      fontFamily: 'inherit',
      fontSize: 13,
      lineHeight: 1.45,
      color: 'var(--dsw-alias-fg-default, #111827)',
    }

    const galleryStyle = {
      display: 'flex',
      flexWrap: 'wrap',
      gap: 8,
      marginTop: 10,
    }

    const imageButtonStyle = {
      padding: 0,
      border: '1px solid var(--dsw-alias-border-subtle, #d1d5db)',
      background: 'var(--dsw-alias-bg-default, #fff)',
      borderRadius: 6,
      overflow: 'hidden',
      cursor: 'zoom-in',
    }

    const imageStyle = {
      display: 'block',
      objectFit: 'cover',
    }

    const imageFallbackStyle = {
      display: 'grid',
      placeItems: 'center',
      width: 180,
      height: 96,
      border: '1px solid var(--dsw-alias-border-subtle, #d1d5db)',
      borderRadius: 6,
      color: 'var(--dsw-alias-fg-muted, #6b7280)',
      fontSize: 12,
    }

    const lightboxBackdropStyle = {
      position: 'fixed',
      inset: 0,
      zIndex: 2147483647,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
      background: 'rgba(0, 0, 0, 0.72)',
      cursor: 'zoom-out',
    }

    const lightboxImageStyle = {
      display: 'block',
      maxWidth: 'calc(100vw - 64px)',
      maxHeight: 'calc(100vh - 64px)',
      objectFit: 'contain',
      borderRadius: 6,
      boxShadow: '0 18px 60px rgba(0, 0, 0, 0.45)',
      cursor: 'default',
    }

    const lightboxCloseStyle = {
      position: 'fixed',
      top: 16,
      right: 16,
      padding: '7px 10px',
      border: '1px solid rgba(255, 255, 255, 0.32)',
      borderRadius: 6,
      background: 'rgba(17, 24, 39, 0.88)',
      color: '#fff',
      fontSize: 13,
      cursor: 'pointer',
    }

    function rowStyle(status) {
      return {
        border: '1px solid var(--dsw-alias-border-subtle, #d1d5db)',
        borderRadius: 8,
        padding: 10,
        background: 'var(--dsw-alias-bg-subtle, #f9fafb)',
        boxShadow: status === 'error' ? 'inset 3px 0 0 #dc2626' : 'none',
      }
    }

    function dotStyle(status) {
      const color = status === 'running' ? '#2563eb' : status === 'error' ? '#dc2626' : status === 'stopped' ? '#d97706' : '#16a34a'
      return {
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: color,
        flex: '0 0 auto',
      }
    }

    exports.inject = inject
    exports.apply = apply
    return module.exports
  },
})
