/** REST + WebSocket 客户端。
 *  - backend/token 来自页面 URL 参数（Electron 以 ?backend=http://127.0.0.1:PORT&token=xx 打开）；
 *    缺省 = 同源（开发模式走 vite 代理）。
 */
import type { Adjustments, AiMaskTarget, MaskState } from './types'

const qs = new URLSearchParams(location.search)
export const backendBase = (qs.get('backend') ?? '').replace(/\/+$/, '')
export const authToken = qs.get('token') ?? ''

function authUrl(path: string): string {
  const sep = path.includes('?') ? '&' : '?'
  return backendBase + path + (authToken ? `${sep}token=${encodeURIComponent(authToken)}` : '')
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(authUrl(path), init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (j?.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch { /* 保留 statusText */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

function post<T>(path: string, payload: unknown): Promise<T> {
  return req<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

// ---- 数据类型 ----
export interface Meta { app: string; version: string; rawpy: boolean; gpu: boolean; width: number; height: number }
export interface HistData { master: number[]; red: number[]; green: number[]; blue: number[] }
export interface PickResult { r: number; g: number; b: number; hue: number }
export type RectNorm = [number, number, number, number]

/** 打开结果（单图入列并选中）。 */
export interface OpenResult {
  id: number; name: string; width: number; height: number; serial: number; library: number
}

export interface LibraryItem {
  id: number; name: string; width: number; height: number; serial: number; current: boolean
}

// ---- REST ----
export const getMeta = () => req<Meta>('/api/meta')
export const getDefaults = () => req<{ adjustments: Adjustments }>('/api/defaults')

export async function openUpload(file: File): Promise<OpenResult> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(authUrl('/api/open-upload'), { method: 'POST', body: fd })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try { detail = (await res.json()).detail ?? detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  return res.json()
}

/** Electron 模式：传本地路径（含 RAW），入列并选中。 */
export const openPath = (path: string) => post<OpenResult>('/api/open', { path })

// ---- 多图图库 ----
export const getLibrary = () => req<{ images: LibraryItem[] }>('/api/library')
export const selectLibrary = (id: number) =>
  post<{ id: number; name: string; width: number; height: number; serial: number; dof?: DofParams | null }>(
    '/api/library/select', { id })
export const removeLibrary = (id: number) =>
  post<{ images: LibraryItem[] }>('/api/library/remove', { id })
export const thumbUrl = (id: number, serial: number) => {
  const base = authUrl(`/api/library/${id}/thumb`)
  return base + (base.includes('?') ? '&' : '?') + `v=${serial}`
}
export const originalUrl = (id: number) => authUrl(`/api/library/${id}/original`)

/** Electron 模式：服务端渲染后直写磁盘（urgent 槽位）。 */
export const saveToPath = (path: string, adjustments: Adjustments) =>
  post<{ rid: number; saved: boolean; path: string }>('/api/save', { path, adjustments })

/** 浏览器模式：同步渲染整幅并触发下载。 */
export async function renderDownload(adjustments: Adjustments, format: string, filename: string): Promise<void> {
  const bytes = await renderBytes(adjustments, format)
  const url = URL.createObjectURL(new Blob([bytes]))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

/** 渲染整幅并返回字节（Electron 原生保存与浏览器下载共用）。 */
export async function renderBytes(adjustments: Adjustments, format: string): Promise<ArrayBuffer> {
  const res = await fetch(authUrl('/api/render'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ adjustments, format }),
  })
  if (!res.ok) {
    let detail = `${res.status}`
    try { detail = (await res.json()).detail ?? detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  return res.arrayBuffer()
}

// ---- Electron 原生桥（desktop/preload.cjs 注入；浏览器里不存在）----
export interface NativeBridge {
  openImageDialog: () => Promise<string[] | null>
  saveImageDialog: (defaultName: string) => Promise<string | null>
  writeFile: (filePath: string, bytes: Uint8Array) => Promise<boolean>
}

export const native: NativeBridge | null =
  (window as unknown as { fernlensNative?: NativeBridge }).fernlensNative ?? null

/** Electron：原生打开对话框（支持多选）→ 按路径逐张入列（支持 RAW）。 */
export async function openDesktop(): Promise<string[] | null> {
  if (!native) return null
  return native.openImageDialog()
}

/** Electron：原生保存对话框 → 渲染 → 落盘。返回保存路径（取消返回 null）。 */
export async function saveDesktop(adjustments: Adjustments, defaultName = 'fernlens.png'): Promise<string | null> {
  if (!native) return null
  const path = await native.saveImageDialog(defaultName)
  if (!path) return null
  const ext = (path.split('.').pop() || 'png').toLowerCase()
  const format = ext === 'jpeg' ? 'jpg' : ext
  const bytes = await renderBytes(adjustments, format)
  await native.writeFile(path, new Uint8Array(bytes))
  return path
}

export const applyStamp = (src_rect: RectNorm, dst_rect: RectNorm, feather: number) =>
  post<{ serial: number }>('/api/stamp', { src_rect, dst_rect, feather })
export const applyStampBrush = (
  src_start: [number, number],
  points: [number, number][],
  radius: number,
  feather: number,
  opacity: number,
) =>
  post<{ serial: number }>('/api/stamp/brush', { src_start, points, radius, feather, opacity })
export const stampUndo = () => post<{ serial: number }>('/api/stamp/undo', {})
export const stampClear = () => post<{ serial: number }>('/api/stamp/clear', {})

export const cropGeometry = (rect: RectNorm, angle: number, masks: MaskState[]) =>
  post<{ masks: MaskState[]; baked: boolean; width: number; height: number }>(
    '/api/geometry/crop', { rect, angle, masks })
export const rotateGeometry = (clockwise: boolean, masks: MaskState[]) =>
  post<{ masks: MaskState[]; width: number; height: number }>(
    '/api/geometry/rotate', { clockwise, masks })

export const pickAt = (x: number, y: number, patch = 5) =>
  post<PickResult>('/api/pick', { x, y, patch })

/** 单个 OpenAI 兼容端点（URL / Key / 模型 ID）。 */
export interface AiEndpoint {
  baseUrl: string
  apiKey: string
  model: string
}

/** 前端「AI 设置」保存的两套端点（localStorage 持久化）：
 *  - llm：对话、调色参数、一键精修、AI 蒙版（视觉大模型）
 *  - image：生图（/images/edits 图生图模型）
 *  两者可以指向不同厂商，互不影响。 */
export interface AiConfig {
  llm: AiEndpoint
  image: AiEndpoint
  /** AI 调整强度 0.2..1（默认 0.6）：统一缩放 AI 返回的参数幅度，越小越克制。 */
  strength: number
}

export const emptyAiEndpoint = (): AiEndpoint => ({ baseUrl: '', apiKey: '', model: '' })

/** AI 调整强度的默认值与允许范围（与服务端 auto.AI_DEFAULT_STRENGTH 对应）。 */
export const AI_STRENGTH_DEFAULT = 0.6
export const AI_STRENGTH_MIN = 0.2
export const AI_STRENGTH_MAX = 1

export const emptyAiConfig = (): AiConfig => ({
  llm: emptyAiEndpoint(), image: emptyAiEndpoint(), strength: AI_STRENGTH_DEFAULT,
})

export const aiEndpointComplete = (e: AiEndpoint | null | undefined): boolean =>
  !!(e && e.baseUrl.trim() && e.apiKey.trim() && e.model.trim())

/** 对话/参数（1、2 号功能）是否可用。 */
export const aiConfigComplete = (c: AiConfig): boolean => aiEndpointComplete(c.llm)

/** 生图（3 号功能）是否可用。 */
export const aiImageComplete = (c: AiConfig): boolean => aiEndpointComplete(c.image)

function endpointPayload(e: AiEndpoint): Record<string, string> {
  return { base_url: e.baseUrl.trim(), api_key: e.apiKey.trim(), model: e.model.trim() }
}

/** 服务端只跑 LLM 侧能力（精修 / 蒙版网格），因此始终传 llm 端点。 */
function aiConfigPayload(c: AiConfig): Record<string, string> {
  return endpointPayload(c.llm)
}

/** 测试单个端点连通性：GET {base}/models，返回可用模型 id 列表。 */
export const testAiEndpoint = (e: AiEndpoint) =>
  post<{ ok: boolean; models: string[] }>('/api/ai/test', { config: endpointPayload(e) })

/** 兼容旧调用：测试 LLM 端点。 */
export const testAiConfig = (c: AiConfig) => testAiEndpoint(c.llm)

/** AI 蒙版识别（本地像素级 SAM2 + U²-Net），返回掩码数据。 */
export interface AiMaskResult {
  target: string
  model: string
  mask_png?: string
  width?: number; height?: number
}
export const aiMask = (target: AiMaskTarget) =>
  post<AiMaskResult>('/api/mask/ai', { target, engine: 'local' })

// ---- 本地分割模型（SAM2 + U2Net 家族，ONNX）----
export interface LocalModelsStatus {
  runtime: boolean
  ready: boolean
  models: Record<string, { file: string; ready: boolean; size: number }>
}
export interface LocalModelsProgress extends LocalModelsStatus {
  running: boolean; done: boolean; error: string
  current: string; received: number; total: number
}
export const getLocalModels = () => req<LocalModelsStatus>('/api/ai/models')
export const downloadLocalModels = () => post<{ started: boolean; reason?: string }>('/api/ai/models/download', {})
export const localModelsProgress = () => req<LocalModelsProgress>('/api/ai/models/progress')

export const autoGrade = (mode: 'rule' | 'ai', cfg?: AiConfig, intent?: string) =>
  post<{ adjustments: Partial<Adjustments>; note: string; model: string }>(
    '/api/auto', { mode, config: cfg ? aiConfigPayload(cfg) : undefined,
      intent: intent ? intent.trim() : undefined, strength: cfg ? cfg.strength : undefined })

export type DepthRelightPreset = 'studio' | 'atmosphere' | 'warm_cool' | 'cinematic'

export interface DofParams {
  focus: number
  aperture: number
  dof_range: number
  bokeh_highlight: number
}

export interface DofPickResult {
  focus: number
  depth_preview_png: string
}

export interface DofResult extends DofParams {
  serial: number
  active: boolean
  depth_preview_png: string
}

export const aiDofPick = (x: number, y: number, dof_range = 20) =>
  post<DofPickResult>('/api/ai/dof/pick', { x, y, dof_range })

export const aiDofApply = (params: DofParams) =>
  post<DofResult>('/api/ai/dof', params)

export const aiDofClear = () =>
  post<{ serial: number; active: boolean }>('/api/ai/dof/clear', {})

export interface DepthRelightResult {
  preset: string
  model: string
  depth_preview_png: string
  preview_w: number
  preview_h: number
  masks: MaskState[]
}

export const aiDepthRelight = (
  preset: DepthRelightPreset,
  strength = 1.0,
  near_bias = 0.0,
  masks_only = false,
) =>
  post<DepthRelightResult>('/api/ai/depth-relight', { preset, strength, near_bias, masks_only })

export interface ColorMatchResult {
  adjustments: Partial<Adjustments>
  ref_palette: string[]
  src_palette: string[]
  protect_subject: boolean
  note: string
}

export async function aiColorMatch(opts: {
  file?: File
  refId?: number
  strength?: number
  protectSubject?: boolean
}): Promise<ColorMatchResult> {
  const fd = new FormData()
  if (opts.file) fd.append('file', opts.file)
  if (opts.refId != null) fd.append('ref_id', String(opts.refId))
  fd.append('strength', String(opts.strength ?? 0.75))
  fd.append('protect_subject', opts.protectSubject ? 'true' : 'false')
  return req<ColorMatchResult>('/api/ai/color-match', { method: 'POST', body: fd })
}

export const fetchHistogram = () => req<HistData>('/api/histogram')

// ---- WebSocket 帧通道 ----
export interface ServerEvent {
  type: 'busy' | 'saved' | 'error'
  busy?: boolean
  rid?: number
  ok?: boolean
  path?: string
  error?: string
  message?: string
  kind?: string
}

export class RenderSocket {
  private ws: WebSocket | null = null
  private retry = 0
  private closed = false
  onFrame: ((rid: number, isFull: boolean, jpeg: ArrayBuffer) => void) | null = null
  onEvent: ((ev: ServerEvent) => void) | null = null
  onState: ((connected: boolean) => void) | null = null

  connect(): void {
    this.closed = false
    const base = (backendBase || location.origin).replace(/^http/, 'ws')
    const url = `${base}/ws${authToken ? `?token=${encodeURIComponent(authToken)}` : ''}`
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    ws.onopen = () => { this.retry = 0; this.onState?.(true) }
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        if (ev.data.byteLength <= 5) return
        const view = new DataView(ev.data)
        const rid = view.getUint32(0)
        const isFull = view.getUint8(4) === 1
        this.onFrame?.(rid, isFull, ev.data.slice(5))
      } else {
        try { this.onEvent?.(JSON.parse(ev.data as string)) } catch { /* 忽略坏消息 */ }
      }
    }
    ws.onclose = () => {
      this.ws = null
      this.onState?.(false)
      if (!this.closed && this.retry < 8) {
        setTimeout(() => this.connect(), 400 * 2 ** this.retry++)
      }
    }
    ws.onerror = () => ws.close()
  }

  sendRender(rid: number, kind: 'preview' | 'full', adjustments: unknown, edge = 0): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'render', rid, kind, edge, adjustments }))
    }
  }

  close(): void {
    this.closed = true
    this.ws?.close()
  }
}
