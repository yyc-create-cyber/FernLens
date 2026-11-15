/** 编辑器全局状态（单例 composable）。
 *  调度语义照抄旧版：拖动 -> preview（服务端最新覆盖），松手/空闲 400ms -> full；
 *  帧按 rid 丢弃过期（与 ProcessorThread.frameReady 接收端一致）。
 */
import { reactive, toRaw } from 'vue'
import {
  Adjustments, CURVE_CHANNELS, GRADING_RANGES, TONE_FIELDS, aiMaskPixels,
  defaultAdjustments,
} from '@/api/types'
import type { AdjustmentsLike, AiMaskTarget, MaskKind, MaskState, PickerState, WheelState } from '@/api/types'
import { defaultMask } from '@/api/types'
import * as api from '@/api/client'
import type { AiConfig, HistData, LibraryItem, LocalModelsStatus, RectNorm, ServerEvent } from '@/api/client'
import { presetColor, CURVE_CHANNEL_LABELS, RANGE_LABELS } from '@/lib/presets'
import {
  BUILTIN_FILTERS, applyLook, downloadJson, filterFileName, filterToFile,
  lookFromAdjustments, parseFilterFile, resetGlobalLook, sanitizeLook,
} from '@/lib/filters'
import type { FilterPreset } from '@/lib/filters'
import { tx } from '@/lib/i18n'

interface Frame { bmp: ImageBitmap; rid: number }

/** AI 端点配置（localStorage 持久化；服务端也支持环境变量回退）。 */
const AI_CONFIG_KEY = 'fernlens.ai-config'
/** 精修提示词（可选，随每次 AI 请求发送；localStorage 持久化）。 */
const AI_INTENT_KEY = 'fernlens.ai-intent'

function loadAiIntent(): string {
  try {
    return localStorage.getItem(AI_INTENT_KEY) || ''
  } catch {
    return ''
  }
}

/** AI 调整强度取值收敛到 [AI_STRENGTH_MIN, AI_STRENGTH_MAX]，缺省用默认值。 */
function clampStrength(v: unknown): number {
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return api.AI_STRENGTH_DEFAULT
  return Math.min(api.AI_STRENGTH_MAX, Math.max(api.AI_STRENGTH_MIN, n))
}

function loadAiConfig(): AiConfig {
  try {
    const raw = JSON.parse(localStorage.getItem(AI_CONFIG_KEY) || '')
    const str = (v: unknown): string => (typeof v === 'string' ? v : '')
    // 旧版是单端点 + imageModel 复用：迁移成 llm / image 两套独立端点
    if (raw && typeof raw === 'object' && raw.llm) {
      return {
        llm: { baseUrl: str(raw.llm.baseUrl), apiKey: str(raw.llm.apiKey),
               model: str(raw.llm.model) },
        image: { baseUrl: str(raw.image?.baseUrl), apiKey: str(raw.image?.apiKey),
                 model: str(raw.image?.model) },
        strength: clampStrength(raw.strength),
      }
    }
    const baseUrl = str(raw?.baseUrl), apiKey = str(raw?.apiKey)
    const model = str(raw?.model), imageModel = str(raw?.imageModel) || model
    return {
      llm: { baseUrl, apiKey, model },
      image: imageModel ? { baseUrl, apiKey, model: imageModel } : api.emptyAiEndpoint(),
      strength: clampStrength(raw?.strength),
    }
  } catch {
    return api.emptyAiConfig()
  }
}

/** 自定义滤镜（localStorage 持久化；导入的 JSON 也存这里）。 */
const FILTERS_KEY = 'fernlens.filters'

function loadCustomFilters(): FilterPreset[] {
  try {
    const raw = localStorage.getItem(FILTERS_KEY)
    if (!raw) return []
    const list = JSON.parse(raw) as unknown
    if (!Array.isArray(list)) return []
    const out: FilterPreset[] = []
    for (const it of list) {
      if (!it || typeof it !== 'object') continue
      const o = it as Record<string, unknown>
      const look = sanitizeLook(o['look'])
      if (!Object.keys(look).length) continue
      out.push({
        id: typeof o['id'] === 'string' && o['id'] ? o['id'] : `custom:${out.length}-${Date.now()}`,
        name: typeof o['name'] === 'string' && o['name'] ? o['name'] : '未命名滤镜',
        group: '自定义',
        note: typeof o['note'] === 'string' ? o['note'] : undefined,
        look,
        createdAt: typeof o['createdAt'] === 'number' ? o['createdAt'] : Date.now(),
      })
    }
    return out
  } catch {
    return []
  }
}

function persistCustomFilters(): void {
  try {
    localStorage.setItem(FILTERS_KEY, JSON.stringify(state.customFilters))
  } catch { /* 忽略配额错误 */ }
}

const state = reactive({
  loaded: false,
  busy: false,
  connected: false,
  width: 0,
  height: 0,
  serial: 0,
  adjustments: defaultAdjustments() as Adjustments,
  previewFrame: null as Frame | null,
  fullFrame: null as Frame | null,
  frameVersion: 0,
  /** 视口需要渲染的长边（像素）：Viewport 按窗口/DPR/缩放写入，渲染按它分级 */
  viewEdge: 900,
  /** 显示几何基准（长边像素）：取最近一帧「全尺寸」结果的尺寸。
   *  草稿帧分辨率更低，但必须拉伸到同一矩形显示，否则拖动时画面会突然缩小。 */
  geomW: 0,
  geomH: 0,
  histogram: null as HistData | null,
  status: '',
  meta: null as api.Meta | null,
  // 多图图库：每张图独立参数（imageParams），当前图 currentImageId
  library: [] as LibraryItem[],
  currentImageId: null as number | null,
  // AI 设置（顶栏 ⚙ 打开；未配置时 AI 功能会提示先配置）
  aiConfig: loadAiConfig(),
  aiIntent: loadAiIntent(),
  aiSettingsOpen: false,
  aiStudioOpen: false,
  prefsOpen: false,
  aiChatOpen: false,
  aiBusy: null as AiMaskTarget | null,
  // 本地像素级引擎（SAM2 + U2Net + Depth Anything V2）
  localAI: null as LocalModelsStatus | null,
  // AI 调节景深（Depth Anything V2 光学虚化）
  dofActive: false,
  dofFocus: 0.75,
  dofAperture: 45,
  dofRange: 20,
  dofBokeh: 25,
  dofPicking: false,
  dofFocusPt: null as [number, number] | null,
  dofPreviewPng: '',
  // 取色模式：null=未激活 | 'new'=新建取色点 | number=重新取色第 i 个
  pickTarget: null as null | 'new' | number,
  pickSelected: null as number | null,
  // 编辑模式（对照旧 PreviewWidget.set_edit_mode / crop / stamp）
  editMode: null as null | 'crop' | 'stamp' | 'mask',
  maskSelected: null as number | null,
  // 仿制图章（Photoshop 式画笔仿制 + 对齐/羽化/不透明度）
  stampSource: null as RectNorm | null,
  stampSourcePt: null as [number, number] | null,
  stampPickingSource: false,
  stampSize: 22,            // 1..100 -> radiusNorm = stampSize * 0.0022
  stampFeather: 35,         // 0..100
  stampOpacity: 100,        // 5..100
  stampAligned: true,       // 对齐取样（松笔后保持相对位移）
  stampOffset: null as [number, number] | null,
  // 裁剪（归一化 rect + 拉直角度 + 比例锁定；应用时服务端权威计算）
  cropRect: null as { x0: number; y0: number; x1: number; y1: number } | null,
  cropAngle: 0,
  cropAspect: 'free' as string,           // free|orig|1|3:2|4:3|16:9（:t=转置）
  cropPending: false,
  // 对比原图（按住 `\`）
  comparing: false,
  originalBmp: null as ImageBitmap | null,
  // 蒙版叠加显示（LR 的 O 键开关；仅影响叠加显示，不影响渲染）
  maskOverlay: true,
  // 蒙版选中后始终显示半透明红色叠加；在针对蒙版做色彩/影调调整时自动隐藏以方便预览
  maskAdjusting: false,
  maskHover: false,
  maskReveal: false,
  // 右侧当前激活的面板 Tab（与 App.vue 同步）
  activeTab: 'light' as 'light' | 'color' | 'effects' | 'filter' | 'curve' | 'grading' | 'mask' | 'picker' | 'history',
  // 自定义多边形选点绘制状态
  polyDrawing: false,
  // 历史记录（每图独立；存完整参数快照，可跳到任意一步）
  history: [] as { label: string; at: number }[],
  historyIndex: -1,
  // 滤镜：lastApplied 用于高亮当前套用的预设（手改参数后自动清除）
  customFilters: loadCustomFilters(),
  activeFilter: '',
})

// 每图独立状态（非响应式：体积大且只在切换时整体读写）
const imageParams = new Map<number, Adjustments>()      // id -> 调色参数
const originalBmps = new Map<number, ImageBitmap>()     // id -> 对比用原位图
const originalBlobs = new Map<number, Blob>()           // id -> 上传的原始字节（浏览器模式）

export function aiConfigured(): boolean {
  return api.aiConfigComplete(state.aiConfig)
}

/** 生图（第 3 个 AI 功能）是否已配置图生图端点。 */
export function aiImageConfigured(): boolean {
  return api.aiImageComplete(state.aiConfig)
}

/** 拉取本地像素级引擎状态（就绪时 AI 蒙版默认走本地引擎）。 */
export async function refreshLocalAI(): Promise<void> {
  try { state.localAI = await api.getLocalModels() } catch { state.localAI = null }
}

export function saveAiConfig(cfg: AiConfig): void {
  state.aiConfig = { ...cfg, strength: clampStrength(cfg.strength) }
  try { localStorage.setItem(AI_CONFIG_KEY, JSON.stringify(state.aiConfig)) } catch { /* 忽略 */ }
}

/** 保存精修提示词（空字符串 = 由模型自选风格）。 */
export function saveAiIntent(intent: string): void {
  state.aiIntent = intent.trim()
  try { localStorage.setItem(AI_INTENT_KEY, state.aiIntent) } catch { /* 忽略 */ }
}

export function openAiSettings(): void {
  state.aiSettingsOpen = true
  void refreshLocalAI()
}

export function openAiStudio(): void {
  state.aiStudioOpen = true
  void refreshLocalAI()
}

export function closeAiStudio(): void {
  state.aiStudioOpen = false
}

export function openPrefs(): void {
  state.prefsOpen = true
}

export function closePrefs(): void {
  state.prefsOpen = false
}

/** 打开/关闭右下角 AI 对话框（对话 / 参数 / 生图）。 */
export function openAiChat(): void {
  state.aiChatOpen = true
}

export function closeAiChat(): void {
  state.aiChatOpen = false
}

let ridCounter = 0
let newestRid = -1
const sock = new api.RenderSocket()

// ------------------------------------------------------------ 帧接收
sock.onFrame = (rid, isFull, jpeg) => {
  if (rid < newestRid) return                       // 过期帧（已被更新的请求取代）
  newestRid = rid
  createImageBitmap(new Blob([jpeg], { type: 'image/jpeg' })).then((bmp) => {
    if (rid < newestRid) { bmp.close(); return }    // 解码期间又来了新帧
    const slot = isFull ? 'fullFrame' : 'previewFrame'
    state[slot]?.bmp.close()
    state[slot] = { bmp, rid }
    if (isFull) { state.geomW = bmp.width; state.geomH = bmp.height }   // 几何基准
    state.frameVersion++
    scheduleHistogram()
  }).catch(() => { /* 坏帧忽略 */ })
}

sock.onEvent = (ev: ServerEvent) => {
  if (ev.type === 'busy') {
    state.busy = !!ev.busy
  } else if (ev.type === 'saved') {
    state.status = ev.ok ? `已保存：${ev.path}` : `保存失败：${ev.error ?? ''}`
  } else if (ev.type === 'error') {
    state.status = '渲染出错（详见服务端日志）'
  }
}

sock.onState = (ok) => {
  state.connected = ok
  if (!ok) state.status = '与服务端断开，正在重连…'
  else state.status = ''
}
sock.connect()

// ------------------------------------------------------------ 调度
/** 显示所需的长边（像素）。Viewport 按窗口尺寸、DPR 与缩放倍数写入。 */
export const PREVIEW_MAX_EDGE = 1200

/** 停手后的目标分辨率：屏幕需要多少就渲染多少，绝不再对整张原图做交互渲染。 */
function targetEdge(): number {
  const e = Math.round(state.viewEdge || 900)
  return Math.min(3200, Math.max(420, e))
}

/** 拖动中的草稿分辨率（约 0.6 倍长边 → 像素量 1/3）。 */
function draftEdge(): number {
  return Math.max(360, Math.round(targetEdge() * 0.62))
}

/** 超过预览图尺寸就必须取原图数据源，否则拉不出更细的细节。 */
function kindFor(edge: number): 'preview' | 'full' {
  return edge > PREVIEW_MAX_EDGE ? 'full' : 'preview'
}

function sendFull(): void {
  const rid = ++ridCounter
  const edge = targetEdge()
  sock.sendRender(rid, kindFor(edge), toRaw(state.adjustments), edge)
  recordHistory()                     // 每次全图渲染 = 一个可回退的状态点
}

function sendPreview(): void {
  const rid = ++ridCounter
  const edge = draftEdge()
  sock.sendRender(rid, kindFor(edge), toRaw(state.adjustments), edge)
}

/**
 * 由 Viewport 调用：告诉调度层“屏幕现在需要多少像素的长边”。
 * 注意只记数不立即渲染——真正发请求仍由 notifyChanged/notifyCommitted 触发，
 * 避免缩放/窗口变化时和参数调整抢同一个渲染线程。
 */
export function setViewEdge(edge: number): void {
  const e = Math.round(edge)
  if (!Number.isFinite(e) || e <= 0) return
  if (Math.abs(e - state.viewEdge) < 40) return     // 轻微变化不重渲染
  state.viewEdge = e
}

/** 视口需要更高分辨率时（放大到 1:1、窗口变大）补一张清晰的图。 */
export function requestDetail(): void {
  if (!state.loaded) return
  const rid = ++ridCounter
  const edge = targetEdge()
  sock.sendRender(rid, kindFor(edge), toRaw(state.adjustments), edge)
}

let previewTimer: ReturnType<typeof setTimeout> | null = null
let idleTimer: ReturnType<typeof setTimeout> | null = null

/** 参数变化（拖动中）：节流发 preview + 空闲 400ms 后发 full。 */
export function notifyChanged(): void {
  if (!state.loaded) return
  if (!filterApplying && state.maskSelected === null) state.activeFilter = ''
  if (state.maskSelected !== null && state.activeTab !== 'mask') {
    state.maskAdjusting = true
    state.maskReveal = false
  }
  if (previewTimer === null) {
    previewTimer = setTimeout(() => { previewTimer = null; sendPreview() }, 30)
  }
  if (idleTimer !== null) clearTimeout(idleTimer)
  idleTimer = setTimeout(() => { idleTimer = null; sendFull() }, 400)
}

/** 参数提交（松手/开关切换）：立即全图。 */
export function notifyCommitted(): void {
  if (!state.loaded) return
  if (!filterApplying && state.maskSelected === null) state.activeFilter = ''
  if (state.maskSelected !== null && state.activeTab !== 'mask') {
    state.maskAdjusting = true
    state.maskReveal = false
  }
  if (idleTimer !== null) { clearTimeout(idleTimer); idleTimer = null }
  sendFull()
}

// ------------------------------------------------------------ 历史记录
/** 一步历史：完整参数快照（JSON 串，含蒙版）+ 描述；全局回退 = 直接跳到第 N 步。 */
interface HistoryStep { label: string; at: number; snap: string }
interface HistoryRec { steps: HistoryStep[]; index: number; bytes: number }

const HISTORY_MAX_STEPS = 120
const HISTORY_MAX_BYTES = 64 * 1024 * 1024
const HISTORY_MERGE_MS = 900          // 同字段连续调整的合并窗口
const HISTORIES = new Map<number, HistoryRec>()
let historyMuted = false
let pendingLabel = ''                     // 下个 sendFull 要用的描述（裁剪/旋转/重置）

const FIELD_LABELS: Record<string, string> = {
  temperature: '色温', tint: '色调', exposure: '曝光', contrast: '对比度',
  highlights: '高光', shadows: '阴影', whites: '白色', blacks: '黑色',
  saturation: '饱和度', vibrance: '自然饱和度', blackwhite: '黑白',
  clarity: '清晰度', denoise: '降噪', sharpen: '锐度', dehaze: '去朦胧',
}

/** 历史描述要覆盖的字段（比 TONE_FIELDS 多效果四件套）。 */
const HISTORY_FIELDS = [
  ...TONE_FIELDS, 'clarity', 'denoise', 'sharpen', 'dehaze',
] as const

function fmtNum(v: number): string {
  const s = (Math.round(v * 100) / 100).toFixed(2).replace(/\.?0+$/, '')
  return s === '' || s === '-0' ? '0' : s
}

/** 对比前后两份参数，生成人话描述（用于历史列表）。 */
function describeChange(prev: Adjustments | null, cur: Adjustments): string {
  if (!prev) return '打开图像'
  const out: string[] = []
  for (const f of HISTORY_FIELDS) {
    const a = prev[f], b = cur[f]
    if (a !== b) out.push(`${FIELD_LABELS[f]} ${b > a ? '+' : ''}${fmtNum(b)}`)
  }
  if (prev.blend !== cur.blend) out.push(`混合 ${fmtNum(cur.blend)}`)
  for (const ch of CURVE_CHANNELS) {
    if (JSON.stringify(prev.curves[ch]) !== JSON.stringify(cur.curves[ch])) {
      out.push(`曲线 · ${CURVE_CHANNEL_LABELS[ch]}`)
    }
  }
  for (const r of GRADING_RANGES) {
    if (JSON.stringify(prev.wheels[r]) !== JSON.stringify(cur.wheels[r])) {
      out.push(`分级 · ${RANGE_LABELS[r]}`)
    }
  }
  if (prev.pickers.length !== cur.pickers.length) {
    out.push(cur.pickers.length > prev.pickers.length ? '新增取色点' : '删除取色点')
  } else if (JSON.stringify(prev.pickers) !== JSON.stringify(cur.pickers)) {
    out.push('取色调整')
  }
  if (prev.masks.length !== cur.masks.length) {
    const added = cur.masks.length > prev.masks.length
    const m = added ? cur.masks[cur.masks.length - 1] : null
    out.push(added ? `新增蒙版「${m?.name || '蒙版'}」` : '删除蒙版')
  } else if (JSON.stringify(prev.masks) !== JSON.stringify(cur.masks)) {
    out.push('蒙版调整')
  }
  if (!out.length) return '调整'
  if (out.length <= 2) return out.join('，')
  return `${out[0]} 等 ${out.length} 项`
}

/** 描述的首段（用于合并同字段的连续调整）。 */
function labelFamily(label: string): string {
  return label.split(/[ ，]/)[0]
}

function histOf(id: number): HistoryRec {
  let h = HISTORIES.get(id)
  if (!h) { h = { steps: [], index: -1, bytes: 0 }; HISTORIES.set(id, h) }
  return h
}

function syncHistoryView(): void {
  const id = state.currentImageId
  const h = id === null ? undefined : HISTORIES.get(id)
  state.history = h ? h.steps.map(s => ({ label: s.label, at: s.at })) : []
  state.historyIndex = h ? h.index : -1
}

/** 打开图像时建立初始快照（第 1 步 = 当前参数）。 */
export function beginHistory(label = '打开图像'): void {
  if (state.currentImageId === null) return
  const h = histOf(state.currentImageId)
  if (!h.steps.length) {
    const snap = JSON.stringify(toRaw(state.adjustments))
    h.steps.push({ label, at: Date.now(), snap })
    h.bytes = snap.length
    h.index = 0
  } else {
    h.index = Math.min(h.index, h.steps.length - 1)
  }
  syncHistoryView()
}

/** 记录一步（由 sendFull 调用；同一快照自动去重）。 */
export function recordHistory(explicitLabel = ''): void {
  if (historyMuted || !state.loaded || state.currentImageId === null) return
  const h = histOf(state.currentImageId)
  const snap = JSON.stringify(toRaw(state.adjustments))
  const cur = h.steps[h.index]
  if (cur && cur.snap === snap) return                      // 无变化
  if (h.index < h.steps.length - 1) {                       // 回退后再调整：丢弃“未来”
    for (const s of h.steps.slice(h.index + 1)) h.bytes -= s.snap.length
    h.steps.length = h.index + 1
  }
  const prev = cur ? JSON.parse(cur.snap) as Adjustments : null
  const label = explicitLabel || pendingLabel || describeChange(prev, JSON.parse(snap) as Adjustments)
  const forced = !!(explicitLabel || pendingLabel)
  const last = h.steps[h.index]
  const merge = !forced && last && Date.now() - last.at < HISTORY_MERGE_MS
    && labelFamily(last.label) === labelFamily(label)
  if (merge) {
    h.bytes += snap.length - last.snap.length
    last.snap = snap
    last.label = label
    last.at = Date.now()
  } else {
    h.steps.push({ label, at: Date.now(), snap })
    h.bytes += snap.length
    h.index = h.steps.length - 1
  }
  while (h.steps.length > HISTORY_MAX_STEPS
         || (h.bytes > HISTORY_MAX_BYTES && h.steps.length > 2)) {
    const dropped = h.steps.shift()
    if (!dropped) break
    h.bytes -= dropped.snap.length
    h.index = Math.max(0, h.index - 1)
  }
  syncHistoryView()
}

/** 跳到第 index 步（0 基）：整体恢复该步的全部参数（滑块/曲线/分级/蒙版/取色）。 */
export function jumpToHistory(index: number): void {
  const id = state.currentImageId
  const h = id === null ? undefined : HISTORIES.get(id)
  if (!h || index < 0 || index >= h.steps.length || index === h.index) return
  const step = h.steps[index]
  historyMuted = true
  try {
    state.adjustments = JSON.parse(step.snap) as Adjustments
    if (state.maskSelected !== null && state.maskSelected >= state.adjustments.masks.length) {
      state.maskSelected = null
    }
    if (state.maskSelected === null && state.editMode === 'mask') state.editMode = null
    state.polyDrawing = false
    polyBackupPoints = null
    if (state.pickSelected !== null && state.pickSelected >= state.adjustments.pickers.length) {
      state.pickSelected = null
    }
    if (typeof state.pickTarget === 'number' && state.pickTarget >= state.adjustments.pickers.length) {
      state.pickTarget = null
    }
    h.index = index
    state.activeFilter = ''
    syncHistoryView()
    sendFull()
    state.status = `已回到第 ${index + 1} 步：${step.label}`
  } finally {
    historyMuted = false
  }
}

export function undoHistory(): void {
  const id = state.currentImageId
  const h = id === null ? undefined : HISTORIES.get(id)
  if (!h || h.index <= 0) return
  jumpToHistory(h.index - 1)
}

export function redoHistory(): void {
  const id = state.currentImageId
  const h = id === null ? undefined : HISTORIES.get(id)
  if (!h || h.index >= h.steps.length - 1) return
  jumpToHistory(h.index + 1)
}

/** 清空当前图的历史，只保留当前状态作为第 1 步。 */
export function clearHistory(): void {
  const id = state.currentImageId
  if (id === null) return
  const snap = JSON.stringify(toRaw(state.adjustments))
  HISTORIES.set(id, { steps: [{ label: '当前状态', at: Date.now(), snap }], index: 0, bytes: snap.length })
  syncHistoryView()
}

export function dropHistory(id: number): void {
  HISTORIES.delete(id)
}

/** 带描述的提交（裁剪/旋转/重置这类非参数差异一步特例）。 */
export function notifyCommittedLabeled(label: string): void {
  pendingLabel = label
  try { notifyCommitted() } finally { pendingLabel = '' }
}

// ---- 直方图：full 帧后节流拉取（对照旧版 0.3s 节流）----
let lastHistAt = 0
function scheduleHistogram(): void {
  const now = performance.now()
  if (now - lastHistAt < 300) return
  lastHistAt = now
  api.fetchHistogram().then((h) => { state.histogram = h }).catch(() => { /* 忽略 */ })
}

// ------------------------------------------------------------ 多图导入 / 切换
function _resetEditState(): void {
  state.editMode = null
  state.maskSelected = null
  state.maskAdjusting = false
  state.maskHover = false
  state.maskReveal = false
  state.polyDrawing = false
  polyBackupPoints = null
  state.pickTarget = null
  state.pickSelected = null
  state.stampSource = null
  state.cropRect = null
  state.cropAngle = 0
  state.comparing = false
  state.dofPicking = false
}

function _stashParams(): void {
  if (state.currentImageId !== null && state.loaded) {
    imageParams.set(state.currentImageId,
      JSON.parse(JSON.stringify(toRaw(state.adjustments))) as Adjustments)
  }
}

/** 对比用原位图：浏览器模式直接解码上传字节，Electron 模式向服务端取原图。 */
async function _ensureOriginal(id: number): Promise<void> {
  if (originalBmps.has(id)) {
    if (state.currentImageId === id) state.originalBmp = originalBmps.get(id)!
    return
  }
  let bmp: ImageBitmap | null = null
  try {
    const blob = originalBlobs.get(id)
      ?? await (await fetch(api.originalUrl(id))).blob()
    bmp = await createImageBitmap(blob)
  } catch { bmp = null }
  if (bmp) originalBmps.set(id, bmp)
  if (state.currentImageId === id) state.originalBmp = bmp     // 竞态防护：迟到的解码不覆盖新图
}

function _applyLoaded(meta: { id: number; name?: string; width: number; height: number;
                              serial: number; dof?: api.DofParams | null }, status = ''): void {
  state.currentImageId = meta.id
  state.width = meta.width
  state.height = meta.height
  state.serial = meta.serial
  if (meta.dof) {
    state.dofActive = true
    state.dofFocus = meta.dof.focus
    state.dofAperture = meta.dof.aperture
    state.dofRange = meta.dof.dof_range
    state.dofBokeh = meta.dof.bokeh_highlight
  } else {
    state.dofActive = false
  }
  state.dofFocusPt = null
  state.dofPreviewPng = ''
  state.previewFrame?.bmp.close()
  state.fullFrame?.bmp.close()
  state.previewFrame = null
  state.fullFrame = null
  state.geomW = 0
  state.geomH = 0
  newestRid = -1
  _resetEditState()
  state.originalBmp = null
  state.adjustments = imageParams.get(meta.id) ?? defaultAdjustments()
  state.activeFilter = ''
  state.loaded = true
  state.status = status
  beginHistory()                       // 每图独立历史：首次打开以当前参数为第 1 步
  state.histogram = null
  void _ensureOriginal(meta.id)
  void refreshLibrary()
  sendFull()
}

export async function refreshLibrary(): Promise<void> {
  try { state.library = (await api.getLibrary()).images } catch { /* 忽略 */ }
}

/** 页面加载时恢复服务端已有图库（F5 后参数不保，图像与文档状态仍在）。 */
export async function initLibrary(): Promise<void> {
  if (state.loaded) return
  try {
    const r = await api.getLibrary()
    state.library = r.images
    const cur = r.images.find(i => i.current)
    if (cur) {
      const meta = await api.selectLibrary(cur.id)
      _applyLoaded(meta)
    }
  } catch { /* 服务端未就绪等情况忽略 */ }
}

/** 浏览器模式：一次导入多张（依次上传入列，选中新导入的最后一张）。 */
export async function openFiles(files: File[]): Promise<void> {
  if (!files.length) return
  _stashParams()
  let last: api.OpenResult | null = null
  try {
    for (const f of files) {
      const meta = await api.openUpload(f)
      originalBlobs.set(meta.id, f)
      last = meta
    }
  } catch (err) {
    state.status = `打开失败：${err instanceof Error ? err.message : err}`
    if (last) _applyLoaded(last)
    return
  }
  if (last) _applyLoaded(last, files.length > 1 ? `已导入 ${files.length} 张图像` : '')
}

/** 兼容入口：单文件打开（自动化测试钩子同款）。 */
export async function openFile(file: File): Promise<void> {
  return openFiles([file])
}

/** Electron 模式：按路径逐张入列（支持 RAW、多选）。 */
export async function openPaths(paths: string[]): Promise<void> {
  if (!paths.length) return
  _stashParams()
  let last: api.OpenResult | null = null
  try {
    for (const p of paths) last = await api.openPath(p)
  } catch (err) {
    state.status = `打开失败：${err instanceof Error ? err.message : err}`
    if (last) _applyLoaded(last)
    return
  }
  if (last) _applyLoaded(last, paths.length > 1 ? `已导入 ${paths.length} 张图像` : '')
}

/** 兼容入口：单路径打开。 */
export async function openByPath(path: string): Promise<void> {
  return openPaths([path])
}

/** 切换当前图：暂存当前参数 -> 服务端切文档 -> 载入目标参数并全图渲染。 */
export async function selectImage(id: number): Promise<void> {
  if (id === state.currentImageId) return
  _stashParams()
  try {
    const meta = await api.selectLibrary(id)
    _applyLoaded(meta)
  } catch (err) {
    state.status = `切换失败：${err instanceof Error ? err.message : err}`
  }
}

/** 从图库删除一张（删当前图时服务端自动切到相邻一张，前端跟随）。 */
export async function removeImage(id: number): Promise<void> {
  try {
    const r = await api.removeLibrary(id)
    imageParams.delete(id)
    dropHistory(id)
    originalBmps.get(id)?.close()
    originalBmps.delete(id)
    originalBlobs.delete(id)
    state.library = r.images
    const cur = r.images.find(i => i.current)
    if (!cur) {                                  // 图库清空
      state.currentImageId = null
      state.loaded = false
      state.adjustments = defaultAdjustments()
      state.previewFrame?.bmp.close()
      state.fullFrame?.bmp.close()
      state.previewFrame = null
      state.fullFrame = null
      state.geomW = 0
      state.geomH = 0
      state.originalBmp?.close()
      state.originalBmp = null
      state.histogram = null
      state.history = []
      state.historyIndex = -1
      newestRid = -1
      _resetEditState()
      state.status = '图库已清空'
    } else if (cur.id !== state.currentImageId) {
      _applyLoaded(cur)
    }
  } catch (err) {
    state.status = `删除失败：${err instanceof Error ? err.message : err}`
  }
}

/** 对比原图（按住 `\`）：显示打开时的原始位图。 */
export function setComparing(on: boolean): void {
  if (state.loaded && state.originalBmp) state.comparing = on
}

export function resetAll(): void {
  state.adjustments = defaultAdjustments()
  state.activeFilter = ''
  exitMaskEdit()
  notifyCommittedLabeled('重置全部')
}

export function toggleBlackwhite(): void {
  state.adjustments.blackwhite = state.adjustments.blackwhite > 0 ? 0 : 100
  notifyCommitted()
}

export async function saveBrowser(format: string): Promise<void> {
  await api.renderDownload(state.adjustments, format, `fernlens.${format}`)
  state.status = `已下载 fernlens.${format}`
}

/** auto.suggest_* 结果按旧版 _apply_adjustment_dict 语义合并（M1 基础影调 + 曲线/色轮/取色点）。 */
export function applyAdjustmentDict(d: Record<string, unknown>): void {
  const a = state.adjustments
  for (const name of TONE_FIELDS) {
    const v = d[name]
    if (typeof v === 'number' && Number.isFinite(v)) (a as unknown as Record<string, number>)[name] = v
  }
  const curves = d['curves'] as Record<string, unknown> | undefined
  if (curves && typeof curves === 'object') {
    for (const ch of CURVE_CHANNELS) {
      const pts = curves[ch]
      if (Array.isArray(pts)) {
        a.curves[ch] = pts
          .filter((p): p is [number, number] => Array.isArray(p) && p.length === 2)
          .map(([x, y]) => [Number(x) || 0, Number(y) || 0] as [number, number])
      }
    }
  }
  const wheels = d['wheels'] as Record<string, unknown> | undefined
  if (wheels && typeof wheels === 'object') {
    for (const range of GRADING_RANGES) {
      const w = wheels[range] as Partial<WheelState> | undefined
      if (w && typeof w === 'object') {
        a.wheels[range] = {
          hue: Number(w.hue) || 0,
          saturation: Number(w.saturation) || 0,
          luminance: Number(w.luminance) || 0,
        }
      }
    }
  }
  if (Array.isArray(d['pickers'])) {
    a.pickers = (d['pickers'] as Record<string, unknown>[]).map((p) => ({
      hue: Number(p.hue) || 0,
      color: Array.isArray(p.color) && p.color.length === 3
        ? [Number(p.color[0]) || 0, Number(p.color[1]) || 0, Number(p.color[2]) || 0]
        : [0.5, 0.5, 0.5],
      span: Number(p.span) || 30,
      hue_shift: Number(p.hue_shift) || 0,
      saturation: Number(p.saturation) || 0,
      luminance: Number(p.luminance) || 0,
      enabled: true,
    } satisfies PickerState))
  }
  notifyCommitted()
}

// ------------------------------------------------------------ 滤镜（预设 look）
/** 套用滤镜时抑制「手改参数清除高亮」的逻辑。 */
let filterApplying = false

/** 基础滤镜 + 自定义滤镜（组件渲染用）。 */
export function allFilters(): FilterPreset[] {
  return [...BUILTIN_FILTERS, ...state.customFilters]
}

/** 套用滤镜：独占语义——先把全局影调/曲线/分级复位到默认，再写入滤镜 look，
 *  所以任意时刻只有一个滤镜生效，换滤镜不会把两者叠在一起。
 *  蒙版、取色点、裁剪/图章仍不受影响（它们属于具体画面，不属于滤镜）。 */
export function applyFilter(p: FilterPreset): void {
  if (!state.loaded) return
  if (state.maskSelected !== null) exitMaskEdit()      // 滤镜是全局的，先退出局部编辑
  filterApplying = true
  try {
    resetGlobalLook(state.adjustments)
    applyLook(state.adjustments, p.look)
    state.activeFilter = p.id
    notifyCommittedLabeled(`滤镜 · ${p.name}`)
    state.status = `已应用滤镜「${p.name}」· 替换之前的滤镜（可撤销）`
  } finally {
    filterApplying = false
  }
}

/** 把当前全局参数存成新滤镜（名称去首尾空格，最长 40 字）。 */
export function saveCurrentAsFilter(name: string): FilterPreset | null {
  const n = name.trim().slice(0, 40)
  if (!n) { state.status = '滤镜名称不能为空'; return null }
  const p: FilterPreset = {
    id: `custom:${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name: n,
    group: '自定义',
    look: lookFromAdjustments(state.adjustments),
    createdAt: Date.now(),
  }
  state.customFilters.unshift(p)
  persistCustomFilters()
  state.activeFilter = p.id
  state.status = `已保存滤镜「${n}」`
  return p
}

export function deleteCustomFilter(id: string): void {
  const i = state.customFilters.findIndex(f => f.id === id)
  if (i < 0) return
  const [removed] = state.customFilters.splice(i, 1)
  if (state.activeFilter === id) state.activeFilter = ''
  persistCustomFilters()
  state.status = `已删除滤镜「${removed.name}」`
}

/** 导出滤镜 JSON：传 id 列表则只导这些，不传则导出全部自定义滤镜。 */
export function exportFilters(ids?: string[]): void {
  const list = ids && ids.length
    ? state.customFilters.filter(f => ids.includes(f.id))
    : state.customFilters
  if (!list.length) { state.status = '没有可导出的自定义滤镜'; return }
  const payload = list.length === 1
    ? filterToFile(list[0])
    : { format: 'fernlens-filter', version: 1, filters: list.map(filterToFile) }
  downloadJson(filterFileName(list.length === 1 ? list[0].name : '全部'), payload)
  state.status = `已导出 ${list.length} 个滤镜`
}

/** 从 JSON 导入滤镜（支持单个 / 数组 / {filters:[]} 三种写法）。 */
export async function importFiltersFromFile(file: File): Promise<number> {
  try {
    const text = await file.text()
    const list = parseFilterFile(text, file.name.replace(/\.json$/i, ''))
    state.customFilters.unshift(...list)
    persistCustomFilters()
    state.status = `已导入 ${list.length} 个滤镜`
    return list.length
  } catch (err) {
    state.status = `导入失败：${err instanceof Error ? err.message : err}`
    throw err
  }
}

export async function autoRule(): Promise<void> {
  const r = await api.autoGrade('rule')
  applyAdjustmentDict(r.adjustments)
  state.status = r.note || '已应用规则自动调色'
}

export async function autoAI(): Promise<void> {
  if (!aiConfigured()) {
    state.status = '未配置 AI 模型：请在「AI 设置」中填写 URL / API Key / 模型 ID'
    openAiSettings()
    return
  }
  state.status = state.aiIntent ? 'AI 精修中…' : 'AI 精修中…（未填写要求，由模型自选风格）'
  const r = await api.autoGrade('ai', state.aiConfig, state.aiIntent)
  applyAdjustmentDict(r.adjustments)
  state.status = (r.note || '已应用 AI 精修') + `（${r.model}）`
}

// ------------------------------------------------------------ 取色器（对照旧 PickerPanel 交互）
export function armNewPick(): void {
  if (!state.loaded) return
  state.pickTarget = 'new'
}

export function armRepick(index: number): void {
  if (!state.loaded) return
  state.pickTarget = index
  state.pickSelected = index
}

export function cancelPick(): void {
  state.pickTarget = null
}

/** 新建预设色相取色点（无需图像取色）。 */
export function newPresetPicker(hueDeg: number): void {
  targetAdj().pickers.push({
    hue: hueDeg, color: presetColor(hueDeg), span: 30,
    hue_shift: 0, saturation: 0, luminance: 0, enabled: true,
  })
  state.pickSelected = targetAdj().pickers.length - 1
  notifyCommitted()
}

export function deletePicker(index: number): void {
  const list = targetAdj().pickers
  if (index < 0 || index >= list.length) return
  list.splice(index, 1)
  if (state.pickSelected !== null && state.pickSelected >= list.length) {
    state.pickSelected = null
  }
  notifyCommitted()
}

/** 视口单击取色：全图像素坐标 -> /api/pick -> 新建或更新取色点。 */
export async function imagePicked(x: number, y: number): Promise<void> {
  if (!state.loaded || state.pickTarget === null) return
  const patch = Math.max(3, Math.round(state.height * 0.01))   // 对照旧 _on_image_picked
  try {
    const r = await api.pickAt(Math.round(x), Math.round(y), patch)
    const color: [number, number, number] = [r.r, r.g, r.b]
    const list = targetAdj().pickers
    if (state.pickTarget === 'new') {
      list.push({
        hue: r.hue, color, span: 30, hue_shift: 0,
        saturation: 25, luminance: 0, enabled: true,
      })
      state.pickSelected = list.length - 1
    } else if (typeof state.pickTarget === 'number') {
      const p = list[state.pickTarget]
      if (p) { p.hue = r.hue; p.color = color }
    }
    state.pickTarget = null
    notifyCommitted()
  } catch (err) {
    state.status = `取色失败：${err instanceof Error ? err.message : err}`
  }
}

export type EditorTab = 'light' | 'color' | 'effects' | 'filter' | 'curve' | 'grading' | 'mask' | 'picker' | 'history'

// ------------------------------------------------------------ 蒙版（对照旧 ui_window 蒙版面板语义）
const MASK_KIND_NAMES: Record<MaskKind, string> = {
  linear: '线性渐变', radial: '径向渐变', brush: '画笔', polygon: '多边形', ai: 'AI 识别',
}

let polyBackupPoints: [number, number][] | null = null

export function setActiveTab(next: EditorTab): void {
  if (state.polyDrawing && next !== 'mask') {
    finishOrDiscardPendingPolygon()
  }
  state.activeTab = next
  if (state.maskSelected !== null) {
    state.maskAdjusting = next !== 'mask'
    if (next === 'mask') state.maskReveal = false
  }
}

/** AI 识别目标的显示名（与 auto.AI_MASK_TARGETS 对应）。 */
export const AI_TARGET_NAMES: Record<AiMaskTarget, string> = {
  subject: '主体',
  person: '人物',
  background: '背景',
  sky: '天空',
  depth_near: '近景深度',
  depth_far: '远景深度',
}

/** AI 识别蒙版（LR 式「选择主体/人物/背景/天空/景深」）。
 *  只走本地像素级引擎（SAM2 + U²-Net + Depth Anything V2）：图像不出本机、免费，不消耗 API。 */
export async function addAiMask(target: AiMaskTarget): Promise<void> {
  if (!state.loaded || state.aiBusy) return
  const isDepth = target === 'depth_near' || target === 'depth_far'
  const ready = isDepth
    ? !!(state.localAI?.models?.depth_anything_v2?.ready || state.localAI?.ready)
    : !!state.localAI?.ready
  if (!ready) {
    state.status = tx(
      '本地蒙版引擎未就绪：请在「AI 设置」里下载 HuggingFace 本地模型',
      'Local model engine not ready: download HuggingFace models in AI Setup',
    )
    openAiSettings()
    return
  }
  state.aiBusy = target
  state.status = tx(
    `AI 识别「${AI_TARGET_NAMES[target]}」中…（本地模型推理中）`,
    `Running local AI mask (${target})…`,
  )
  try {
    const r = await api.aiMask(target)
    finishOrDiscardPendingPolygon()
    const idx = state.adjustments.masks.length + 1
    const m: MaskState = aiMaskPixels(`AI ${AI_TARGET_NAMES[target]} ${idx}`, target,
      r.mask_png ?? '', r.width ?? 0, r.height ?? 0)
    state.adjustments.masks.push(m)
    state.maskSelected = state.adjustments.masks.length - 1
    state.editMode = 'mask'
    state.maskAdjusting = false
    state.polyDrawing = false
    state.status = tx(
      `已创建「${AI_TARGET_NAMES[target]}」蒙版（像素级），可直接在右侧做局部调整`,
      `Created pixel-level AI mask (${target})`,
    )
    notifyCommittedLabeled(`AI ${target}`)
  } catch (err) {
    state.status = `AI Mask error: ${err instanceof Error ? err.message : err}`
  } finally {
    state.aiBusy = null
  }
}

/** AI 景深调节（Depth Anything V2）：画布点选对焦 + 光学散景虚化应用/清除。 */
export function setDofPicking(on: boolean): void {
  state.dofPicking = on
  if (on) {
    state.pickTarget = null
    if (state.editMode === 'stamp') state.editMode = null
  }
}

export async function pickDofFocusAt(nx: number, ny: number): Promise<void> {
  if (!state.loaded) return
  const cx = Math.max(0, Math.min(1, nx))
  const cy = Math.max(0, Math.min(1, ny))
  state.status = tx('正在用 Depth Anything V2 识别对焦点深度…', 'Sampling focal plane with Depth Anything V2…')
  try {
    const res = await api.aiDofPick(cx, cy, state.dofRange)
    state.dofFocus = res.focus
    state.dofFocusPt = [cx, cy]
    state.dofPreviewPng = res.depth_preview_png
    state.dofPicking = false
    if (state.dofAperture > 0) {
      await applyDof()
    } else {
      state.status = tx(`对焦深度已设为 ${Math.round(res.focus * 100)}%`, `Focal plane set to ${Math.round(res.focus * 100)}%`)
    }
  } catch (err) {
    state.dofPicking = false
    state.status = `${tx('对焦取点失败：', 'Focus pick failed: ')}${err instanceof Error ? err.message : err}`
    throw err
  }
}

export async function applyDof(opts?: Partial<api.DofParams>): Promise<api.DofResult | null> {
  if (!state.loaded) return null
  if (opts?.focus !== undefined) state.dofFocus = opts.focus
  if (opts?.aperture !== undefined) state.dofAperture = opts.aperture
  if (opts?.dof_range !== undefined) state.dofRange = opts.dof_range
  if (opts?.bokeh_highlight !== undefined) state.dofBokeh = opts.bokeh_highlight
  state.status = tx('AI 景深虚化计算中…', 'Rendering AI Depth of Field bokeh…')
  try {
    const res = await api.aiDofApply({
      focus: state.dofFocus,
      aperture: state.dofAperture,
      dof_range: state.dofRange,
      bokeh_highlight: state.dofBokeh,
    })
    state.serial = res.serial
    state.dofActive = res.active
    state.dofPreviewPng = res.depth_preview_png
    state.status = res.active
      ? tx(
          `已应用 AI 景深虚化（光圈强度 ${Math.round(res.aperture)}%，对焦 ${Math.round(res.focus * 100)}%）`,
          `Applied AI Depth of Field (blur ${Math.round(res.aperture)}%, focus ${Math.round(res.focus * 100)}%)`,
        )
      : tx('AI 景深强度为 0，保持原图清晰', 'AI Depth of Field blur set to 0')
    notifyCommittedLabeled(tx('AI 景深虚化', 'AI Depth of Field'))
    return res
  } catch (err) {
    state.status = `${tx('AI 景深虚化失败：', 'AI Depth of Field failed: ')}${err instanceof Error ? err.message : err}`
    throw err
  }
}

export async function clearDof(): Promise<void> {
  if (!state.loaded) return
  try {
    const res = await api.aiDofClear()
    state.serial = res.serial
    state.dofActive = false
    state.dofFocusPt = null
    state.dofPicking = false
    state.status = tx('已清除 AI 景深虚化，恢复原图清晰', 'Cleared AI Depth of Field')
    notifyCommittedLabeled(tx('清除 AI 景深', 'Clear AI DOF'))
  } catch (err) {
    state.status = `${tx('清除景深失败：', 'Clear DOF failed: ')}${err instanceof Error ? err.message : err}`
    throw err
  }
}

/** AI 景深立体调色（Depth Anything V2）：生成/更新近景与远景蒙版并注入预设参数。 */
export async function runDepthRelight(opts: {
  preset: api.DepthRelightPreset
  strength: number
  nearBias: number
  masksOnly?: boolean
}): Promise<api.DepthRelightResult | null> {
  if (!state.loaded) return null
  state.status = tx('AI 景深立体调色计算中…', 'Computing AI Depth Relight…')
  try {
    const res = await api.aiDepthRelight(
      opts.preset,
      opts.strength,
      opts.nearBias,
      Boolean(opts.masksOnly),
    )
    // 替换之前生成的 AI 景深蒙版，保留用户其他蒙版
    finishOrDiscardPendingPolygon()
    const kept = state.adjustments.masks.filter(
      (m) => !(m.kind === 'ai' && (m.ai_target === 'depth_near' || m.ai_target === 'depth_far')),
    )
    state.adjustments.masks = [...kept, ...res.masks]
    state.maskSelected = kept.length < state.adjustments.masks.length ? kept.length : null
    if (state.maskSelected !== null) {
      state.editMode = 'mask'
      state.maskAdjusting = state.activeTab !== 'mask'
    }
    state.polyDrawing = false
    state.status = tx(
      `已应用 AI 景深立体调色（${res.preset}，共生成 ${res.masks.length} 个深度蒙版）`,
      `Applied AI Depth Relight (${res.preset}) with ${res.masks.length} depth masks`,
    )
    notifyCommittedLabeled(`AI Depth (${res.preset})`)
    return res
  } catch (err) {
    state.status = `${tx('AI 景深失败：', 'Depth Relight failed: ')}${err instanceof Error ? err.message : err}`
    throw err
  }
}

/** AI 参考仿色：提取参考图色调/三路分级/影调参数并应用到当前照片（支持 Ctrl+Z 撤销）。 */
export async function runColorMatch(opts: {
  file?: File
  refId?: number
  strength: number
  protectSubject: boolean
}): Promise<api.ColorMatchResult | null> {
  if (!state.loaded) return null
  state.status = tx('AI 参考仿色提取中…', 'Extracting reference color profile…')
  try {
    const res = await api.aiColorMatch(opts)
    const patch = res.adjustments || {}
    const a = state.adjustments
    for (const [k, v] of Object.entries(patch)) {
      if (v === undefined) continue
      if (k === 'curves' && v && typeof v === 'object') {
        a.curves = { ...a.curves, ...(v as Adjustments['curves']) }
      } else if (k === 'wheels' && v && typeof v === 'object') {
        a.wheels = { ...a.wheels, ...(v as Adjustments['wheels']) }
      } else if (k === 'bw' || k === 'bw_mix' || k === 'masks' || k === 'pickers') {
        // preserve unless explicitly provided cleanly
      } else {
        ;(a as unknown as Record<string, unknown>)[k] = v
      }
    }
    state.activeFilter = ''
    state.status = res.note || tx('已应用 AI 参考仿色', 'Applied AI Color Match')
    notifyCommittedLabeled('AI Color Match')
    return res
  } catch (err) {
    state.status = `${tx('参考仿色失败：', 'Color Match failed: ')}${err instanceof Error ? err.message : err}`
    throw err
  }
}

/** 面板目标：选中蒙版时其局部参数，否则全局参数（曲线/色轮/取色器/滑块共用）。 */
export function targetAdj(): AdjustmentsLike {
  const i = state.maskSelected
  if (i !== null && state.adjustments.masks[i]) return state.adjustments.masks[i].adj
  return state.adjustments
}

function finishOrDiscardPendingPolygon(): void {
  if (!state.polyDrawing) return
  const idx = state.maskSelected
  state.polyDrawing = false
  if (idx !== null) {
    const m = state.adjustments.masks[idx]
    if (m && m.kind === 'polygon' && m.points.length < 3) {
      if (polyBackupPoints && polyBackupPoints.length >= 3) {
        m.points = polyBackupPoints
      } else {
        state.adjustments.masks.splice(idx, 1)
        state.maskSelected = null
        if (state.editMode === 'mask') state.editMode = null
      }
    } else if (m && m.kind === 'polygon' && m.points.length >= 3) {
      notifyCommitted()
    }
  }
  polyBackupPoints = null
}

export function addMask(kind: MaskKind): void {
  if (!state.loaded) return
  finishOrDiscardPendingPolygon()
  const m = defaultMask(kind, `${MASK_KIND_NAMES[kind]} ${state.adjustments.masks.length + 1}`)
  state.adjustments.masks.push(m)
  state.maskSelected = state.adjustments.masks.length - 1
  state.editMode = 'mask'
  state.maskAdjusting = false
  state.maskReveal = false
  if (kind === 'polygon') {
    polyBackupPoints = null
    state.polyDrawing = true
    state.status = tx(
      '点击图片选取多边形顶点（至少 3 个点），点击起点、双击或按「确认」完成',
      'Click on image to pick polygon vertices (min 3); click start, double-click, or Confirm',
    )
    return
  }
  state.polyDrawing = false
  notifyCommitted()
}

export function selectMask(index: number | null): void {
  // 再点一次已选中的蒙版 = 退出局部编辑（对照 LR 的 Done）
  if (index !== null && state.maskSelected === index) {
    exitMaskEdit()
    return
  }
  if (state.polyDrawing) {
    const prevIdx = state.maskSelected
    const removedBefore = prevIdx !== null
      && state.adjustments.masks[prevIdx]?.kind === 'polygon'
      && state.adjustments.masks[prevIdx].points.length < 3
      && !(polyBackupPoints && polyBackupPoints.length >= 3)
    finishOrDiscardPendingPolygon()
    if (removedBefore && index !== null && prevIdx !== null && index > prevIdx) {
      index -= 1
    }
  }
  state.maskSelected = index
  state.polyDrawing = false
  if (index !== null) {
    state.editMode = 'mask'
    state.pickTarget = null
    state.maskAdjusting = state.activeTab !== 'mask'
    state.maskReveal = false
  } else if (state.editMode === 'mask') {
    state.editMode = null
    state.maskAdjusting = false
  }
}

export function exitMaskEdit(): void {
  finishOrDiscardPendingPolygon()
  state.maskSelected = null
  state.maskHover = false
  state.maskReveal = false
  state.maskAdjusting = false
  state.polyDrawing = false
  polyBackupPoints = null
  if (state.editMode === 'mask') state.editMode = null
  state.status = ''
}

export function toggleMaskOverlay(): void {
  if (state.maskOverlay && state.maskAdjusting) {
    state.maskAdjusting = false
    state.maskHover = false
    state.maskReveal = false
    return
  }
  state.maskOverlay = !state.maskOverlay
  if (!state.maskOverlay) {
    state.maskHover = false
    state.maskReveal = false
  } else {
    state.maskAdjusting = false
  }
}

/** 短暂显示红色叠加（预览按钮使用）。 */
let maskRevealTimer: ReturnType<typeof setTimeout> | undefined
export function revealMaskOverlay(ms = 1600): void {
  state.maskReveal = true
  if (maskRevealTimer) clearTimeout(maskRevealTimer)
  maskRevealTimer = setTimeout(() => { state.maskReveal = false }, ms)
}

/** 鼠标悬停蒙版条目/图钉时临时显示叠加。 */
export function setMaskHover(on: boolean): void {
  state.maskHover = on
}

/** 当前是否应绘制半透明红色叠加：
 *  蒙版选中后始终显示；在针对蒙版做色彩/影调调整时自动隐藏以方便预览；退出后不显示。 */
export function maskOverlayVisible(): boolean {
  if (state.editMode !== 'mask' || state.maskSelected === null) return false
  if (!state.maskOverlay) return false
  if (state.maskHover || state.maskReveal) return true
  return !state.maskAdjusting
}

export function deleteMask(index: number): void {
  const list = state.adjustments.masks
  if (index < 0 || index >= list.length) return
  if (state.maskSelected === index) {
    state.polyDrawing = false
    polyBackupPoints = null
  }
  list.splice(index, 1)
  if (state.maskSelected !== null) {
    if (state.maskSelected === index) {
      state.maskSelected = list.length ? Math.min(index, list.length - 1) : null
    } else if (state.maskSelected > index) {
      state.maskSelected -= 1
    }
    if (state.maskSelected === null) {
      state.editMode = null
      state.maskAdjusting = false
    }
  }
  notifyCommitted()
}

export function undoPolygonPoint(): void {
  if (!state.polyDrawing || state.maskSelected === null) return
  const m = state.adjustments.masks[state.maskSelected]
  if (m?.kind === 'polygon' && m.points.length) {
    m.points.pop()
  }
}

export function clearPolygonPoints(): void {
  if (!state.polyDrawing || state.maskSelected === null) return
  const m = state.adjustments.masks[state.maskSelected]
  if (m?.kind === 'polygon') {
    m.points = []
  }
}

export function confirmPolygonMask(): void {
  if (!state.polyDrawing || state.maskSelected === null) return
  const m = state.adjustments.masks[state.maskSelected]
  if (!m || m.kind !== 'polygon' || m.points.length < 3) return
  state.polyDrawing = false
  polyBackupPoints = null
  state.maskAdjusting = false
  state.status = tx(
    `已生成多边形蒙版（${m.points.length} 个顶点），可调整羽化或切换面板进行局部调色`,
    `Polygon mask created (${m.points.length} vertices); adjust feather or switch tabs to grade`,
  )
  notifyCommitted()
}

export function cancelPolygonDrawing(): void {
  if (!state.polyDrawing || state.maskSelected === null) return
  const idx = state.maskSelected
  const m = state.adjustments.masks[idx]
  state.polyDrawing = false
  if (polyBackupPoints && polyBackupPoints.length >= 3 && m?.kind === 'polygon') {
    m.points = polyBackupPoints
    polyBackupPoints = null
    state.status = ''
    return
  }
  polyBackupPoints = null
  if (m?.kind === 'polygon') {
    state.adjustments.masks.splice(idx, 1)
    state.maskSelected = null
    if (state.editMode === 'mask') state.editMode = null
  }
  state.status = ''
}

export function redrawPolygonMask(index: number): void {
  const m = state.adjustments.masks[index]
  if (!m || m.kind !== 'polygon') return
  finishOrDiscardPendingPolygon()
  polyBackupPoints = m.points.map(([x, y]) => [x, y] as [number, number])
  m.points = []
  state.maskSelected = index
  state.editMode = 'mask'
  state.maskAdjusting = false
  state.polyDrawing = true
  state.status = tx(
    '请重新在图片上选择多边形顶点（至少 3 个点）',
    'Click on image to re-pick polygon vertices (min 3)',
  )
}

/** 蒙版几何变化（视口拖动手柄/端点）：只需 preview；松手时 commit。 */
export function maskGeometryChanged(): void {
  notifyChanged()
}
export function maskGeometryCommitted(): void {
  notifyCommitted()
}

// ---- 线性渐变旋转 ----
/** 线性渐变角度（起点->终点方向，0°=水平向右，逆时针为正，0..360）。 */
export function linearMaskAngle(m: MaskState): number {
  const deg = -Math.atan2(m.y2 - m.y1, m.x2 - m.x1) * 180 / Math.PI
  return (deg + 360) % 360
}

/** 把线性渐变绕中线中心旋转到绝对角度（保持长度与中心；端点允许出画布）。 */
export function setLinearMaskAngle(index: number, deg: number): void {
  const m = state.adjustments.masks[index]
  if (!m || m.kind !== 'linear') return
  const cx = (m.x1 + m.x2) / 2, cy = (m.y1 + m.y2) / 2
  const half = Math.max(Math.hypot(m.x2 - m.x1, m.y2 - m.y1) / 2, 1e-3)
  const rad = -(deg % 360) * Math.PI / 180   // 视觉逆时针为正 -> 屏幕 y 向下取负
  m.x1 = cx - Math.cos(rad) * half
  m.y1 = cy - Math.sin(rad) * half
  m.x2 = cx + Math.cos(rad) * half
  m.y2 = cy + Math.sin(rad) * half
  notifyChanged()
}

/** 相对旋转（键盘 Q/E：±1°，Shift ±15°）。 */
export function rotateLinearMask(index: number, deltaDeg: number): void {
  const m = state.adjustments.masks[index]
  if (!m || m.kind !== 'linear') return
  setLinearMaskAngle(index, linearMaskAngle(m) + deltaDeg)
}

/** 画笔：笔画即几何，笔画结束提交全图。 */
export function brushStrokeCommitted(index: number): void {
  void index
  notifyCommitted()
}
export function undoBrushStroke(index: number): void {
  const m = state.adjustments.masks[index]
  if (m?.strokes.length) {
    m.strokes.pop()
    notifyCommitted()
  }
}
export function clearBrushStrokes(index: number): void {
  const m = state.adjustments.masks[index]
  if (m?.strokes.length) {
    m.strokes = []
    notifyCommitted()
  }
}

// ------------------------------------------------------------ 图章（Photoshop 式画笔仿制 + 兼容旧矩形）
export function stampBrushRadiusNorm(): number {
  return Math.max(0.004, Math.min(0.16, state.stampSize * 0.0022))
}

export function setStampSourcePt(pt: [number, number]): void {
  state.stampSourcePt = [
    Math.max(0, Math.min(1, pt[0])),
    Math.max(0, Math.min(1, pt[1])),
  ]
  state.stampPickingSource = false
  state.stampOffset = null
  state.status = tx(
    '已设定仿制取样点：左键拖拽涂抹覆盖（Alt+点击重新取样，[ / ] 调笔刷大小）',
    'Clone source set: drag to paint over targets (Alt+Click to re-sample, [ / ] resize brush)',
  )
}

export function toggleStampPicking(): void {
  state.stampPickingSource = !state.stampPickingSource
  if (state.stampPickingSource) {
    state.status = tx('请在画面上点击设置仿制取样点', 'Click on the image to set clone sample point')
  }
}

export function adjustStampSize(delta: number): void {
  state.stampSize = Math.max(2, Math.min(100, Math.round(state.stampSize + delta)))
}

export async function applyStampStroke(
  srcStart: [number, number],
  points: [number, number][],
): Promise<void> {
  if (!points.length) return
  try {
    await api.applyStampBrush(
      srcStart,
      points,
      stampBrushRadiusNorm(),
      Math.max(0, Math.min(1, state.stampFeather / 100)),
      Math.max(0.05, Math.min(1, state.stampOpacity / 100)),
    )
    state.serial++
    notifyCommittedLabeled(tx('仿制图章', 'Clone Stamp'))
  } catch (err) {
    state.status = `${tx('图章失败：', 'Clone stamp failed: ')}${err instanceof Error ? err.message : err}`
  }
}

export function setStampSource(rect: RectNorm): void {
  state.stampSource = rect
  state.status = tx('已选源区域：在目标处拖出矩形覆盖', 'Source area selected')
}

export async function applyStampAt(dst: RectNorm): Promise<void> {
  if (!state.stampSource) {
    state.status = tx('请先按住 Alt 设置取样点', 'Please Alt+Click to set sample source')
    return
  }
  try {
    await api.applyStamp(state.stampSource, dst, state.stampFeather)
    state.serial++
    notifyCommitted()
  } catch (err) {
    state.status = `${tx('图章失败：', 'Clone stamp failed: ')}${err instanceof Error ? err.message : err}`
  }
}

export async function stampUndo(): Promise<void> {
  try {
    await api.stampUndo()
    state.serial++
    notifyCommitted()
  } catch (err) {
    state.status = `${tx('撤销失败：', 'Undo failed: ')}${err instanceof Error ? err.message : err}`
  }
}

export async function stampClear(): Promise<void> {
  try {
    await api.stampClear()
    state.serial++
    state.stampSource = null
    state.stampSourcePt = null
    state.stampOffset = null
    notifyCommitted()
    state.status = tx('已清除全部图章修改', 'Cleared all clone stamp strokes')
  } catch (err) {
    state.status = `${tx('清除失败：', 'Clear failed: ')}${err instanceof Error ? err.message : err}`
  }
}

// ------------------------------------------------------------ 裁剪/旋转（服务端 crop_rotate 权威计算）
const ASPECT_MAP: Record<string, number | null> = {
  free: null, orig: 0, '1:1': 1, '3:2': 3 / 2, '4:3': 4 / 3, '16:9': 16 / 9,
}

export function enterCrop(): void {
  if (!state.loaded) return
  finishOrDiscardPendingPolygon()
  state.editMode = 'crop'
  state.cropRect = { x0: 0, y0: 0, x1: 1, y1: 1 }
  state.cropAngle = 0
  state.cropAspect = 'free'
  state.status = '拖动调整裁剪框；Enter 应用，Esc 取消'
}

export function cancelCrop(): void {
  if (state.editMode === 'crop') {
    state.editMode = null
    state.cropRect = null
    state.cropAngle = 0
    state.status = ''
  }
}

export async function applyCrop(): Promise<void> {
  if (state.editMode !== 'crop' || !state.cropRect || state.cropPending) return
  state.cropPending = true
  try {
    const r = await api.cropGeometry(
      [state.cropRect.x0, state.cropRect.y0, state.cropRect.x1, state.cropRect.y1],
      state.cropAngle, toRaw(state.adjustments.masks) as MaskState[])
    state.adjustments.masks = r.masks
    state.width = r.width
    state.height = r.height
    state.serial++
    state.editMode = null
    state.cropRect = null
    state.cropAngle = 0
    state.status = r.baked ? '已应用裁剪；图章修改已一并合并' : '已应用裁剪'
    notifyCommittedLabeled('裁剪/拉直')
  } catch (err) {
    state.status = `裁剪失败：${err instanceof Error ? err.message : err}`
  } finally {
    state.cropPending = false
  }
}

export async function rotate90(clockwise: boolean): Promise<void> {
  if (!state.loaded) return
  try {
    const r = await api.rotateGeometry(clockwise,
      toRaw(state.adjustments.masks) as MaskState[])
    state.adjustments.masks = r.masks
    state.width = r.width
    state.height = r.height
    state.serial++
    notifyCommittedLabeled(clockwise ? '右转 90°' : '左转 90°')
    state.status = clockwise ? '已右转 90°' : '已左转 90°'
  } catch (err) {
    state.status = `旋转失败：${err instanceof Error ? err.message : err}`
  }
}

/** 比例锁定换算（裁剪框拖动时由视口调用）：返回约束后的 rect。 */
export function constrainCropRect(
  rect: { x0: number; y0: number; x1: number; y1: number },
): { x0: number; y0: number; x1: number; y1: number } {
  const key = state.cropAspect.replace(':t', '')
  let ratio = ASPECT_MAP[key]
  if (ratio === undefined) ratio = null
  if (ratio === null) return rect
  const transposed = state.cropAspect.endsWith(':t')
  const w = state.width, h = state.height
  let target: number
  if (ratio === 0) {
    target = w / h                                  // 原始比例
  } else {
    target = ratio
  }
  if (transposed) target = 1 / target
  // 图像像素空间的比例（rect 为归一化坐标）
  const pixelW = (rect.x1 - rect.x0) * w
  const pixelH = (rect.y1 - rect.y0) * h
  const cur = pixelW / Math.max(pixelH, 1e-6)
  let nw = pixelW, nh = pixelH
  if (Math.abs(cur - target) > 1e-6) {
    if (cur > target) nw = nh * target
    else nh = nw / target
  }
  const cxN = (rect.x0 + rect.x1) / 2, cyN = (rect.y0 + rect.y1) / 2
  let x0 = cxN - nw / w / 2, x1 = cxN + nw / w / 2
  let y0 = cyN - nh / h / 2, y1 = cyN + nh / h / 2
  // 夹回 [0,1]
  if (x0 < 0) { x1 -= x0; x0 = 0 }
  if (y0 < 0) { y1 -= y0; y0 = 0 }
  if (x1 > 1) { x0 -= x1 - 1; x1 = 1 }
  if (y1 > 1) { y0 -= y1 - 1; y1 = 1 }
  return { x0: Math.max(0, x0), y0: Math.max(0, y0), x1: Math.min(1, x1), y1: Math.min(1, y1) }
}

// ---- 导出状态 ----
export function useEditor() {
  return state
}

// 自动化测试/调试钩子（生产保留：体积可忽略，供 UI 端到端测试驱动打开文件）
declare global {
  interface Window {
    __fernlens?: { openFile: typeof openFile; state: typeof state }
  }
}
window.__fernlens = { openFile, state }
