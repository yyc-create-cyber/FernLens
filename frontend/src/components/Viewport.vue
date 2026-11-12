<script setup lang="ts">
/** 主视口：图像显示 + 交互叠加（蒙版/裁剪/图章/取色）。
 *  行为对照旧 PreviewWidget 与 LR：
 *  - 线性渐变 = LR 原版：三条垂直线（起点/中心/终点），渐变沿轴投影、垂直方向
 *    无限延伸（与 core._linear_mask 语义一致）；拖端线移动、拖中线整体平移。
 *  - 画笔：正在涂/悬停时显示低透明度红色涂刷区域 + 光标圈；O 键开关叠加。
 *  - 未选中的蒙版只显示编号 pin（LR 式）；AI 蒙版红色叠加改悬停/刚创建时短显。
 *  - 径向：中心/长轴/短轴缩放/旋转手柄。
 *  缩放平移/快捷键对照旧 PreviewWidget：0/1/+/-、双击适应、滚轮以光标为中心。 */
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useEditor, imagePicked, setStampSource, applyStampAt,
         maskGeometryChanged, maskGeometryCommitted, brushStrokeCommitted,
         constrainCropRect, setLinearMaskAngle,
         maskOverlayVisible, setMaskHover, revealMaskOverlay,
         setViewEdge, requestDetail } from '@/stores/editor'
import Icon from '@/components/Icon'
import type { MaskState } from '@/api/types'
import type { RectNorm } from '@/api/client'

const store = useEditor()
const wrap = ref<HTMLDivElement | null>(null)
const canvas = ref<HTMLCanvasElement | null>(null)

const view = reactive({ manual: false, zoom: 1, x: 0, y: 0 })
const zoomPct = ref(100)
const MIN_ZOOM = 0.02
const MAX_ZOOM = 32

interface CropRect { x0: number; y0: number; x1: number; y1: number }

type Drag =
  | { kind: 'pan'; lastX: number; lastY: number }
  | { kind: 'mask-line'; index: number; which: 0 | 1 | 2 }   // 0=起点线 1=中线 2=终点线
  | { kind: 'mask-line-rotate'; index: number }              // 线性渐变旋转手柄
  | { kind: 'mask-radial-center'; index: number }
  | { kind: 'mask-radial-scale'; index: number; axis: 'long' | 'short' }
  | { kind: 'mask-radial-rotate'; index: number }
  | { kind: 'mask-brush'; index: number; strokeIndex: number }
  | { kind: 'crop-move'; start: { x: number; y: number }; orig: CropRect }
  | { kind: 'crop-handle'; handle: string; start: { x: number; y: number }; orig: CropRect }
  | { kind: 'stamp-src'; start: { x: number; y: number }; rect?: RectNorm }
  | { kind: 'stamp-dst'; start: { x: number; y: number }; cur: { x: number; y: number } }

let drag: Drag | null = null
let dragMoved = 0
let mouseImg: { x: number; y: number } | null = null
let dstRectPreview: { a: { x: number; y: number }; b: { x: number; y: number } } | null = null

// ------------------------------------------------------------ 帧选择与变换
function displayFrame(): { bmp: ImageBitmap; w: number; h: number } | null {
  if (store.comparing && store.originalBmp) {
    return { bmp: store.originalBmp, w: store.originalBmp.width, h: store.originalBmp.height }
  }
  const f = !store.previewFrame ? store.fullFrame
    : !store.fullFrame ? store.previewFrame
      : (store.fullFrame.rid >= store.previewFrame.rid ? store.fullFrame : store.previewFrame)
  if (!f) return null
  // 几何基准 = 源图像素尺寸（打开/裁剪/旋转后更新），不能用渲染帧的宽度：
  // 渲染帧按 viewEdge 分级（1200/2400/3200…），拿帧宽当基准会让「适应窗口」
  // 的比例随每次请求的分辨率跳变——打开图章/蒙版后画面会突然放大或缩小。
  // 草稿帧分辨率更低，也必须拉伸到同一矩形，否则画面会在草稿帧到达时跳动。
  const gw = store.width || f.bmp.width
  const gh = store.height || f.bmp.height
  return { bmp: f.bmp, w: gw, h: gh }
}

function fitScale(): number {
  const el = wrap.value
  const d = displayFrame()
  if (!el || !d || el.clientWidth < 4 || el.clientHeight < 4) return 1
  return Math.min(el.clientWidth / d.w, el.clientHeight / d.h)
}

function fitOffsets(z: number, d: { w: number; h: number }): { x: number; y: number } {
  const el = wrap.value!
  return { x: (el.clientWidth - d.w * z) / 2, y: (el.clientHeight - d.h * z) / 2 }
}

function curTransform(): { z: number; ox: number; oy: number } {
  const d = displayFrame()
  const z = view.manual ? view.zoom : fitScale()
  if (view.manual || !d) return { z, ox: view.x, oy: view.y }
  const off = fitOffsets(z, d)
  return { z, ox: off.x, oy: off.y }
}

function screenToImg(sx: number, sy: number): { x: number; y: number } {
  const { z, ox, oy } = curTransform()
  return { x: (sx - ox) / z, y: (sy - oy) / z }
}

function imgToScreen(ix: number, iy: number): { x: number; y: number } {
  const { z, ox, oy } = curTransform()
  return { x: ix * z + ox, y: iy * z + oy }
}

function shortSide(d: { w: number; h: number }): number {
  return Math.min(d.w, d.h)
}

// ------------------------------------------------------------ 蒙版几何
function radialGeom(m: MaskState, d: { w: number; h: number }) {
  const short = shortSide(d)
  const R = Math.max(m.radius, 0.01) * short
  const asp = Math.max(m.aspect, 0.05)
  return { cx: m.cx * d.w, cy: m.cy * d.h, rx: R * asp, ry: R / asp, asp, angRad: m.angle * Math.PI / 180 }
}

/** LR 式线性渐变：起点/终点/中点 + 轴向单位向量 + 垂直向量（图像像素空间）。 */
function linearGeom(m: MaskState, d: { w: number; h: number }) {
  const p1 = { x: m.x1 * d.w, y: m.y1 * d.h }
  const p2 = { x: m.x2 * d.w, y: m.y2 * d.h }
  const dx = p2.x - p1.x, dy = p2.y - p1.y
  const len = Math.max(Math.hypot(dx, dy), 1e-3)
  const u = { x: dx / len, y: dy / len }
  const v = { x: -u.y, y: u.x }
  const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 }
  return { p1, p2, mid, u, v, len, L: Math.hypot(d.w, d.h) }
}

/** 线性渐变旋转手柄：终点外侧沿轴延长 26 屏幕像素（imagePt 为图像像素坐标）。 */
function lineRotateHandleImg(g: ReturnType<typeof linearGeom>, z: number): { x: number; y: number } {
  const r = 26 / z
  return { x: g.p2.x + g.u.x * r, y: g.p2.y + g.u.y * r }
}

function maskPinPoint(m: MaskState, d: { w: number; h: number }): { x: number; y: number } {
  if (m.kind === 'linear') {
    return { x: (m.x1 + m.x2) / 2 * d.w, y: (m.y1 + m.y2) / 2 * d.h }
  }
  if (m.kind === 'radial') return { x: m.cx * d.w, y: m.cy * d.h }
  if (m.kind === 'ai') {
    // 网格质心（全 0 时回退中心）
    let sx = 0, sy = 0, n = 0
    for (let j = 0; j < Math.min(m.grid_h, m.grid.length); j++) {
      const row = m.grid[j]
      for (let i = 0; i < Math.min(m.grid_w, row.length); i++) {
        if (row[i] === '1') { sx += i + 0.5; sy += j + 0.5; n++ }
      }
    }
    if (!n) return { x: d.w / 2, y: d.h / 2 }
    return { x: sx / n / m.grid_w * d.w, y: sy / n / m.grid_h * d.h }
  }
  let sx = 0, sy = 0, n = 0
  for (const s of m.strokes) for (const [x, y] of s) { sx += x; sy += y; n++ }
  return n ? { x: sx / n * d.w, y: sy / n * d.h } : { x: d.w / 2, y: d.h / 2 }
}

/** AI 蒙版 -> 红色 30% 透明叠加画布（drawImage 缩放自然软边）。
 *  像素级掩码（mask_png，base64 PNG）与 VLM 网格（grid）统一处理；
 *  按蒙版对象 + 签名缓存，redraw 每帧触发时不重复构建。 */
const aiCanvasCache = new WeakMap<object, { sig: string; canvas: HTMLCanvasElement }>()
const pngImageCache = new Map<string, HTMLImageElement>()

/** 红色叠加强度（alpha）：0.30 → 0.50，小模型硬边蒙版也能看清边界。 */
const TINT_ALPHA = 128

/** 任意掩码来源 -> 红色着色画布（掩码亮度 -> alpha）。 */
function tintMask(src: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')!
  ctx.drawImage(src, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h)
  for (let k = 0; k < w * h; k++) {
    const v = data.data[k * 4]
    data.data[k * 4] = 255
    data.data[k * 4 + 1] = 56
    data.data[k * 4 + 2] = 56
    data.data[k * 4 + 3] = Math.round(v / 255 * TINT_ALPHA)
  }
  ctx.putImageData(data, 0, 0)
  return c
}

function aiMaskCanvas(m: MaskState): HTMLCanvasElement | null {
  // ---- 像素级掩码（本地 SAM2）----
  if (m.mask_png && m.mask_w > 0 && m.mask_h > 0) {
    const sig = `png:${m.mask_w}x${m.mask_h}:${m.mask_png.length}:${m.mask_png.slice(-48)}`
    const hit = aiCanvasCache.get(m)
    if (hit && hit.sig === sig) return hit.canvas
    let img = pngImageCache.get(sig)
    if (!img) {
      img = new Image()
      img.onload = () => redraw()                    // 解码完成后下一帧出叠加
      img.src = `data:image/png;base64,${m.mask_png}`
      pngImageCache.set(sig, img)
      return null                                    // 首帧未解码，先不画
    }
    if (!img.complete || !img.naturalWidth) return null
    const canvas = tintMask(img, m.mask_w, m.mask_h)
    aiCanvasCache.set(m, { sig, canvas })
    return canvas
  }
  // ---- VLM 网格 ----
  if (m.grid_w <= 0 || m.grid_h <= 0 || !m.grid.length) return null
  const sig = `grid:${m.grid_w}x${m.grid_h}:${m.grid.join('')}`
  const hit = aiCanvasCache.get(m)
  if (hit && hit.sig === sig) return hit.canvas
  const c = document.createElement('canvas')
  c.width = m.grid_w
  c.height = m.grid_h
  const ctx = c.getContext('2d')
  if (!ctx) return null
  const img = ctx.createImageData(m.grid_w, m.grid_h)
  for (let j = 0; j < m.grid_h; j++) {
    const row = m.grid[j] || ''
    for (let i = 0; i < m.grid_w; i++) {
      const k = (j * m.grid_w + i) * 4
      if (row[i] === '1') {
        img.data[k] = 255; img.data[k + 1] = 56; img.data[k + 2] = 56
        img.data[k + 3] = TINT_ALPHA        // ≈ 0.50，与像素级涂色一致
      }
    }
  }
  ctx.putImageData(img, 0, 0)
  aiCanvasCache.set(m, { sig, canvas: c })
  return c
}

function drawPin(ctx: CanvasRenderingContext2D, index: number, imgPt: { x: number; y: number }): void {
  const s = imgToScreen(imgPt.x, imgPt.y)
  ctx.beginPath(); ctx.arc(s.x, s.y, 9, 0, Math.PI * 2)
  ctx.fillStyle = 'rgba(23,23,28,0.85)'; ctx.fill()
  ctx.lineWidth = 1.4; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.stroke()
  ctx.fillStyle = '#e2e2e2'
  ctx.font = '10px sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(String(index + 1), s.x, s.y + 0.5)
}

function drawHandle(ctx: CanvasRenderingContext2D, x: number, y: number, color: string): void {
  ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2)
  ctx.fillStyle = color; ctx.fill()
  ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 1; ctx.stroke()
}

function clipToImage(ctx: CanvasRenderingContext2D, d: { w: number; h: number }): void {
  const { z, ox, oy } = curTransform()
  ctx.beginPath()
  ctx.rect(ox, oy, d.w * z, d.h * z)
  ctx.clip()
}

// ------------------------------------------------------------ 蒙版叠加绘制
function drawMaskOverlay(ctx: CanvasRenderingContext2D, d: { w: number; h: number }): void {
  const masks = store.adjustments.masks
  const z = curTransform().z
  for (let i = 0; i < masks.length; i++) {
    const m = masks[i]
    if (!m.enabled) continue
    const selected = i === store.maskSelected && store.editMode === 'mask'
    const showOverlay = selected && store.maskOverlay          // 几何手柄：选中即可见
    const showTint = selected && maskOverlayVisible()          // 红色涂色：悬停/预览时短显

    if (m.kind === 'linear') {
      if (!showOverlay) { drawPin(ctx, i, maskPinPoint(m, d)); continue }
      const g = linearGeom(m, d)
      ctx.save()
      clipToImage(ctx, d)
      ctx.strokeStyle = 'rgba(255,255,255,0.92)'
      ctx.lineWidth = 1.6
      for (const c of [g.p1, g.mid, g.p2]) {
        const s = imgToScreen(c.x, c.y)
        ctx.beginPath()
        ctx.moveTo(s.x - g.v.x * g.L * z, s.y - g.v.y * g.L * z)
        ctx.lineTo(s.x + g.v.x * g.L * z, s.y + g.v.y * g.L * z)
        ctx.stroke()
      }
      ctx.restore()
      for (const c of [g.p1, g.mid, g.p2]) {
        const s = imgToScreen(c.x, c.y)
        drawHandle(ctx, s.x, s.y, '#e2e2e2')
      }
      // 旋转手柄：终点外侧（橙色，与径向旋转手柄同约定）
      const rh = imgToScreen(g.p2.x + g.u.x * 26 / z, g.p2.y + g.u.y * 26 / z)
      drawHandle(ctx, rh.x, rh.y, '#f0a030')
    } else if (m.kind === 'radial') {
      if (!showOverlay) { drawPin(ctx, i, maskPinPoint(m, d)); continue }
      const g = radialGeom(m, d)
      const c = imgToScreen(g.cx, g.cy)
      ctx.strokeStyle = '#e2e2e2'
      ctx.lineWidth = 1.6
      ctx.beginPath()
      ctx.ellipse(c.x, c.y, g.rx * z, g.ry * z, g.angRad, 0, Math.PI * 2)
      ctx.stroke()
      const dirX = Math.cos(g.angRad), dirY = Math.sin(g.angRad)
      const longP = { x: c.x + dirX * g.rx * z, y: c.y + dirY * g.rx * z }
      const shortP = { x: c.x - dirY * g.ry * z, y: c.y + dirX * g.ry * z }
      const rotP = { x: c.x + dirX * (g.rx * z + 22), y: c.y + dirY * (g.rx * z + 22) }
      drawHandle(ctx, c.x, c.y, '#e2e2e2')
      drawHandle(ctx, longP.x, longP.y, '#4a9eff')
      drawHandle(ctx, shortP.x, shortP.y, '#4a9eff')
      drawHandle(ctx, rotP.x, rotP.y, '#f0a030')
    } else if (m.kind === 'ai') {
      // AI 蒙版：悬停蒙版条目/刚创建时短显红色叠加（网格放大即软边预览）
      if (showTint) {
        const src = aiMaskCanvas(m)
        if (src) {
          ctx.save()
          clipToImage(ctx, d)
          ctx.imageSmoothingEnabled = true
          const p0 = imgToScreen(0, 0)
          const p1 = imgToScreen(d.w, d.h)
          ctx.drawImage(src, p0.x, p0.y, p1.x - p0.x, p1.y - p0.y)
          ctx.restore()
        }
      }
      if (!showTint) drawPin(ctx, i, maskPinPoint(m, d))
    } else {
      // 画笔：正在涂/悬停时显示实际涂刷区域（低透明度，不遮挡画面）
      if (showTint && m.strokes.length) {
        const radius = m.size / 100 * shortSide(d) * 0.5
        ctx.save()
        clipToImage(ctx, d)
        ctx.strokeStyle = 'rgba(255,56,56,0.42)'
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.lineWidth = Math.max(3, radius * 2 * z)
        for (const stroke of m.strokes) {
          if (!stroke.length) continue
          ctx.beginPath()
          stroke.forEach(([nx, ny], k) => {
            const p = imgToScreen(nx * d.w, ny * d.h)
            if (k === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y)
          })
          if (stroke.length === 1) {
            const p = imgToScreen(stroke[0][0] * d.w, stroke[0][1] * d.h)
            ctx.lineTo(p.x + 0.01, p.y)
          }
          ctx.stroke()
        }
        ctx.restore()
      }
      if (selected) {
        if (mouseImg && showTint) {
          const p = imgToScreen(mouseImg.x, mouseImg.y)
          const r = (m.size / 100 * shortSide(d) * 0.5) * z
          ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(3, r), 0, Math.PI * 2)
          ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1.2; ctx.stroke()
        }
        if (!showTint && m.strokes.length) drawPin(ctx, i, maskPinPoint(m, d))
      } else {
        drawPin(ctx, i, maskPinPoint(m, d))
      }
    }
  }
}

// ------------------------------------------------------------ 裁剪叠加
function cropRectPx(d: { w: number; h: number }): CropRect {
  const r = store.cropRect!
  return { x0: r.x0 * d.w, y0: r.y0 * d.h, x1: r.x1 * d.w, y1: r.y1 * d.h }
}

function drawCropOverlay(ctx: CanvasRenderingContext2D, d: { w: number; h: number }): void {
  const r = cropRectPx(d)
  const a = imgToScreen(r.x0, r.y0)
  const b = imgToScreen(r.x1, r.y1)
  const el = wrap.value!
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.fillRect(0, 0, el.clientWidth, a.y)
  ctx.fillRect(0, b.y, el.clientWidth, el.clientHeight - b.y)
  ctx.fillRect(0, a.y, a.x, b.y - a.y)
  ctx.fillRect(b.x, a.y, el.clientWidth - b.x, b.y - a.y)
  ctx.strokeStyle = 'rgba(255,255,255,0.28)'
  ctx.lineWidth = 1
  for (let i = 1; i <= 2; i++) {
    const gx = a.x + (b.x - a.x) * i / 3
    const gy = a.y + (b.y - a.y) * i / 3
    ctx.beginPath(); ctx.moveTo(gx, a.y); ctx.lineTo(gx, b.y); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(a.x, gy); ctx.lineTo(b.x, gy); ctx.stroke()
  }
  ctx.strokeStyle = '#e2e2e2'
  ctx.lineWidth = 1.6
  ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y)
  ctx.fillStyle = '#e2e2e2'
  for (const [hx, hy] of [[a.x, a.y], [b.x, a.y], [a.x, b.y], [b.x, b.y]] as const) {
    ctx.fillRect(hx - 4, hy - 4, 8, 8)
  }
  ctx.fillRect((a.x + b.x) / 2 - 4, a.y - 4, 8, 8)
  ctx.fillRect((a.x + b.x) / 2 - 4, b.y - 4, 8, 8)
  ctx.fillRect(a.x - 4, (a.y + b.y) / 2 - 4, 8, 8)
  ctx.fillRect(b.x - 4, (a.y + b.y) / 2 - 4, 8, 8)
}

function cropHitHandle(sx: number, sy: number, d: { w: number; h: number }): string | null {
  const r = cropRectPx(d)
  const a = imgToScreen(r.x0, r.y0)
  const b = imgToScreen(r.x1, r.y1)
  const R = 12
  const near = (px: number, py: number) => Math.abs(sx - px) <= R && Math.abs(sy - py) <= R
  const midX = (a.x + b.x) / 2, midY = (a.y + b.y) / 2
  if (near(a.x, a.y)) return 'nw'
  if (near(b.x, a.y)) return 'ne'
  if (near(a.x, b.y)) return 'sw'
  if (near(b.x, b.y)) return 'se'
  if (near(midX, a.y)) return 'n'
  if (near(midX, b.y)) return 's'
  if (near(a.x, midY)) return 'w'
  if (near(b.x, midY)) return 'e'
  if (sx > a.x && sx < b.x && sy > a.y && sy < b.y) return 'move'
  return null
}

function applyCropDrag(handle: string, orig: CropRect, dxImg: number, dyImg: number): void {
  const d = displayFrame()!
  const w = d.w, h = d.h
  const { x0, y0, x1, y1 } = orig
  const nrm = { x0: x0 / w, y0: y0 / h, x1: x1 / w, y1: y1 / h }
  const dxn = dxImg / w, dyn = dyImg / h
  if (handle === 'move') {
    const nx0 = Math.max(0, Math.min(1 - (nrm.x1 - nrm.x0), nrm.x0 + dxn))
    const ny0 = Math.max(0, Math.min(1 - (nrm.y1 - nrm.y0), nrm.y0 + dyn))
    store.cropRect = { x0: nx0, y0: ny0, x1: nx0 + (nrm.x1 - nrm.x0), y1: ny0 + (nrm.y1 - nrm.y0) }
    return
  }
  const E = 0.02
  const n = { ...nrm }
  if (handle.includes('n')) n.y0 = Math.min(n.y1 - E, n.y0 + dyn)
  if (handle.includes('s')) n.y1 = Math.max(n.y0 + E, n.y1 + dyn)
  if (handle.includes('w')) n.x0 = Math.min(n.x1 - E, n.x0 + dxn)
  if (handle.includes('e')) n.x1 = Math.max(n.x0 + E, n.x1 + dxn)
  n.x0 = Math.max(0, n.x0); n.y0 = Math.max(0, n.y0)
  n.x1 = Math.min(1, n.x1); n.y1 = Math.min(1, n.y1)
  store.cropRect = constrainCropRect(n)
}

// ------------------------------------------------------------ 绘制
/** 告诉渲染调度层：屏幕现在真正需要多少像素（窗口尺寸 × DPR × 放大倍数）。 */
function reportViewEdge(z: number, cw: number, ch: number, dpr: number): void {
  const fit = fitScale()
  const ratio = Math.max(1, z / Math.max(fit, 1e-6))   // 相对“适应窗口”的放大倍数
  let need = Math.max(cw, ch) * dpr * ratio
  const srcLong = Math.max(store.width, store.height)
  if (srcLong > 0) need = Math.min(need, srcLong)
  setViewEdge(need)
}

function redraw(): void {
  const cv = canvas.value
  const el = wrap.value
  if (!cv || !el) return
  const dpr = window.devicePixelRatio || 1
  const cw = el.clientWidth
  const ch = el.clientHeight
  if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) {
    cv.width = Math.round(cw * dpr)
    cv.height = Math.round(ch * dpr)
  }
  const ctx = cv.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.fillStyle = '#0a0c10'
  ctx.fillRect(0, 0, cw, ch)
  const d = displayFrame()
  if (!d) return
  const { z, ox, oy } = curTransform()

  zoomPct.value = Math.round(z * 100)
  reportViewEdge(z, cw, ch, dpr)
  // 草稿帧被拉伸显示时（位图比目标矩形小）也必须平滑，否则会看到马赛克
  ctx.imageSmoothingEnabled = z < 2 || d.bmp.width < d.w * z - 1
  ctx.imageSmoothingQuality = 'high'
  if (store.editMode === 'crop' && store.cropAngle) {
    const c = imgToScreen(d.w / 2, d.h / 2)
    ctx.save()
    ctx.translate(c.x, c.y)
    ctx.rotate(-store.cropAngle * Math.PI / 180)
    ctx.drawImage(d.bmp, -d.w * z / 2, -d.h * z / 2, d.w * z, d.h * z)
    ctx.restore()
  } else {
    ctx.drawImage(d.bmp, ox, oy, d.w * z, d.h * z)
  }

  if (store.adjustments.masks.length) drawMaskOverlay(ctx, d)
  if (store.editMode === 'crop' && store.cropRect) drawCropOverlay(ctx, d)
  if (store.editMode === 'stamp') {
    if (store.stampSource) {
      const s = store.stampSource
      const p0 = imgToScreen(s[0] * d.w, s[1] * d.h)
      const p1 = imgToScreen(s[2] * d.w, s[3] * d.h)
      ctx.strokeStyle = '#4dd07a'
      ctx.lineWidth = 1.6
      ctx.setLineDash([6, 4])
      ctx.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y)
      ctx.setLineDash([])
      const cxm = (p0.x + p1.x) / 2, cym = (p0.y + p1.y) / 2
      ctx.beginPath(); ctx.moveTo(cxm - 7, cym); ctx.lineTo(cxm + 7, cym)
      ctx.moveTo(cxm, cym - 7); ctx.lineTo(cxm, cym + 7); ctx.stroke()
    }
    if (dstRectPreview) {
      ctx.strokeStyle = '#e2e2e2'
      ctx.lineWidth = 1.4
      ctx.strokeRect(
        Math.min(dstRectPreview.a.x, dstRectPreview.b.x),
        Math.min(dstRectPreview.a.y, dstRectPreview.b.y),
        Math.abs(dstRectPreview.b.x - dstRectPreview.a.x),
        Math.abs(dstRectPreview.b.y - dstRectPreview.a.y))
    }
  }
}

// ------------------------------------------------------------ 缩放/平移
function zoomAt(cx: number, cy: number, factor: number): void {
  const d = displayFrame()
  if (!d) return
  const z1 = view.manual ? view.zoom : fitScale()
  if (!view.manual) {
    const off = fitOffsets(z1, d)
    view.x = off.x
    view.y = off.y
  }
  const z2 = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z1 * factor))
  if (z2 === z1) return
  view.x = cx - (cx - view.x) * (z2 / z1)
  view.y = cy - (cy - view.y) * (z2 / z1)
  view.zoom = z2
  view.manual = true
  redraw()
}

function zoomFit(): void { view.manual = false; redraw() }
function zoomOneToOne(): void {
  const d = displayFrame()
  if (!d) return
  const el = wrap.value!
  view.zoom = 1
  view.x = (el.clientWidth - d.w) / 2
  view.y = (el.clientHeight - d.h) / 2
  view.manual = true
  redraw()
}
function zoomStep(factor: number): void {
  const el = wrap.value
  if (el) zoomAt(el.clientWidth / 2, el.clientHeight / 2, factor)
}

// ------------------------------------------------------------ 交互
function localPos(e: { clientX: number; clientY: number }): { x: number; y: number } {
  const rect = canvas.value!.getBoundingClientRect()
  return { x: e.clientX - rect.left, y: e.clientY - rect.top }
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function onWheel(e: WheelEvent): void {
  e.preventDefault()
  const p = localPos(e)
  zoomAt(p.x, p.y, e.deltaY < 0 ? 1.25 : 0.8)
}

function onPointerDown(e: PointerEvent): void {
  if (e.button !== 0) return
  const d = displayFrame()
  if (!d) return
  const lp = localPos(e)
  dragMoved = 0
  const z = curTransform().z
  const hitTol = 10 / z

  if (store.editMode === 'stamp') {
    canvas.value!.setPointerCapture(e.pointerId)
    if (e.altKey) drag = { kind: 'stamp-src', start: lp }
    else {
      drag = { kind: 'stamp-dst', start: lp, cur: lp }
      dstRectPreview = { a: lp, b: lp }
    }
    redraw()
    return
  }

  if (store.editMode === 'crop' && store.cropRect) {
    const handle = cropHitHandle(lp.x, lp.y, d)
    if (handle) {
      canvas.value!.setPointerCapture(e.pointerId)
      drag = { kind: handle === 'move' ? 'crop-move' : 'crop-handle',
               handle, start: lp, orig: cropRectPx(d) }
    }
    return
  }

  if (store.editMode === 'mask' && store.maskSelected !== null) {
    const m = store.adjustments.masks[store.maskSelected]
    const img = screenToImg(lp.x, lp.y)
    const idx = store.maskSelected
    if (m.kind === 'linear' && store.maskOverlay) {
      const g = linearGeom(m, d)
      // 旋转手柄优先（终点外侧橙色点）
      const rh = lineRotateHandleImg(g, z)
      if (dist(img, rh) < hitTol) {
        drag = { kind: 'mask-line-rotate', index: idx }
        canvas.value!.setPointerCapture(e.pointerId)
        return
      }
      const centers: [{ x: number; y: number }, 0 | 1 | 2][] =
        [[g.p1, 0], [g.mid, 1], [g.p2, 2]]
      for (const [c, which] of centers) {
        if (dist(img, c) < hitTol) {
          drag = { kind: 'mask-line', index: idx, which }
          canvas.value!.setPointerCapture(e.pointerId)
          return
        }
      }
      const nearLine = (c: { x: number; y: number }) =>
        Math.abs((img.x - c.x) * g.v.x + (img.y - c.y) * g.v.y) < hitTol
      for (const [c, which] of centers) {
        if (nearLine(c)) {
          drag = { kind: 'mask-line', index: idx, which }
          canvas.value!.setPointerCapture(e.pointerId)
          return
        }
      }
    } else if (m.kind === 'radial' && store.maskOverlay) {
      const g = radialGeom(m, d)
      const dirX = Math.cos(g.angRad), dirY = Math.sin(g.angRad)
      const center = { x: g.cx, y: g.cy }
      const longP = { x: g.cx + dirX * g.rx, y: g.cy + dirY * g.rx }
      const shortP = { x: g.cx - dirY * g.ry, y: g.cy + dirX * g.ry }
      const rotP = { x: g.cx + dirX * (g.rx + 22 / z), y: g.cy + dirY * (g.rx + 22 / z) }
      if (dist(img, rotP) < hitTol) {
        drag = { kind: 'mask-radial-rotate', index: idx }
        canvas.value!.setPointerCapture(e.pointerId); return
      }
      if (dist(img, longP) < hitTol) {
        drag = { kind: 'mask-radial-scale', index: idx, axis: 'long' }
        canvas.value!.setPointerCapture(e.pointerId); return
      }
      if (dist(img, shortP) < hitTol) {
        drag = { kind: 'mask-radial-scale', index: idx, axis: 'short' }
        canvas.value!.setPointerCapture(e.pointerId); return
      }
      if (dist(img, center) < hitTol) {
        drag = { kind: 'mask-radial-center', index: idx }
        canvas.value!.setPointerCapture(e.pointerId); return
      }
    } else if (m.kind === 'brush') {
      const stroke: [number, number][] = [[img.x / d.w, img.y / d.h]]
      m.strokes.push(stroke)
      drag = { kind: 'mask-brush', index: idx, strokeIndex: m.strokes.length - 1 }
      canvas.value!.setPointerCapture(e.pointerId)
      setMaskHover(true)             // 涂刷过程中保持红色可见
      maskGeometryChanged()
      redraw()
      return
    }
  }

  canvas.value!.setPointerCapture(e.pointerId)
  drag = { kind: 'pan', lastX: e.clientX, lastY: e.clientY }
}

function onPointerMove(e: PointerEvent): void {
  const d = displayFrame()
  const lp = localPos(e)
  mouseImg = screenToImg(lp.x, lp.y)

  if (!drag) {
    if (store.editMode === 'mask' && store.maskSelected !== null) {
      const m = store.adjustments.masks[store.maskSelected]
      if (m?.kind === 'brush') redraw()
    }
    return
  }
  dragMoved += Math.abs(e.movementX) + Math.abs(e.movementY)

  switch (drag.kind) {
    case 'pan': {
      if (!view.manual) {
        const z = fitScale()
        const off = fitOffsets(z, d!)
        view.x = off.x; view.y = off.y; view.zoom = z
      }
      view.manual = true
      view.x += e.clientX - drag.lastX
      view.y += e.clientY - drag.lastY
      drag.lastX = e.clientX
      drag.lastY = e.clientY
      break
    }
    case 'mask-line': {
      const m = store.adjustments.masks[drag.index]
      if (!m || !d) break
      const g = linearGeom(m, d)
      const img = screenToImg(lp.x, lp.y)
      if (drag.which === 1) {
        // 中线：整体平移
        const dxn = (img.x - g.mid.x) / d.w
        const dyn = (img.y - g.mid.y) / d.h
        m.x1 = Math.min(1, Math.max(0, m.x1 + dxn))
        m.x2 = Math.min(1, Math.max(0, m.x2 + dxn))
        m.y1 = Math.min(1, Math.max(0, m.y1 + dyn))
        m.y2 = Math.min(1, Math.max(0, m.y2 + dyn))
      } else {
        // 端线：沿轴移动该线（另一端固定，保持最小长度）
        const anchor = drag.which === 0 ? g.p2 : g.p1
        let t = (img.x - anchor.x) * g.u.x + (img.y - anchor.y) * g.u.y
        t = Math.max(2, t)
        const nx = anchor.x + g.u.x * t
        const ny = anchor.y + g.u.y * t
        if (drag.which === 0) {
          m.x1 = Math.min(1, Math.max(0, nx / d.w))
          m.y1 = Math.min(1, Math.max(0, ny / d.h))
        } else {
          m.x2 = Math.min(1, Math.max(0, nx / d.w))
          m.y2 = Math.min(1, Math.max(0, ny / d.h))
        }
      }
      maskGeometryChanged()
      break
    }
    case 'mask-line-rotate': {
      const m = store.adjustments.masks[drag.index]
      if (!m || !d) break
      const g = linearGeom(m, d)
      const img = screenToImg(lp.x, lp.y)
      // 绕中线中心旋转到鼠标方向（视觉逆时针为正）；Shift 吸附 15°
      let deg = -Math.atan2(img.y - g.mid.y, img.x - g.mid.x) * 180 / Math.PI
      if (e.shiftKey) deg = Math.round(deg / 15) * 15
      setLinearMaskAngle(drag.index, (deg + 360) % 360)   // 内部已 notifyChanged
      break
    }
    case 'mask-radial-center': {
      const m = store.adjustments.masks[drag.index]
      if (!m || !d) break
      const img = screenToImg(lp.x, lp.y)
      m.cx = Math.min(1, Math.max(0, img.x / d.w))
      m.cy = Math.min(1, Math.max(0, img.y / d.h))
      maskGeometryChanged()
      break
    }
    case 'mask-radial-scale': {
      const m = store.adjustments.masks[drag.index]
      if (!m || !d) break
      const g = radialGeom(m, d)
      const img = screenToImg(lp.x, lp.y)
      const dxp = img.x - g.cx, dyp = img.y - g.cy
      if (drag.axis === 'long') {
        const rx = Math.max(4, dist({ x: 0, y: 0 }, { x: dxp, y: dyp }))
        const R = Math.sqrt(rx * g.ry)
        m.radius = Math.max(0.01, R / shortSide(d))
        m.aspect = Math.max(0.05, Math.min(20, rx / Math.max(g.ry, 1e-3)))
      } else {
        const ry = Math.max(4, dist({ x: 0, y: 0 }, { x: dxp, y: dyp }))
        const R = Math.sqrt(g.rx * ry)
        m.radius = Math.max(0.01, R / shortSide(d))
        m.aspect = Math.max(0.05, Math.min(20, g.rx / Math.max(ry, 1e-3)))
      }
      maskGeometryChanged()
      break
    }
    case 'mask-radial-rotate': {
      const m = store.adjustments.masks[drag.index]
      if (!m || !d) break
      const g = radialGeom(m, d)
      const img = screenToImg(lp.x, lp.y)
      m.angle = ((Math.atan2(img.y - g.cy, img.x - g.cx) * 180) / Math.PI + 360) % 360
      maskGeometryChanged()
      break
    }
    case 'mask-brush': {
      const m = store.adjustments.masks[drag.index]
      if (!m || !d) break
      const stroke = m.strokes[drag.strokeIndex]
      const last = stroke[stroke.length - 1]
      const nx = mouseImg!.x / d.w, ny = mouseImg!.y / d.h
      if (!last || Math.abs(nx - last[0]) * d.w + Math.abs(ny - last[1]) * d.h > 3) {
        stroke.push([nx, ny])
        maskGeometryChanged()
      }
      break
    }
    case 'crop-move':
    case 'crop-handle': {
      if (!d) break
      const dxImg = (lp.x - drag.start.x) / curTransform().z
      const dyImg = (lp.y - drag.start.y) / curTransform().z
      applyCropDrag(drag.kind === 'crop-move' ? 'move' : drag.handle, drag.orig, dxImg, dyImg)
      break
    }
    case 'stamp-src': {
      if (!d) break
      const a = screenToImg(Math.min(lp.x, drag.start.x), Math.min(lp.y, drag.start.y))
      const b = screenToImg(Math.max(lp.x, drag.start.x), Math.max(lp.y, drag.start.y))
      dstRectPreview = { a: imgToScreen(a.x, a.y), b: imgToScreen(b.x, b.y) }
      drag.rect = normRect(a, b, d)
      redraw()
      return
    }
    case 'stamp-dst': {
      if (!d) break
      dstRectPreview = { a: drag.start, b: lp }
      drag.cur = lp
      redraw()
      return
    }
  }
  redraw()
}

function normRect(a: { x: number; y: number }, b: { x: number; y: number },
                 d: { w: number; h: number }): RectNorm {
  const x0 = Math.max(0, Math.min(a.x, b.x)) / d.w
  const y0 = Math.max(0, Math.min(a.y, b.y)) / d.h
  const x1 = Math.min(d.w, Math.max(a.x, b.x)) / d.w
  const y1 = Math.min(d.h, Math.max(a.y, b.y)) / d.h
  return [x0, y0, x1, y1]
}

function onPointerUp(e: PointerEvent): void {
  if (!drag) {
    canvas.value?.releasePointerCapture(e.pointerId)
    return
  }
  const d = displayFrame()
  if (drag.kind === 'pan' && dragMoved < 4 && store.pickTarget !== null && d) {
    const lp = localPos(e)
    const p = screenToImg(lp.x, lp.y)
    if (p.x >= 0 && p.x < d.w && p.y >= 0 && p.y < d.h) {
      void imagePicked(p.x / d.w * store.width, p.y / d.h * store.height)
    }
  }
  if (drag.kind === 'stamp-src') {
    const rect = drag.rect
    if (rect && d && (rect[2] - rect[0]) * d.w > 4 && (rect[3] - rect[1]) * d.h > 4) {
      setStampSource(rect)
    }
  }
  if (drag.kind === 'stamp-dst') {
    const a = screenToImg(Math.min(drag.start.x, drag.cur.x), Math.min(drag.start.y, drag.cur.y))
    const b = screenToImg(Math.max(drag.start.x, drag.cur.x), Math.max(drag.start.y, drag.cur.y))
    if (d) {
      const rect = normRect(a, b, d)
      if ((rect[2] - rect[0]) * d.w > 4 && (rect[3] - rect[1]) * d.h > 4) {
        void applyStampAt(rect)
      }
    }
  }
  if (drag.kind === 'mask-brush') brushStrokeCommitted(drag.index)
  if (drag.kind === 'mask-line' || drag.kind === 'mask-line-rotate'
      || drag.kind === 'mask-radial-center'
      || drag.kind === 'mask-radial-scale' || drag.kind === 'mask-radial-rotate') {
    maskGeometryCommitted()
  }
  if (drag.kind.startsWith('mask-')) {
    setMaskHover(false)              // 松手后收起叠加，调参数时不再挡画面
    revealMaskOverlay(900)
  }
  dstRectPreview = null
  drag = null
  canvas.value?.releasePointerCapture(e.pointerId)
  redraw()
}

function onDblClick(): void { zoomFit() }

function onKey(e: KeyboardEvent): void {
  if (e.target instanceof HTMLInputElement) return
  if (e.key === '0') zoomFit()
  else if (e.key === '1') zoomOneToOne()
  else if (e.key === '+' || e.key === '=') zoomStep(1.25)
  else if (e.key === '-') zoomStep(0.8)
}

let ro: ResizeObserver | null = null
let detailTimer: number | undefined

onMounted(() => {
  ro = new ResizeObserver(() => redraw())
  ro.observe(wrap.value!)
  window.addEventListener('keydown', onKey)
  redraw()
})

onBeforeUnmount(() => {
  ro?.disconnect()
  window.removeEventListener('keydown', onKey)
  if (detailTimer) window.clearTimeout(detailTimer)
})

// 视口需要的分辨率变了（放大到 1:1、窗口变大）：停手后补一张更清晰的图。
// 拖动/缩放过程中不发请求，避免和参数渲染抢同一个渲染线程。
watch(() => store.viewEdge, () => {
  if (detailTimer) window.clearTimeout(detailTimer)
  detailTimer = window.setTimeout(() => {
    detailTimer = undefined
    requestDetail()
  }, 260)
})

watch(() => store.frameVersion, () => redraw())
watch(() => [store.editMode, store.maskSelected, store.maskOverlay,
            store.maskHover, store.maskReveal, store.cropRect, store.comparing],
      () => redraw(), { deep: true })

const cursorClass = computed(() => {
  if (store.pickTarget !== null || store.editMode === 'stamp') return 'cursor-cross'
  if (store.editMode === 'mask' || store.editMode === 'crop') return 'cursor-mode'
  return 'cursor-grab'
})
</script>

<template>
  <div ref="wrap" class="viewport" :class="cursorClass">
    <canvas
      ref="canvas"
      @wheel="onWheel"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @dblclick="onDblClick"
      @contextmenu.prevent
    />

    <div class="vignette" />

    <div v-if="store.loaded" class="zoom-bar">
      <button class="mini" title="适应窗口（0）" @click="zoomFit">适应</button>
      <button class="mini" title="1:1 像素（1）" @click="zoomOneToOne">100%</button>
      <span class="zoom-val num">{{ zoomPct }}%</span>
    </div>

    <div v-if="!store.loaded" class="viewport-hint">
      <div class="hint-mark">
        <Icon name="image" :size="30" :stroke="1.1" />
      </div>
      <p class="hint-title">PhotoLab</p>
      <p class="hint-sub">把照片拖进窗口，或点击左上角「打开」</p>
      <div class="hint-keys">
        <span><b class="kbd">Ctrl</b><b class="kbd">O</b> 打开</span>
        <span><b class="kbd">\</b> 对比原图</span>
        <span><b class="kbd">0</b> 适应窗口</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.viewport {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: var(--bg-canvas);
}
canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  touch-action: none;
}
.cursor-grab canvas { cursor: grab; }
.cursor-grab canvas:active { cursor: grabbing; }
.cursor-cross canvas { cursor: crosshair; }
.cursor-mode canvas { cursor: crosshair; }

/* 暗角：给画面一个轻微的“取景器”包裹感，不参与交互 */
.vignette {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(120% 100% at 50% 45%,
    rgba(0, 0, 0, 0) 52%, rgba(0, 0, 0, 0.34) 100%);
}

/* 右下角缩放条 */
.zoom-bar {
  position: absolute;
  right: 16px;
  bottom: 14px;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 7px 4px 5px;
  border-radius: 10px;
  background: rgba(16, 14, 12, 0.82);
  box-shadow: inset 0 0 0 1px var(--line-2), 0 8px 22px rgba(0, 0, 0, 0.45);
  backdrop-filter: blur(10px) saturate(1.1);
  user-select: none;
}
.zoom-val {
  min-width: 44px;
  text-align: right;
  font-family: var(--font-display);
  font-size: 10.5px;
  color: var(--text-2);
  letter-spacing: 0.04em;
  padding-right: 4px;
}

/* 空态 */
.viewport-hint {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  color: var(--text-3);
  user-select: none;
  pointer-events: none;
  animation: fade-in 0.3s var(--ease);
}
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
.hint-mark {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 62px;
  height: 62px;
  margin-bottom: 14px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.035);
  box-shadow: inset 0 0 0 1px var(--line);
  color: var(--text-3);
}
.hint-title {
  margin: 0;
  font-size: 19px;
  font-weight: 600;
  letter-spacing: 5px;
  color: var(--text-2);
}
.hint-sub {
  margin: 6px 0 0;
  font-size: 11.5px;
  color: var(--text-3);
}
.hint-keys {
  display: flex;
  gap: 16px;
  margin-top: 22px;
  font-size: 10.5px;
  color: var(--text-3);
}
.hint-keys span { display: inline-flex; align-items: center; gap: 4px; }
.hint-keys .kbd { font-weight: 400; }
</style>
