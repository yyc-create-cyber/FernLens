/** 滤镜（预设 look）：只包含「全局影调 + 曲线 + 分级色轮」，
 *  不含蒙版（几何属于单张照片）与取色点（颜色取自具体画面），
 *  这样同一份滤镜可以安全地套到任意照片、也能导出成 JSON 分享。
 *  数值范围照抄 core.py（曝光 -5..5 EV，其余 -100..100，黑白 0..100）。 */
import type {
  Adjustments, Curves, GradingRange, WheelState,
} from '@/api/types'
import { CURVE_CHANNELS, GRADING_RANGES, TONE_FIELDS, defaultCurves } from '@/api/types'

/** 滤镜覆盖的全部全局字段 = TONE_FIELDS（11 个）+ 效果四件套。 */
export const LOOK_FIELDS = [
  ...TONE_FIELDS, 'clarity', 'denoise', 'sharpen', 'dehaze',
] as const
export type LookField = (typeof LOOK_FIELDS)[number]

export type FilterLook = Partial<Record<LookField, number>> & {
  curves?: Partial<Curves>
  wheels?: Partial<Record<GradingRange, WheelState>>
}

export interface FilterPreset {
  id: string
  name: string
  group: string            // 基础 / 人像 / 风光 / 胶片 / 黑白 / 自定义
  note?: string
  look: FilterLook
  builtin?: boolean
  createdAt?: number
}

/** 导出文件格式标识（导入时据此判断，也兼容裸 look / 数组 / {filters:[]}）。 */
export const FILTER_FORMAT = 'photolab-filter'
export const FILTER_VERSION = 1

// ------------------------------------------------------------ 数值域
const RANGE: Record<LookField, [number, number]> = {
  temperature: [-100, 100], tint: [-100, 100],
  exposure: [-5, 5], contrast: [-100, 100], highlights: [-100, 100], shadows: [-100, 100],
  whites: [-100, 100], blacks: [-100, 100],
  saturation: [-100, 100], vibrance: [-100, 100], blackwhite: [0, 100],
  clarity: [-100, 100], denoise: [0, 100], sharpen: [0, 100], dehaze: [-100, 100],
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

function num(v: unknown, fallback: number): number {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n : fallback
}

// ------------------------------------------------------------ look 读写
/** 从当前参数提取滤镜 look（影调 + 曲线 + 色轮；丢弃蒙版/取色点）。 */
export function lookFromAdjustments(adj: Adjustments): FilterLook {
  const look: FilterLook = {}
  for (const f of LOOK_FIELDS) look[f] = adj[f]
  look.curves = {}
  for (const ch of CURVE_CHANNELS) {
    look.curves[ch] = adj.curves[ch].map(([x, y]) => [x, y] as [number, number])
  }
  look.wheels = {}
  for (const r of GRADING_RANGES) look.wheels[r] = { ...adj.wheels[r] }
  return look
}

/** 把 look 合并进参数（只覆盖 look 里出现的字段；蒙版/取色点保持不动）。 */
export function applyLook(adj: Adjustments, look: FilterLook): void {
  for (const f of LOOK_FIELDS) {
    const v = look[f]
    if (typeof v === 'number' && Number.isFinite(v)) {
      ;(adj as unknown as Record<string, number>)[f] = clamp(v, RANGE[f][0], RANGE[f][1])
    }
  }
  if (look.curves) {
    for (const ch of CURVE_CHANNELS) {
      const pts = look.curves[ch]
      if (Array.isArray(pts) && pts.length >= 2) {
        adj.curves[ch] = pts.map(([x, y]) => [
          clamp(num(x, 0), 0, 1), clamp(num(y, 0), 0, 1),
        ] as [number, number])
      }
    }
  }
  if (look.wheels) {
    for (const r of GRADING_RANGES) {
      const w = look.wheels[r]
      if (w) {
        adj.wheels[r] = {
          hue: clamp(num(w.hue, 0), 0, 360),
          saturation: clamp(num(w.saturation, 0), -100, 100),
          luminance: clamp(num(w.luminance, 0), -100, 100),
        }
      }
    }
  }
}

/** 校正一份来路不明的 look（导入 JSON 用）：只保留已知字段，数值夹到合法范围。 */
export function sanitizeLook(raw: unknown): FilterLook {
  const src = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const look: FilterLook = {}
  for (const f of LOOK_FIELDS) {
    if (src[f] === undefined || src[f] === null) continue
    look[f] = clamp(num(src[f], 0), RANGE[f][0], RANGE[f][1])
  }
  const curves = src['curves'] as Record<string, unknown> | undefined
  if (curves && typeof curves === 'object') {
    const out: Partial<Curves> = {}
    for (const ch of CURVE_CHANNELS) {
      const pts = curves[ch]
      if (!Array.isArray(pts)) continue
      const clean = pts
        .filter((p): p is [unknown, unknown] => Array.isArray(p) && p.length === 2)
        .map(([x, y]) => [clamp(num(x, 0), 0, 1), clamp(num(y, 0), 0, 1)] as [number, number])
        .sort((a, b) => a[0] - b[0])
      if (clean.length >= 2) out[ch] = clean
    }
    if (Object.keys(out).length) look.curves = out
  }
  const wheels = src['wheels'] as Record<string, unknown> | undefined
  if (wheels && typeof wheels === 'object') {
    const out: Partial<Record<GradingRange, WheelState>> = {}
    for (const r of GRADING_RANGES) {
      const w = wheels[r] as Record<string, unknown> | undefined
      if (!w || typeof w !== 'object') continue
      out[r] = {
        hue: clamp(num(w.hue, 0), 0, 360),
        saturation: clamp(num(w.saturation, 0), -100, 100),
        luminance: clamp(num(w.luminance, 0), -100, 100),
      }
    }
    if (Object.keys(out).length) look.wheels = out
  }
  return look
}

/** look 是否完全等于默认（「原图」这类空滤镜判断）。 */
export function isNeutralLook(look: FilterLook): boolean {
  for (const f of LOOK_FIELDS) if ((look[f] ?? 0) !== 0) return false
  if (look.curves) {
    const diag = defaultCurves()
    for (const ch of CURVE_CHANNELS) {
      const pts = look.curves[ch]
      if (!pts) continue
      if (JSON.stringify(pts) !== JSON.stringify(diag[ch])) return false
    }
  }
  if (look.wheels) {
    for (const r of GRADING_RANGES) {
      const w = look.wheels[r]
      if (!w) continue
      if (w.hue || w.saturation || w.luminance) return false
    }
  }
  return true
}

// ------------------------------------------------------------ 缩略色块
/** 用 look 的冷暖/饱和/明度推一个渐变缩略图（不用真渲染，切换滤镜零延迟）。 */
export function filterSwatch(look: FilterLook): string {
  const exposure = look.exposure ?? 0
  const contrast = look.contrast ?? 0
  if ((look.blackwhite ?? 0) > 0) {
    const base = clamp(12 + exposure * 7 - contrast * 0.07, 6, 34)
    const top = clamp(base + 34 + contrast * 0.1, base + 12, 66)
    return `linear-gradient(140deg, hsl(0 0% ${base.toFixed(1)}%), hsl(0 0% ${top.toFixed(1)}%))`
  }
  const t = look.temperature ?? 0        // 负=偏蓝，正=偏黄
  const g = look.tint ?? 0               // 负=偏绿，正=偏品红
  const hue = t >= 0 ? 38 + g * 0.7 : 212 + g * 0.7
  // 分级色轮参与缩略图：阴影/高光有饱和度时用它们的色相，青橙、霓虹一眼可分
  const ws = look.wheels?.shadows
  const wh = look.wheels?.highlights
  const wheelSat = (ws?.saturation ?? 0) + (wh?.saturation ?? 0)
  const darkHue = ws && ws.saturation ? ws.hue : hue
  const lightHue = wh && wh.saturation ? wh.hue : hue + 18
  const sat = clamp(8 + Math.abs(t) * 0.85 + (look.vibrance ?? 0) * 0.75
    + (look.saturation ?? 0) * 0.45 + Math.abs(wheelSat) * 0.5, 6, 88)
  const lift = exposure * 6
  const dark = clamp(9 + lift - contrast * 0.06, 3, 30)
  const light = clamp(50 + lift + contrast * 0.09, 30, 72)
  return `linear-gradient(140deg, hsl(${darkHue.toFixed(1)} ${sat.toFixed(1)}% ${dark.toFixed(1)}%), `
    + `hsl(${lightHue.toFixed(1)} ${(sat * 0.82).toFixed(1)}% ${light.toFixed(1)}%))`
}

// ------------------------------------------------------------ 基础滤镜
/** S 曲线（提对比）：暗部压一点、高光提一点。 */
function sCurve(amount = 0.08): [number, number][] {
  return [[0, 0], [0.25, clamp(0.25 - amount, 0, 1)], [0.75, clamp(0.75 + amount, 0, 1)], [1, 1]]
}

/** 褪色曲线（黑位抬高、白位压低），胶片感。 */
function fadeCurve(lift = 0.06, roll = 0.04): [number, number][] {
  return [[0, lift], [0.5, 0.5], [1, 1 - roll]]
}

function zeroTone(): Record<LookField, number> {
  const o = {} as Record<LookField, number>
  for (const f of LOOK_FIELDS) o[f] = 0
  return o
}

function zeroWheels(): Record<GradingRange, WheelState> {
  return {
    shadows: { hue: 0, saturation: 0, luminance: 0 },
    midtones: { hue: 0, saturation: 0, luminance: 0 },
    highlights: { hue: 0, saturation: 0, luminance: 0 },
    global: { hue: 0, saturation: 0, luminance: 0 },
  }
}

export const BUILTIN_FILTERS: FilterPreset[] = [
  {
    id: 'builtin:neutral', name: '原图', group: '基础', builtin: true,
    note: '清空全局影调、曲线与分级（不动蒙版）',
    look: { ...zeroTone(), curves: defaultCurves(), wheels: zeroWheels() },
  },
  {
    id: 'builtin:clean', name: '通透', group: '基础', builtin: true,
    note: '抬暗部、压高光，中等清晰度',
    look: {
      contrast: 14, highlights: -20, shadows: 22, whites: 8, blacks: -10,
      clarity: 12, vibrance: 14, dehaze: 8, curves: { master: sCurve(0.05) },
    },
  },
  {
    id: 'builtin:soft', name: '柔光', group: '基础', builtin: true,
    note: '低对比、低清晰度，柔和过渡',
    look: {
      exposure: 0.1, contrast: -18, highlights: -8, shadows: 20, blacks: 16,
      clarity: -14, dehaze: -8, saturation: -6, vibrance: 6,
      curves: { master: fadeCurve(0.04, 0.03) },
    },
  },
  {
    id: 'builtin:punch', name: '浓郁', group: '基础', builtin: true,
    note: '高对比高饱和，适合色彩丰富的场景',
    look: {
      contrast: 20, highlights: -18, shadows: 14, whites: 10, blacks: -16,
      clarity: 14, vibrance: 24, saturation: 6, dehaze: 12,
      curves: { master: sCurve(0.07) },
    },
  },
  {
    id: 'builtin:warm', name: '暖阳', group: '基础', builtin: true,
    note: '加暖加亮，适合日出/室内暖光',
    look: {
      temperature: 24, tint: 6, exposure: 0.14, contrast: 6, highlights: -14,
      shadows: 14, vibrance: 16, saturation: 4,
      curves: { master: sCurve(0.03) },
    },
  },
  {
    id: 'builtin:cool', name: '冷调', group: '基础', builtin: true,
    note: '冷蓝基调，压低饱和',
    look: {
      temperature: -22, tint: -4, exposure: 0.04, contrast: 8, highlights: -16,
      shadows: 8, blacks: -8, saturation: -12, vibrance: 6,
      wheels: { shadows: { hue: 220, saturation: 14, luminance: 0 } },
    },
  },
  {
    id: 'builtin:portrait', name: '柔肤人像', group: '人像', builtin: true,
    note: '降清晰度与高光，肤色通透',
    look: {
      temperature: 8, tint: 4, exposure: 0.18, contrast: -8, highlights: -22,
      shadows: 16, blacks: 6, clarity: -12, sharpen: 6, vibrance: 12, saturation: -4,
      curves: { master: [[0, 0.02], [0.35, 0.37], [0.8, 0.84], [1, 0.99]] },
    },
  },
  {
    id: 'builtin:landscape', name: '清透风光', group: '风光', builtin: true,
    note: '去朦胧 + 自然饱和度，天空更干净',
    look: {
      contrast: 12, highlights: -26, shadows: 24, whites: 6, blacks: -8,
      clarity: 16, dehaze: 16, vibrance: 26, saturation: 4,
      curves: { master: sCurve(0.06) },
    },
  },
  {
    id: 'builtin:film', name: '复古胶片', group: '胶片', builtin: true,
    note: '黑位抬高、白位压低，暖中带品红',
    look: {
      temperature: 12, tint: 8, contrast: -6, highlights: -14, shadows: 12,
      blacks: 12, saturation: -18, vibrance: 10, clarity: -4,
      curves: { master: fadeCurve(0.07, 0.05) },
      wheels: { shadows: { hue: 40, saturation: 10, luminance: 0 } },
    },
  },
  {
    id: 'builtin:cinema', name: '电影青橙', group: '胶片', builtin: true,
    note: '阴影青、高光橙的经典电影配色',
    look: {
      temperature: 2, tint: 2, contrast: 16, highlights: -20, shadows: -4,
      blacks: -14, saturation: -8, vibrance: 14, clarity: 6,
      wheels: {
        shadows: { hue: 196, saturation: 18, luminance: -4 },
        highlights: { hue: 32, saturation: 16, luminance: 2 },
      },
      curves: { master: sCurve(0.05) },
    },
  },
  {
    id: 'builtin:jp', name: '日系清透', group: '胶片', builtin: true,
    note: '高亮度低饱和，蓝绿偏冷',
    look: {
      temperature: -6, tint: -3, exposure: 0.24, contrast: -12, highlights: -10,
      shadows: 26, blacks: 12, saturation: -14, vibrance: 10, clarity: 2,
      curves: { master: [[0, 0.05], [0.5, 0.54], [1, 1]] },
    },
  },
  {
    id: 'builtin:neon', name: '夜景霓虹', group: '风光', builtin: true,
    note: '压黑位、提饱和，青紫氛围',
    look: {
      temperature: -8, tint: 8, exposure: -0.08, contrast: 22, highlights: -12,
      shadows: -8, blacks: -20, saturation: 14, vibrance: 20, clarity: 8,
      wheels: {
        shadows: { hue: 252, saturation: 20, luminance: 0 },
        highlights: { hue: 320, saturation: 14, luminance: 0 },
      },
    },
  },
  {
    id: 'builtin:bw', name: '经典黑白', group: '黑白', builtin: true,
    note: '中性黑白，中等对比',
    look: {
      blackwhite: 100, contrast: 14, highlights: -8, shadows: 8, clarity: 10,
      curves: { master: sCurve(0.04) },
    },
  },
  {
    id: 'builtin:bw-hard', name: '高反差黑白', group: '黑白', builtin: true,
    note: '重对比、压黑位，街头纪实感',
    look: {
      blackwhite: 100, contrast: 34, highlights: -6, shadows: -10, whites: 12,
      blacks: -22, clarity: 22,
      curves: { master: [[0, 0], [0.22, 0.12], [0.78, 0.9], [1, 1]] },
    },
  },
]

// ------------------------------------------------------------ JSON 导入/导出
export interface FilterFile {
  format: string
  version: number
  name: string
  createdAt?: string
  note?: string
  look: FilterLook
}

/** 单个滤镜 -> 导出用的 JSON 对象。 */
export function filterToFile(p: FilterPreset): FilterFile {
  const out: FilterFile = {
    format: FILTER_FORMAT,
    version: FILTER_VERSION,
    name: p.name,
    createdAt: new Date(p.createdAt ?? Date.now()).toISOString(),
    look: p.look,
  }
  if (p.note) out.note = p.note
  return out
}

let importSeq = 0

/** 解析导入的 JSON：支持单个滤镜、数组、{filters:[...]} 三种写法。 */
export function parseFilterFile(text: string, fallbackName = '导入的滤镜'): FilterPreset[] {
  const data = JSON.parse(text) as unknown
  const items: unknown[] = Array.isArray(data)
    ? data
    : (data && typeof data === 'object' && Array.isArray((data as { filters?: unknown[] }).filters))
      ? (data as { filters: unknown[] }).filters
      : [data]
  const out: FilterPreset[] = []
  for (const it of items) {
    if (!it || typeof it !== 'object') continue
    const o = it as Record<string, unknown>
    const look = sanitizeLook(o['look'] ?? o['adjustments'] ?? o)
    if (!Object.keys(look).length) continue
    const name = typeof o['name'] === 'string' && o['name'].trim()
      ? o['name'].trim().slice(0, 40)
      : fallbackName
    out.push({
      id: `custom:${Date.now().toString(36)}-${(importSeq++).toString(36)}`,
      name,
      group: '自定义',
      note: typeof o['note'] === 'string' ? o['note'].slice(0, 80) : undefined,
      look,
      createdAt: Date.now(),
    })
  }
  if (!out.length) throw new Error('文件里没有可用的滤镜配置')
  return out
}

/** 导出文件名：photolab-filter-<名字>.json（去掉路径非法字符）。 */
export function filterFileName(name: string): string {
  const safe = name.replace(/[\\/:*?"<>|\s]+/g, '-').replace(/^-+|-+$/g, '') || 'filter'
  return `photolab-filter-${safe}.json`
}

/** 下载 JSON（浏览器模式 / Electron 渲染进程都可用）。 */
export function downloadJson(fileName: string, payload: unknown): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = fileName
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
