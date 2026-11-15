/** 参数模型 —— 与 core.py 的 dataclass 一一对应（Adjustments/MaskState/...）。
 *  数值范围与默认值一律照抄旧代码，禁止自创。 */

export type CurveChannel = 'master' | 'red' | 'green' | 'blue'
export const CURVE_CHANNELS: readonly CurveChannel[] = ['master', 'red', 'green', 'blue']

export type GradingRange = 'shadows' | 'midtones' | 'highlights' | 'global'
export const GRADING_RANGES: readonly GradingRange[] = ['shadows', 'midtones', 'highlights', 'global']

export type MaskKind = 'linear' | 'radial' | 'brush' | 'polygon' | 'ai'

export interface WheelState {
  hue: number          // 色相角度（度）
  saturation: number   // -100..100
  luminance: number    // -100..100
}

export interface PickerState {
  hue: number
  color: [number, number, number]   // 0..1 sRGB，UI 色块用
  span: number                      // 影响色相范围（度）
  hue_shift: number                 // -60..60
  saturation: number                // -100..100
  luminance: number                 // -100..100
  enabled: boolean
}

export type CurvePoints = [number, number][]
export type Curves = Record<CurveChannel, CurvePoints>

export interface MaskAdjustments {
  exposure: number; contrast: number; highlights: number; shadows: number
  whites: number; blacks: number; temperature: number; tint: number
  saturation: number; vibrance: number
  clarity: number; denoise: number; sharpen: number; dehaze: number
  curves: Curves
  blend: number
  wheels: Record<GradingRange, WheelState>
  pickers: PickerState[]
}

export type AiMaskTarget = 'subject' | 'person' | 'background' | 'sky' | 'depth_near' | 'depth_far'

export interface MaskState {
  kind: MaskKind
  name: string
  enabled: boolean
  invert: boolean
  feather: number          // 0..100
  x1: number; y1: number; x2: number; y2: number    // 线性渐变（归一化）
  cx: number; cy: number                            // 径向中心
  radius: number; aspect: number; angle: number     // 径向椭圆
  size: number             // 画笔 1..100
  strokes: [number, number][][]
  points: [number, number][] // 自定义多边形顶点（归一化 0..1）
  ai_target: string        // AI 蒙版目标（subject/person/background/sky）
  grid: string[]           // VLM 网格掩码：每行 grid_w 个 '0'/'1' 字符
  grid_w: number
  grid_h: number
  mask_png: string         // 本地像素级掩码：单通道 PNG base64（256 级软边界）
  mask_w: number
  mask_h: number
  adj: MaskAdjustments
}

export interface Adjustments {
  temperature: number; tint: number
  exposure: number; contrast: number; highlights: number; shadows: number
  whites: number; blacks: number
  saturation: number; vibrance: number; blackwhite: number
  clarity: number; denoise: number; sharpen: number; dehaze: number
  curves: Curves
  blend: number
  wheels: Record<GradingRange, WheelState>
  pickers: PickerState[]
  masks: MaskState[]
}

// ---- 默认值（与 core.py dataclass 默认一致）----
export function defaultWheel(): WheelState {
  return { hue: 0, saturation: 0, luminance: 0 }
}

export function defaultCurves(): Curves {
  const diag: CurvePoints = [[0, 0], [1, 1]]
  return { master: [...diag], red: [...diag], green: [...diag], blue: [...diag] }
}

export function defaultWheels(): Record<GradingRange, WheelState> {
  return {
    shadows: defaultWheel(), midtones: defaultWheel(),
    highlights: defaultWheel(), global: defaultWheel(),
  }
}

export function defaultMaskAdjustments(): MaskAdjustments {
  return {
    exposure: 0, contrast: 0, highlights: 0, shadows: 0,
    whites: 0, blacks: 0, temperature: 0, tint: 0,
    saturation: 0, vibrance: 0, clarity: 0, denoise: 0, sharpen: 0, dehaze: 0,
    curves: defaultCurves(), blend: 50, wheels: defaultWheels(), pickers: [],
  }
}

export function defaultMask(kind: MaskKind, name: string): MaskState {
  return {
    kind, name, enabled: true, invert: false, feather: 30,
    x1: 0.3, y1: 0.5, x2: 0.7, y2: 0.5,
    cx: 0.5, cy: 0.5, radius: 0.30, aspect: 1.0, angle: 0,
    size: 12, strokes: [], points: [],
    ai_target: '', grid: [], grid_w: 0, grid_h: 0,
    mask_png: '', mask_w: 0, mask_h: 0,
    adj: defaultMaskAdjustments(),
  }
}

/** AI 识别蒙版（VLM 返回 0/1 网格，归一化覆盖整幅图像）。 */
export function aiMask(
  name: string, target: AiMaskTarget, grid: string[], gridW: number, gridH: number,
): MaskState {
  const m = defaultMask('ai', name)
  m.feather = 0           // AI 蒙版按识别边界作用，不羽化
  m.ai_target = target
  m.grid = grid
  m.grid_w = gridW
  m.grid_h = gridH
  return m
}

/** 本地像素级 AI 蒙版（SAM2 掩码 base64 PNG，256 级软边界）。 */
export function aiMaskPixels(
  name: string, target: AiMaskTarget, png: string, w: number, h: number,
): MaskState {
  const m = defaultMask('ai', name)
  m.feather = 0           // AI 蒙版按识别边界作用，不羽化
  m.ai_target = target
  m.mask_png = png
  m.mask_w = w
  m.mask_h = h
  return m
}

export function defaultAdjustments(): Adjustments {
  return {
    temperature: 0, tint: 0,
    exposure: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
    saturation: 0, vibrance: 0, blackwhite: 0,
    clarity: 0, denoise: 0, sharpen: 0, dehaze: 0,
    curves: defaultCurves(), blend: 50, wheels: defaultWheels(),
    pickers: [], masks: [],
  }
}

/** 全局参数与蒙版局部参数的公共形状（面板目标切换用）。 */
export type AdjustmentsLike = Adjustments | MaskAdjustments

/** auto.suggest_* 返回的字典按旧版 _apply_adjustment_dict 语义合并进当前参数。 */
export const TONE_FIELDS = [
  'temperature', 'tint', 'exposure', 'contrast', 'highlights', 'shadows',
  'whites', 'blacks', 'saturation', 'vibrance', 'blackwhite',
] as const
export type ToneField = (typeof TONE_FIELDS)[number]

/**
 * 按强度（0..1）缩放 AI 返回的调整参数，避免模型给过量数值导致过度后期。
 * 标量与曝光直接乘；曲线向恒等线收缩（y' = x + (y-x)*gain）；
 * 色轮与取色器的饱和度/明度乘、色相本身不变。
 */
export function scaleAiAdjustments(
  d: Record<string, unknown>, strength: number,
): Record<string, unknown> {
  const gain = Math.min(1, Math.max(0, Number(strength)))
  if (!Number.isFinite(gain) || gain >= 1) return d
  const out: Record<string, unknown> = { ...d }
  for (const name of TONE_FIELDS) {
    const v = out[name]
    if (typeof v === 'number' && Number.isFinite(v)) out[name] = v * gain
  }
  const curves = out['curves']
  if (curves && typeof curves === 'object') {
    const scaled: Record<string, unknown> = {}
    for (const [ch, pts] of Object.entries(curves as Record<string, unknown>)) {
      if (!Array.isArray(pts)) continue
      scaled[ch] = pts
        .filter((p): p is [number, number] => Array.isArray(p) && p.length === 2)
        .map(([x, y]) => [x, x + (y - x) * gain] as [number, number])
    }
    out['curves'] = scaled
  }
  const wheels = out['wheels']
  if (wheels && typeof wheels === 'object') {
    const scaled: Record<string, unknown> = {}
    for (const [range, w] of Object.entries(wheels as Record<string, unknown>)) {
      if (!w || typeof w !== 'object') continue
      const wheel = w as Record<string, unknown>
      scaled[range] = {
        hue: Number(wheel.hue) || 0,
        saturation: (Number(wheel.saturation) || 0) * gain,
        luminance: (Number(wheel.luminance) || 0) * gain,
      }
    }
    out['wheels'] = scaled
  }
  const pickers = out['pickers']
  if (Array.isArray(pickers)) {
    out['pickers'] = (pickers as Record<string, unknown>[]).map((p) => ({
      ...p,
      hue_shift: (Number(p.hue_shift) || 0) * gain,
      saturation: (Number(p.saturation) || 0) * gain,
      luminance: (Number(p.luminance) || 0) * gain,
    }))
  }
  return out
}
