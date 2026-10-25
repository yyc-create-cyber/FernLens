"""FernLens 核心：纯 NumPy / OpenCV 图像处理。

本模块不依赖 Qt，输入输出都是 np.ndarray（float32 线性光，HxWx3）。
内容分区（卷标注释可快速定位）：
  1. 色彩空间工具      2. 参数模型        3. 基础调整
  4. 曲线              5. 颜色分级        6. 取色器
  7. 效果调节（清晰度/降噪/锐度/去朦胧）  8. 蒙版（含曲线/色轮/取色器局部化）
  8b. 仿制图章         9. 直方图          10. 处理流水线
  蒙版支持线性渐变 / 径向渐变 / 画笔 / 自定义多边形。

流水线顺序（float32 线性光空间）：
    白平衡 -> 基础 -> 效果 -> 曲线 -> 颜色分级 -> 蒙版局部调整 -> 取色

纯逐像素的阶段（白平衡/基础/黑白/曲线/颜色分级）可走 OpenCL GPU 路径
（gpu.py，可选），含效果/蒙版/取色器时自动回退 CPU 分块并行。
"""
from __future__ import annotations

import colorsys
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

# ======================================================================
# 1. 色彩空间工具
# ======================================================================

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> 线性光。"""
    x = np.asarray(x, dtype=np.float32)
    out = (x + np.float32(0.055)) * np.float32(1.0 / 1.055)
    low = x * np.float32(1.0 / 12.92)
    small = x <= np.float32(0.04045)
    np.power(out, np.float32(2.4), out=out)
    np.copyto(out, low, where=small)
    return out


def to_srgb(x: np.ndarray) -> np.ndarray:
    """线性光 -> sRGB [0,1]（内部裁剪，避免负数开方产生 NaN）。"""
    out = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    small = out <= np.float32(0.0031308)
    low = out * np.float32(12.92)
    np.power(out, np.float32(1.0 / 2.4), out=out)
    out *= np.float32(1.055)
    out -= np.float32(0.055)
    np.copyto(out, low, where=small)
    return out


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.709 相对亮度，返回 (H, W) float32。"""
    return img @ LUMA


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """平滑阶梯函数 smoothstep(t)=t²(3-2t)。"""
    t = (x - np.float32(e0)) * np.float32(1.0 / (e1 - e0))
    np.clip(t, 0.0, 1.0, out=t)
    t2 = t * t
    c = np.float32(3.0) - np.float32(2.0) * t
    return t2 * c


def rgb8_to_linear(rgb8: np.ndarray) -> np.ndarray:
    """uint8 RGB (H,W,3) -> float32 线性光 [0,1]。"""
    return to_linear(rgb8.astype(np.float32) * np.float32(1.0 / 255.0))


def linear_to_rgb8(lin: np.ndarray) -> np.ndarray:
    """float32 线性光 -> uint8 sRGB。"""
    out = to_srgb(lin)
    out *= np.float32(255.0)
    out += np.float32(0.5)
    return out.astype(np.uint8)


# ======================================================================
# 2. 参数模型
# ======================================================================

CURVE_CHANNELS = ("master", "red", "green", "blue")
GRADING_RANGES = ("shadows", "midtones", "highlights", "global")

# 效果字段（清晰度/降噪/锐度/去朦胧）
EFFECT_FIELDS = ("clarity", "denoise", "sharpen", "dehaze")

# 取色器预设色相：名称 -> 色相（度）
HS_PRESETS = (("红", 0.0), ("橙", 30.0), ("黄", 60.0), ("绿", 120.0),
              ("青", 180.0), ("蓝", 240.0), ("紫", 280.0), ("洋红", 300.0))

_DIAGONAL = [(0.0, 0.0), (1.0, 1.0)]


def default_curves() -> dict:
    """四个通道各自独立的控制点集合。"""
    return {ch: [tuple(p) for p in _DIAGONAL] for ch in CURVE_CHANNELS}


@dataclass
class WheelState:
    """一个颜色分级色轮的状态。"""

    hue: float = 0.0            # 色相角度（度）
    saturation: float = 0.0     # -100..100，正值=该色相，负值=相反色相
    luminance: float = 0.0      # -100..100


@dataclass
class PickerState:
    """一个 HSL 取色点。"""

    hue: float = 0.0                    # 取色得到的中心色相（度）
    color: tuple = (0.5, 0.5, 0.5)      # 采样到的 sRGB 颜色（0..1），用于 UI 色块
    span: float = 30.0                  # 影响色相范围（度），两侧羽化
    hue_shift: float = 0.0              # 色相旋转（度，-60..60，可令绿偏黄/偏青）
    saturation: float = 0.0             # 饱和度调整 -100..100
    luminance: float = 0.0              # 明亮度调整 -100..100
    enabled: bool = True


@dataclass
class MaskAdjustments:
    """蒙版内部署的局部调整（字段名与 Adjustments 一致，便于直接复用算子）。

    除基础影调与效果外，还带曲线/颜色分级/取色器——蒙版内跑与全局相同的
    mini 流水线，只是作用范围被蒙版限制。
    """

    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    clarity: float = 0.0
    denoise: float = 0.0
    sharpen: float = 0.0
    dehaze: float = 0.0

    curves: dict = field(default_factory=default_curves)
    blend: float = 50.0
    wheels: dict = field(default_factory=lambda: {r: WheelState() for r in GRADING_RANGES})
    pickers: list = field(default_factory=list)

    def is_default(self) -> bool:
        if any(getattr(self, f) != 0.0 for f in (
                "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
                "temperature", "tint", "saturation", "vibrance",
                "clarity", "denoise", "sharpen", "dehaze")):
            return False
        for pts in self.curves.values():
            if [tuple(p) for p in pts] != [tuple(p) for p in _DIAGONAL]:
                return False
        if any(w.saturation or w.luminance for w in self.wheels.values()):
            return False
        if any(p.hue_shift or p.saturation or p.luminance for p in self.pickers):
            return False
        return True


@dataclass
class MaskState:
    """一个局部蒙版：linear(线性渐变) / radial(径向渐变) / brush(画笔) / polygon(自定义多边形)。

    几何坐标全部归一化到 0..1（相对图像宽高），便于缩放与传输；
    feather / size 为 0..100 的 UI 档位，渲染时换算成像素。
    """

    kind: str = "linear"                 # 'linear' | 'radial' | 'brush' | 'polygon'
    name: str = ""
    enabled: bool = True
    invert: bool = False                 # 反向：作用于蒙版之外
    feather: float = 30.0                # 0..100 羽化

    # 线性渐变：起点 -> 终点
    x1: float = 0.3
    y1: float = 0.5
    x2: float = 0.7
    y2: float = 0.5

    # 径向渐变（椭圆）：中心 / 半径 / 长宽比 / 旋转角
    cx: float = 0.5
    cy: float = 0.5
    radius: float = 0.30
    aspect: float = 1.0
    angle: float = 0.0            # 椭圆长轴角度（度）

    # 画笔：笔刷大小 + 笔画（每段笔画是 [(x, y), ...] 归一化点序列）
    size: float = 12.0                   # 1..100
    strokes: list = field(default_factory=list)

    # 自定义多边形：至少 3 个顶点 [(x, y), ...]（归一化）；
    # closed 仅用于画布交互（False = 仍在选点，True = 已完成）。
    points: list = field(default_factory=list)
    closed: bool = False

    adj: MaskAdjustments = field(default_factory=MaskAdjustments)

    def is_trivial(self) -> bool:
        return (not self.enabled) or self.adj.is_default()


@dataclass
class Adjustments:
    """全部调色参数（顺序即流水线顺序）。"""

    # ---- 白平衡 ----
    temperature: float = 0.0   # -100..100 蓝 <-> 黄
    tint: float = 0.0          # -100..100 绿 <-> 品红

    # ---- 基础 ----
    exposure: float = 0.0      # -5..5 EV
    contrast: float = 0.0      # -100..100
    highlights: float = 0.0    # -100..100
    shadows: float = 0.0       # -100..100
    whites: float = 0.0        # -100..100
    blacks: float = 0.0        # -100..100

    # ---- 颜色基础 ----
    saturation: float = 0.0    # -100..100
    vibrance: float = 0.0      # -100..100
    blackwhite: float = 0.0    # 0..100 黑白混合（100 = 完全灰度）

    # ---- 效果 ----
    clarity: float = 0.0       # -100..100 清晰度（中频局部对比）
    denoise: float = 0.0       # 0..100    降噪（感知域双边缘滤波）
    sharpen: float = 0.0       # 0..100    锐度（USM）
    dehaze: float = 0.0        # -100..100 去朦胧

    # ---- 曲线 ----
    curves: dict = field(default_factory=default_curves)

    # ---- 颜色分级 ----
    blend: float = 50.0        # 0..100 过渡混合（越大过渡越柔和）
    wheels: dict = field(default_factory=lambda: {r: WheelState() for r in GRADING_RANGES})

    # ---- 取色器 ----
    pickers: list = field(default_factory=list)

    # ---- 蒙版 ----
    masks: list = field(default_factory=list)

    def copy(self) -> "Adjustments":
        import copy
        return copy.deepcopy(self)

    @classmethod
    def defaults(cls) -> "Adjustments":
        return cls()

    def is_default(self) -> bool:
        """全部为默认值时可直接跳过处理（快速预览 / 保存原图）。"""
        for name in ("temperature", "tint", "exposure", "contrast", "highlights",
                     "shadows", "whites", "blacks", "saturation", "vibrance",
                     "clarity", "denoise", "sharpen", "dehaze", "blackwhite"):
            if getattr(self, name) != 0.0:
                return False
        for pts in self.curves.values():
            if [tuple(p) for p in pts] != [tuple(p) for p in _DIAGONAL]:
                return False
        if any(w.saturation or w.luminance for w in self.wheels.values()):
            return False
        if any(p.hue_shift or p.saturation or p.luminance for p in self.pickers):
            return False
        if any(not m.is_trivial() for m in self.masks):
            return False
        return True


# ======================================================================
# 3. 基础调整（白平衡 / 影调 / 颜色）
# ======================================================================
# 公开函数不修改输入；内部尽量原地计算（大数组临时内存是主要瓶颈）。


def _mul_(out: np.ndarray, w: np.ndarray, k: float):
    """out *= (1 + k*w)，w 为 (H,W) 蒙版。"""
    t = w * np.float32(k)
    t += np.float32(1.0)
    out *= t[..., None]


def _wb_gains(temperature: float, tint: float) -> np.ndarray:
    """色温/色调 -> RGB 增益（几何均值归一化保护曝光）。"""
    t = float(temperature) / 100.0
    ti = float(tint) / 100.0
    gains = np.array([
        1.0 + t * 0.35 + ti * 0.10,   # R：偏黄/品红
        1.0 - ti * 0.22,              # G：品红方向压绿
        1.0 - t * 0.35 + ti * 0.10,   # B：偏蓝方向
    ], dtype=np.float32)
    gains /= np.cbrt(np.prod(gains))  # 保持整体亮度大致不变
    return gains


def _wb_(out: np.ndarray, temperature: float, tint: float):
    out *= _wb_gains(temperature, tint)


def apply_white_balance(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """色温（蓝<->黄）与色调（绿<->品红），以通道增益实现。"""
    if temperature == 0 and tint == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    _wb_(out, temperature, tint)
    return out


def apply_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """曝光，单位为 EV（挡）。"""
    if ev == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    out *= np.float32(2.0 ** float(ev))
    return out


def apply_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """对比度：绕 0.5 的 S 曲线，保证 0/1 端点不动。

    正值用 smoothstep(smoothstep(v))（枢轴斜率 2.25、端点趋平，接近 LR 的 +100 手感）；
    负值向 smoothstep(v) 混合（枢轴斜率降到 0.5，画面发灰）。
    """
    if amount == 0:
        return img
    a = np.float32(np.clip(amount / 100.0, -1.0, 1.0))
    out = np.array(img, dtype=np.float32, copy=True)
    v = out

    # smoothstep(v) = 3v² - 2v³
    t2 = v * v
    t2 *= np.float32(2.0)
    t2 *= v                          # t2 = 2v³
    s = v * v
    s *= np.float32(3.0)             # s = 3v²
    s -= t2                          # s = smoothstep(v)

    if a > 0:
        # smoothstep(smoothstep(v))：枢轴斜率 2.25、端点趋平，接近 LR 的 +100 手感
        u2 = s * s
        u2 *= np.float32(2.0)
        u2 *= s                      # 2s³
        s2 = s * s
        s2 *= np.float32(3.0)        # 3s²
        s2 -= u2                     # smoothstep(smoothstep(v))
        out = v * (np.float32(1.0) - a) + s2 * a
    else:
        # v + c·v(1-v)(v-0.5)，c = 2a ∈ [-2, 0]：枢轴斜率降到 0.5，端点斜率 ≤ 2，保持单调
        c = np.float32(2.0) * a
        t = v * (np.float32(1.0) - v)
        t *= (v - np.float32(0.5))
        out = v + c * t
    return out


def apply_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    """高光：按亮度权重对亮部做增益（负值恢复高光细节）。"""
    if amount == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    _mul_(out, smoothstep(0.35, 0.95, luminance(out)), (amount / 100.0) * 1.8)
    return out


def apply_shadows(img: np.ndarray, amount: float) -> np.ndarray:
    """阴影：暗部乘性+加性提亮/压暗。"""
    if amount == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    w = np.float32(1.0) - smoothstep(0.0, 0.45, luminance(out))
    _mul_(out, w, (amount / 100.0) * 0.6)
    out += np.float32((amount / 100.0) * 0.10) * w[..., None]
    return out


def apply_whites(img: np.ndarray, amount: float) -> np.ndarray:
    """白色：调整白场（推向纯白或拉回灰度）。"""
    if amount == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    w = smoothstep(0.55, 1.0, luminance(out))
    _mul_(out, w, (amount / 100.0) * 0.9)
    out += np.float32((amount / 100.0) * 0.06) * w[..., None]
    return out


def apply_blacks(img: np.ndarray, amount: float) -> np.ndarray:
    """黑色：调整黑场（正值抬黑，负值加深）。

    数学上等价于 out*(1+k*w) + k*w*(1-out) = out + k*w，故只需一次加性偏移。
    """
    if amount == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    w = np.float32(1.0) - smoothstep(0.0, 0.40, luminance(out))
    out += np.float32((amount / 100.0) * 0.09) * w[..., None]
    return out


def _chroma(img: np.ndarray) -> np.ndarray:
    """归一化彩度估计 (max-min)/(max+min)，返回 (H,W)。"""
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    t = mx - mn
    t /= (mx + mn + np.float32(1e-5))
    return t


def apply_saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """全局饱和度：把像素向亮度轴收缩/扩张。-100 为灰度。"""
    if amount == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    lum = luminance(out)
    out -= lum[..., None]
    out *= np.float32(1.0 + float(amount) / 100.0)
    out += lum[..., None]
    return out


def apply_vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """自然饱和度：对低饱和像素作用更强，保护皮肤等已饱和的颜色。"""
    if amount == 0:
        return img
    out = np.array(img, dtype=np.float32, copy=True)
    lum = luminance(out)
    a = float(amount) / 100.0
    sat = _chroma(out)
    w = (np.float32(1.0) - sat) if a > 0 else sat
    w *= np.float32(a)
    w += np.float32(1.0)
    out -= lum[..., None]
    out *= w[..., None]
    out += lum[..., None]
    return out


def apply_blackwhite(img: np.ndarray, amount: float) -> np.ndarray:
    """黑白模式：按 Rec.709 亮度混合到灰度。amount 0..100（强度）。"""
    if amount == 0:
        return img
    k = np.float32(np.clip(amount / 100.0, 0.0, 1.0))
    lum = luminance(img)[..., None]
    return img * (np.float32(1.0) - k) + lum * k


def apply_basic(img: np.ndarray, params) -> np.ndarray:
    """按 Lightroom 的基础面板顺序应用全部基础调整。

    params 只需具备 exposure/contrast/highlights/shadows/whites/blacks/
    temperature/tint/saturation/vibrance 字段，MaskAdjustments 亦可直接传入。
    """
    if (params.temperature == 0 and params.tint == 0 and params.exposure == 0
            and params.contrast == 0 and params.highlights == 0 and params.shadows == 0
            and params.whites == 0 and params.blacks == 0
            and params.saturation == 0 and params.vibrance == 0):
        return img

    out = np.array(img, dtype=np.float32, copy=True)
    if params.temperature or params.tint:
        _wb_(out, params.temperature, params.tint)
    if params.exposure:
        out *= np.float32(2.0 ** float(params.exposure))
    if params.contrast:
        out = apply_contrast(out, params.contrast)
    if params.highlights:
        _mul_(out, smoothstep(0.35, 0.95, luminance(out)), (params.highlights / 100.0) * 1.8)
    if params.shadows:
        w = np.float32(1.0) - smoothstep(0.0, 0.45, luminance(out))
        _mul_(out, w, (params.shadows / 100.0) * 0.6)
        out += np.float32((params.shadows / 100.0) * 0.10) * w[..., None]
    if params.whites:
        w = smoothstep(0.55, 1.0, luminance(out))
        _mul_(out, w, (params.whites / 100.0) * 0.9)
        out += np.float32((params.whites / 100.0) * 0.06) * w[..., None]
    if params.blacks:
        w = np.float32(1.0) - smoothstep(0.0, 0.40, luminance(out))
        out += np.float32((params.blacks / 100.0) * 0.09) * w[..., None]
    if params.saturation:
        lum = luminance(out)
        out -= lum[..., None]
        out *= np.float32(1.0 + params.saturation / 100.0)
        out += lum[..., None]
    if params.vibrance:
        lum = luminance(out)
        a = params.vibrance / 100.0
        sat = _chroma(out)
        w = (np.float32(1.0) - sat) if a > 0 else sat
        w *= np.float32(a)
        w += np.float32(1.0)
        out -= lum[..., None]
        out *= w[..., None]
        out += lum[..., None]
    return out


# ======================================================================
# 4. 曲线（单调三次插值 + LUT）
# ======================================================================

LUT_SIZE = 2048          # 插值网格精度
MIN_POINT_GAP = 1 / 512  # 控制点最小水平间距


def sanitize_points(points) -> list:
    """把任意控制点集合规整为：x 严格递增、y ∈ [0,1]、端点固定在 x=0 / x=1。"""
    pts = sorted(((float(x), float(np.clip(y, 0.0, 1.0))) for x, y in points),
                 key=lambda p: p[0])
    out: list = []
    for x, y in pts:
        if out and (x - out[-1][0]) < MIN_POINT_GAP:
            continue  # 丢弃水平方向过近的点
        out.append((x, y))
    if not out:
        return [(0.0, 0.0), (1.0, 1.0)]
    if len(out) == 1:
        out = [(0.0, out[0][1]), (1.0, out[0][1])]
    out[0] = (0.0, out[0][1])
    out[-1] = (1.0, out[-1][1])
    return out


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson 单调保形三次插值的节点导数。"""
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h                      # 分段割线斜率
    m = np.zeros(n, dtype=np.float64)
    if n == 2:
        m[:] = d[0]
        return m
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0                      # 局部极值点，导数为零 -> 保单调
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    # 两端使用单侧三点估计，并做形状保护
    dd = ((2.0 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    if np.sign(dd) != np.sign(d[0]):
        dd = 0.0
    elif np.sign(d[0]) != np.sign(d[1]) and abs(dd) > 3.0 * abs(d[0]):
        dd = 3.0 * d[0]
    m[0] = dd
    dd = ((2.0 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    if np.sign(dd) != np.sign(d[-1]):
        dd = 0.0
    elif np.sign(d[-1]) != np.sign(d[-2]) and abs(dd) > 3.0 * abs(d[-1]):
        dd = 3.0 * d[-1]
    m[-1] = dd
    return m


def build_lut(points, size: int = LUT_SIZE) -> np.ndarray:
    """由控制点生成长度为 size 的 float32 查找表。"""
    pts = sanitize_points(points)
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    m = _pchip_slopes(x, y)

    grid = np.linspace(0.0, 1.0, size, dtype=np.float64)
    seg = np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(x) - 2)
    out = np.empty(size, dtype=np.float64)
    for i in range(len(x) - 1):
        sel = seg == i
        if not sel.any():
            continue
        h = x[i + 1] - x[i]
        t = (grid[sel] - x[i]) / h
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        out[sel] = (h00 * y[i] + h10 * h * m[i]
                    + h01 * y[i + 1] + h11 * h * m[i + 1])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compose_lut(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """LUT 复合：先 inner 后 outer（用于主通道曲线 + 单色通道曲线合并）。"""
    grid = np.linspace(0.0, 1.0, outer.size, dtype=np.float64)
    return np.interp(inner, grid, outer).astype(np.float32)


def channel_luts(curves: dict) -> dict:
    """返回每个输出通道最终使用的 LUT（已复合主通道）。"""
    master = build_lut(curves.get("master") or default_curves()["master"])
    luts = {}
    for ch in ("red", "green", "blue"):
        pts = curves.get(ch) or default_curves()[ch]
        luts[ch] = compose_lut(master, build_lut(pts))
    return luts


def apply_curves(img: np.ndarray, curves: dict) -> np.ndarray:
    """应用 RGB 曲线（线性光空间）。"""
    luts = channel_luts(curves)
    idx = np.clip((img * np.float32(LUT_SIZE - 1)).astype(np.int32), 0, LUT_SIZE - 1)
    out = np.empty_like(img)
    for c, ch in enumerate(("red", "green", "blue")):
        out[..., c] = luts[ch].take(idx[..., c])
    return out


def apply_curves_fast(img: np.ndarray, curves: dict) -> np.ndarray:
    """apply_curves 的短路版本：曲线全默认时不做查表。"""
    if curves_are_default(curves):
        return img
    return apply_curves(img, curves)


def curves_are_default(curves: dict) -> bool:
    for ch in CURVE_CHANNELS:
        if [tuple(p) for p in sanitize_points(curves.get(ch) or [])] != _DIAGONAL:
            return False
    return True


# ======================================================================
# 5. 颜色分级（四个色轮）
# ======================================================================

SAT_STRENGTH = 0.35
LUM_STRENGTH = 0.25


def wheel_direction(hue_deg: float, strength: float) -> np.ndarray:
    """色轮到 RGB 方向向量（零均值，|分量| <= 1）。

    三轴余弦色轮：hue=0 红 / 120 绿 / 240 蓝，60/180/300 为黄/青/品红。
    strength ∈ [-1,1]，负值相当于把色相旋转 180°。
    """
    if strength == 0:
        return np.zeros(3, dtype=np.float32)
    h = np.deg2rad(float(hue_deg) % 360.0)
    d = np.array([
        np.cos(h),
        np.cos(h - 2.0 * np.pi / 3.0),
        np.cos(h + 2.0 * np.pi / 3.0),
    ], dtype=np.float64)
    d /= np.sqrt(1.5)  # 归一化，保证最大分量为 1
    return (d * float(strength)).astype(np.float32)


def range_masks(lum: np.ndarray) -> dict:
    """四个影调范围的蒙版（可重叠，中间调为钟形）。"""
    shadows = np.float32(1.0) - smoothstep(0.00, 0.50, lum)
    highlights = smoothstep(0.50, 1.00, lum)
    midtones = np.float32(1.0) - np.abs(np.float32(2.0) * lum - np.float32(1.0)) * np.float32(1.6)
    np.clip(midtones, 0.0, 1.0, out=midtones)
    return {
        "shadows": shadows,
        "midtones": midtones,
        "highlights": highlights,
        "global": np.ones_like(lum),
    }


def blend_exponent(blend: float) -> float:
    """blend 0..100 -> 蒙版指数。数值越大过渡越柔和（1..4）。"""
    return 1.0 + 3.0 * (1.0 - float(np.clip(blend, 0.0, 100.0)) / 100.0)


def apply_color_grading(img: np.ndarray, wheels: dict, blend: float) -> np.ndarray:
    """应用四个色轮。wheels: {range: WheelState}。"""
    active = {name: w for name, w in wheels.items() if w.saturation or w.luminance}
    if not active:
        return img
    lum = luminance(img)
    masks = range_masks(lum)
    exp = blend_exponent(blend)
    out = np.array(img, dtype=np.float32, copy=True)

    for name, w in active.items():
        strength = float(w.saturation) / 100.0
        lum_amt = float(w.luminance) / 100.0
        mask = masks[name]
        if exp != 1.0:
            mask = np.power(mask, np.float32(exp))
        offset = wheel_direction(w.hue, strength) * np.float32(SAT_STRENGTH)
        offset = offset + LUMA * np.float32(lum_amt * LUM_STRENGTH)
        for c in range(3):
            k = float(offset[c])
            if k == 0.0:
                continue
            # mask (H,W) * 标量 -> (H,W)，避免 HxWx3 临时数组
            out[..., c] += mask * np.float32(k)
    return out


# ======================================================================
# 6. 取色器（目标色 / HSL）
# ======================================================================


def rgb_to_hsv(rgb: np.ndarray):
    """矢量化 RGB -> HSV。rgb ∈ [0,1]，返回 (h, s, v)，h∈[0,1)。"""
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn
    nz = d > 1e-7
    ds = np.where(nz, d, np.float32(1.0))
    h = np.zeros_like(mx)
    sel = nz & (mx == r)
    h = np.where(sel, ((g - b) / ds) % 6.0, h)
    sel = nz & (mx == g)
    h = np.where(sel, (b - r) / ds + 2.0, h)
    sel = nz & (mx == b)
    h = np.where(sel, (r - g) / ds + 4.0, h)
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-7, d / np.maximum(mx, 1e-7), np.float32(0.0))
    return h.astype(np.float32), s.astype(np.float32), mx


def hue_degrees(color) -> float:
    """单个 sRGB 颜色（0..1）的色相（度）。"""
    h, _, _ = rgb_to_hsv(np.asarray(color, dtype=np.float32).reshape(1, 1, 3))
    return float(h[0, 0]) * 360.0


def preset_color(hue_deg: float, sat: float = 0.85, val: float = 0.95) -> tuple:
    """预设色相对应的展示色（0..1 RGB），用于面板色块。"""
    r, g, b = colorsys.hsv_to_rgb((float(hue_deg) % 360.0) / 360.0, sat, val)
    return (float(r), float(g), float(b))


def sample_patch(rgb8: np.ndarray, x: int, y: int, patch: int = 5):
    """从原图采样一小块的平均颜色，返回 sRGB (r,g,b) 0..1 的 float32 数组。"""
    h, w = rgb8.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    r = max(1, patch // 2)
    region = rgb8[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    if region.size == 0:
        region = rgb8[y:y + 1, x:x + 1]
    return region.reshape(-1, 3).astype(np.float32).mean(axis=0) / np.float32(255.0)


# ---- 绕灰度轴旋转色相（正交基分解，避免 HxWx3 临时数组）----
_LU = LUMA / np.linalg.norm(LUMA)
_U = np.cross(_LU, np.array([0.0, 0.0, 1.0]))
_U = _U / np.linalg.norm(_U)
_V = np.cross(_LU, _U)
_BASIS = (_LU.astype(np.float32), _U.astype(np.float32), _V.astype(np.float32))


def rotate_hue(rgb: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """按逐像素角度 theta（弧度，(H,W)）绕灰度轴旋转色度，保持亮度不变。"""
    l, u, v = _BASIS
    cu = rgb @ u
    cv = rgb @ v
    ct = np.cos(theta)
    st = np.sin(theta)
    du = cu * ct - cv * st
    dv = cu * st + cv * ct
    du -= cu
    dv -= cv
    out = np.array(rgb, dtype=np.float32, copy=True)
    out += du[..., None] * u
    out += dv[..., None] * v
    return out


def picker_mask(hsv, hue_deg: float, span_deg: float) -> np.ndarray:
    """色相环形距离蒙版 × 彩度蒙版。"""
    h, s, _ = hsv
    center = (float(hue_deg) % 360.0) / 360.0
    half = max(float(span_deg), 2.0) / 360.0
    dist = np.abs(h - np.float32(center))
    dist = np.minimum(dist, np.float32(1.0) - dist)      # 环形距离，最大 0.5
    w_hue = np.float32(1.0) - smoothstep(half * np.float32(0.35), half, dist)
    w_sat = smoothstep(np.float32(0.04), np.float32(0.30), s)  # 不影响灰像素
    return (w_hue * w_sat).astype(np.float32)


def apply_pickers(img: np.ndarray, pickers) -> np.ndarray:
    """按顺序应用全部取色点调整。"""
    active = [p for p in pickers
              if p.enabled and (p.hue_shift or p.saturation or p.luminance)]
    if not active:
        return img

    hsv = rgb_to_hsv(to_srgb(img))  # 在感知（gamma）空间估计色相/彩度
    masks = [picker_mask(hsv, p.hue, p.span) for p in active]

    # 重叠区域归一化，避免多个取色点叠加过量
    total = np.zeros_like(masks[0])
    for m in masks:
        total = total + m
    scale = np.where(total > 1.0, total, np.float32(1.0))
    masks = [m / scale for m in masks]

    out = np.array(img, dtype=np.float32, copy=True)
    for p, m in zip(active, masks):
        if p.hue_shift:
            out = rotate_hue(out, np.deg2rad(float(p.hue_shift)) * m)
        if p.saturation:
            lum = luminance(out)
            k = m * np.float32(float(p.saturation) / 100.0)
            k += np.float32(1.0)                # k = 1 + a*m，蒙版外为 1（不变）
            out -= lum[..., None]
            out *= k[..., None]
            out += lum[..., None]
        if p.luminance:
            out += np.float32(float(p.luminance) / 100.0 * 0.2) * m[..., None]
    return out


# ======================================================================
# 7. 效果调节（清晰度 / 降噪 / 锐度 / 去朦胧）
# ======================================================================


def apply_clarity(img: np.ndarray, amount: float, low: np.ndarray | None = None) -> np.ndarray:
    """清晰度：中频局部对比度（低频差分叠加）。

    low 为预先算好的同尺寸低频图（由 LowResRenderer 提供，分块安全）；
    为 None 时在本地计算（单图 / 独立调用）。
    """
    if amount == 0:
        return img
    if low is None:
        h, w = img.shape[:2]
        s = max(1, max(h, w) // 512)             # 降采样倍率
        sh, sw = max(1, h // s), max(1, w // s)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.5, sw // 96.0))
        low = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    a = np.float32(np.clip(amount / 100.0, -1.0, 1.0) * 0.85)
    out = img - low
    out *= a
    out += img
    return out


def apply_sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    """锐度：小半径 Unsharp Masking。"""
    if amount == 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
    a = np.float32(np.clip(amount / 100.0, 0.0, 1.0) * 1.2)
    return img + (img - blur) * a


def _gamma_encode(x: np.ndarray) -> np.ndarray:
    """sRGB 伽马编码，不裁剪上界（>1 的线性值可逆映射，供域转换用）。

    与 to_linear 严格互逆：lin = ((γ+0.055)/1.055)^2.4  <=>  γ = 1.055·lin^(1/2.4) - 0.055。
    """
    x = np.maximum(np.asarray(x, dtype=np.float32), np.float32(0.0))
    out = x ** np.float32(1.0 / 2.4)
    out *= np.float32(1.055)
    out -= np.float32(0.055)
    low = x * np.float32(12.92)
    np.copyto(out, low, where=x <= np.float32(0.0031308))
    return out


def _denoise_sigmas(amount: float) -> tuple:
    """降噪强度 -> 双边滤波 (sigmaColor, sigmaSpace)，均在感知域。"""
    a = float(np.clip(amount / 100.0, 0.0, 1.0))
    return 0.025 + 0.175 * a, 2.0 + 8.0 * a


def apply_denoise(img: np.ndarray, amount: float) -> np.ndarray:
    """降噪：感知域双边缘滤波（bilateral），保边去噪。

    在 sRGB 感知域滤波（噪声的视觉强度在该域近似均匀），再转回线性光；
    sigma 随强度增长，0 为恒等。分块并行时依赖 halo（见 mask_halo）。
    """
    if amount == 0:
        return img
    sigma_color, sigma_space = _denoise_sigmas(amount)
    x = _gamma_encode(img)
    out = cv2.bilateralFilter(x, 0, sigma_color, sigma_space)
    return to_linear(out)


def apply_dehaze(img: np.ndarray, amount: float, A: float | None = None,
                 sigma: float | None = None) -> np.ndarray:
    """去朦胧：暗通道雾模型的稳健近似。

    amount > 0 去雾（增强对比，尤其远处）；amount < 0 反向加雾（降低对比、抬黑位）。
    A 为整幅图统一估计的大气光，sigma 为统一的暗通道平滑半径，
    二者都由 GlobalStats 提供，保证分块结果逐位一致。

    雾模型 I = J·t + A(1-t)  =>  J = (I - A)/t + A，再按 |amount| 强度与原图混合，
    避免小档位也出现剧烈跳变。
    """
    if amount == 0:
        return img
    h, w = img.shape[:2]
    a = float(np.clip(amount / 100.0, -1.0, 1.0))

    if A is None:                                  # 独立调用时本地估计
        lum = luminance(img)
        sample = lum[::4, ::4]
        A = float(np.percentile(sample, 99.9)) if sample.size > 16 else float(lum.max())
    A = max(A, 0.15)
    if sigma is None:
        sigma = max(1.0, min(h, w) / 240.0)

    # 暗通道（逐像素通道最小值 + 平滑，抑制单像素噪声）
    # 注意：cv2.GaussianBlur 会把单通道的尾维度去掉，故保持 (H,W) 再手动升维
    dark = np.min(img, axis=-1)                     # (H,W)
    dark = cv2.GaussianBlur(dark, (0, 0), sigmaX=sigma)

    omega = 0.85
    t = np.float32(1.0) - np.float32(omega * a) * (dark / np.float32(A))
    t = np.clip(t, 0.15, 4.0).astype(np.float32)[..., None]      # (H,W,1)

    # 标准雾模型求解，并裁剪到 [0,1] 防止暗部被过度压黑
    J = (img - np.float32(A)) / t + np.float32(A)
    np.clip(J, 0.0, 1.0, out=J)

    # 强度混合：amount=±100 为完全生效，中间档位平滑过渡
    k = np.float32(abs(a))
    out = img * (np.float32(1.0) - k) + J * k
    return out.astype(np.float32)


def apply_effects(img: np.ndarray, params, low: np.ndarray | None = None,
                  stats: "GlobalStats | None" = None) -> np.ndarray:
    """按顺序应用效果调节；params 只需具备
    clarity/denoise/sharpen/dehaze 字段。

    low 为预先算好的低频图（清晰度用，见 LowResRenderer）；
    stats 提供全局统计（去朦胧的大气光与平滑半径，见 GlobalStats）。
    """
    if not (params.clarity or params.denoise or params.sharpen or params.dehaze):
        return img
    out = apply_clarity(img, params.clarity, low)
    out = apply_denoise(out, params.denoise)
    out = apply_sharpen(out, params.sharpen)
    if stats is not None:
        out = apply_dehaze(out, params.dehaze, stats.A, stats.sigma)
    else:
        out = apply_dehaze(out, params.dehaze)
    return out


class LowResRenderer:
    """清晰度所需的低频图：整幅图只算一次，再按行区间切片（分块安全、无接缝）。"""

    def __init__(self, lin: np.ndarray, params, edge: int = 512):
        self.low = None
        if not params.clarity:
            return
        h, w = lin.shape[:2]
        s = max(1, max(h, w) // edge)
        sh, sw = max(1, h // s), max(1, w // s)
        small = cv2.resize(lin, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.5, sw // 96.0))
        self.low = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def band(self, y0: int, y1: int):
        if self.low is None:
            return None
        return self.low[y0:y1]


class GlobalStats:
    """一次作业内复用的全局统计（去朦胧的大气光与暗通道平滑半径）。

    两者都必须按整幅图的尺寸计算：分块若各自统计会产生横向接缝与整体差异。
    """

    def __init__(self, lin: np.ndarray, params):
        self.A = None
        self.sigma = 1.0
        if not params.dehaze:
            return
        h, w = lin.shape[:2]
        self.sigma = max(1.0, min(h, w) / 240.0)
        sample = (lin[::4, ::4] @ LUMA)
        self.A = float(np.percentile(sample, 99.9)) if sample.size > 16 \
            else float(luminance(lin).max())


# ======================================================================
# 8. 蒙版（线性渐变 / 径向渐变 / 画笔 / 自定义多边形）
# ======================================================================


def _axis_coords(shape) -> tuple:
    """归一化坐标轴：(xs, ys)（一维），配合广播即可得到任意 (H,W) 结果，避免 mgrid。"""
    h, w = shape
    xs = np.arange(w, dtype=np.float32) / np.float32(max(w - 1, 1))
    ys = np.arange(h, dtype=np.float32) / np.float32(max(h - 1, 1))
    return xs, ys


def _linear_mask(m: "MaskState", shape) -> np.ndarray:
    """线性渐变蒙版：从起点 0 平滑过渡到终点 1，过渡带宽度 = 羽化，带两端居中。"""
    h, w = shape
    x1, y1 = m.x1 * w, m.y1 * h
    x2, y2 = m.x2 * w, m.y2 * h
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return np.zeros((h, w), np.float32)
    ux, uy = dx / length, dy / length

    xx = np.arange(w, dtype=np.float32)
    yy = np.arange(h, dtype=np.float32)
    # proj 是"列方向"与"行方向"两个一维投影之和（外层和 -> (H,W)）
    proj = ((xx - np.float32(x1)) * np.float32(ux))[None, :] \
        + ((yy - np.float32(y1)) * np.float32(uy))[:, None]
    fpx = m.feather / 100.0 * length * 0.5          # 过渡带宽度
    half = max(length - 2.0 * fpx, 0.0) / 2.0
    base = (proj - np.float32(half)) * np.float32(1.0 / max(2.0 * fpx, 1e-3))
    np.clip(base, 0.0, 1.0, out=base)
    return smoothstep(np.float32(0.0), np.float32(1.0), base).astype(np.float32)


def _radial_mask(m: "MaskState", shape) -> np.ndarray:
    """径向渐变蒙版：椭圆内为 1，边缘按羽化过渡到 0。

    rx = radius*aspect（长轴）、ry = radius/aspect（短轴），angle 为长轴方向（度）。
    """
    h, w = shape
    xs, ys = _axis_coords(shape)
    short = max(min(w, h), 1)
    R = max(m.radius, 0.01) * short
    asp = max(float(m.aspect), 0.05)
    px = (xs - np.float32(m.cx)) * np.float32(w)      # (W,)
    py = (ys - np.float32(m.cy)) * np.float32(h)      # (H,)
    ang = np.deg2rad(float(m.angle))
    ca, sa = np.float32(np.cos(ang)), np.float32(np.sin(ang))
    # 旋转 -ang，把椭圆轴对齐到坐标轴
    xr = px * ca + py[:, None] * sa                    # (H,W) 广播
    yr = -px * sa + py[:, None] * ca
    rx = R * asp
    ry = R / asp
    d = np.sqrt((xr / np.float32(rx)) ** 2 + (yr / np.float32(ry)) ** 2)
    d = d.astype(np.float32)                            # 1 = 椭圆边界
    f = float(np.clip(m.feather / 100.0, 0.0, 1.0))
    if f <= 0.0:
        return (d <= np.float32(1.0)).astype(np.float32)     # 硬边，避免 smoothstep 除零
    return np.float32(1.0) - smoothstep(np.float32(1.0 - f), np.float32(1.0), d)


def _polygon_mask(m: "MaskState", shape) -> np.ndarray:
    """自定义多边形蒙版：按顶点顺序填充多边形，边缘向内羽化。

    顶点少于 3 个时返回全 0（与画笔空笔迹一致）。羽化用距离变换实现：
    距边界 feather_px 以内平滑过渡到 1，内部保持 1，外部保持 0；
    因此反向、局部调整、分块渲染都与其它蒙版完全一致。
    """
    h, w = shape
    pts = np.asarray(m.points, dtype=np.float32) if m.points else None
    if pts is None or len(pts) < 3:
        return np.zeros((h, w), np.float32)
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * np.float32(max(w - 1, 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * np.float32(max(h - 1, 1))
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [poly], 255, lineType=cv2.LINE_8)

    f = float(np.clip(m.feather / 100.0, 0.0, 1.0))
    if f <= 0.0:
        return canvas.astype(np.float32) * np.float32(1.0 / 255.0)

    fpx = max(f * min(w, h) * 0.25, 0.5)
    dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
    return smoothstep(np.float32(0.0), np.float32(fpx), dist).astype(np.float32)


def _brush_mask(m: "MaskState", shape) -> np.ndarray:
    """画笔蒙版：硬边笔画 + 可分离高斯羽化（先 1/8 降采样模糊再上采样，限制邻域半径）。"""
    h, w = shape
    short = max(min(w, h), 1)
    radius = max(1.0, m.size / 100.0 * short * 0.5)
    canvas = np.zeros((h, w), np.float32)
    thick = max(2, int(round(radius * 2.0)))
    for stroke in m.strokes:
        if not stroke:
            continue
        pts = np.asarray(stroke, np.float32) * np.array([w, h], np.float32)
        r_i = int(round(radius))
        if len(pts) == 1:
            cv2.circle(canvas, (int(pts[0, 0]), int(pts[0, 1])), r_i, 1.0, -1)
            continue
        ip = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [ip], False, 1.0, thickness=thick, lineType=cv2.LINE_8)

    fpx = min(m.feather / 100.0 * short * 0.15, 160.0)
    if fpx > 0.6 and canvas.max() > 0:
        s = 8
        sh, sw = max(1, h // s), max(1, w // s)
        small = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(fpx / s, 0.5))
        canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(canvas, 0.0, 1.0)


def render_mask(m: "MaskState", shape, max_edge: int | None = None) -> np.ndarray:
    """渲染蒙版为 (H,W) float32 0..1（已处理反向）。

    max_edge 指定时先在更低分辨率渲染再上采样（蒙版本身平滑，视觉等价但更快），
    常用于 UI 叠加层显示。
    """
    if max_edge:
        h, w = shape
        long_edge = max(h, w)
        if long_edge > max_edge:
            s = max_edge / float(long_edge)
            coarse = render_mask(m, (max(1, int(round(h * s))),
                                     max(1, int(round(w * s)))))
            return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    h, w = shape
    if m.kind == "polygon" and len(m.points) < 3:
        # 未完成的多边形（顶点不足 3 个）始终为空，包含反向时也不覆盖全图
        return np.zeros((h, w), np.float32)
    if m.kind == "linear":
        mask = _linear_mask(m, shape)
    elif m.kind == "radial":
        mask = _radial_mask(m, shape)
    elif m.kind == "polygon":
        mask = _polygon_mask(m, shape)
    else:
        mask = _brush_mask(m, shape)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if m.invert:
        mask = np.float32(1.0) - mask
    return mask.astype(np.float32)


class MaskRenderer:
    """一次作业内复用的蒙版渲染器。

    思路与 LowResRenderer 一致：先在低分辨率（默认最长边 768）渲染，再一次性
    上采样到全尺寸，之后按行区间切片返回。这样每个分块拿到的蒙版与整图处理
    完全一致（逐位相同），既没有接缝，也不需要 halo。
    """

    COARSE_EDGE = 768

    def __init__(self, masks, shape):
        self.masks = masks
        self.shape = shape
        h, w = shape
        self._full: list = []
        self._coarse: list = []
        for m in masks:
            if m.is_trivial():
                self._coarse.append(None)
                self._full.append(None)
                continue
            s = min(1.0, self.COARSE_EDGE / float(max(h, w)))
            if s < 1.0:
                self._coarse.append(render_mask(m, (max(1, int(h * s)),
                                                    max(1, int(w * s)))))
            else:
                self._coarse.append(render_mask(m, (h, w)))
            self._full.append(None)

    def get(self, index: int, y0: int, y1: int, width: int) -> np.ndarray:
        """原图行区间 [y0, y1) 对应的蒙版。"""
        if self._full[index] is None:
            coarse = self._coarse[index]
            if coarse is None:
                coarse = render_mask(self.masks[index], (y1 - y0, width))
                self._full[index] = coarse
            else:
                h, w = self.shape
                self._full[index] = cv2.resize(coarse, (w, h),
                                               interpolation=cv2.INTER_LINEAR)
                self._coarse[index] = None      # 释放低分辨率副本
        mask = self._full[index]
        return mask[y0:y1] if mask.shape[0] > (y1 - y0) else \
            cv2.resize(mask, (width, y1 - y0), interpolation=cv2.INTER_LINEAR)


def mask_cache_key(m: "MaskState", shape) -> tuple:
    """缓存键：只与几何/羽化/反向有关，与调整数值无关。"""
    h, w = shape
    base = (m.kind, bool(m.invert), round(float(m.feather), 4), h, w)
    if m.kind == "linear":
        return base + (round(m.x1, 5), round(m.y1, 5), round(m.x2, 5), round(m.y2, 5))
    if m.kind == "radial":
        return base + (round(m.cx, 5), round(m.cy, 5), round(m.radius, 5),
                       round(m.aspect, 5), round(m.angle, 4))
    if m.kind == "polygon":
        points = tuple((round(x, 5), round(y, 5)) for x, y in m.points)
        return base + (len(points), points)
    strokes = tuple(tuple((round(x, 5), round(y, 5)) for x, y in s) for s in m.strokes)
    return base + (round(m.size, 4), strokes)


def apply_masked_adjustments(img: np.ndarray, masks, renderer: "MaskRenderer | None" = None,
                              y0: int = 0, y1: int | None = None,
                              low: np.ndarray | None = None,
                              stats: "GlobalStats | None" = None) -> np.ndarray:
    """按顺序应用各蒙版的局部调整（蒙版内做 mini 流水线后按蒙版混合）。

    mini 流水线与全局一致：基础 -> 效果 -> 曲线 -> 颜色分级 -> 取色器；
    renderer 提供时按行区间复用低分辨率蒙版（分块安全）；
    low / stats 分别提供低频图与全局统计，供蒙版内的清晰度/去朦胧使用。
    曲线/色轮/取色器均为逐像素算子，分块无需额外 halo。
    """
    out = img
    if y1 is None:
        y1 = y0 + img.shape[0]
    for i, m in enumerate(masks):
        if m.is_trivial():
            continue
        if renderer is not None:
            mask = renderer.get(i, y0, y1, img.shape[1])
        else:
            mask = render_mask(m, img.shape[:2])
        local = apply_basic(img, m.adj)
        local = apply_effects(local, m.adj, low, stats)
        local = apply_curves_fast(local, m.adj.curves)
        local = apply_color_grading(local, m.adj.wheels, m.adj.blend)
        local = apply_pickers(local, m.adj.pickers)
        if out is img:
            out = np.array(img, dtype=np.float32, copy=True)
        w3 = mask[..., None]
        out = out * (np.float32(1.0) - w3) + local * w3
    return out


def mask_halo(params, h: int, w: int) -> int:
    """分块时所需的重叠像素：效果算子的模糊需要邻域。

    清晰度/锐度/去朦胧取 64；降噪按双边滤波半径（≈2σ）另计，二者取大。
    蒙版由 MaskRenderer 在低分辨率全局渲染，不产生接缝，因此不需要 halo。
    """
    halo = 64 if (params.clarity or params.sharpen or params.dehaze) else 0
    if getattr(params, "denoise", 0):
        halo = max(halo, int(_denoise_sigmas(params.denoise)[1] * 2.0) + 6)
    return halo


# ======================================================================
# 8b. 仿制图章（像素级内容修改，在流水线之前作用于源图）
# ======================================================================


def _disc_alpha(radius: int, feather: float = 0.45) -> np.ndarray:
    """圆形羽化 alpha 盘 (2r+1, 2r+1)：1 = 完全复制，边缘 smoothstep 到 0。"""
    r = max(int(radius), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy) / np.float32(r)
    a = np.float32(1.0) - smoothstep(np.float32(1.0 - feather), np.float32(1.0), d)
    return a.astype(np.float32)


def clone_stamp(img: np.ndarray, src_px, dst_px, radius: float,
                feather: float = 0.45) -> None:
    """仿制图章：把以 src_px 为中心的圆形区域羽化复制到 dst_px（原地修改）。

    坐标为像素 (x, y)；radius 为像素半径。以目标窗口为基准求三重交集
    （图像范围 × 源窗口范围 × 目标窗口范围），源/目标重叠时先取快照再混合，
    越界部分自动裁剪。
    """
    h, w = img.shape[:2]
    r = max(1, int(round(radius)))
    alpha = _disc_alpha(r, feather)
    sx, sy = int(round(src_px[0])), int(round(src_px[1]))
    dx, dy = int(round(dst_px[0])), int(round(dst_px[1]))
    oy, ox = sy - dy, sx - dx                     # 目标像素 -> 源像素偏移
    dy0, dx0 = dy - r, dx - r
    y0 = max(dy0, 0, -oy)
    y1 = min(dy0 + 2 * r + 1, h, h - oy)
    x0 = max(dx0, 0, -ox)
    x1 = min(dx0 + 2 * r + 1, w, w - ox)
    if y0 >= y1 or x0 >= x1:
        return
    a = alpha[y0 - dy0:y1 - dy0, x0 - dx0:x1 - dx0][..., None]
    patch = img[y0 + oy:y1 + oy, x0 + ox:x1 + ox].astype(np.float32)  # 先快照
    dst = img[y0:y1, x0:x1]
    blended = dst.astype(np.float32)
    blended *= (np.float32(1.0) - a)
    blended += patch * a
    dst[...] = np.clip(blended + np.float32(0.5), 0, 255).astype(np.uint8)


# ======================================================================
# 9. 直方图（曲线编辑器背景）
# ======================================================================

HIST_BINS = 64


def linear_histograms(rgb8: np.ndarray) -> dict:
    """返回 {'master': 亮度直方图, 'red'/'green'/'blue': 通道直方图}（归一化到 0..1）。"""
    lin = np.clip(rgb8_to_linear(rgb8), 0.0, 1.0)
    master = np.histogram(luminance(lin), bins=HIST_BINS, range=(0.0, 1.0))[0].astype(np.float32)
    ch = {}
    for name, idx in (("red", 0), ("green", 1), ("blue", 2)):
        ch[name] = np.histogram(lin[..., idx], bins=HIST_BINS, range=(0.0, 1.0))[0].astype(np.float32)
    peak = max(float(master.max()), 1.0)
    master /= peak
    for k in ch:
        peak = max(float(ch[k].max()), 1.0)
        ch[k] /= peak
    return {"master": master, **ch}


# ======================================================================
# 10. 处理流水线（可分块并行；纯逐像素阶段可选 OpenCL GPU 路径，见 gpu.py）
# ======================================================================

# 拖动时处理的降采样图最长边
PREVIEW_MAX_EDGE = 1200

# 超过该像素数才启用多线程（小图开销不划算；预览图 1200px ≈ 1~2MP 也能并行）
PARALLEL_MIN_PIXELS = 600_000
_MAX_WORKERS = max(1, min(32, os.cpu_count() or 1))

_EXECUTOR: ThreadPoolExecutor | None = None

# 环境变量 FERNLENS_NO_GPU=1 强制关闭 GPU 路径
_GPU_ENABLED = os.environ.get("FERNLENS_NO_GPU", "") != "1"


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS,
                                       thread_name_prefix="fernlens")
    return _EXECUTOR


def downscale_max_edge(rgb8: np.ndarray, max_edge: int = PREVIEW_MAX_EDGE) -> np.ndarray:
    """把 uint8 RGB 图降采样到最长边 max_edge（保持宽高比，INTER_AREA）。"""
    if max_edge <= 0:
        return rgb8
    h, w = rgb8.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return rgb8
    scale = max_edge / float(long_edge)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(rgb8, size, interpolation=cv2.INTER_AREA)


def rgb8_to_linear_mt(rgb8: np.ndarray) -> np.ndarray:
    """rgb8_to_linear 的多线程版本（逐像素算子，分块结果与单线程一致）。

    打开大图时 sRGB->线性光转换本身就要上百毫秒，分块到线程池可明显缩短
    预览首帧延迟；小图直接走单线程。
    """
    h, w = rgb8.shape[:2]
    if h * w < PARALLEL_MIN_PIXELS or _MAX_WORKERS <= 1:
        return rgb8_to_linear(rgb8)
    out = np.empty((h, w, 3), np.float32)
    bounds = np.linspace(0, h, _MAX_WORKERS + 1).astype(int)

    def work(i: int):
        y0, y1 = int(bounds[i]), int(bounds[i + 1])
        if y1 > y0:
            out[y0:y1] = rgb8_to_linear(rgb8[y0:y1])

    list(_executor().map(work, range(_MAX_WORKERS)))
    return out


def _gpu_eligible(params) -> bool:
    """纯逐像素阶段（无效果/蒙版/取色器）才交给 GPU，避免来回搬运抵消收益。"""
    return not (params.masks or params.pickers
                or params.clarity or params.denoise
                or params.sharpen or params.dehaze)


def _try_gpu(params, lin: np.ndarray) -> np.ndarray | None:
    """尝试 GPU 路径（gpu.py 可选依赖）；任何失败都静默回退 CPU。"""
    try:
        from gpu import process_simple
    except Exception:
        return None
    return process_simple(params, lin)


def _process_rows(params, lin: np.ndarray, y0: int, y1: int, pad: int,
                  renderer: "MaskRenderer | None" = None,
                  low_res: "LowResRenderer | None" = None,
                  stats: "GlobalStats | None" = None) -> np.ndarray:
    """处理行区间 [y0, y1)（含 pad 像素的上下文），返回该区间的结果。

    pad 为锐度/去朦胧模糊所需的邻域重叠；清晰度（低频图）、蒙版（粗渲染上采样）
    与去朦胧（大气光统计）都由整图级对象提供，保证分块之间逐位一致、无接缝。
    """
    h = lin.shape[0]
    a = max(0, y0 - pad)
    b = min(h, y1 + pad)
    low = low_res.band(a, b) if low_res is not None else None
    out = apply_basic(lin[a:b], params)                               # 1. 白平衡 + 基础
    out = apply_effects(out, params, low, stats)                      # 2. 效果
    out = apply_curves_fast(out, params.curves)                       # 3. 曲线
    out = apply_color_grading(out, params.wheels, params.blend)       # 4. 颜色分级
    out = apply_masked_adjustments(out, params.masks, renderer=renderer,
                                   y0=a, y1=b, low=low, stats=stats)   # 5. 蒙版局部调整
    out = apply_pickers(out, params.pickers)                          # 6. 取色器
    if params.blackwhite:                                            # 7. 黑白
        out = apply_blackwhite(out, params.blackwhite)
    np.clip(out, 0.0, 1.0, out=out)
    result = linear_to_rgb8(out)
    return result[y0 - a:result.shape[0] - (b - y1)]


def _process_band(params, lin: np.ndarray) -> np.ndarray:
    """整幅图的完整流水线（单线程路径，pad=0）。"""
    renderer = MaskRenderer(params.masks, lin.shape[:2]) if params.masks else None
    low_res = LowResRenderer(lin, params) if params.clarity else None
    stats = GlobalStats(lin, params) if params.dehaze else None
    return _process_rows(params, lin, 0, lin.shape[0], 0, renderer, low_res, stats)


def process_linear(params, lin: np.ndarray) -> np.ndarray:
    """线性光图像 -> 处理后的 uint8 sRGB RGB 图。"""
    if params.is_default():
        return linear_to_rgb8(lin)

    if _GPU_ENABLED and _gpu_eligible(params):
        out = _try_gpu(params, lin)
        if out is not None:
            return out

    h, w = lin.shape[:2]
    pad = mask_halo(params, h, w)
    workers = 1
    if h * w >= PARALLEL_MIN_PIXELS:
        workers = _MAX_WORKERS
    parallel = workers > 1 and h > 2 * pad

    # 自己按行分块并行时，限制 OpenCV 内部线程，避免 8x16 线程互相争抢
    cv2.setNumThreads(1 if parallel else 0)

    if not parallel:
        return _process_band(params, lin)

    out = np.empty(lin.shape, dtype=np.uint8)
    bounds = np.linspace(0, h, workers + 1).astype(int)
    renderer = MaskRenderer(params.masks, (h, w)) if params.masks else None
    low_res = LowResRenderer(lin, params) if params.clarity else None
    stats = GlobalStats(lin, params) if params.dehaze else None

    def work(i: int):
        y0, y1 = int(bounds[i]), int(bounds[i + 1])
        if y1 <= y0:
            return
        out[y0:y1] = _process_rows(params, lin, y0, y1, pad, renderer, low_res, stats)

    list(_executor().map(work, range(workers)))
    return out


def process(params, rgb8: np.ndarray) -> np.ndarray:
    """uint8 sRGB RGB -> 处理后的 uint8 sRGB RGB。"""
    return process_linear(params, rgb8_to_linear(rgb8))
