"""FernLens 核心：纯 NumPy / OpenCV 图像处理。

本模块不依赖 Qt，输入输出都是 np.ndarray（float32 线性光，HxWx3）。
内容分区（卷标注释可快速定位）：
  1. 色彩空间工具      2. 参数模型        3. 基础调整
  4. 曲线              5. 颜色分级        6. 取色器
  7. 效果调节（清晰度/降噪/锐度/去朦胧）  8. 蒙版（含曲线/色轮/取色器局部化）
  8b. 仿制图章         9. 直方图          10. 处理流水线

流水线顺序（float32 线性光空间）：
    白平衡 -> 基础 -> 效果 -> 曲线 -> 颜色分级 -> 蒙版局部调整 -> 取色

纯逐像素的阶段（白平衡/基础/黑白/曲线/颜色分级）可走 OpenCL GPU 路径
（gpu.py，可选），含效果/蒙版/取色器时自动回退 CPU 分块并行。
"""
from __future__ import annotations

import colorsys
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

# ======================================================================
# 1. 色彩空间工具
# ======================================================================

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# sRGB 编解码查表（16 位量化）：gamma 域转换在降噪/编码里各跑两次，
# 逐点 np.power 在 0.96MP 上要 20ms，查表只要 3ms。
_LUT_F_LOCK = threading.Lock()
_ENC_LUT_F: np.ndarray | None = None
_DEC_LUT_F: np.ndarray | None = None


def _build_lut_f() -> None:
    global _ENC_LUT_F, _DEC_LUT_F
    v = np.arange(65536, dtype=np.float64) / 65535.0
    enc = np.where(v <= 0.0031308, v * 12.92,
                   np.float32(1.055) * np.power(v, 1.0 / 2.4) - 0.055)
    dec = np.where(v <= 0.04045, v / 12.92,
                   np.power((v + 0.055) / 1.055, 2.4))
    _ENC_LUT_F = np.ascontiguousarray(enc.astype(np.float32).reshape(-1, 1))
    _DEC_LUT_F = np.ascontiguousarray(dec.astype(np.float32).reshape(-1, 1))


def _encode_lut_f() -> np.ndarray:
    if _ENC_LUT_F is None:
        with _LUT_F_LOCK:
            if _ENC_LUT_F is None:
                _build_lut_f()
    return _ENC_LUT_F


def _decode_lut() -> np.ndarray:
    if _DEC_LUT_F is None:
        with _LUT_F_LOCK:
            if _DEC_LUT_F is None:
                _build_lut_f()
    return _DEC_LUT_F


def to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> 线性光。"""
    x = np.asarray(x, dtype=np.float32)
    fast = _decode_lut()
    # 只在 [0,1] 内查表：YCrCb 的 Cr/Cb 会超出这个区间，裁剪会带来看得见的偏色
    if fast is not None and x.size >= 20000 \
            and float(x.min()) >= 0.0 and float(x.max()) <= 1.0:
        # 16 位量化查表：0.96MP 约 3ms，np.power 要 19ms（误差 < 4e-5）
        idx = np.multiply(x, np.float32(65535.0))
        np.rint(idx, out=idx)
        return cv2.LUT(idx.astype(np.uint16), fast)
    out = (x + np.float32(0.055)) * np.float32(1.0 / 1.055)
    low = x * np.float32(1.0 / 12.92)
    small = x <= np.float32(0.04045)
    # x 可能略小于 0（YCrCb 转回 RGB 时高饱和像素会溢出）：这些位置 out < 0，
    # power 会算出 NaN 并报警，但随后被下面的 copyto（small 分支）整段覆盖，
    # 结果不受影响，只是警告噪音——这里显式静音，不做多余的整幅裁剪
    with np.errstate(invalid="ignore"):
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


_SRGB_LUT: np.ndarray | None = None
_SRGB_LUT_LOCK = threading.Lock()


def _srgb_encode_lut() -> np.ndarray:
    """sRGB 编码查表：线性值按 16 位量化 -> uint8。

    np.power 是整条流水线最贵的逐像素算子之一（0.96MP 约 40ms）；
    量化到 16 位后查表误差 < 0.05/255，落成 uint8 后与逐点计算一致
    （极端贴边的像素最多差 1 级，视觉不可辨）。
    """
    global _SRGB_LUT
    if _SRGB_LUT is None:
        with _SRGB_LUT_LOCK:
            if _SRGB_LUT is None:
                v = np.arange(65536, dtype=np.float32) / np.float32(65535.0)
                hi = np.float32(1.055) * np.power(v, np.float32(1.0 / 2.4)) - np.float32(0.055)
                lo = v * np.float32(12.92)
                s = np.where(v <= np.float32(0.0031308), lo, hi)
                lut = np.clip(np.rint(s * np.float32(255.0)), 0, 255).astype(np.uint8)
                _SRGB_LUT = np.ascontiguousarray(lut.reshape(-1, 1))
    return _SRGB_LUT


def linear_to_rgb8(lin: np.ndarray) -> np.ndarray:
    """float32 线性光 -> uint8 sRGB（16 位量化 + cv2.LUT，比逐点 pow 快约 4 倍）。"""
    lut = _srgb_encode_lut()
    idx = np.asarray(lin, dtype=np.float32)
    if idx is lin:
        idx = np.array(lin, dtype=np.float32, copy=True)
    idx *= np.float32(65535.0)
    np.clip(idx, 0.0, 65535.0, out=idx)
    # 量化取最近邻（不是截断）：线性空间 1/65535 的偏移在暗部会被 sRGB 陡坡放大，
    # 取整可把误差压到半级以内，保证与 GPU 内核（逐点 pow）仍在 ±2 级内一致
    np.rint(idx, out=idx)
    return cv2.LUT(idx.astype(np.uint16), lut)


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
    """一个局部蒙版：linear(线性渐变) / radial(径向渐变) / brush(画笔) / polygon(自定义多边形) / ai(AI 识别)。

    几何坐标全部归一化到 0..1（相对图像宽高），便于缩放与传输；
    feather / size 为 0..100 的 UI 档位，渲染时换算成像素。
    """

    kind: str = "linear"                 # 'linear' | 'radial' | 'brush' | 'polygon' | 'ai'
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

    # 自定义多边形：顶点列表 [(x, y), ...] 归一化坐标 0..1
    points: list = field(default_factory=list)

    # AI 蒙版：识别目标（subject/person/background/sky）+ 掩码数据（两种形态）。
    # ① grid：'0'/'1' 字符行 + 尺寸（视觉大模型网格分割，粗边界）
    # ② mask_png：单通道 8bit PNG 的 base64（本地分割模型，像素级软边界）
    # mask_png 非空时优先渲染；两者都以归一化坐标覆盖整幅图像。
    ai_target: str = ""
    grid: list = field(default_factory=list)
    grid_w: int = 0
    grid_h: int = 0
    mask_png: str = ""
    mask_w: int = 0
    mask_h: int = 0

    adj: MaskAdjustments = field(default_factory=MaskAdjustments)

    def is_trivial(self) -> bool:
        if (not self.enabled) or self.adj.is_default():
            return True
        if self.kind == "polygon" and len(self.points) < 3:
            return True
        return False


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


# 对比度查表：域 [0, 4]（覆盖 +2EV 内的过曝值），按 a 值缓存
_CONTRAST_DOMAIN = 4.0
_CONTRAST_LUTS: dict = {}


def _contrast_lut(a: float) -> np.ndarray | None:
    """按对比度系数生成 65536 项查表（返回 None 时走逐点解析式）。

    与 apply_contrast 的解析式逐点等价：量化误差 < 1e-4，落在 8 位输出上
    基本不可见；域 [0,4] 之外按端点取值（这些值最终都会被裁剪掉）。
    """
    key = round(float(a), 5)
    hit = _CONTRAST_LUTS.get(key)
    if hit is not None:
        return hit
    v = np.linspace(0.0, _CONTRAST_DOMAIN, 65536, dtype=np.float64)
    s = v * v * (3.0 - 2.0 * v)                      # smoothstep(v)
    if a > 0:
        s2 = s * s * (3.0 - 2.0 * s)                 # smoothstep(smoothstep(v))
        out = v * (1.0 - a) + s2 * a
    else:
        out = v + (2.0 * a) * v * (1.0 - v) * (v - 0.5)
    lut = np.ascontiguousarray(out.astype(np.float32).reshape(-1, 1))
    if len(_CONTRAST_LUTS) > 24:
        _CONTRAST_LUTS.clear()
    _CONTRAST_LUTS[key] = lut
    return lut


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
    lut = _contrast_lut(float(a))
    if lut is not None and out.ndim == 3 and out.size >= 4096:
        # 对比度是纯逐通道一元函数：量化到 [0,4] 域查表（0.96MP 约 6ms，
        # 逐点 smoothstep 复合要 36ms）。域外按端点取值，最终都会被裁剪掉
        idx = np.multiply(out, np.float32(65535.0 / 4.0))
        np.clip(idx, 0.0, 65535.0, out=idx)
        np.rint(idx, out=idx)
        return cv2.LUT(idx.astype(np.uint16), lut)
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
    # 按通道视图做 maximum/minimum：比 np.max(axis=-1) 的跨步归约快 5 倍
    # （0.37MP 下 5ms vs 29ms），结果逐位相同
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
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


def apply_saturation_vibrance(img: np.ndarray, saturation: float,
                              vibrance: float) -> np.ndarray:
    """饱和 + 自然饱和度：一次整幅遍历完成两段（两者都非零时走这里）。

    两段都是「绕亮度线性缩放色度」，而亮度是通道的线性组合、缩放后不变，
    所以第二段的色度权重可以用缩放前的极值直接推出：
        v2 = lum + (v - lum) * ks
        mx2 = lum + (mx - lum) * ks ,  mn2 = lum + (mn - lum) * ks
    逐元素与「先 apply_saturation 再 apply_vibrance」完全同序同舍入
    （同一串 float32 运算），省掉 3 次整幅读写。
    """
    if not vibrance:
        return apply_saturation(img, saturation)
    if not saturation:
        return apply_vibrance(img, vibrance)

    v = np.array(img, dtype=np.float32, copy=True)
    ks = np.float32(1.0 + float(saturation) / 100.0)
    lum = luminance(v)
    r, g, b = v[..., 0], v[..., 1], v[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)

    mx -= lum
    mx *= ks
    mx += lum
    mn -= lum
    mn *= ks
    mn += lum
    sat = mx - mn
    sat /= (mx + mn + np.float32(1e-5))

    a = np.float32(float(vibrance) / 100.0)
    w = (np.float32(1.0) - sat) if a > 0 else sat
    w *= a
    w += np.float32(1.0)
    w *= ks

    v -= lum[..., None]
    v *= w[..., None]
    v += lum[..., None]
    return v


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
    # 高光/阴影、白色/黑色共用一次亮度计算：亮度矩阵乘是这一段的固定开销，
    # 而四档权重本来就只差一个平滑窗，视觉上等价的省略让整段少算 2 次亮度
    if params.highlights or params.shadows:
        lum = luminance(out)
        if params.highlights:
            _mul_(out, smoothstep(0.35, 0.95, lum), (params.highlights / 100.0) * 1.8)
        if params.shadows:
            w = np.float32(1.0) - smoothstep(0.0, 0.45, lum)
            _mul_(out, w, (params.shadows / 100.0) * 0.6)
            out += np.float32((params.shadows / 100.0) * 0.10) * w[..., None]
    if params.whites or params.blacks:
        lum = luminance(out)
        if params.whites:
            w = smoothstep(0.55, 1.0, lum)
            _mul_(out, w, (params.whites / 100.0) * 0.9)
            out += np.float32((params.whites / 100.0) * 0.06) * w[..., None]
        if params.blacks:
            w = np.float32(1.0) - smoothstep(0.0, 0.40, lum)
            out += np.float32((params.blacks / 100.0) * 0.09) * w[..., None]
    if params.saturation or params.vibrance:
        out = apply_saturation_vibrance(out, params.saturation, params.vibrance)
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
    """应用 RGB 曲线（线性光空间）。

    曲线本身是 2048 点的表；cv2.LUT 要求 16 位输入配 65536 项表，所以把
    下标放大 32 倍（2047*32=65504，仍在 uint16 内），表项按 k>>5 复制。
    这样下标语义与原来的「floor(v*2047) 查 2048 点表」逐位一致，
    与 GPU 内核的 convert_int_sat 截断也一致，同时保留了 LUT 的速度
    （0.96MP 约 8ms，np.take 要 50ms 以上）。
    """
    tabs = _curve_tables(curves)
    idx = np.array(img, dtype=np.float32, copy=True)
    idx *= np.float32(LUT_SIZE - 1)
    np.clip(idx, 0.0, LUT_SIZE - 1, out=idx)
    idx16 = idx.astype(np.uint16)      # 正数截断 = floor，与 GPU 一致
    idx16 *= np.uint16(32)
    # 三通道一次查表：省掉 split/merge 两次整幅拷贝（0.96MP 省约 8ms）
    return cv2.LUT(idx16, _curve_lut3(tabs))


# 曲线表缓存：分块渲染时每个分块都要同一组表，缓存后只做一次插值
_CURVE_TABS: OrderedDict = OrderedDict()
_CURVE_TABS_LOCK = threading.Lock()
_CURVE_TABS_MAX = 8


def _curve_key(curves: dict) -> tuple:
    return tuple(
        tuple((round(float(p[0]), 6), round(float(p[1]), 6))
              for p in sanitize_points(curves.get(ch) or default_curves()[ch]))
        for ch in CURVE_CHANNELS)


def _curve_tables(curves: dict) -> list:
    """把每条曲线细化为 65536 项 float32 查表（按控制点缓存）。

    表内容由 2048 点原始查表按下标换算复制而来（table[k] = lut[k*2047//65535]），
    因此与旧的「2048 点表 + np.take」逐位一致，只是换成 cv2.LUT 提速。
    """
    key = _curve_key(curves)
    with _CURVE_TABS_LOCK:
        hit = _CURVE_TABS.get(key)
        if hit is not None:
            _CURVE_TABS.move_to_end(key)
            return hit
    luts = channel_luts(curves)
    tabs = [np.ascontiguousarray(
        luts[ch].astype(np.float32)[np.arange(65536) >> 5].reshape(-1, 1))
        for ch in ("red", "green", "blue")]
    with _CURVE_TABS_LOCK:
        _CURVE_TABS[key] = tabs
        _CURVE_TABS.move_to_end(key)
        while len(_CURVE_TABS) > _CURVE_TABS_MAX:
            _CURVE_TABS.popitem(last=False)
    return tabs


_CURVE_LUT3: dict = {}


def _curve_lut3(tabs: list) -> np.ndarray:
    """把三张 65536 项单通道表并成一张三通道表，供 cv2.LUT 一次处理 RGB。

    cv2.LUT 要求「表通道数 == 图像通道数或为 1」，所以并表后可以省掉
    split/merge 两次整幅数据拷贝（0.96MP 约省 8ms）。
    """
    key = id(tabs)
    hit = _CURVE_LUT3.get(key)
    if hit is not None and hit[0] is tabs:
        return hit[1]
    lut3 = np.ascontiguousarray(
        np.concatenate(tabs, axis=1).reshape(1, 65536, 3), dtype=np.float32)
    if len(_CURVE_LUT3) > 16:
        _CURVE_LUT3.clear()
    _CURVE_LUT3[key] = (tabs, lut3)
    return lut3


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
    return {
        "shadows": _range_mask("shadows", lum),
        "midtones": _range_mask("midtones", lum),
        "highlights": _range_mask("highlights", lum),
        "global": np.ones_like(lum),
    }


def _range_mask(name: str, lum: np.ndarray) -> np.ndarray:
    """单个影调范围的权重（range_masks 的单档版本，避免算用不到的档）。"""
    if name == "shadows":
        return np.float32(1.0) - smoothstep(0.00, 0.50, lum)
    if name == "highlights":
        return smoothstep(0.50, 1.00, lum)
    if name == "global":
        return np.ones_like(lum)
    midtones = np.float32(1.0) - np.abs(np.float32(2.0) * lum - np.float32(1.0)) * np.float32(1.6)
    np.clip(midtones, 0.0, 1.0, out=midtones)
    return midtones


def blend_exponent(blend: float) -> float:
    """blend 0..100 -> 蒙版指数。数值越大过渡越柔和（1..4）。"""
    return 1.0 + 3.0 * (1.0 - float(np.clip(blend, 0.0, 100.0)) / 100.0)


def apply_color_grading(img: np.ndarray, wheels: dict, blend: float) -> np.ndarray:
    """应用四个色轮。wheels: {range: WheelState}。"""
    active = {name: w for name, w in wheels.items() if w.saturation or w.luminance}
    if not active:
        return img
    lum = luminance(img)
    exp = blend_exponent(blend)
    out = np.array(img, dtype=np.float32, copy=True)
    masks: dict = {}

    for name, w in active.items():
        strength = float(w.saturation) / 100.0
        lum_amt = float(w.luminance) / 100.0
        offset = wheel_direction(w.hue, strength) * np.float32(SAT_STRENGTH)
        offset = offset + LUMA * np.float32(lum_amt * LUM_STRENGTH)
        if name == "global":
            # 全局档权重恒为 1：直接加偏移，省掉一次 HxW 乘法
            for c in range(3):
                k = float(offset[c])
                if k != 0.0:
                    out[..., c] += np.float32(k)
            continue
        mask = masks.get(name)
        if mask is None:
            mask = _range_mask(name, lum)
            if exp != 1.0:
                mask = np.power(mask, np.float32(exp))
            masks[name] = mask
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
    fast = _encode_lut_f()
    if fast is not None and x.size >= 20000 and float(x.max()) <= 1.0:
        # 与 to_linear 同一套 16 位量化查表（0.96MP 约 3ms，逐点 pow 要 21ms）
        idx = np.multiply(x, np.float32(65535.0))
        np.rint(idx, out=idx)
        return cv2.LUT(idx.astype(np.uint16), fast)
    out = x ** np.float32(1.0 / 2.4)
    out *= np.float32(1.055)
    out -= np.float32(0.055)
    low = x * np.float32(12.92)
    np.copyto(out, low, where=x <= np.float32(0.0031308))
    return out


def _denoise_sigmas(amount: float) -> tuple:
    """降噪强度 -> (sigma_space, sigma_y, sigma_c)，均在感知域。

    亮度与色度分开：色度可重滤（色彩噪点不构成纹理感），
    亮度必须轻滤（纹理/细节都在亮度通道，重滤即“肉”）。

    亮度改成「两遍轻滤」而不是「一遍重滤」：同样能压噪声（噪声在平坦区
    逐遍指数衰减），但中低对比的纹理每遍都低于 sigma_color 而被保住，
    实测比单遍大 sigma 的纹理相关度高 0.06~0.10（见 apply_denoise）。
    """
    a = float(np.clip(amount / 100.0, 0.0, 1.0))
    return 2.0 + 3.0 * a, 0.02 + 0.035 * a, 0.06 + 0.24 * a


def apply_denoise(img: np.ndarray, amount: float,
                  noise_ref: float | None = None) -> np.ndarray:
    """降噪：感知域亮色分离双边滤波 + 亮度细节回注。

    1. sRGB 感知域（gamma 编码）转 YCrCb；
    2. 亮度与色度各走一遍双边滤波：亮度用「两遍」轻滤（噪声逐遍衰减，
       中低对比纹理保住），色度用大 sigmaColor 一次滤净（色斑去得干净）；
       注意不能用单通道 bilateralFilter：OpenCV 5 该路径走 IPP 派发，
       跨线程结果不一致（差 ~1e-7），会破坏分块并行的逐位一致验收；
    3. 被滤掉的亮度高频层按「结构权重」回注：高频在若干像素上同号聚集
       说明是真结构（边缘/纹理），回注比例高（0.35）；散乱高频是噪声，
       只按 1-0.9a 回注。平坦区因此保持干净，纹理区不会发“肉”；
    4. 合并转回线性光。0 为恒等。分块并行时依赖 halo（见 mask_halo）。

    noise_ref 为整幅图的噪声基线（见 GlobalStats.noise_ref）：分块时各块必须
    用同一个值，否则细节掩膜不一致、接缝处会出现横条。None 表示就地估计。
    """
    if amount == 0:
        return img
    a = float(np.clip(amount / 100.0, 0.0, 1.0))
    sigma_space, sigma_y, sigma_c = _denoise_sigmas(amount)
    x = _gamma_encode(img)
    ycc = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    # 两次双边滤波串行执行：cv2 的 IPP 分块结果依赖当前可用线程数，
    # 并行跑会让输出与串行差到 2%，破坏逐位一致验收
    y_src = ycc[..., 0]
    y_den = y_src
    # 三遍轻滤：半径小（≈1.5σ）但串起来等效支持域够大。单通道 float32 的
    # bilateral 走慢路径（0.37MP 单遍 300ms，三通道只要 95ms），所以复制成
    # 三通道再滤；三通道的颜色距离是各通道欧氏距离，等价于 σ/√3，
    # 因此这里把 sigma 乘回 √3 才能得到与单通道相同的强度。
    sy3 = sigma_y * 1.7320508
    for _ in range(3):
        y_den = cv2.bilateralFilter(cv2.merge([y_den, y_den, y_den]),
                                    0, sy3, sigma_space)[..., 0]
    c_den = cv2.bilateralFilter(ycc, 0, sigma_c, sigma_space)

    detail = y_src - y_den
    gain = np.float32(1.0 - 0.9 * a)         # 噪声（散乱高频）的回注比例
    # 结构判定：高频在 3px 尺度上同号聚集 = 真结构；噪声零均值，平滑后趋零
    struct = cv2.GaussianBlur(detail, (0, 0), 3.0)
    np.abs(struct, out=struct)
    if noise_ref is not None:
        med = float(noise_ref)
    else:
        med = float(np.median(struct[::4, ::4]))  # 就地估计（采样子集，省排序开销）
    if med > 1e-7:
        t0, t1 = np.float32(1.2 * med), np.float32(4.2 * med)
        struct -= t0
        struct *= np.float32(1.0 / (t1 - t0))
        np.clip(struct, 0.0, 1.0, out=struct)
        struct *= np.float32(3.0) - struct * np.float32(2.0)   # smoothstep
        # 映射到 [gain, kmax]：kmax 至少与 gain 相同，低强度档不会被“提亮”细节
        kmax = max(0.35, float(gain))
        struct *= np.float32(kmax - float(gain))
        struct += gain
    else:
        struct = None
    y_out = y_den + detail * (struct if struct is not None else gain)
    out = cv2.merge([y_out, c_den[..., 1], c_den[..., 2]])
    return to_linear(cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB))


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
    out = apply_denoise(out, params.denoise,
                        stats.noise_ref if stats is not None else None)
    out = apply_sharpen(out, params.sharpen)
    if stats is not None:
        out = apply_dehaze(out, params.dehaze, stats.A, stats.sigma)
    else:
        out = apply_dehaze(out, params.dehaze)
    return out


class LowResRenderer:
    """清晰度所需的低频图：整幅图只算一次，再按行区间切片（分块安全、无接缝）。

    clarity_needed 应在整图级别判断：全局清晰度或任一蒙版局部清晰度非零都要创建，
    否则蒙版内的清晰度会按分块自身算低频图，产生分块间差异。
    """

    def __init__(self, lin: np.ndarray, clarity_needed: bool, edge: int = 512):
        self.low = None
        if not clarity_needed:
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
    dehaze_needed 同样要覆盖蒙版局部的去朦胧（见 LowResRenderer）。

    denoise_amount > 0 时再算一个「噪声基线」：降噪的细节掩膜阈值要按整幅图
    的噪声水平定，让分块各自统计会出现横条接缝。估法是在最长边 384 的缩略图上
    跑同一套双边 + 结构平滑，取 |结构| 的中位数（噪声统计与分辨率无关）。
    """

    def __init__(self, lin: np.ndarray, dehaze_needed: bool,
                 denoise_amount: float = 0.0):
        self.A = None
        self.sigma = 1.0
        self.noise_ref = None
        if denoise_amount:
            self.noise_ref = _noise_reference(lin, denoise_amount)
        if not dehaze_needed:
            return
        h, w = lin.shape[:2]
        self.sigma = max(1.0, min(h, w) / 240.0)
        sample = (lin[::4, ::4] @ LUMA)
        self.A = float(np.percentile(sample, 99.9)) if sample.size > 16 \
            else float(luminance(lin).max())


def _noise_reference(lin: np.ndarray, amount: float) -> float | None:
    """整幅图的噪声基线（降噪细节掩膜阈值用）：在缩略图上跑同一套前两步。"""
    h, w = lin.shape[:2]
    s = max(1.0, max(h, w) / 384.0)
    if s > 1.5:
        small = cv2.resize(lin, (max(8, int(w / s)), max(8, int(h / s))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = lin
    sigma_space, sigma_y, _ = _denoise_sigmas(amount)
    y = cv2.cvtColor(_gamma_encode(np.clip(small, 0.0, 1.0)),
                     cv2.COLOR_RGB2YCrCb)[..., 0]
    yd = y
    sy3 = sigma_y * 1.7320508
    for _ in range(3):
        yd = cv2.bilateralFilter(cv2.merge([yd, yd, yd]), 0, sy3, sigma_space)[..., 0]
    struct = cv2.GaussianBlur(y - yd, (0, 0), 3.0)
    np.abs(struct, out=struct)
    return float(np.median(struct)) if struct.size > 16 else None


# ======================================================================
# 8. 蒙版（线性渐变 / 径向渐变 / 画笔）
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


def _feather_canvas_mask(canvas: np.ndarray, feather: float) -> np.ndarray:
    """对二值/抗锯齿画布蒙版应用可分离高斯羽化（大图先降采样再上采样）。"""
    h, w = canvas.shape[:2]
    short = max(min(w, h), 1)
    fpx = min(float(feather) / 100.0 * short * 0.15, 160.0)
    if fpx > 0.6 and canvas.max() > 0:
        s = 8 if min(h, w) >= 160 else 1
        if s > 1:
            sh, sw = max(1, h // s), max(1, w // s)
            small = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(fpx / s, 0.5))
            canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=max(fpx, 0.5))
    return np.clip(canvas, 0.0, 1.0)


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

    return _feather_canvas_mask(canvas, m.feather)


def _polygon_mask(m: "MaskState", shape) -> np.ndarray:
    """自定义多边形蒙版：按归一化顶点填充多边形 + 高斯羽化。"""
    h, w = shape
    if not m.points or len(m.points) < 3:
        return np.zeros((h, w), np.float32)
    pts = np.asarray(m.points, np.float32) * np.array([w, h], np.float32)
    ip = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    u8 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(u8, [ip], 255, lineType=cv2.LINE_AA)
    canvas = u8.astype(np.float32) / np.float32(255.0)
    return _feather_canvas_mask(canvas, m.feather)


def grid_to_array(rows, gw: int, gh: int) -> np.ndarray:
    """['0101...', ...] -> (gh, gw) float32 0/1 数组；非法输入安全降级。

    只认 '1' 字符，其余（杂散字符/补位）一律为 0。
    """
    gw, gh = int(gw), int(gh)
    if gw <= 0 or gh <= 0 or not rows:
        return np.zeros((max(gh, 0), max(gw, 0)), np.float32)
    data = "".join(
        "".join(ch for ch in str(r) if ch in "01")[:gw].ljust(gw, "0")
        for r in list(rows)[:gh])
    data = data.ljust(gw * gh, "0")[:gw * gh]
    arr = np.frombuffer(data.encode("ascii"), np.uint8)
    return (arr == ord("1")).astype(np.float32).reshape(gh, gw)


def array_to_grid(arr: np.ndarray) -> list:
    """(H,W) 0/1 数组 -> ['0101...', ...] 字符串行。"""
    if arr.size == 0:
        return []
    return ["".join("1" if v >= 0.5 else "0" for v in row)
            for row in np.asarray(arr >= 0.5)]


def encode_mask_png(arr: np.ndarray) -> tuple:
    """(H,W) float 0..1 掩码 -> (base64 PNG 字符串, w, h)；保留 256 级软边界。"""
    import base64
    h, w = arr.shape[:2]
    u8 = np.clip(np.asarray(arr, np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        return "", 0, 0
    return base64.b64encode(buf.tobytes()).decode("ascii"), int(w), int(h)


def decode_mask_png(b64: str) -> np.ndarray:
    """base64 单通道 PNG -> (H,W) float 0..1；失败返回空数组。"""
    import base64
    if not b64:
        return np.zeros((0, 0), np.float32)
    try:
        data = np.frombuffer(base64.b64decode(b64), np.uint8)
        u8 = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return np.zeros((0, 0), np.float32)
    if u8 is None:
        return np.zeros((0, 0), np.float32)
    return u8.astype(np.float32) / np.float32(255.0)


def _ai_mask(m: "MaskState", shape) -> np.ndarray:
    """AI 蒙版：mask_png（像素级，优先）或 0/1 网格放大到整幅。

    AI 蒙版不做羽化：识别到哪里就作用到哪里，边界只保留 1~2px 抗锯齿，
    避免局部调整渗到主体之外的背景上（旧参数里的 feather 对 AI 蒙版不再生效）。
    """
    h, w = shape
    if m.mask_png:
        fine = decode_mask_png(m.mask_png)
        if fine.size:
            mask = cv2.resize(fine, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = np.zeros((h, w), np.float32)
    else:
        cells = grid_to_array(m.grid, m.grid_w, m.grid_h)
        if cells.size == 0:
            return np.zeros((h, w), np.float32)
        mask = cv2.resize(cells, (w, h), interpolation=cv2.INTER_LINEAR)
    # 插值/概率图会留下几像素到半格的"蹭边"软过渡，收窄到边界处：
    # 背景侧严格为 0，主体内部严格为 1，只在 1~2px 内抗锯齿。
    mask = smoothstep(0.35, 0.65, np.asarray(mask, np.float32))
    return np.clip(mask, 0.0, 1.0)


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
    if m.kind == "linear":
        mask = _linear_mask(m, shape)
    elif m.kind == "radial":
        mask = _radial_mask(m, shape)
    elif m.kind == "polygon":
        mask = _polygon_mask(m, shape)
    elif m.kind == "ai":
        mask = _ai_mask(m, shape)
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

    低分辨率结果走跨任务 LRU（_render_mask_cached）：拖动滑块/改参数时每个
    渲染任务都会重建 MaskRenderer，而蒙版几何未变——重复渲染纯属浪费。
    蒙版只依赖几何与形状（与图像像素无关），缓存天然跨图安全。
    """

    COARSE_EDGE = 768

    def __init__(self, masks, shape):
        self.masks = masks
        self.shape = shape
        h, w = shape
        self._full: list = []
        self._coarse: list = []
        self._lock = threading.Lock()   # 并行分块首用时只做一次上采样
        for m in masks:
            if m.is_trivial():
                self._coarse.append(None)
                self._full.append(None)
                continue
            s = min(1.0, self.COARSE_EDGE / float(max(h, w)))
            if s < 1.0:
                self._coarse.append(_render_mask_cached(m, (max(1, int(h * s)),
                                                            max(1, int(w * s)))))
            else:
                self._coarse.append(_render_mask_cached(m, (h, w)))
            self._full.append(None)

    def get(self, index: int, y0: int, y1: int, width: int) -> np.ndarray:
        """原图行区间 [y0, y1) 对应的蒙版。"""
        if self._full[index] is None:
            with self._lock:
                if self._full[index] is None:   # 双重检查：并行时只上采样一次
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
    if m.kind == "ai":
        # 掩码内容决定渲染结果；ai_target 只是语义标签，不影响像素
        return base + (int(m.grid_w), int(m.grid_h), tuple(m.grid),
                       hash(m.mask_png) if m.mask_png else 0)
    if m.kind == "polygon":
        pts = tuple((round(float(x), 5), round(float(y), 5)) for x, y in (m.points or []))
        return base + (pts,)
    strokes = tuple(tuple((round(x, 5), round(y, 5)) for x, y in s) for s in m.strokes)
    return base + (round(m.size, 4), strokes)


# ---- 蒙版渲染跨任务 LRU ------------------------------------------------
# 每条缓存 ≈ 768×576×4B ≈ 1.7MB；12 条足够单图常用蒙版集合。
_MASK_CACHE_MAX = 12
_mask_cache: dict = {}
_mask_cache_order: list = []
_mask_cache_lock = threading.Lock()
mask_cache_stats = {"hit": 0, "miss": 0}


def _render_mask_cached(m: "MaskState", shape) -> np.ndarray:
    """render_mask 的 LRU 版本：蒙版几何未变时直接复用上一次的渲染结果。

    渲染在锁外执行（避免长持锁），写入时再查重（并发下最多多算一次，无害）。
    返回的是共享数组，调用方只读、不得原地修改（MaskRenderer.get 上采样即拷贝）。
    """
    key = mask_cache_key(m, shape)
    with _mask_cache_lock:
        ent = _mask_cache.get(key)
        if ent is not None:
            _mask_cache_order.remove(key)
            _mask_cache_order.append(key)
            mask_cache_stats["hit"] += 1
            return ent
    arr = render_mask(m, shape)
    with _mask_cache_lock:
        if key not in _mask_cache:
            _mask_cache[key] = arr
            _mask_cache_order.append(key)
            while len(_mask_cache_order) > _MASK_CACHE_MAX:
                _mask_cache.pop(_mask_cache_order.pop(0), None)
        mask_cache_stats["miss"] += 1
    return _mask_cache[key]


def clear_mask_cache() -> None:
    """清空蒙版缓存（测试隔离用）。"""
    with _mask_cache_lock:
        _mask_cache.clear()
        _mask_cache_order.clear()


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


def _mask_effect_strengths(params) -> dict:
    """效果强度的整图级记账：全局与所有蒙版局部取最大（分块/halo 依据）。"""
    masks = getattr(params, "masks", [])
    out = {f: abs(float(getattr(params, f, 0.0)))
           for f in EFFECT_FIELDS}
    for m in masks:
        for f in EFFECT_FIELDS:
            out[f] = max(out[f], abs(float(getattr(m.adj, f, 0.0))))
    return out


def mask_halo(params, h: int, w: int) -> int:
    """分块时所需的重叠像素：效果算子的模糊需要邻域。

    全局与蒙版局部效果一并计入：清晰度/锐度取 64；去朦胧的暗通道高斯
    实际半径为 4σ（σ = 短边/240），大图时超过 64，按 4σ+1 动态计算；
    降噪按双边滤波半径（≈2σ）另计，三者取大。
    蒙版由 MaskRenderer 在低分辨率全局渲染，不产生接缝，因此不需要 halo。
    """
    eff = _mask_effect_strengths(params)
    halo = 0
    if eff["clarity"] or eff["sharpen"] or eff["dehaze"]:
        halo = 64
    if eff["dehaze"]:
        sigma = max(1.0, min(h, w) / 240.0)
        halo = max(halo, int(4.0 * sigma) + 1)
    if eff["denoise"]:
        # 亮度跑三遍双边：边界影响半径按三遍叠加（≈1.5σ×3）计，否则分块会出接缝
        halo = max(halo, int(_denoise_sigmas(eff["denoise"])[0] * 6.0) + 8)
    return halo


# ======================================================================
# 8b. 仿制图章（矩形取样覆盖，在流水线之前作用于源图）
# ======================================================================


def copy_patch(img: np.ndarray, src_rect, dst_rect, feather: float = 0.2) -> None:
    """矩形取样覆盖：把源矩形内容缩放到目标矩形尺寸后羽化覆盖（原地修改）。

    src_rect / dst_rect 为像素 (x0, y0, x1, y1)，越界部分自动裁剪；
    feather ∈ [0,1] 为边缘羽化宽度（相对目标短边的一半，上限 64px）。
    先把源区域 resize 成独立数组再混合，源/目标重叠安全。
    """
    h, w = img.shape[:2]

    def clamp_rect(r):
        x0, x1 = sorted((float(r[0]), float(r[2])))
        y0, y1 = sorted((float(r[1]), float(r[3])))
        ix0 = int(round(max(0.0, x0)))
        iy0 = int(round(max(0.0, y0)))
        ix1 = int(round(min(float(w), x1)))
        iy1 = int(round(min(float(h), y1)))
        return ix0, iy0, max(ix0 + 2, ix1), max(iy0 + 2, iy1)

    sx0, sy0, sx1, sy1 = clamp_rect(src_rect)
    dx0, dy0, dx1, dy1 = clamp_rect(dst_rect)
    dw, dh = dx1 - dx0, dy1 - dy0
    patch = cv2.resize(img[sy0:sy1, sx0:sx1], (dw, dh),
                       interpolation=cv2.INTER_LINEAR)

    fpx = min(float(np.clip(feather, 0.0, 1.0)) * min(dw, dh) * 0.5, 64.0)

    def ramp(n: int) -> np.ndarray:
        """沿一维的边缘羽化：边缘 0 -> 内部精确 1（smoothstep 过渡）。"""
        if fpx <= 0.5:
            return np.ones(n, np.float32)
        t = np.minimum(np.arange(n, dtype=np.float32),
                       np.arange(n - 1, -1, -1, dtype=np.float32))
        return smoothstep(np.float32(0.0), np.float32(fpx), t + np.float32(0.5))

    alpha = ramp(dh)[:, None] * ramp(dw)[None, :]

    dst = img[dy0:dy1, dx0:dx1]
    blended = dst.astype(np.float32)
    blended *= (np.float32(1.0) - alpha[..., None])
    blended += patch.astype(np.float32) * alpha[..., None]
    dst[...] = np.clip(blended + np.float32(0.5), 0, 255).astype(np.uint8)


def norm_rect_px(rect_norm, h: int, w: int) -> tuple:
    """归一化矩形 (x0,y0,x1,y1) -> 像素矩形（按高 h、宽 w 换算）。"""
    return (rect_norm[0] * w, rect_norm[1] * h,
            rect_norm[2] * w, rect_norm[3] * h)


def stamp_brush(img: np.ndarray, src_start, points, radius_norm: float,
                feather: float = 0.35, opacity: float = 1.0) -> None:
    """Photoshop 风格仿制图章画笔涂抹（原地修改）：

    - src_start: (sx, sy) 归一化起点采样坐标
    - points: [(x, y), ...] 归一化笔迹坐标序列，偏移量为 src_start - points[0]
    - radius_norm: 笔刷半径（相对图像短边）
    - feather: 边缘羽化比例 0..1（0=硬边带 1px 抗锯齿，1=从中心柔和衰减到边缘）
    - opacity: 单笔不透明度 0..1
    在笔迹外接框内用 distanceTransform 计算精确欧氏距离，一次采样混合避免重叠累加与自涂抹。
    """
    if not points or radius_norm <= 0.0 or opacity <= 0.0:
        return
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return
    short = float(min(h, w))
    r_px = max(1.5, float(radius_norm) * short)
    op = float(np.clip(opacity, 0.0, 1.0))
    fe = float(np.clip(feather, 0.0, 1.0))

    pts_px = np.asarray([(float(p[0]) * w, float(p[1]) * h) for p in points], np.float32)
    off_x = int(round(float(src_start[0]) * w - pts_px[0, 0]))
    off_y = int(round(float(src_start[1]) * h - pts_px[0, 1]))

    pad = int(np.ceil(r_px)) + 2
    bx0 = max(0, int(np.floor(pts_px[:, 0].min())) - pad)
    by0 = max(0, int(np.floor(pts_px[:, 1].min())) - pad)
    bx1 = min(w, int(np.ceil(pts_px[:, 0].max())) + pad + 1)
    by1 = min(h, int(np.ceil(pts_px[:, 1].max())) + pad + 1)
    bw, bh = bx1 - bx0, by1 - by0
    if bw <= 0 or bh <= 0:
        return

    centerline = np.full((bh, bw), 255, np.uint8)
    local_pts = np.round(pts_px - np.array([bx0, by0], np.float32)).astype(np.int32)
    if len(local_pts) == 1:
        px, py = int(local_pts[0, 0]), int(local_pts[0, 1])
        if 0 <= px < bw and 0 <= py < bh:
            centerline[py, px] = 0
        else:
            # 点在局部框边界外时改用解析距离
            yy, xx = np.mgrid[by0:by1, bx0:bx1]
            dist = np.hypot(xx.astype(np.float32) - pts_px[0, 0],
                            yy.astype(np.float32) - pts_px[0, 1])
    else:
        cv2.polylines(centerline, [local_pts.reshape(-1, 1, 2)], False, 0,
                      thickness=1, lineType=cv2.LINE_8)

    if len(local_pts) > 1 or (0 <= int(local_pts[0, 0]) < bw and 0 <= int(local_pts[0, 1]) < bh):
        dist = cv2.distanceTransform(centerline, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    inner = min(r_px * (1.0 - fe * 0.95), max(0.0, r_px - 1.2))
    alpha = (np.float32(1.0) - smoothstep(np.float32(inner), np.float32(r_px), dist)) * np.float32(op)
    if float(alpha.max()) <= 1e-4:
        return

    sy = np.clip(np.arange(by0, by1, dtype=np.int32) + off_y, 0, h - 1)
    sx = np.clip(np.arange(bx0, bx1, dtype=np.int32) + off_x, 0, w - 1)
    patch = img[np.ix_(sy, sx)].astype(np.float32)

    dst = img[by0:by1, bx0:bx1]
    blended = dst.astype(np.float32) * (np.float32(1.0) - alpha[..., None]) + patch * alpha[..., None]
    dst[...] = np.clip(blended + np.float32(0.5), 0, 255).astype(np.uint8)


# ======================================================================
# 8b2. AI 景深散景（基于 Depth Anything V2 深度图的光学镜头虚化）
# ======================================================================


def apply_depth_of_field(
    rgb8: np.ndarray,
    depth01: np.ndarray,
    focus: float = 0.75,
    aperture: float = 45.0,
    dof_range: float = 20.0,
    bokeh_highlight: float = 25.0,
) -> np.ndarray:
    """基于深度图模拟真实光学大光圈景深虚化（线性光空间 + 预乘防渗边金字塔）。

    - focus: 对焦平面深度 0..1（1=最近前景，0=最远背景）
    - aperture: 虚化强度/光圈 0..100（0 为原图不变）
    - dof_range: 清晰景深宽度 0..80
    - bokeh_highlight: 焦外高光散景增强 0..100
    """
    ap = float(np.clip(aperture, 0.0, 100.0))
    if ap <= 0.5:
        return np.ascontiguousarray(rgb8)

    h, w = rgb8.shape[:2]
    if h < 4 or w < 4:
        return np.ascontiguousarray(rgb8)

    d = np.asarray(depth01, dtype=np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    d = np.clip(d, 0.0, 1.0)

    focus_c = np.float32(np.clip(focus, 0.0, 1.0))
    half_zone = np.float32(np.clip(dof_range, 0.0, 80.0) / 200.0)
    delta = np.abs(d - focus_c)
    excess = np.maximum(np.float32(0.0), delta - half_zone)
    denom = np.float32(max(0.10, 1.0 - float(half_zone)))
    coc = np.clip(excess / denom, 0.0, 1.0)
    coc = np.power(coc, np.float32(1.12)).astype(np.float32)
    if float(coc.max()) <= 1e-4:
        return np.ascontiguousarray(rgb8)

    short = float(min(h, w))
    max_sigma = max(0.8, min(56.0, (ap / 100.0) * 0.036 * short))

    lin = rgb8_to_linear(rgb8)
    hl_str = float(np.clip(bokeh_highlight, 0.0, 100.0)) / 100.0 * 1.8
    if hl_str > 1e-4:
        lum = luminance(lin)
        hl = smoothstep(np.float32(0.55), np.float32(0.95), lum) * coc * np.float32(hl_str)
        lin_src = lin * (np.float32(1.0) + hl[..., None])
    else:
        lin_src = lin

    # 预乘焦外遮罩，防止锐利前景颜色渗透到焦外散景形成暗边光晕
    oob = smoothstep(np.float32(0.015), np.float32(0.14), coc)
    src4 = np.empty((h, w, 4), dtype=np.float32)
    src4[..., :3] = lin_src * oob[..., None]
    src4[..., 3] = oob

    levels = 5
    idx_f = coc * np.float32(levels)
    out_lin = np.zeros_like(lin)

    # k = 0: 清晰合焦平面
    w0 = np.maximum(np.float32(0.0), np.float32(1.0) - idx_f)
    if np.any(w0 > 1e-4):
        out_lin += lin * w0[..., None]

    for k in range(1, levels + 1):
        wk = np.maximum(np.float32(0.0), np.float32(1.0) - np.abs(idx_f - np.float32(k)))
        if not np.any(wk > 1e-4):
            continue
        sig = max_sigma * ((k / float(levels)) ** 1.35)
        if k == 1:
            # 焦平面边缘柔顺过渡：低半径不掩码，避免 0..0.15 coc 出现台阶
            lk = cv2.GaussianBlur(lin_src, (0, 0), sigmaX=sig, sigmaY=sig,
                                  borderType=cv2.BORDER_REPLICATE)
        else:
            if sig > 9.0 and h >= 64 and w >= 64:
                dh, dw = max(16, h // 2), max(16, w // 2)
                small = cv2.resize(src4, (dw, dh), interpolation=cv2.INTER_AREA)
                b_small = cv2.GaussianBlur(small, (0, 0), sigmaX=sig * 0.5, sigmaY=sig * 0.5,
                                           borderType=cv2.BORDER_REPLICATE)
                b4 = cv2.resize(b_small, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                b4 = cv2.GaussianBlur(src4, (0, 0), sigmaX=sig, sigmaY=sig,
                                      borderType=cv2.BORDER_REPLICATE)
            alpha = b4[..., 3:4]
            safe_a = np.maximum(alpha, np.float32(1e-3))
            cover = np.clip(alpha * np.float32(10.0), 0.0, 1.0)
            lk = cover * (b4[..., :3] / safe_a) + (np.float32(1.0) - cover) * lin
        out_lin += lk * wk[..., None]

    if hl_str > 1e-4:
        # 高光柔和压缩肩部，保住散景光斑色彩与圆润感
        over = np.maximum(np.float32(0.0), out_lin - np.float32(0.92))
        out_lin = np.where(
            out_lin > np.float32(0.92),
            np.float32(0.92) + over / (np.float32(1.0) + over * np.float32(6.0)),
            out_lin,
        )
    return linear_to_rgb8(np.clip(out_lin, 0.0, 1.0))


def depth_focus_preview(depth01: np.ndarray, focus: float = 0.75,
                        dof_range: float = 20.0, max_edge: int = 280) -> str:
    """渲染紧凑深度图 + 绿色清晰焦平面高亮提示 PNG（base64）。"""
    import base64

    d = np.asarray(depth01, dtype=np.float32)
    h, w = d.shape[:2]
    scale = min(1.0, float(max_edge) / float(max(h, w, 1)))
    if scale < 1.0:
        nh = max(8, int(round(h * scale)))
        nw = max(8, int(round(w * scale)))
        d = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_AREA)
    d = np.clip(d, 0.0, 1.0)

    u8 = np.clip(np.rint(d * 255.0), 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO).astype(np.float32)

    half_zone = np.float32(np.clip(dof_range, 0.0, 80.0) / 200.0)
    delta = np.abs(d - np.float32(np.clip(focus, 0.0, 1.0)))
    in_focus = np.float32(1.0) - smoothstep(
        half_zone, np.float32(float(half_zone) + 0.06), delta)
    # 合焦带叠加青绿高亮，直观标识清晰范围
    focus_bgr = np.array([175.0, 255.0, 95.0], dtype=np.float32)
    blend = in_focus[..., None] * np.float32(0.55)
    vis = vis * (np.float32(1.0) - blend) + focus_bgr * blend
    ok, buf = cv2.imencode(".png", np.clip(vis, 0, 255).astype(np.uint8))
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


# ======================================================================
# 8c. 几何变换（裁剪 / 旋转 / 拉直）——与图章一样在流水线之前作用于源图
# ======================================================================


def rotate90(rgb: np.ndarray, quarter_turns: int = 1) -> np.ndarray:
    """旋转 90° 的整数倍（quarter_turns>0 为逆时针，与 np.rot90 一致），无损。"""
    return np.ascontiguousarray(np.rot90(rgb, quarter_turns))


def rotation_matrix(shape, angle_deg: float) -> np.ndarray:
    """绕图像中心的旋转矩阵（2x3，旧像素坐标 -> 新像素坐标，OpenCV 约定）。"""
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)


def rotate_image(rgb: np.ndarray, angle_deg: float,
                 interpolation: int = cv2.INTER_LINEAR):
    """绕中心旋转 angle_deg 度（OpenCV 约定：正值逆时针），画布尺寸不变。

    返回 (旋转结果, M)；M 为 2x3 仿射矩阵，把旧像素坐标映射到新坐标，
    供蒙版坐标同步换算（见 remap_mask_geometry）。旋转后画布四角可能落在
    原图之外，调用方应把裁剪框约束在有效区内（见 fit_rect_in_rotation）。
    """
    m = rotation_matrix(rgb.shape, angle_deg)
    out = cv2.warpAffine(rgb, m, (rgb.shape[1], rgb.shape[0]),
                         flags=interpolation, borderMode=cv2.BORDER_REPLICATE)
    return out, m


def _rect_corners_inside_rotation(rect, shape, angle_deg: float) -> bool:
    """矩形（像素 x0,y0,x1,y1）的四角是否全部落在旋转后的原图有效区内。"""
    x0, y0, x1, y1 = rect
    h, w = shape[:2]
    t = float(np.radians(angle_deg))
    ca, sa = float(np.cos(t)), float(np.sin(t))
    cx, cy = w / 2.0, h / 2.0
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        dx, dy = x - cx, y - cy
        ux = ca * dx + sa * dy          # 逆旋转回原图坐标
        uy = -sa * dx + ca * dy
        if abs(ux) > w / 2.0 + 1e-6 or abs(uy) > h / 2.0 + 1e-6:
            return False
    return True


def fit_rect_in_rotation(rect, shape, angle_deg: float,
                         min_px: int = 16) -> tuple:
    """把像素矩形向中心收缩，直到完全位于旋转 angle_deg 后的有效区内。

    θ=0 时只做画布边界裁剪；否则二分收缩比例（对任意角度/宽高比稳健，
    避免内接矩闭式解的分段讨论）。返回 (x0, y0, x1, y1) 像素坐标。
    """
    w, h = shape[1], shape[0]
    x0, y0, x1, y1 = (max(0.0, min(rect[0], w)), max(0.0, min(rect[1], h)),
                      min(float(w), max(rect[2], 0.0)), min(float(h), max(rect[3], 0.0)))
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if abs(angle_deg) < 1e-9:
        return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

    def scaled(k: float) -> tuple:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) / 2.0 * k, (y1 - y0) / 2.0 * k
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    if _rect_corners_inside_rotation((x0, y0, x1, y1), shape, angle_deg):
        best = (x0, y0, x1, y1)
    else:
        lo, hi = 0.0, 1.0            # 收缩比例的下/上界
        best = scaled(lo)
        for _ in range(28):          # 二分 ~1e-8
            mid = (lo + hi) / 2.0
            r = scaled(mid)
            if _rect_corners_inside_rotation(r, shape, angle_deg):
                lo, best = mid, r
            else:
                hi = mid
    # 保证最小尺寸、取整；取整可能把边缘放大零点几像素，再向内收缩到有效区
    cx, cy = (best[0] + best[2]) / 2.0, (best[1] + best[3]) / 2.0
    hw = max((best[2] - best[0]) / 2.0, min_px / 2.0)
    hh = max((best[3] - best[1]) / 2.0, min_px / 2.0)
    ix0, iy0 = int(round(cx - hw)), int(round(cy - hh))
    ix1, iy1 = int(round(cx + hw)), int(round(cy + hh))
    while ix1 - ix0 > 4 and iy1 - iy0 > 4 and not _rect_corners_inside_rotation(
            (ix0, iy0, ix1, iy1), shape, angle_deg):
        ix0 += 1
        iy0 += 1
        ix1 -= 1
        iy1 -= 1
    return (max(0, ix0), max(0, iy0), min(w, ix1), min(h, iy1))


def _remap_ai_grid(m: "MaskState", map_point, swap_dims: bool = False) -> None:
    """几何变换后同步 AI 网格（原地重写 m.grid）。

    逆映射采样：数值拟合 map_point 的仿射矩阵并求逆，对新网格每格中心
    查旧格取值（最近邻，保持 0/1）。前向重采样在裁剪（放大映射）下会把
    旧格中心稀疏地撒到新网格上产生空洞，故与 _remap_ai_mask_png 一致
    采用逆映射；出界格丢弃。swap_dims 用于 90° 旋转：图像宽高互换时
    网格维度随之转置（gw/gh 互换），否则网格纵横比与图像失配且重采样丢格。
    """
    cells = grid_to_array(m.grid, m.grid_w, m.grid_h)
    gh, gw = cells.shape
    if gh == 0 or gw == 0:
        return
    M = _fit_affine(map_point)                     # 旧归一化 -> 新归一化
    if M is None:
        return
    Minv = np.linalg.inv(
        np.concatenate([M, [[0.0, 0.0, 1.0]]], axis=0))             # 新 -> 旧
    ngw, ngh = (gh, gw) if swap_dims else (gw, gh)
    xs = (np.arange(ngw, dtype=np.float32) + 0.5) / ngw
    ys = (np.arange(ngh, dtype=np.float32) + 0.5) / ngh
    nx, ny = np.meshgrid(xs, ys)
    ox = Minv[0, 0] * nx + Minv[0, 1] * ny + Minv[0, 2]
    oy = Minv[1, 0] * nx + Minv[1, 1] * ny + Minv[1, 2]
    inside = (ox >= 0.0) & (ox < 1.0) & (oy >= 0.0) & (oy < 1.0)
    oi = np.clip((ox * gw).astype(np.int32), 0, gw - 1)
    oj = np.clip((oy * gh).astype(np.int32), 0, gh - 1)
    out = cells[oj, oi] * inside.astype(np.float32)
    m.grid_w, m.grid_h = int(ngw), int(ngh)
    m.grid = array_to_grid(out.astype(np.float32))


def _fit_affine(map_point) -> tuple:
    """从 map_point 采样 4 点最小二乘拟合仿射（旧归一化 -> 新归一化）。

    裁剪/拉直/90° 旋转的映射均为仿射；返回 2x3 矩阵，失败返回 None。
    """
    src = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
                   np.float32)
    dst = np.array([map_point(x, y) for x, y in src], np.float32)
    A = np.concatenate([src, np.ones((4, 1), np.float32)], axis=1)   # (4,3)
    try:
        coef, *_ = np.linalg.lstsq(A, dst, rcond=None)               # (3,2)
    except np.linalg.LinAlgError:
        return None
    return coef.T                                                    # (2,3) [a b c; d e f]


def _remap_ai_mask_png(m: "MaskState", map_point) -> None:
    """几何变换后同步像素级 AI 掩码（原地重写 m.mask_png）。

    与网格的前向重采样不同：像素掩码用逆映射重采样——数值拟合 map_point
    的仿射矩阵求逆，对新掩码每个像素查旧掩码坐标（cv2.remap），保留
    256 级软边界且无洞。
    """
    fine = decode_mask_png(m.mask_png)
    h, w = fine.shape[:2]
    if h == 0 or w == 0:
        return
    M = _fit_affine(map_point)                     # 旧坐标 = M · [新坐标; 1] 的逆？
    if M is None:
        return
    Minv = np.linalg.inv(
        np.concatenate([M, [[0.0, 0.0, 1.0]]], axis=0))             # 新 -> 旧
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h
    nx, ny = np.meshgrid(xs, ys)
    ox = Minv[0, 0] * nx + Minv[0, 1] * ny + Minv[0, 2]
    oy = Minv[1, 0] * nx + Minv[1, 1] * ny + Minv[1, 2]
    map_x = ox * w - 0.5
    map_y = oy * h - 0.5
    out = cv2.remap(fine, map_x.astype(np.float32), map_y.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    b64, pw, ph = encode_mask_png(out)
    if b64:
        m.mask_png, m.mask_w, m.mask_h = b64, pw, ph


def remap_mask_geometry(masks, map_point, old_shape, new_shape,
                        angle_delta: float = 0.0,
                        swap_grid_dims: bool = False) -> None:
    """几何变换后同步蒙版坐标（原地修改蒙版列表）。

    map_point: (nx, ny) -> (nx', ny')，旧图归一化坐标 -> 新图归一化坐标；
    径向蒙版角度加 angle_delta（屏幕顺时针为正，与 MaskState.angle 同约定；
    刚体变换下椭圆主轴长度不变，故 aspect 无需改动）；
    半径 / 笔刷大小按两图短边像素比缩放，保持物理尺寸不变；
    AI 蒙版的网格按同一映射前向重采样。
    """
    short_old = min(old_shape[0], old_shape[1])
    short_new = max(min(new_shape[0], new_shape[1]), 1)
    ratio = short_old / short_new
    for m in masks:
        if m.kind == "linear":
            m.x1, m.y1 = map_point(m.x1, m.y1)
            m.x2, m.y2 = map_point(m.x2, m.y2)
        elif m.kind == "radial":
            m.cx, m.cy = map_point(m.cx, m.cy)
            m.radius = float(np.clip(m.radius * ratio, 0.01, 5.0))
            m.angle = (float(m.angle) + angle_delta) % 360.0
        elif m.kind == "ai":
            _remap_ai_grid(m, map_point, swap_dims=swap_grid_dims)
            _remap_ai_mask_png(m, map_point)
        elif m.kind == "polygon":
            m.points = [tuple(map_point(x, y)) for x, y in (m.points or [])]
        else:
            m.strokes = [[tuple(map_point(x, y)) for x, y in s]
                         for s in m.strokes]
            m.size = float(np.clip(m.size * ratio, 1.0, 100.0))


def remap_masks_rot90(masks, quarter_turns: int, old_shape) -> tuple:
    """rotate90 之后同步蒙版，返回新图像形状。

    旋转是刚体变换：椭圆主轴长度不变（aspect 不变），半径按短边比缩放，
    角度随内容旋转（屏幕顺时针为正：逆时针旋转取负）。
    """
    k = quarter_turns % 4
    oh, ow = old_shape[:2]
    new_shape = (ow, oh) if k % 2 else (oh, ow)
    if k == 0:
        return new_shape
    if k == 1:            # 逆时针：nx' = ny, ny' = 1-nx
        f = lambda nx, ny: (ny, 1.0 - nx)
        delta = -90.0
    elif k == 3:          # 顺时针
        f = lambda nx, ny: (1.0 - ny, nx)
        delta = 90.0
    else:                 # 180°
        f = lambda nx, ny: (1.0 - nx, 1.0 - ny)
        delta = 180.0
    remap_mask_geometry(masks, f, old_shape, new_shape, angle_delta=delta,
                        swap_grid_dims=(k % 2 == 1))
    return new_shape


def crop_rotate(rgb: np.ndarray, rect_norm, angle_deg: float = 0.0,
                interpolation: int = cv2.INTER_LINEAR):
    """按裁剪框裁剪（可先绕中心旋转拉直），返回 (结果, map_point, 新形状)。

    rect_norm 为 (x0, y0, x1, y1) 归一化坐标，定义在**旋转后的视图**里
    （拉直时预览所见的坐标系）。map_point 把原图归一化坐标映射到结果
    归一化坐标，供 remap_mask_geometry 同步蒙版。
    """
    h, w = rgb.shape[:2]
    if abs(angle_deg) > 1e-9:
        rotated, m = rotate_image(rgb, angle_deg, interpolation)
    else:
        rotated, m = rgb, None
    x0 = int(round(float(np.clip(rect_norm[0], 0.0, 1.0)) * w))
    y0 = int(round(float(np.clip(rect_norm[1], 0.0, 1.0)) * h))
    x1 = int(round(float(np.clip(rect_norm[2], 0.0, 1.0)) * w))
    y1 = int(round(float(np.clip(rect_norm[3], 0.0, 1.0)) * h))
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    out = np.ascontiguousarray(rotated[y0:y1, x0:x1])
    nw, nh = max(x1 - x0, 1), max(y1 - y0, 1)

    def map_point(nx: float, ny: float):
        px, py = nx * w, ny * h
        if m is not None:
            ox, oy = px, py
            px = float(m[0, 0]) * ox + float(m[0, 1]) * oy + float(m[0, 2])
            py = float(m[1, 0]) * ox + float(m[1, 1]) * oy + float(m[1, 2])
        return ((px - x0) / nw, (py - y0) / nh)

    return out, map_point, out.shape[:2]


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

# 超过该像素数才启用多线程（小图开销不划算）。
# 120k ≈ 400x300：拖动中的草稿帧也在这个量级，必须走多核才能压到 10~20ms。
PARALLEL_MIN_PIXELS = 120_000
# 行分块数：实测（8 逻辑核 15W CPU）2 块最优——块数再多，halo 重复计算与
# 线程争抢的代价超过并行收益（8 块比 1 块慢 1.5 倍）。
_MAX_WORKERS = 2 if (os.cpu_count() or 1) > 1 else 1
# 每块留给 OpenCV 自己的线程数：双边/高斯滤波在 cv2 内是多线程的，
# 关成 1 反而更慢（0.96MP 全链 974ms @1 线程 / 755ms @4 线程）。
_CV2_THREADS = max(1, (os.cpu_count() or 4) // 2)

_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()

# 环境变量 FERNLENS_NO_GPU=1（兼容 PHOTOLAB_NO_GPU=1）强制关闭 GPU 路径
_GPU_ENABLED = (os.environ.get("FERNLENS_NO_GPU") or os.environ.get("PHOTOLAB_NO_GPU", "")) != "1"

# 进入并行渲染前记录 OpenCV 默认线程数，结束后恢复（setNumThreads 是进程级全局设置）
_CV2_DEFAULT_THREADS = cv2.getNumThreads()


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                # 池子要能容纳「无效果的纯逐像素链」用的多块并行
                _EXECUTOR = ThreadPoolExecutor(
                    max_workers=max(_MAX_WORKERS, min(8, os.cpu_count() or 2)),
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


_BASIC_FIELDS = ("temperature", "tint", "exposure", "contrast", "highlights",
                 "shadows", "whites", "blacks", "saturation", "vibrance")


def _basic_is_default(params) -> bool:
    return all(not getattr(params, k, 0) for k in _BASIC_FIELDS)


# GPU 只做基础影调时，像素数太小不值得来回搬运（实测 40k 以下 transfer 占主导）
GPU_BASIC_MIN_PIXELS = 20_000


def _try_gpu_basic(params, lin: np.ndarray) -> np.ndarray | None:
    """用 GPU 只算基础影调（返回 float32 线性光）；不可用/失败返回 None。

    效果、曲线、分级、蒙版仍按原顺序在 CPU 上跑，所以画面语义与纯 CPU 一致，
    只是把 apply_basic 那一段整幅 numpy 运算换成了 GPU。
    """
    try:
        from gpu import process_basic_linear
    except Exception:
        return None
    return process_basic_linear(params, lin)


def _process_rows(params, lin: np.ndarray, y0: int, y1: int, pad: int,
                  renderer: "MaskRenderer | None" = None,
                  low_res: "LowResRenderer | None" = None,
                  stats: "GlobalStats | None" = None,
                  tone: "np.ndarray | None" = None) -> np.ndarray:
    """处理行区间 [y0, y1)（含 pad 像素的上下文），返回该区间的结果。

    pad 为锐度/去朦胧模糊所需的邻域重叠；清晰度（低频图）、蒙版（粗渲染上采样）
    与去朦胧（大气光统计）都由整图级对象提供，保证分块之间逐位一致、无接缝。

    tone 为整图级的「基础影调」结果（GPU 算出，float32 线性光）。给出时跳过
    apply_basic；低频图/蒙版/统计仍基于原始 lin，与 CPU 路径完全同源。
    """
    h = lin.shape[0]
    a = max(0, y0 - pad)
    b = min(h, y1 + pad)
    low = low_res.band(a, b) if low_res is not None else None
    if tone is not None:                                              # 1. 白平衡 + 基础
        # GPU 结果是整图共享的：拷一份再改，避免相邻块（含 pad 重叠行）互相写坏。
        # 基础参数全默认时 tone 就是 lin，与旧行为一致（不复制，交给下游只读）
        out = tone[a:b] if tone is lin else tone[a:b].copy()
    else:
        out = apply_basic(lin[a:b], params)
    out = apply_effects(out, params, low, stats)                      # 2. 效果
    out = apply_curves_fast(out, params.curves)                       # 3. 曲线
    out = apply_color_grading(out, params.wheels, params.blend)       # 4. 颜色分级
    out = apply_masked_adjustments(out, params.masks, renderer=renderer,
                                   y0=a, y1=b, low=low, stats=stats)   # 5. 蒙版局部调整
    out = apply_pickers(out, params.pickers)                          # 6. 取色器
    if params.blackwhite:                                            # 7. 黑白
        out = apply_blackwhite(out, params.blackwhite)
    if out is lin or out is tone or out.base is lin or out.base is tone:
        # 全链恒等时 out 仍是 lin（或其切片视图），裁剪不能写进输入缓存
        out = out.copy()
    np.clip(out, 0.0, 1.0, out=out)
    result = linear_to_rgb8(out)
    return result[y0 - a:result.shape[0] - (b - y1)]


class CancelledError(RuntimeError):
    """渲染被协作取消（如窗口关闭时停止后台线程）；调用方应静默丢弃结果。"""


def _process_band(params, lin: np.ndarray, tone=None,
                  auto_tone: bool = True) -> np.ndarray:
    """整幅图的完整流水线（单线程路径，pad=0）。

    auto_tone 为真且确实有效果算子要跑时，自行取整图级基础影调（与
    process_linear 走同一条 GPU 路径），这样「单块整幅」与「分块并行」
    的对照才是在比同一套算子，而不是比 CPU/GPU 的舍入差。
    """
    eff = _mask_effect_strengths(params)
    if tone is None and auto_tone and any(eff[f] for f in EFFECT_FIELDS):
        tone = _compute_tone(params, lin)
    renderer = MaskRenderer(params.masks, lin.shape[:2]) if params.masks else None
    low_res = LowResRenderer(lin, bool(eff["clarity"]))
    stats = GlobalStats(lin, bool(eff["dehaze"]), eff["denoise"])
    return _process_rows(params, lin, 0, lin.shape[0], 0, renderer, low_res,
                         stats, tone)


def _compute_tone(params, lin: np.ndarray) -> np.ndarray | None:
    """整图级「基础影调」：GPU 算得出就用 GPU，否则返回 None（调用方走 CPU）。

    基础参数全默认时直接返回 lin（既不复制也不计算）——这比 CPU 上的
    apply_basic 还省一次整幅拷贝，数值上完全等价（该函数原样返回 img）。
    """
    if _basic_is_default(params):
        return lin
    h, w = lin.shape[:2]
    if not _GPU_ENABLED or h * w < GPU_BASIC_MIN_PIXELS:
        return None
    return _try_gpu_basic(params, lin)


def process_linear(params, lin: np.ndarray, should_cancel=None) -> np.ndarray:
    """线性光图像 -> 处理后的 uint8 sRGB RGB 图。

    should_cancel 为可选回调：返回 True 时尽快抛出 CancelledError 中止渲染
    （在分块之间检查；GPU 内核与单个分块内部不可中断）。
    """
    if params.is_default():
        return linear_to_rgb8(lin)

    if should_cancel is not None and should_cancel():
        raise CancelledError("已取消")

    if _GPU_ENABLED and _gpu_eligible(params):
        out = _try_gpu(params, lin)
        if out is not None:
            return out

    h, w = lin.shape[:2]
    pad = mask_halo(params, h, w)
    # 有模糊类效果（清晰度/降噪/锐化/去朦胧）时，cv2 内部并行已经吃满核心，
    # 再按行分太多块只会让 halo 重复计算和线程争抢（实测 8 块慢 1.5 倍）；
    # 纯逐像素链（无效果）反而是 numpy 单线程算子，分块越多越快。
    eff_pre = _mask_effect_strengths(params)
    heavy = any(eff_pre[f] for f in EFFECT_FIELDS)

    # 基础影调整段可以整幅交给 GPU（效果/曲线/分级/蒙版仍在 CPU、顺序不变）。
    # 只有「真的有效果算子要跑」时才值得：否则上面的整链 GPU 路径已经覆盖。
    tone = None
    if heavy and _GPU_ENABLED:
        tone = _compute_tone(params, lin)

    workers = 1
    if h * w >= PARALLEL_MIN_PIXELS:
        workers = _MAX_WORKERS if heavy else max(2, min(8, os.cpu_count() or 2))
    parallel = workers > 1 and h > 2 * pad

    # 自己按行分块并行时，限制 OpenCV 内部线程，避免 8x16 线程互相争抢；
    # 渲染结束后恢复默认值，避免影响 GUI 线程的解码/缩放等 cv2 调用
    if parallel:
        cv2.setNumThreads(_CV2_THREADS if heavy else 1)
    try:
        if not parallel:
            if should_cancel is not None and should_cancel():
                raise CancelledError("已取消")
            return _process_band(params, lin, tone, auto_tone=False)

        eff = eff_pre
        out = np.empty(lin.shape, dtype=np.uint8)
        bounds = np.linspace(0, h, workers + 1).astype(int)
        renderer = MaskRenderer(params.masks, (h, w)) if params.masks else None
        low_res = LowResRenderer(lin, bool(eff["clarity"]))
        stats = GlobalStats(lin, bool(eff["dehaze"]), eff["denoise"])

        def work(i: int):
            if should_cancel is not None and should_cancel():
                raise CancelledError("已取消")
            y0, y1 = int(bounds[i]), int(bounds[i + 1])
            if y1 <= y0:
                return
            out[y0:y1] = _process_rows(params, lin, y0, y1, pad, renderer,
                                       low_res, stats, tone)

        list(_executor().map(work, range(workers)))
        if should_cancel is not None and should_cancel():
            raise CancelledError("已取消")
        return out
    finally:
        if parallel:
            cv2.setNumThreads(_CV2_DEFAULT_THREADS)


def process(params, rgb8: np.ndarray, should_cancel=None) -> np.ndarray:
    """uint8 sRGB RGB -> 处理后的 uint8 sRGB RGB。"""
    return process_linear(params, rgb8_to_linear(rgb8), should_cancel)
