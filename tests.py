"""PhotoLab 自检。

用法：
    python tests.py              # 核心算法自检（无需 Qt）
    python tests.py --ui         # 追加 UI 冒烟测试（自动 offscreen + 同步处理）
"""
from __future__ import annotations

import os
import sys

RUN_UI = "--ui" in sys.argv

if RUN_UI:                                   # 必须在导入 Qt 之前设置
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PHOTOLAB_SYNC", "1")

# Windows 控制台默认 GBK，避免中文/符号输出报 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import core

FAILS: list = []


def check(name, cond):
    if isinstance(cond, np.ndarray):
        cond = bool(np.all(cond))
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILS.append(name)


# ======================================================================
# 一、核心算法
# ======================================================================

def test_colorspaces():
    x = np.linspace(0, 1, 256, dtype=np.float32)
    img = np.stack([x, x, x], axis=-1)[None, :, :]
    back = core.to_srgb(core.to_linear(img))
    check("sRGB->线性->sRGB 往返误差 < 1e-4", float(np.abs(back - img).max()) < 1e-4)
    check("线性亮度单调", bool(np.all(np.diff(core.luminance(core.to_linear(img))[0]) > 0)))


def test_basic():
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out = core.to_srgb(core.apply_exposure(img, 1.0))
    check("曝光 +1EV 使 sRGB 输出变亮", float(out.mean()) > 0.6)
    check("曝光 0 为恒等", np.array_equal(core.apply_exposure(img, 0.0), img))

    lo = np.full((8, 8, 3), 0.2, np.float32)
    hi = np.full((8, 8, 3), 0.8, np.float32)
    c_lo, c_hi = core.apply_contrast(lo, 100), core.apply_contrast(hi, 100)
    check("对比度 +100 拉开暗/亮", float(c_lo.mean()) < 0.2 and float(c_hi.mean()) > 0.8)
    check("对比度固定 0 和 1 端点", float(core.apply_contrast(np.float32(0.0), 100)) == 0.0
          and abs(float(core.apply_contrast(np.float32(1.0), 100)) - 1.0) < 1e-6)

    dark = np.zeros((16, 16, 3), np.float32) + 0.02
    check("阴影提亮只影响暗部", float(core.apply_shadows(dark, 100).mean()) > 0.02)

    wb = core.apply_white_balance(np.full((4, 4, 3), 0.5, np.float32), 100, 0)
    check("色温 +100 偏暖(R>B)", wb[0, 0, 0] > wb[0, 0, 2])

    gray = np.full((8, 8, 3), 0.5, np.float32)
    gray[..., 1] = 0.9
    sat = core.apply_saturation(gray, 100)
    check("饱和度增大色差", float(sat[..., 1].mean() - sat[..., 0].mean()) > 0.4)
    check("饱和度 -100 得到灰度", float(np.abs(core.apply_saturation(gray, -100).std(axis=-1).max())) < 1e-5)
    check("自然饱和度 0 为恒等", np.array_equal(core.apply_vibrance(gray, 0), gray))
    # 合并实现必须与「先饱和再自然饱和度」逐元素等价（推导见 apply_saturation_vibrance）
    rnd = np.random.default_rng(7).random((48, 64, 3)).astype(np.float32)
    for sa, vi in ((15, 12), (-40, 30), (60, -50), (100, 100)):
        seq = core.apply_vibrance(core.apply_saturation(rnd, sa), vi)
        fus = core.apply_saturation_vibrance(rnd, sa, vi)
        # 2e-6 容差：分步版会对「已饱和」的图再算一次亮度（数学上等于原亮度，
        # 浮点上差 1 ULP，±100 极限档放大到 1.9e-6），8 位输出逐位相同
        check(f"饱和+自然饱和度合并与分步等价({sa},{vi})",
              float(np.abs(seq - fus).max()) < 2e-6)
        check(f"合并后 8 位输出逐位相同({sa},{vi})",
              np.array_equal(core.linear_to_rgb8(np.clip(seq, 0.0, 1.0)),
                             core.linear_to_rgb8(np.clip(fus, 0.0, 1.0))))
    check("黑色抬升暗部", float(core.apply_blacks(dark, 100).mean()) > 0.02)


def test_curves():
    pts = [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]
    lut = core.build_lut(pts, 512)
    check("对角线为恒等 LUT",
          float(np.abs(core.build_lut([(0.0, 0.0), (1.0, 1.0)], 512)
                       - np.linspace(0, 1, 512)).max()) < 1e-5)
    check("LUT 经过控制点", abs(float(lut[256]) - 0.7) < 0.05)

    mono = [(0.0, 0.0), (0.25, 0.25), (0.75, 0.75), (1.0, 1.0)]
    check("单调控制点 -> 单调 LUT（无过冲）", float(np.diff(core.build_lut(mono, 1024)).min()) >= -1e-6)

    sp = core.sanitize_points([(0.5, 0.5), (0.2, 0.2), (0.5, 0.9), (0.0, 0.0), (1.0, 1.0)])
    check("控制点按 x 排序去重且端点固定",
          sp[0][0] == 0.0 and sp[-1][0] == 1.0
          and all(sp[i][0] < sp[i + 1][0] for i in range(len(sp) - 1)))

    img = np.full((8, 8, 3), 0.5, np.float32)
    curves = core.default_curves()
    curves["master"] = [(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)]
    check("主通道曲线变暗中间调", float(core.apply_curves(img, curves).mean()) < 0.5)
    curves_r = core.default_curves()
    curves_r["red"] = [(0.0, 0.0), (1.0, 0.2)]
    out_r = core.apply_curves(img, curves_r)
    check("红色通道曲线只压 R", out_r[0, 0, 0] < out_r[0, 0, 1]
          and abs(float(out_r[0, 0, 1]) - 0.5) < 0.05)
    check("默认曲线判定", core.curves_are_default(core.default_curves()))
    check("LUT 复合数量正确", set(core.channel_luts(core.default_curves())) == {"red", "green", "blue"})


def test_grading():
    img = np.zeros((32, 32, 3), np.float32)
    img[:16] = 0.05     # 阴影区
    img[16:] = 0.85     # 高光区
    zero = {r: core.WheelState() for r in core.GRADING_RANGES}
    check("零参数颜色分级为恒等", np.array_equal(core.apply_color_grading(img, zero, 50), img))
    wheels = {r: core.WheelState() for r in core.GRADING_RANGES}
    wheels["shadows"] = core.WheelState(hue=0.0, saturation=80.0)
    out = core.apply_color_grading(img, wheels, 50.0)
    check("阴影色轮只影响暗部(R 偏移)",
          float(out[:16, :, 0].mean() - img[:16, :, 0].mean()) > 0.01
          and abs(float(out[16:, :, 0].mean() - img[16:, :, 0].mean())) < 0.01)
    check("色轮方向：hue=0 为红", core.wheel_direction(0, 1)[0] > 0.8
          and core.wheel_direction(0, 1)[1] < 0)
    check("色轮方向：hue=120 为绿", core.wheel_direction(120, 1)[1] > 0.8)


def test_picker():
    rng = np.random.default_rng(7)
    img = rng.random((48, 48, 3)).astype(np.float32)
    img[:16, :16] = (0.9, 0.05, 0.05)
    img[16:32, :16] = (0.05, 0.8, 0.05)
    img[32:, :16] = 0.5
    rgb8 = core.linear_to_rgb8(img)

    check("色相采样：红≈0°，绿≈120°",
          abs(core.hue_degrees(core.sample_patch(rgb8, 2, 2))) < 5
          and abs(core.hue_degrees(core.sample_patch(rgb8, 2, 18)) - 120.0) < 5)

    p = core.PickerState(hue=core.hue_degrees(core.sample_patch(rgb8, 2, 2)),
                         span=25.0, saturation=-100.0)
    out = core.apply_pickers(img, [p])
    check("取色器去饱和只影响红色区域",
          float(out[:16, :16, 1].mean() - img[:16, :16, 1].mean()) > 0.1
          and abs(float(out[16:32, :16, 1].mean() - img[16:32, :16, 1].mean())) < 0.01
          and float(np.abs(out[32:, :16] - img[32:, :16]).max()) < 1e-4)

    p2 = core.PickerState(hue=120.0, span=30.0, hue_shift=120.0)
    out2 = core.apply_pickers(img, [p2])
    check("色相旋转改变绿色区域色相",
          core.hue_degrees(out2[16:32, :16].reshape(-1, 3).mean(axis=0)) > 100)
    check("无取色点为恒等", np.array_equal(core.apply_pickers(img, []), img))


def test_effects():
    # 清晰度：应增强局部对比（对比两块区域的平均差）
    img = np.zeros((64, 64, 3), np.float32)
    img[:32] = 0.20
    img[32:] = 0.30
    clear = core.apply_clarity(img, 80)
    check("清晰度增强局部对比",
          abs(float(clear[:32].mean() - clear[32:].mean())) >
          abs(float(img[:32].mean() - img[32:].mean())))
    check("清晰度 0 为恒等", np.array_equal(core.apply_clarity(img, 0), img))

    # 锐度：USM 应增强边缘梯度
    strip = np.zeros((32, 32, 3), np.float32)
    strip[:, 14:18] = 0.8
    sharp = core.apply_sharpen(strip, 80)
    check("锐度增强边缘", float(sharp[:, 15].mean() - sharp[:, 13].mean()) >
          float(strip[:, 15].mean() - strip[:, 13].mean()))
    check("锐度 0 为恒等", np.array_equal(core.apply_sharpen(strip, 0), strip))

    # 去朦胧：合成"雾图"应被恢复（对比提升）
    rng = np.random.default_rng(5)
    base = (rng.random((120, 160, 3)) * 0.6).astype(np.float32)
    hazy = base * 0.65 + 0.25                     # 加雾：压对比、抬黑位
    dehazed = core.apply_dehaze(hazy, 70)
    check("去朦胧提升对比",
          float(dehazed.std()) > float(hazy.std()))
    fogged = core.apply_dehaze(base, -60)
    check("负值去朦胧 = 加雾（对比下降）", float(fogged.std()) < float(base.std()))
    check("去朦胧 0 为恒等", np.array_equal(core.apply_dehaze(hazy, 0), hazy))
    check("去朦胧不产生 NaN/无穷", bool(np.all(np.isfinite(dehazed))))
    dark = np.zeros((32, 32, 3), np.float32) + 0.002
    check("近黑图去朦胧数值稳定", bool(np.all(np.isfinite(core.apply_dehaze(dark, 100)))))


def test_masks():
    shape = (300, 400)                             # h, w
    h, w = shape

    linear = core.MaskState(kind="linear", x1=0.2, y1=0.5, x2=0.8, y2=0.5, feather=40)
    lm = core.render_mask(linear, shape)
    check("线性蒙版尺寸/范围", lm.shape == (h, w) and 0 <= lm.min() and lm.max() <= 1)
    check("线性蒙版左端 ~0 右端 ~1",
          float(lm[150, 10]) < 0.05 and float(lm[150, 390]) > 0.95)
    check("线性蒙版端点连线中点为 ~0.5", abs(float(lm[150, 200]) - 0.5) < 0.1)
    lin_inv = core.MaskState(kind="linear", x1=0.2, y1=0.5, x2=0.8, y2=0.5,
                             feather=40, invert=True)
    check("线性蒙版反向", abs(float((core.render_mask(lin_inv, shape) + lm).max()) - 1.0) < 1e-5)
    lin_hard = core.MaskState(kind="linear", x1=0.2, y1=0.5, x2=0.8, y2=0.5, feather=0)
    lh = core.render_mask(lin_hard, shape)
    check("线性蒙版羽化 0 为硬边", abs(float(lh[150, 199])) < 0.02 and abs(float(lh[150, 201]) - 1) < 0.02)

    radial = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25, feather=30)
    rm = core.render_mask(radial, shape)
    check("径向蒙版中心为 1", float(rm[150, 200]) > 0.99)
    check("径向蒙版四角为 0", float(rm[2, 2]) < 0.01 and float(rm[297, 397]) < 0.01)
    radial_inv = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25,
                                feather=30, invert=True)
    check("径向蒙版反向 = 除椭圆外全部覆盖",
          float(core.render_mask(radial_inv, shape)[150, 200]) < 0.01)
    check("径向长宽比生效",
          core.render_mask(core.MaskState(kind="radial", radius=0.25, feather=0, aspect=2.5),
                           shape)[150, 300] > 0.5)

    brush = core.MaskState(kind="brush", size=10, feather=20,
                           strokes=[[(0.5, 0.5), (0.6, 0.55)], [(0.3, 0.7)]])
    bm = core.render_mask(brush, shape)
    check("画笔蒙版覆盖笔迹", float(bm[150, 200]) > 0.5 and float(bm[210, 120]) > 0.5)
    check("画笔蒙版不影响远处", float(bm[20, 380]) < 0.01)
    check("空画笔蒙版全 0", float(core.render_mask(
        core.MaskState(kind="brush"), shape).max()) == 0.0)
    brush_hard = core.MaskState(kind="brush", size=10, feather=0,
                                 strokes=[[(0.5, 0.5)]])
    bh = core.render_mask(brush_hard, shape)
    check("画笔羽化 0 为硬边", float(bh.max()) > 0.999)

    # 蒙版缓存键：几何变化 -> 键变化；仅改调整 -> 键不变
    k1 = core.mask_cache_key(brush, shape)
    brush2 = core.MaskState(kind="brush", size=10, feather=20,
                            strokes=[[(0.5, 0.5), (0.6, 0.55)], [(0.3, 0.7)]])
    brush2.adj.exposure = 1.0
    k2 = core.mask_cache_key(brush2, shape)
    brush3 = core.MaskState(kind="brush", size=12, feather=20, strokes=brush.strokes)
    check("蒙版缓存键与调整数值无关", k1 == k2)
    check("蒙版缓存键随几何变化", k1 != core.mask_cache_key(brush3, shape))

    # 蒙版局部调整：只作用于蒙版内
    img = np.full((64, 64, 3), 0.3, np.float32)
    m = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=10)
    m.adj.exposure = 1.0
    out = core.apply_masked_adjustments(img, [m])
    check("蒙版内曝光 +1EV", out[32, 32].mean() > 0.55)
    check("蒙版外保持原值", abs(float(out[2, 2].mean()) - 0.3) < 1e-5)
    m2 = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=10, invert=True)
    m2.adj.exposure = 1.0
    out2 = core.apply_masked_adjustments(img, [m2])
    check("反向蒙版作用于蒙版外", out2[2, 2].mean() > 0.55
          and abs(float(out2[32, 32].mean()) - 0.3) < 1e-5)
    check("无调整的蒙版为恒等",
          np.array_equal(core.apply_masked_adjustments(img, [core.MaskState()]), img))
    check("蒙版蒙版叠加不过曝", core.apply_masked_adjustments(
        img, [m, m2]).max() <= 0.3 * 2 + 1e-6)


def test_denoise():
    rng = np.random.default_rng(9)
    # 平坦区去噪：噪声图与干净参考都从 uint8 出发（与真实管线一致），
    # 避免把 uint8 量化偏差算进噪声里
    clean8 = np.full((96, 96, 3), 162, np.uint8)
    ref_lin = core.rgb8_to_linear(clean8)
    noisy8 = np.clip(clean8.astype(np.float32)
                     + rng.normal(0, 14, clean8.shape).astype(np.float32),
                     0, 255).astype(np.uint8)
    lin = core.rgb8_to_linear(noisy8)
    den = core.apply_denoise(lin, 80)
    dev_before = float(np.abs(lin - ref_lin).mean())
    dev_after = float(np.abs(den - ref_lin).mean())
    check(f"降噪降低噪声偏差（{dev_before:.4f} -> {dev_after:.4f}）",
          dev_after < dev_before * 0.4)
    check("降噪 0 为恒等", np.array_equal(core.apply_denoise(lin, 0), lin))
    check("降噪输出无 NaN", bool(np.all(np.isfinite(den))))

    # 边缘保持：强边缘不应被抹平
    edge = np.zeros((96, 96, 3), np.float32)
    edge[:, :48] = 0.2
    edge[:, 48:] = 0.8
    noisy_edge = edge + rng.normal(0, 0.03, edge.shape).astype(np.float32)
    den_e = core.apply_denoise(noisy_edge, 60)
    gap = abs(float(den_e[:, 16].mean() - den_e[:, 80].mean()))
    check(f"降噪保留强边缘（两侧差 {gap:.2f}）", gap > 0.35)

    # 流水线与 halo：denoise 参与 is_default，halo 覆盖滤波半径
    p = core.Adjustments(denoise=50)
    check("降噪参数使流水线生效", not p.is_default())
    check("降噪 halo 合理", 6 <= core.mask_halo(p, 3000, 2000)
          and core.mask_halo(p, 3000, 2000) < 64)
    rgb = core.linear_to_rgb8(noisy_edge)
    out = core.process(p, rgb)
    check("降噪流水线输出范围合法", int(out.min()) >= 0 and int(out.max()) <= 255)


def test_auto_rule():
    import auto
    # 偏暗、低对比、偏暖的合成图 -> 曝光为正、对比度为正、色温为负
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.10 + 0.06 * (xx / w).astype(np.float32)
    img = np.stack([base * 1.6, base * 1.05, base * 0.55], axis=-1)
    rgb8 = core.linear_to_rgb8(img)
    d = auto.suggest_rule(rgb8)
    check(f"规则自动:暗图曝光为正（{d.get('exposure')}）", d.get("exposure", 0) > 0.2)
    check(f"规则自动:低对比图对比度为正（{d.get('contrast')}）", d.get("contrast", 0) > 3)
    check(f"规则自动:暖偏色被校正（{d.get('temperature')}）", d.get("temperature", 0) < -5)
    check("规则自动:输出范围合法", all(-105 <= v <= 105 for v in d.values()))

    # 曝光良好的中性渐变图 -> 不应有大动作
    base2 = 0.08 + 0.80 * (xx / w).astype(np.float32)
    rgb2 = core.linear_to_rgb8(np.stack([base2] * 3, axis=-1))
    d2 = auto.suggest_rule(rgb2)
    check(f"规则自动:正常图曝光接近零（{d2.get('exposure')}）",
          abs(d2.get("exposure", 0)) < 0.5)
    check("规则自动:正常图无极端建议", max(abs(v) for v in d2.values()) <= 40)


def test_ai_sanitize():
    import json as _json

    import auto
    raw = "```json\n" + _json.dumps({
        "exposure": 99, "contrast": "x", "temperature": -25.6,
        "curves": {"master": [[0, 0], [0.5, 0.7], [1, 1]], "red": "bad"},
        "wheels": {"shadows": {"hue": 400, "saturation": 30, "luminance": -10},
                   "midtones": "bad"},
        "pickers": [{"hue": 120, "span": 500, "hue_shift": -30,
                     "saturation": 20, "luminance": 0},
                    {"hue": "x"}],
        "note": "提亮并压暖",
        "unknown_field": 1,
    }, ensure_ascii=False) + "\n```"
    # strength=1.0：先验「纯钳制」行为，不看缩放
    d = auto.sanitize_ai_params(auto._extract_json(raw), 1.0)
    check("AI 解析:exposure 99 钳到 5", d.get("exposure") == 5.0)
    check("AI 解析:非法字段被丢弃", "contrast" not in d and "unknown_field" not in d)
    check("AI 解析:色温保留", d.get("temperature") == -25.6)
    check("AI 解析:曲线被规整且非法通道被丢弃",
          d["curves"]["master"][1] == (0.5, 0.7) and "red" not in d["curves"])
    check("AI 解析:色轮 hue 归一化", d["wheels"]["shadows"]["hue"] == 40.0
          and "midtones" not in d["wheels"])
    check("AI 解析:取色器钳制且无效项被丢弃", len(d["pickers"]) == 1
          and d["pickers"][0]["span"] == 180 and d["pickers"][0]["hue_shift"] == -30)
    check("AI 解析:空输入安全", auto.sanitize_ai_params(None) == {}
          and auto.sanitize_ai_params("bad") == {})

    # ---- 强度缩放：默认 0.6，统一收敛幅度（防过度后期）----
    base = {"exposure": 2.0, "contrast": 60, "blackwhite": 100,
            "curves": {"master": [[0, 0], [0.5, 0.7], [1, 1]]},
            "wheels": {"shadows": {"hue": 40, "saturation": 40, "luminance": -20}},
            "pickers": [{"hue": 120, "span": 60, "hue_shift": -40,
                         "saturation": 50, "luminance": 10}]}
    full = auto.sanitize_ai_params(base, 1.0)
    check("AI 强度:1.0 与纯钳制等价", full["contrast"] == 60.0
          and full["wheels"]["shadows"]["saturation"] == 40.0)
    half = auto.sanitize_ai_params(base, 0.5)
    check("AI 强度:标量按比例缩小", half["contrast"] == 30.0
          and half["exposure"] == 1.0 and half["blackwhite"] == 50.0)
    check("AI 强度:曲线向恒等线收缩",
          abs(half["curves"]["master"][1][1] - 0.6) < 1e-6)
    check("AI 强度:色轮/取色器只缩幅度不改色相",
          half["wheels"]["shadows"]["saturation"] == 20.0
          and half["wheels"]["shadows"]["hue"] == 40.0
          and half["pickers"][0]["hue_shift"] == -20.0
          and half["pickers"][0]["hue"] == 120.0)
    soft = auto.sanitize_ai_params(base)      # 默认强度
    check("AI 强度:默认值 0.6", abs(soft["contrast"] - 36.0) < 1e-6
          and abs(auto.AI_DEFAULT_STRENGTH - 0.6) < 1e-9)
    zero = auto.sanitize_ai_params(base, 0.0)
    check("AI 强度:0 等于不做任何调整", zero == {})

    # json.loads 默认接受 NaN/Infinity 字面量，必须全部被钳制层丢弃
    nan_case = {"exposure": float("nan"), "contrast": float("inf"),
                "wheels": {"shadows": {"hue": float("nan"), "saturation": 30}},
                "pickers": [{"hue": float("inf"), "span": 30, "saturation": -40}]}
    check("AI 解析:NaN/Infinity 全部被丢弃", auto.sanitize_ai_params(nan_case) == {})


def test_ai_mask():
    import auto

    # ---- 像素级掩码（mask_png）：编解码往返 / 渲染优先级 / 缓存键 / 重映射 ----
    prob = np.zeros((100, 150), np.float32)
    prob[20:80, 30:120] = np.linspace(0, 1, 90)[None, :]    # 软渐变（256 级）
    b64, pw, ph = core.encode_mask_png(prob)
    back = core.decode_mask_png(b64)
    check("AI 像素掩码:编解码往返（形状/软边界/量化<1/255）",
          back.shape == (100, 150) and abs(float(back.max()) - 1.0) < 0.01
          and len(np.unique(back)) >= 90
          and float(np.abs(back - np.round(prob * 255) / 255).max()) < 0.005)
    check("AI 像素掩码:空/坏输入安全",
          core.decode_mask_png("").size == 0
          and core.decode_mask_png("not-base64!!").size == 0)

    # 渲染：mask_png 优先于 grid（采样点取渐变内部，避免边界插值稀释）
    m = core.MaskState(kind="ai", feather=0,
                       grid=["1100"] * 4, grid_w=4, grid_h=4,
                       mask_png=b64, mask_w=pw, mask_h=ph)
    r = core.render_mask(m, (300, 450))
    check("AI 像素掩码:渲染 mask_png 优先（中值≈0.5）",
          r.shape == (300, 450) and abs(float(r[150, 226]) - 0.5) < 0.03
          and float(r[150, 96]) < 0.05 and float(r[150, 356]) > 0.95)
    m2 = core.MaskState(kind="ai", feather=0,
                        grid=["1100"] * 4, grid_w=4, grid_h=4)
    r2 = core.render_mask(m2, (300, 450))
    check("AI 像素掩码:空 mask_png 回退 grid",
          float(r2[:, :112].min()) > 0.9 and float(r2[:, 300:].max()) < 0.1)

    k0 = core.mask_cache_key(m, (300, 450))
    m.mask_png = core.encode_mask_png(1.0 - prob)[0]
    check("AI 像素掩码:缓存键随掩码内容变化",
          k0 != core.mask_cache_key(m, (300, 450)))

    # 几何重映射：裁掉右半（nx'=2nx）——新图右端对应原列 75（值 0.5）
    m3 = core.MaskState(kind="ai", feather=0, mask_png=b64, mask_w=pw, mask_h=ph)
    core.remap_mask_geometry([m3], lambda nx, ny: (nx * 2.0, ny), (450, 300), (225, 300))
    r3 = core.decode_mask_png(m3.mask_png)
    check("AI 像素掩码:裁剪右半后右端≈0.5/左端≈0",
          abs(float(r3[50, -1]) - 0.5) < 0.06 and float(r3[50, 0]) < 0.06)
    m4 = core.MaskState(kind="ai", feather=0, mask_png=b64, mask_w=pw, mask_h=ph)
    core.remap_masks_rot90([m4], 3, (450, 300))
    r4 = core.decode_mask_png(m4.mask_png)
    check("AI 像素掩码:90° 旋转后掩码有效",
          m4.mask_w == pw and m4.mask_h == ph and float(r4.max()) > 0.95)

    # ---- 网格 <-> 数组往返与容错 ----
    rows = ["1100", "0110", "0011"]
    arr = core.grid_to_array(rows, 4, 3)
    check("AI 网格:解析形状与内容",
          arr.shape == (3, 4) and arr[0, 0] == 1 and arr[0, 2] == 0)
    check("AI 网格:数组回编码无损", core.array_to_grid(arr) == rows)
    bad = core.grid_to_array(["1x0", "0"], 3, 2)     # 杂散字符/缺行补 0
    check("AI 网格:杂散字符与短行安全降级",
          core.array_to_grid(bad) == ["100", "000"])
    check("AI 网格:空网格安全", core.grid_to_array([], 0, 0).size == 0)

    # ---- 渲染 ----
    shape = (180, 240)
    m = core.MaskState(kind="ai", ai_target="sky", grid=rows, grid_w=4, grid_h=3,
                       feather=30)
    am = core.render_mask(m, shape)
    check("AI 蒙版:渲染尺寸与范围",
          am.shape == shape and 0 <= am.min() and am.max() <= 1.0)
    check("AI 蒙版:左上网 1 -> 左上区域接近 1", float(am[:40, :40].mean()) > 0.9)
    check("AI 蒙版:右上网 0 -> 右上区域接近 0", float(am[:40, -40:].mean()) < 0.05)
    check("AI 蒙版:反向互补",
          abs(float((core.render_mask(
              core.MaskState(kind="ai", grid=rows, grid_w=4, grid_h=3,
                             feather=0, invert=True), shape)
              + core.render_mask(core.MaskState(
                  kind="ai", grid=rows, grid_w=4, grid_h=3, feather=0),
                  shape)).max()) - 1.0) < 1e-5)
    check("AI 蒙版:空网格渲染全 0", float(core.render_mask(
        core.MaskState(kind="ai"), shape).max()) == 0.0)

    # ---- 缓存键 ----
    k0 = core.mask_cache_key(m, shape)
    m_diff = core.MaskState(kind="ai", grid=["1100", "0110", "0010"],
                            grid_w=4, grid_h=3, feather=30)
    check("AI 蒙版:缓存键随网格变化", k0 != core.mask_cache_key(m_diff, shape))
    check("AI 蒙版:缓存键随羽化变化",
          k0 != core.mask_cache_key(
              core.MaskState(kind="ai", grid=rows, grid_w=4, grid_h=3,
                             feather=31), shape))
    check("AI 蒙版:同网格同羽化键稳定", k0 == core.mask_cache_key(
        core.MaskState(kind="ai", grid=list(rows), grid_w=4, grid_h=3,
                       feather=30), shape))

    # ---- 进流水线（蒙版局部调整作用于网格覆盖区）----
    img = np.full(shape + (3,), 0.35, np.float32)
    m.adj = core.MaskAdjustments(exposure=1.0)
    out = core.apply_masked_adjustments(img, [m])
    check("AI 蒙版:局部调整作用于网格 1 区",
          float(out[10, 10].mean()) > float(out[10, -10].mean()))

    # ---- 几何同步 ----
    m2 = core.MaskState(kind="ai", grid=rows, grid_w=4, grid_h=3, feather=0)
    core.remap_mask_geometry([m2], lambda nx, ny: (nx * 2.0, ny), (240, 180), (120, 180))
    a2 = core.grid_to_array(m2.grid, 4, 3)
    check("AI 蒙版:裁剪右半后出界格丢弃、保留格重定位",
          a2[0, 1] == 1 and a2[1, 3] == 1 and a2[2, :].max() == 0)
    check("AI 蒙版:裁剪重映射无空洞（逆映射连续覆盖）",
          np.array_equal(a2[0], [1, 1, 1, 1]) and np.array_equal(a2[1], [0, 0, 1, 1]))
    m3 = core.MaskState(kind="ai", grid=rows, grid_w=4, grid_h=3, feather=0)
    core.remap_masks_rot90([m3], 3, (240, 180))
    before = core.grid_to_array(rows, 4, 3).sum()
    after = core.grid_to_array(m3.grid, m3.grid_w, m3.grid_h).sum()
    check("AI 蒙版:90° 旋转网格重排（1 的数量不减少）", after >= before - 1)
    check("AI 蒙版:90° 旋转网格零丢失且维度转置",
          int(after) == int(before) and (m3.grid_w, m3.grid_h) == (3, 4))

    # ---- auto 层：配置解析 ----

    import os
    envs = ("PHOTOLAB_AI_BASE_URL", "PHOTOLAB_AI_KEY", "PHOTOLAB_AI_MODEL")
    saved = {k: os.environ.get(k) for k in envs}
    try:
        for k in envs:
            os.environ.pop(k, None)
        check("AI 配置:未配置返回 None", auto.resolve_config() is None)
        check("AI 配置:UI 传入优先",
              auto.resolve_config({"base_url": "http://x/v1/", "api_key": "k",
                                   "model": "m"}) == ("http://x/v1", "k", "m"))
        os.environ.update({"PHOTOLAB_AI_BASE_URL": "http://env", "PHOTOLAB_AI_KEY": "ek",
                           "PHOTOLAB_AI_MODEL": "em"})
        check("AI 配置:环境变量后备", auto.resolve_config(None) == ("http://env", "ek", "em"))
        check("AI 配置:UI 传入覆盖环境变量",
              auto.resolve_config({"base_url": "http://x", "api_key": "k",
                                   "model": "m"}) == ("http://x", "k", "m"))
        check("AI 配置:旧入口等价", auto.resolve_candidates() == [("http://env", "ek", "em")])
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_mask_cache():
    """蒙版跨任务 LRU：几何不变时第二次渲染必须命中缓存且结果逐位一致。"""
    core.clear_mask_cache()
    shape = (600, 800)
    masks = [
        core.MaskState(kind="ai", ai_target="sky",
                       grid=["11001100"] * 9, grid_w=8, grid_h=9, feather=30),
        core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=25),
        core.MaskState(kind="linear", x1=0.1, y1=0.2, x2=0.9, y2=0.8, feather=20),
    ]
    for m in masks:                        # 局部调整非默认 -> 非平凡，才会渲染
        m.adj = core.MaskAdjustments(exposure=0.3)
    r1 = core.MaskRenderer(masks, shape)
    core.mask_cache_stats["hit"] = core.mask_cache_stats["miss"] = 0
    r2 = core.MaskRenderer(masks, shape)          # 参数变化 -> 新任务，蒙版未变
    check("蒙版缓存:第二次任务全部命中", core.mask_cache_stats["hit"] == 3
          and core.mask_cache_stats["miss"] == 0)
    for i in range(3):
        a = r1.get(i, 0, 40, 800)
        b = r2.get(i, 0, 40, 800)
        check(f"蒙版缓存:蒙版{i} 命中结果与首次逐位一致", np.array_equal(a, b))

    # 几何变化 -> 键变化 -> 不命中旧值
    masks[1].radius = 0.4
    core.MaskRenderer(masks, shape)
    check("蒙版缓存:几何变化触发重算", core.mask_cache_stats["miss"] == 1)

    # 缓存淘汰不破坏正确性：塞满后重渲染首个蒙版仍逐位一致
    for k in range(core._MASK_CACHE_MAX + 2):
        m = core.MaskState(kind="radial", cx=0.1 * (k % 10), cy=0.5, radius=0.3)
        m.adj = core.MaskAdjustments(exposure=0.3)
        core.MaskRenderer([m], shape)
    r3 = core.MaskRenderer(masks[:1], shape)
    check("蒙版缓存:淘汰后重算结果一致",
          np.array_equal(r3.get(0, 0, 40, 800), r1.get(0, 0, 40, 800)))
    core.clear_mask_cache()


def test_gpu_matches_cpu():
    """有 OpenCL GPU 时，GPU 结果必须与 CPU 接近（浮点实现允许微小舍入差）。"""
    try:
        import gpu
    except Exception:
        gpu = None
    if gpu is None or not gpu.available():
        print("  SKIP  （无可用 OpenCL GPU）")
        return
    print(f"  （设备：{gpu.device_name()}）")
    rng = np.random.default_rng(13)
    rgb = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    lin = core.rgb8_to_linear(rgb)

    p = core.Adjustments()
    p.temperature = 40
    p.exposure = 0.5
    p.contrast = 35
    p.highlights = -20
    p.shadows = 25
    p.whites = 15
    p.blacks = -10
    p.saturation = 20
    p.vibrance = 30
    p.curves["master"] = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
    p.wheels["shadows"] = core.WheelState(200, 50, 15)
    p.wheels["highlights"] = core.WheelState(45, -30, -10)
    g = core.process_linear(p, lin)          # 应自动走 GPU
    s = core._process_band(p, lin)           # 单线程 CPU
    d = int(np.abs(g.astype(int) - s.astype(int)).max())
    # 容差 3/255：sRGB 编码表已与 CPU 共用同一张 16 位量化表，残余差异只来自
    # float32 与 numpy 的末位舍入在曲线/色轮查表边界上偶尔跨一级（实测 7.7 万像素中 1 个）
    check(f"GPU 与 CPU 结果一致（最大差 {d}/255）", d <= 3)

    p2 = core.Adjustments(blackwhite=100)
    g2 = core.process_linear(p2, lin)
    s2 = core._process_band(p2, lin)
    d2 = int(np.abs(g2.astype(int) - s2.astype(int)).max())
    check(f"黑白模式 GPU 与 CPU 一致（最大差 {d2}/255）", d2 <= 2)
    check("GPU 路径输出为灰度",
          float(np.abs(g2[..., 0].astype(int) - g2[..., 1]).max()) <= 2)

    # 混合路径：基础影调走 GPU（float32 输出），效果/曲线/分级仍留 CPU
    p3 = core.Adjustments(temperature=20, tint=5, exposure=0.3, contrast=20,
                          highlights=-30, shadows=30, whites=10, blacks=-10,
                          saturation=15, vibrance=12, clarity=25, sharpen=20,
                          dehaze=10)
    g3 = core.process_linear(p3, lin)
    was = core._GPU_ENABLED
    core._GPU_ENABLED = False
    try:
        s3 = core.process_linear(p3, lin)
    finally:
        core._GPU_ENABLED = was
    d3 = int(np.abs(g3.astype(int) - s3.astype(int)).max())
    check(f"GPU 基础影调 + CPU 效果与纯 CPU 一致（最大差 {d3}/255）", d3 <= 3)


def test_mask_advanced():
    """蒙版内的曲线 / 色轮 / 取色器局部调整。"""
    img = np.full((64, 64, 3), 0.5, np.float32)
    m = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.4, feather=5)
    m.adj.curves["master"] = [(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)]
    out = core.apply_masked_adjustments(img, [m])
    check("蒙版内曲线压暗中间调", out[32, 32].mean() < 0.4)
    check("蒙版外曲线不生效", abs(float(out[2, 2].mean()) - 0.5) < 1e-5)
    check("仅曲线的蒙版非默认", not m.is_trivial())

    img2 = np.full((64, 64, 3), 0.3, np.float32)
    m2 = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.4, feather=5)
    m2.adj.wheels["shadows"] = core.WheelState(hue=0.0, saturation=80.0)
    out2 = core.apply_masked_adjustments(img2, [m2])
    check("蒙版内色轮偏红", float(out2[32, 32, 0] - img2[32, 32, 0]) > 0.01
          and float(out2[32, 32, 0] - out2[32, 32, 2]) > 0.02)
    check("蒙版外色轮不生效", abs(float(out2[2, 2].mean()) - 0.3) < 1e-5)

    img3 = np.zeros((64, 64, 3), np.float32)
    img3[:, :32] = (0.9, 0.05, 0.05)
    # 线性蒙版从右向左：左半权重≈1（起点的值为 0，终点的值为 1）
    m3 = core.MaskState(kind="linear", x1=1.0, y1=0.5, x2=0.0, y2=0.5, feather=5)
    m3.adj.pickers = [core.PickerState(hue=0.0, span=60.0, saturation=-100.0)]
    out3 = core.apply_masked_adjustments(img3, [m3])
    check("蒙版内取色器去饱和", float(np.abs(out3[32, 8, 0] - out3[32, 8, 1]).mean()) <
          float(np.abs(img3[32, 8, 0] - img3[32, 8, 1]).mean()) * 0.5)
    check("蒙版外取色器不生效",
          float(np.abs(out3[32, 60] - img3[32, 60]).max()) < 1e-4)
    p = core.Adjustments()
    p.masks = [m]
    check("含曲线蒙版使流水线生效", not p.is_default())


def test_patch_copy():
    """矩形取样覆盖：内容复制、缩放、羽化、越界与源目标重叠安全。"""
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = 200
    core.copy_patch(img, (4, 4, 28, 28), (36, 36, 60, 60), 0.0)
    check("矩形取样复制内容", img[48, 48, 0] == 200)
    check("目标外不受影响", img[2, 60, 0] == 0)
    check("无羽化时目标边角完全覆盖",
          img[36, 36, 0] == 200 and img[59, 59, 0] == 200)

    img2 = np.zeros((50, 50, 3), np.uint8)
    img2[10:30, 10:30] = 255                       # 大源块
    core.copy_patch(img2, (10, 10, 30, 30), (5, 5, 15, 15), 0.0)   # 缩小覆盖
    check("覆盖自动缩放到目标尺寸", img2[6, 6, 0] == 255 and img2[14, 14, 0] == 255)

    img3 = np.full((40, 40, 3), 128, np.uint8)
    core.copy_patch(img3, (5, 5, 20, 20), (5, 5, 20, 20), 0.3)     # 源=目标
    check("源目标重叠安全", bool(np.all(np.isfinite(img3))) and float(img3.mean()) > 0)

    img4 = np.zeros((40, 40, 3), np.uint8)
    img4[:20] = 200
    core.copy_patch(img4, (0, 0, 40, 20), (30, 30, 60, 60), 0.0)   # 目标越界
    check("目标越界自动裁剪不崩溃", img4[39, 39, 0] == 200)

    img5 = np.zeros((40, 40, 3), np.uint8)
    img5[:20] = 200
    core.copy_patch(img5, (0, 0, 40, 20), (0, 20, 40, 40), 0.5)    # 羽化边缘
    check("羽化边缘平滑过渡", 0 < img5[20, 20, 0] < 200
          and img5[33, 33, 0] == 200)


def test_contrast_strength():
    """对比度必须有可见效果（修复"滑块似乎不工作"）。"""
    xs = np.array([0.1, 0.25, 0.5, 0.75, 0.9], np.float32).reshape(1, -1, 1).repeat(3, 2)
    hi = core.apply_contrast(xs, 100)
    lo = core.apply_contrast(xs, -100)
    mid = core.apply_contrast(xs, 50)
    check("对比度 +100 明显压暗暗部", float(hi[0, 0, 0]) < 0.02)
    check("对比度 +100 明显提亮亮部", float(hi[0, 4, 0]) > 0.95)
    check("对比度 +100 枢轴不变", abs(float(hi[0, 2, 0]) - 0.5) < 1e-6)
    check("对比度 -100 向中间收缩", float(lo[0, 0, 0]) > 0.12 and float(lo[0, 4, 0]) < 0.88)
    check("对比度 50 介于恒等与 100 之间",
          float(mid[0, 0, 0]) > float(hi[0, 0, 0])
          and float(mid[0, 4, 0]) < float(hi[0, 4, 0]))
    check("对比度单调（暗->亮递增）",
          bool(np.all(np.diff(hi[0, :, 0]) > 0)) and bool(np.all(np.diff(lo[0, :, 0]) > 0)))
    # 真实感图像上的标准差提升
    rng = np.random.default_rng(3)
    rgb = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
    base = core.process(core.Adjustments(), rgb)
    strong = core.process(core.Adjustments(contrast=100), rgb)
    check("对比度 +100 显著提升对比（std +25% 以上）",
          float(strong.std()) > float(base.std()) * 1.25)


def test_dehaze_no_blackout():
    """去朦胧不得把画面压黑（修复"调节导致画面变黑"）。"""
    h, w = 600, 800
    img = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    img[..., 0] = 0.55 + 0.1 * (xx / w)
    img[..., 1] = 0.62 + 0.05 * (yy / h)
    img[..., 2] = 0.75 - 0.08 * (yy / h)
    img[400:, :] = (0.10, 0.18, 0.08)
    rgb = (img * 255).astype(np.uint8)
    base = core.process(core.Adjustments(), rgb)

    for v in (25, 50, 75, 100):
        out = core.process(core.Adjustments(dehaze=v), rgb)
        black = 100.0 * float((out.mean(axis=-1) < 6).mean())
        check(f"去朦胧 +{v} 不全黑（黑像素 {black:.1f}%）", black < 3.0)
        check(f"去朦胧 +{v} 输出范围合法", int(out.min()) >= 0 and int(out.max()) <= 255)
    check("去朦胧 +100 提升对比", float(core.process(core.Adjustments(dehaze=100), rgb).std())
          > float(base.std()))
    for v in (-50, -100):
        out = core.process(core.Adjustments(dehaze=v), rgb)
        check(f"去朦胧 {v}（加雾）降低对比", float(out.std()) < float(base.std()))


def test_geometry():
    """裁剪 / 旋转 / 拉直：几何变换与蒙版坐标同步。"""
    import cv2
    rng = np.random.default_rng(21)
    rgb = (rng.random((200, 300, 3)) * 255).astype(np.uint8)

    def make_masks():
        return [
            core.MaskState(kind="linear", x1=0.2, y1=0.8, x2=0.7, y2=0.3,
                           feather=25),
            core.MaskState(kind="radial", cx=0.4, cy=0.6, radius=0.2, aspect=1.6,
                           angle=30.0, feather=20),
            core.MaskState(kind="brush", size=8, feather=15,
                           strokes=[[(0.3, 0.3), (0.5, 0.45)], [(0.7, 0.8)]]),
        ]

    # ---- 90° 旋转：无损 + 蒙版同步（与光栅旋转对比；容差涵盖半像素/羽化网格差异）
    r90 = core.rotate90(rgb, 1)
    check("rot90 形状互换", r90.shape == (300, 200, 3))
    check("rot90 四次还原", np.array_equal(core.rotate90(rgb, 4), rgb))
    for k in (1, 2, 3):
        ms = make_masks()
        r0 = [core.render_mask(m, rgb.shape[:2]) for m in ms]
        ns = core.remap_masks_rot90(ms, k, rgb.shape[:2])
        diffs = [float(np.max(np.abs(np.rot90(r, k) - core.render_mask(m, ns))))
                 for r, m in zip(r0, ms)]
        check(f"rot90 k={k} 蒙版同步（最大差 {max(diffs):.3f}）", max(diffs) < 0.2)

    # ---- 裁剪：像素切片 + 蒙版同步
    out, map_point, ns = core.crop_rotate(rgb, (0.1, 0.2, 0.6, 0.9), 0.0)
    check("裁剪尺寸与内容正确", out.shape == (140, 150, 3)
          and np.array_equal(out, rgb[40:180, 30:180]))
    ms = make_masks()
    r0 = [core.render_mask(m, rgb.shape[:2]) for m in ms]
    core.remap_mask_geometry(ms, map_point, rgb.shape[:2], ns)
    diffs = [float(np.max(np.abs(r0[i][40:180, 30:180]
                                 - core.render_mask(ms[i], ns))))
             for i in range(3)]
    check(f"裁剪蒙版同步（最大差 {max(diffs):.3f}）", max(diffs) < 0.2)

    # ---- 拉直：画布不变、点映射与矩阵一致（最近邻 + 单点验证）
    dot = np.zeros((200, 300, 3), np.uint8)
    dot[60, 100] = 255
    rot10, m10 = core.rotate_image(dot, 10.0, interpolation=cv2.INTER_NEAREST)
    exp = m10 @ np.array([100.0, 60.0, 1.0])
    ys, xs = np.nonzero(rot10[..., 0])
    check("拉直画布尺寸不变", rot10.shape == dot.shape)
    check("拉直点映射与矩阵一致",
          len(xs) == 1 and abs(xs[0] - exp[0]) <= 1.0 and abs(ys[0] - exp[1]) <= 1.0)

    # ---- 拉直 + 蒙版同步（仅在旋转有效区内对比）
    a = 12.0
    out, map_point, ns = core.crop_rotate(rgb, (0, 0, 1, 1), a)
    check("拉直后画布不变", ns == (200, 300))
    ms = make_masks()
    r0 = [core.render_mask(m, rgb.shape[:2]) for m in ms]
    core.remap_mask_geometry(ms, map_point, rgb.shape[:2], ns, angle_delta=-a)
    x0, y0, x1, y1 = core.fit_rect_in_rotation((0, 0, 300, 200), (200, 300), a)
    check("最大有效区在画布内", 0 <= x0 < x1 <= 300 and 0 <= y0 < y1 <= 200)
    ma = core.rotation_matrix(rgb.shape[:2], a)
    raster = [cv2.warpAffine(r, ma, (300, 200), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
              for r in r0]
    diffs = [float(np.max(np.abs(raster[i][y0:y1, x0:x1]
                                 - core.render_mask(ms[i], ns)[y0:y1, x0:x1])))
             for i in range(3)]
    check(f"拉直蒙版同步（有效区内最大差 {max(diffs):.3f}）", max(diffs) < 0.25)

    # ---- 有效区收缩：角度越大越小
    r5 = core.fit_rect_in_rotation((0, 0, 300, 200), (200, 300), 5.0)
    r30 = core.fit_rect_in_rotation((0, 0, 300, 200), (200, 300), 30.0)
    area5 = (r5[2] - r5[0]) * (r5[3] - r5[1])
    area30 = (r30[2] - r30[0]) * (r30[3] - r30[1])
    check("拉直角度越大有效区越小", area5 > area30 > 0)


def test_radial_shape():
    """径向蒙版的形状（长宽比）与旋转。"""
    shape = (400, 600)
    m0 = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25, feather=0)
    r0 = core.render_mask(m0, shape)
    check("径向默认正圆（半径 0.25 短边=100px）",
          float(r0[200, 300]) > 0.99 and float(r0[200, 100]) < 0.01)
    # 拉长：aspect=2 时水平方向应覆盖更远
    mx = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25, aspect=2.0, feather=0)
    rx = core.render_mask(mx, shape)
    check("长宽比 2 时水平覆盖更远", float(rx[200, 130]) > 0.5 and float(r0[200, 130]) < 0.01)
    check("长宽比 2 时垂直覆盖不变", float(rx[260, 300]) < 0.01)
    # 旋转 90° 后长轴变垂直
    mrot = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25, aspect=2.0,
                          feather=0, angle=90.0)
    rr = core.render_mask(mrot, shape)
    check("旋转 90° 后长轴变垂直", float(rr[280, 300]) > 0.5 and float(rx[280, 300]) < 0.01)
    check("旋转 45° 覆盖对角", float(core.render_mask(
        core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25, aspect=2.0,
                       feather=0, angle=45.0), shape)[280, 380]) > 0.5)
    check("旋转不影响中心", float(rr[200, 300]) > 0.99)
    check("羽化 0 无 NaN", bool(np.all(np.isfinite(rx))))
    check("旋转角度参与缓存键",
          core.mask_cache_key(mx, shape) != core.mask_cache_key(mrot, shape))
def test_blackwhite():
    img = np.zeros((32, 32, 3), np.float32)
    img[:16] = (0.8, 0.2, 0.2)
    img[16:] = (0.2, 0.6, 0.9)
    check("黑白 0 为恒等", np.array_equal(core.apply_blackwhite(img, 0), img))
    gray = core.apply_blackwhite(img, 100)
    check("黑白 100 三通道相等", float(np.abs(gray[..., 0] - gray[..., 1]).max()) < 1e-6
          and float(np.abs(gray[..., 1] - gray[..., 2]).max()) < 1e-6)
    lum = core.luminance(img)
    check("黑白 100 等于亮度", float(np.abs(gray[..., 0] - lum).max()) < 1e-5)
    half = core.apply_blackwhite(img, 50)
    check("黑白 50 介于彩色与灰度之间",
          float(np.abs(half[..., 0] - half[..., 1]).max()) > 1e-6
          and float(np.abs(half[..., 0] - half[..., 1]).max())
          < float(np.abs(img[..., 0] - img[..., 1]).max()))
    # 流水线：黑白参数参与判定与渲染
    p = core.Adjustments(blackwhite=100)
    check("黑白参数使流水线生效", not p.is_default())
    rgb = (core.to_srgb(img) * 255).astype(np.uint8)
    out = core.process(p, rgb)
    check("黑白流水线输出灰度",
          float(np.abs(out[..., 0].astype(int) - out[..., 1]).max()) <= 1)


def test_pipeline():
    rng = np.random.default_rng(3)
    rgb = (rng.random((200, 320, 3)) * 255).astype(np.uint8)

    p = core.Adjustments()
    check("默认参数 -> 原图直出", np.array_equal(core.process(p, rgb), rgb))

    p.exposure = 0.5
    p.clarity = 30
    p.sharpen = 20
    p.dehaze = 25
    p.curves["master"] = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
    p.wheels["shadows"] = core.WheelState(hue=210, saturation=60, luminance=10)
    p.wheels["highlights"] = core.WheelState(hue=40, saturation=30, luminance=-10)
    p.blend = 60
    p.pickers.append(core.PickerState(hue=0.0, span=40.0, saturation=30.0))
    p.masks.append(core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=25))
    p.masks[0].adj.exposure = -0.5
    out = core.process(p, rgb)
    check("流水线输出 shape/dtype 正确", out.shape == rgb.shape and out.dtype == np.uint8)
    check("流水线输出范围合法", int(out.min()) >= 0 and int(out.max()) <= 255)
    check("流水线两次结果一致", np.array_equal(out, core.process(p, rgb)))
    check("降采样不放大", core.downscale_max_edge(rgb, 1200).shape[0] <= core.PREVIEW_MAX_EDGE)
    check("小图不降采样", core.downscale_max_edge(rgb, 4000) is rgb)

    prev = core.process(p, core.downscale_max_edge(rgb, 1200))
    check("预览/全图尺寸比例正确", abs(rgb.shape[0] / prev.shape[0] - 200 / prev.shape[0]) < 0.01)
    check("halo 计算合理", core.mask_halo(p, 3000, 2000) >= 64
          and core.mask_halo(core.Adjustments(), 3000, 2000) == 0)


def test_parallel_equals_serial():
    """大图走多线程分块路径，结果必须与单线程完全一致。"""
    rng = np.random.default_rng(11)
    rgb = (rng.random((1600, 3000, 3)) * 255).astype(np.uint8)   # 4.8MP，触发多线程
    p = core.Adjustments()
    p.exposure = 0.4
    p.contrast = 25
    p.highlights = -15
    p.shadows = 20
    p.temperature = 50
    p.saturation = 10
    p.vibrance = 30
    p.curves["master"] = [(0.0, 0.0), (0.3, 0.4), (0.7, 0.65), (1.0, 1.0)]
    p.wheels["shadows"] = core.WheelState(210, 70, 20)
    p.wheels["highlights"] = core.WheelState(45, -40, -15)
    p.blend = 65
    p.pickers.append(core.PickerState(0.0, (0.8, 0.2, 0.2), 45, 40, -30))
    p.pickers.append(core.PickerState(120.0, (0.2, 0.8, 0.2), 35, 0, 25))

    lin = core.rgb8_to_linear(rgb)
    par = core.process_linear(p, lin)
    ser = core._process_band(p, lin)
    check("多线程结果与单线程逐位一致", np.array_equal(par, ser))
    check("多线程输出尺寸/类型正确", par.shape == rgb.shape and par.dtype == np.uint8)

    # 蒙版 + 效果一起并行，验证低分辨率蒙版按行切片不产生接缝
    p2 = p.copy()
    p2.clarity = 30
    p2.sharpen = 20
    p2.dehaze = 15
    rm = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=25)
    rm.adj.exposure = -0.5
    bm = core.MaskState(kind="brush", size=12, feather=30,
                        strokes=[[(0.2, 0.2), (0.4, 0.35), (0.6, 0.2)],
                                 [(0.7, 0.8), (0.85, 0.9)]])
    bm.adj.shadows = 35
    lm = core.MaskState(kind="linear", x1=0.15, y1=0.9, x2=0.85, y2=0.15,
                        feather=35, invert=True)
    lm.adj.temperature = 25
    lm.adj.pickers = [core.PickerState(hue=210.0, span=50.0, saturation=-40)]
    rm.adj.curves["master"] = [(0.0, 0.0), (0.5, 0.65), (1.0, 1.0)]
    rm.adj.wheels["highlights"] = core.WheelState(40, 30, -10)
    p2.masks = [rm, bm, lm]
    par2 = core.process_linear(p2, lin)
    ser2 = core._process_band(p2, lin)
    check("蒙版+效果并行结果逐位一致（无接缝）", np.array_equal(par2, ser2))
    check("蒙版流水线输出范围合法", int(par2.min()) >= 0 and int(par2.max()) <= 255)
    check("蒙版流水线确定", np.array_equal(par2, core.process_linear(p2, lin)))
    check("蒙版渲染器切片有效",
          core.MaskRenderer(p2.masks, rgb.shape[:2]).get(1, 800, 1000, rgb.shape[1]).shape
          == (200, rgb.shape[1]))

    # 仅蒙版局部效果（全局效果全 0）：halo / 低频图 / 全局统计必须同样覆盖蒙版 adj，
    # 否则分块各自为政产生接缝甚至整体不一致
    p3 = core.Adjustments()
    mm = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.5, feather=10)
    mm.adj.clarity = 60
    mm.adj.sharpen = 80
    mm.adj.denoise = 100
    mm.adj.dehaze = 40
    p3.masks = [mm]
    rgb3 = (np.random.default_rng(3).random((900, 700, 3)) * 255).astype(np.uint8)
    lin3 = core.rgb8_to_linear(rgb3)
    par3 = core.process_linear(p3, lin3)
    ser3 = core._process_band(p3, lin3)
    check("仅蒙版局部效果并行逐位一致（无接缝）", np.array_equal(par3, ser3))

    # 大图去朦胧：暗通道高斯半径 4σ 超过固定 64 后，halo 必须按 4σ+1 动态扩展
    p4 = core.Adjustments()
    p4.dehaze = 60
    rgb4 = (np.random.default_rng(4).random((3900, 4100, 3)) * 255).astype(np.uint8)
    lin4 = core.rgb8_to_linear(rgb4)
    par4 = core.process_linear(p4, lin4)
    ser4 = core._process_band(p4, lin4)
    check("大图去朦胧并行逐位一致（动态 halo）", np.array_equal(par4, ser4))


# ======================================================================
# 二、UI 冒烟（--ui）
# ======================================================================

def run_ui_smoke() -> int:
    import tempfile

    import cv2
    from PySide6.QtWidgets import QApplication

    import core
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
    import ui_window
    from ui_window import MainWindow
    from ui_widgets import apply_dark_theme

    IMG_W, IMG_H = 2000, 3000
    RED_CENTER = (1000, 1500)
    GREEN_CENTER = (400, 600)

    def make_test_image(path: str):
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
        img = np.empty((IMG_H, IMG_W, 3), np.uint8)
        img[..., 0] = (xx * 255 // IMG_W).astype(np.uint8)
        img[..., 1] = (yy * 255 // IMG_H).astype(np.uint8)
        img[..., 2] = 90
        img[1400:1600, 900:1100] = (200, 30, 30)     # 红块
        img[450:750, 250:550] = (30, 200, 40)        # 绿块
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def image_to_widget(window, px, py):
        prev = window.preview
        off = prev._offset()
        scale = prev._scale()
        ratio = prev.current_frame_size()[0] / IMG_W
        return int(off.x() + px * ratio * scale), int(off.y() + py * ratio * scale)

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    window.resize(1460, 920)
    app.processEvents()

    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.png")
        make_test_image(img_path)
        window.load_image(img_path)

        check("图像已加载", window.source is not None
              and window.source.shape[:2] == (IMG_H, IMG_W))
        check("预览控件有帧", window.preview._pixmap is not None)
        check("加载后渲染全图", window._last_frame.shape[:2] == (IMG_H, IMG_W))

        # ---- 基础 ----
        row = window.panel.basic.rows["exposure"]
        before = window._last_frame.astype(np.int32).mean()
        row.slider.setValue(int(1.0 * row._mult))          # 模拟拖动滑块
        check("曝光滑块写入参数", window.params.exposure == 1.0)
        check("拖动时处理降采样图(预览)", window._last_frame.shape[0] == 1200)
        check("曝光 +1EV 画面变亮", window._last_frame.astype(np.int32).mean() > before)
        window._on_params_committed()
        check("松手后处理全图", window._last_frame.shape[:2] == (IMG_H, IMG_W))

        for key, val in (("contrast", 50), ("highlights", -20), ("shadows", 15),
                         ("temperature", 60), ("tint", -10), ("saturation", 30),
                         ("vibrance", 20)):
            window.panel.basic.rows[key].slider.setValue(val)
        check("基础参数全部生效", window.params.contrast == 50
              and window.params.temperature == 60 and window.params.vibrance == 20)

        # ---- 曲线 ----
        window.panel.curve._on_points_changed("master", [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)])
        check("主通道曲线参数已写入", len(window.params.curves["master"]) == 3)
        window.panel.curve._on_points_committed("green", [(0.0, 0.1), (1.0, 0.9)])
        check("绿色通道曲线参数已写入", window.params.curves["green"][0] == (0.0, 0.1))
        window.panel.curve._apply_reset("all")
        check("曲线可重置", all(len(v) == 2 for v in window.params.curves.values()))

        # ---- 颜色分级 ----
        window.panel.grading._on_wheel("shadows", 200.0, 0.7)
        check("色轮参数已写入", abs(window.params.wheels["shadows"].saturation - 70) < 1e-3)
        window.panel.grading._on_lum("highlights", -20)
        window.panel.grading._on_sat("global", -25)
        window.panel.grading._on_blend(70)
        check("混合滑块已写入", window.params.blend == 70)
        frame_graded = window._last_frame.copy()

        # ---- 取色器 ----
        wx, wy = image_to_widget(window, *RED_CENTER)
        window._on_image_picked(wx, wy)
        check("图像点击生成取色点", len(window.params.pickers) == 1)
        p = window.params.pickers[0]
        check("取色点色相≈红色", min(p.hue, 360 - p.hue) < 25.0)
        check("取色点采样颜色为红", p.color[0] > 0.6 and p.color[1] < 0.3)
        p.hue_shift = 120
        p.saturation = -50
        window._on_params_changed(force_full=True)
        frame_picked = window._last_frame.copy()
        check("取色器调整改变画面", not np.array_equal(frame_picked, frame_graded))

        gx, gy = image_to_widget(window, *GREEN_CENTER)
        window._on_image_picked(gx, gy)
        check("支持多个取色点", len(window.params.pickers) == 2)
        window.panel.picker.delete_picker(1)
        check("可删除取色点", len(window.params.pickers) == 1)

        # ---- 新功能：预设颜色 / 效果调节 ----
        window.panel.picker.new_preset(120.0)                     # 预设"绿"
        check("预设颜色创建取色点", len(window.params.pickers) == 2
              and abs(window.params.pickers[1].hue - 120.0) < 1e-6)
        window.params.pickers[1].hue_shift = 45            # 让绿偏黄
        window.params.pickers[1].saturation = -30
        window._on_params_changed(force_full=True)         # 提交渲染后再比较
        check("预设色可单独调色相", window.params.pickers[1].hue_shift == 45)
        frame_preset = window._last_frame.copy()
        check("预设取色点影响画面", not np.array_equal(frame_preset, frame_picked))

        window.panel.basic.rows["clarity"].slider.setValue(35)
        window.panel.basic.rows["sharpen"].slider.setValue(25)
        window.panel.basic.rows["dehaze"].slider.setValue(20)
        check("效果参数生效", window.params.clarity == 35
              and window.params.sharpen == 25 and window.params.dehaze == 20)
        window._on_params_committed()
        frame_effects = window._last_frame.copy()
        check("效果调节改变画面", not np.array_equal(frame_effects, frame_preset))
        window.panel.basic.rows["denoise"].slider.setValue(45)
        check("降噪参数生效", window.params.denoise == 45)
        window.panel.basic.rows["denoise"].slider.setValue(0)
        for k in ("clarity", "sharpen", "dehaze"):
            window.panel.basic.rows[k].slider.setValue(0)

        # ---- 蒙版 ----
        wx, wy = image_to_widget(window, *RED_CENTER)
        window._on_image_picked(wx, wy)
        check("恢复图像取色", len(window.params.pickers) == 3)

        window.panel.mask.create_mask("radial")
        check("创建径向蒙版", len(window.params.masks) == 1
              and window.params.masks[0].kind == "radial")
        check("创建后自动进入编辑", window.preview.edit_mode() == "radial"
              and window._mask_edit == 0)

        # 拖动蒙版中心（在画布上模拟）
        window.preview.radialCenterMoved.emit(0.4, 0.6)
        check("径向蒙版中心可拖", abs(window.params.masks[0].cx - 0.4) < 1e-6)
        # 半径/长宽比/反向
        window.panel.mask._on_radius(45.0)
        window.panel.mask._on_aspect(150.0)
        check("径向蒙版半径/长宽比", abs(window.params.masks[0].radius - 0.45) < 1e-6
              and abs(window.params.masks[0].aspect - 1.5) < 1e-6)
        window.panel.mask._set_attr("invert", True)
        check("蒙版可反向", window.params.masks[0].invert is True)
        window.panel.mask._set_attr("invert", False)
        # 局部调整
        window.panel.mask._on_adj("exposure", -1.0)
        check("蒙版局部调整生效", window.params.masks[0].adj.exposure == -1.0)
        window._on_params_committed()
        frame_mask = window._last_frame.copy()
        check("蒙版局部调整改变画面", not np.array_equal(frame_mask, frame_effects))
        check("蒙版叠加层已生成", len(window.preview._overlays) == 1)
        check("蒙版缓存复用", len(window._overlay_cache) >= 1)

        # ESC 退出编辑
        window._exit_mask_edit()
        check("Esc 退出蒙版编辑", window.preview.edit_mode() == "none")

        # 画笔蒙版 + 笔画 + 撤销/清除
        window.panel.mask.create_mask("brush")
        window._on_brush_point(0.45, 0.45)
        window._on_brush_point(0.55, 0.55)
        window._on_brush_finished()
        window._on_brush_point(0.2, 0.2)
        window._on_brush_finished()
        check("画笔蒙版记录笔画", len(window.params.masks[1].strokes) == 2
              and len(window.params.masks[1].strokes[0]) == 2)
        window.panel.mask._on_adj("shadows", 30)
        window._on_params_committed()
        check("画笔蒙版局部调整改变画面",
              not np.array_equal(window._last_frame, frame_mask))
        window._on_brush_undo()
        check("撤销一笔", len(window.params.masks[1].strokes) == 1)
        window._on_brush_cleared()
        check("清除笔迹", len(window.params.masks[1].strokes) == 0)

        # 线性渐变蒙版（从画布拖手柄）
        window.panel.mask.create_mask("linear")
        window.preview.linearHandleMoved.emit(0, 0.15, 0.8, 0.85, 0.3)
        check("线性蒙版端点可拖", abs(window.params.masks[2].x1 - 0.15) < 1e-6
              and abs(window.params.masks[2].y2 - 0.3) < 1e-6)
        window.panel.mask.delete_mask(2)
        check("可删除蒙版", len(window.params.masks) == 2)

        # ---- 真实鼠标事件：画布上拖拽编辑蒙版 ----
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        def wxy(nx, ny):
            pt = window.preview.frame_point_to_widget(nx, ny)
            return QPoint(int(pt.x()), int(pt.y()))

        # 画笔：按住并拖动应记录一笔（两个点）
        window._exit_mask_edit()
        window.panel.mask.create_mask("brush")
        check("画笔模式已进入", window.preview.edit_mode() == "brush")
        p0, p1 = wxy(0.45, 0.4), wxy(0.55, 0.5)
        QTest.mouseMove(window.preview, p0)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, p0)
        QTest.mouseMove(window.preview, p1)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, p1)
        brush = window.params.masks[2]
        check("鼠标拖动绘制一笔", len(brush.strokes) == 1 and len(brush.strokes[0]) == 2)
        check("笔迹坐标归一化", abs(brush.strokes[0][0][0] - 0.45) < 0.02
              and abs(brush.strokes[0][-1][1] - 0.5) < 0.02)

        # 线性渐变：拖动起点手柄
        window._exit_mask_edit()
        window.panel.mask.delete_mask(2)
        window.panel.mask.create_mask("linear")
        handle = wxy(window.params.masks[2].x1, window.params.masks[2].y1)
        target = wxy(0.2, 0.25)
        QTest.mouseMove(window.preview, handle)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, handle)
        QTest.mouseMove(window.preview, target)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, target)
        lin = window.params.masks[2]
        check("拖动线性手柄改变端点", abs(lin.x1 - 0.2) < 0.02 and abs(lin.y1 - 0.25) < 0.02)

        # 径向渐变：拖动中心
        window._exit_mask_edit()
        window.panel.mask.delete_mask(2)
        window.panel.mask.create_mask("radial")
        center = wxy(window.params.masks[2].cx, window.params.masks[2].cy)
        target = wxy(0.65, 0.35)
        QTest.mouseMove(window.preview, center)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, center)
        QTest.mouseMove(window.preview, target)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, target)
        rad = window.params.masks[2]
        check("拖动径向中心", abs(rad.cx - 0.65) < 0.02 and abs(rad.cy - 0.35) < 0.02)

        # 径向：拖动长轴手柄拉长椭圆（保持短轴）
        info = window.preview.mask_handles_info()
        hx = wxy(info["hx"][0], info["hx"][1])
        before = (rad.radius, rad.aspect)
        QTest.mouseMove(window.preview, hx)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, hx)
        mid = QPoint(int(hx.x() + 0.10 * info["hx"][0] * window.preview.width()),
                     int(hx.y()))
        QTest.mouseMove(window.preview, mid)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, mid)
        check("拖动长轴手柄拉长椭圆", rad.aspect > before[1] + 0.02)

        # 成比例手感：同样再拖一小段，缩放比例应与拖拽距离成比例（缓慢、可预期）
        info2 = window.preview.mask_handles_info()
        hx2 = wxy(info2["hx"][0], info2["hx"][1])
        radius_before = rad.radius
        QTest.mouseMove(window.preview, hx2)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, hx2)
        step = QPoint(int(hx2.x() + 6), int(hx2.y()))
        QTest.mouseMove(window.preview, step)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, step)
        # 6px 拖拽在 1:1 缩放下只应带来很小的半径变化
        check("小幅度拖拽只产生小变化（灵敏度合理）",
              abs(rad.radius - radius_before) < 0.08)

        # 径向：拖动旋转手柄改变角度
        info = window.preview.mask_handles_info()
        rot = wxy(info["rot"][0], info["rot"][1])
        QTest.mouseMove(window.preview, rot)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, rot)
        rot_to = wxy(0.2, 0.5)
        QTest.mouseMove(window.preview, rot_to)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, rot_to)
        check("拖动旋转手柄改变角度", abs(rad.angle) > 1.0)

        # 面板滑块：长宽比 / 旋转
        window.panel.mask._on_aspect(220.0)
        window.panel.mask._on_angle(45.0)
        check("面板可设长宽比与旋转", abs(rad.aspect - 2.2) < 1e-6 and rad.angle == 45.0)
        check("旋转后的蒙版渲染有效",
              core.render_mask(rad, window.source.shape[:2]).shape
              == window.source.shape[:2])

        # 编辑模式下左键不应平移画面
        pan_before = window.preview._pan
        QTest.mouseMove(window.preview, center)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, center)
        QTest.mouseMove(window.preview, QPoint(center.x() + 60, center.y() + 40))
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier,
                           QPoint(center.x() + 60, center.y() + 40))
        check("编辑模式不触发平移", window.preview._pan == pan_before)

        window._exit_mask_edit()
        for m in list(range(len(window.params.masks) - 1, -1, -1)):
            window.panel.mask.delete_mask(m)
        check("清理全部蒙版", not window.params.masks)

        # ---- 蒙版目标切换：选中蒙版时曲线/色轮/取色器编辑蒙版参数 ----
        window.panel.mask.create_mask("radial")
        m0 = window.params.masks[0]
        window.panel.curve._on_points_changed("master", [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)])
        check("选中蒙版时曲线写入蒙版参数", len(m0.adj.curves["master"]) == 3
              and len(window.params.curves["master"]) == 2)
        window.panel.grading._on_sat("shadows", 40)
        check("选中蒙版时色轮写入蒙版参数", m0.adj.wheels["shadows"].saturation == 40)
        window.panel.picker.new_preset(120.0)
        check("选中蒙版时取色点写入蒙版参数", len(m0.adj.pickers) == 1)
        window.panel.mask.delete_mask(0)
        check("删除蒙版后曲线面板回到全局",
              window.panel.curve._params is window.params)
        check("全局曲线未被污染", len(window.params.curves["master"]) == 2)

        # ---- 仿制图章（矩形取样覆盖）----
        window._toggle_stamp(True)
        check("图章模式进入", window.preview.stamp_mode())
        window.preview.cloneApplyRequested.emit((0.1, 0.1, 0.2, 0.2))
        check("无源矩形时安全拦截", window._retouch_full is None)
        window.preview.cloneSourcePicked.emit((0.25, 0.25, 0.45, 0.45))
        check("源矩形已记录", window._stamp_src_rect == (0.25, 0.25, 0.45, 0.45)
              and window.preview.stamp_source() == (0.25, 0.25, 0.45, 0.45))
        window.preview.cloneApplyRequested.emit((0.55, 0.55, 0.75, 0.75))
        check("覆盖修改像素", window._retouch_full is not None
              and not np.array_equal(window._retouch_full, window.source))
        check("覆盖记录一笔", len(window._clone_ops) == 1)
        check("目标内容来自源区域（内部逐位一致）",
              np.array_equal(window._retouch_full[1710:2190, 1160:1440],
                             window.source[810:1290, 560:840]))
        full_src = window.processor._sources.get("full")
        check("图章后全图源已切换（线程持有私有副本）",
              full_src is not window._retouch_full
              and np.array_equal(full_src, window._retouch_full))
        window._on_stamp_undo()
        check("图章撤销还原", window._retouch_full is None
              and np.array_equal(window.processor._sources.get("full"),
                                 window.source)
              and window.processor._sources.get("full") is not window.source)
        window._on_escape()
        check("Esc 退出图章模式", not window.preview.stamp_mode())

        # ---- 黑白模式（顶栏开关）----
        window.act_bw.trigger()                       # 打开
        check("顶栏黑白开关生效", window.params.blackwhite == 100)
        bw_frame = window._last_frame.copy()
        check("黑白模式输出为灰度",
              float(np.abs(bw_frame[..., 0].astype(int) - bw_frame[..., 1]).max()) <= 1)
        window.act_bw.trigger()                       # 关闭
        check("再次点击恢复彩色", window.params.blackwhite == 0)

        # ---- 对比原图 ----
        window._toggle_compare(True)
        check("对比原图显示降采样原图", window.preview.current_frame_size()[1] == 1200)
        window._toggle_compare(False)
        check("退出对比恢复处理结果", window.preview.current_frame_size()[0] == IMG_W)

        # ---- 保存 ----
        out_path = os.path.join(tmp, "out.jpg")
        window._save_path = out_path
        window._save_rid = 10 ** 6
        window._write_image(window._last_frame, out_path)
        check("保存 JPEG 成功", os.path.isfile(out_path) and os.path.getsize(out_path) > 1000)
        check("保存内容尺寸正确", cv2.imread(out_path).shape[:2] == (IMG_H, IMG_W))

        # ---- 自动调色（规则版，不联网）----
        window._auto_rule()
        check("规则自动调色生效", window.params.exposure != 0
              or window.params.contrast != 0 or window.params.temperature != 0)
        check("AI 调色按钮存在", window.act_ai is not None)

        # ---- 重置 ----
        window.reset_adjustments()
        check("重置后参数归零", window.params.is_default() and not window.params.pickers)
        check("重置后输出回到原图", np.array_equal(window._last_frame, window.source))

        # ---- 视图交互 ----
        window.preview.zoom_at(2.0)
        check("滚轮缩放生效", abs(window.preview.zoom - 2.0) < 1e-6)
        window.preview.zoom_fit()
        window.preview.zoom_one_to_one()
        check("1:1 缩放生效", window.preview.zoom > 1.0)
        window.preview.zoom_fit()
        hist = core.linear_histograms(window._last_frame)
        window.panel.set_histograms(hist)
        check("直方图可更新", window.panel.curve.editor._hist is hist)
        window.panel.set_histograms(None)
        check("直方图可清空", window.panel.curve.editor._hist is None)

        # ---- 色轮方向（环与拖点必须一致，不能上下镜像）----
        import threading
        import time as _time

        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "legacy"))
        import ui_widgets
        from PySide6.QtCore import QPointF
        from ui_widgets import _DISC_R, ColorWheel

        cw = ColorWheel()
        cw.resize(cw.sizeHint())
        c0 = QPointF(cw.width() / 2.0, cw.height() / 2.0)
        cw._update_from_pos(QPointF(c0.x() + 20, c0.y()))
        check("色轮右侧拖点取 0°", abs(cw.hue_sat()[0]) < 1.0)
        cw._update_from_pos(QPointF(c0.x(), c0.y() - 20))
        check("色轮顶部拖点取 90°", abs(cw.hue_sat()[0] - 90.0) < 1.0)
        cw.set_hue_sat(90.0, 1.0)
        cw_img = cw.grab().toImage()
        rr = int(_DISC_R - 3)
        top_px = cw_img.pixelColor(int(c0.x()), int(c0.y()) - rr)
        bot_px = cw_img.pixelColor(int(c0.x()), int(c0.y()) + rr)
        ring_px = cw_img.pixelColor(int(c0.x()), int(c0.y()) - int(_DISC_R + 5))
        check("色轮 90° 拖点画在顶部（与色环一致）",
              top_px.saturation() > 100 and 60 <= top_px.hue() <= 140)
        check("色轮底部无拖点", bot_px.saturation() < 60)
        check("色环顶部为绿色系", ring_px.saturation() > 100
              and 60 <= ring_px.hue() <= 140)

        # ---- 蒙版叠加层缓存上限 ----
        window.panel.mask.create_mask("linear")
        for k in range(40):
            window.params.masks[0].x2 = 0.3 + 0.001 * k
            window._refresh_mask_overlays()
        check("叠加层缓存有上限", len(window._overlay_cache)
              <= window._OVERLAY_CACHE_MAX)
        window.reset_adjustments()

        # ---- 图章源矩形保存归一化坐标 ----
        window.preview.set_stamp_source((0.25, 0.5, 0.5, 0.75))
        window.preview.zoom_at(2.0)
        check("图章源矩形保存归一化坐标",
              window.preview.stamp_source() == (0.25, 0.5, 0.5, 0.75))
        window.preview.set_stamp_source(None)
        window.preview.zoom_fit()

        # ---- ProcessorThread 异步行为（真实线程，非同步模式）----
        def _wait_until(pred, timeout=5.0):
            t0 = _time.monotonic()
            while not pred():
                if _time.monotonic() - t0 > timeout:
                    return False
                app.processEvents()
                _time.sleep(0.01)
            return True

        small_src = (np.random.default_rng(5).random((40, 60, 3))
                     * 255).astype(np.uint8)
        pt = ui_widgets.ProcessorThread()
        frames: list = []
        pt.frameReady.connect(lambda arr, rid, full: frames.append(rid))
        pt.set_source(ui_widgets.PREVIEW, small_src)
        check("set_source 持有私有副本",
              pt._sources[ui_widgets.PREVIEW] is not small_src
              and np.array_equal(pt._sources[ui_widgets.PREVIEW], small_src))

        gate = threading.Event()
        entered: list = []
        orig_process = ui_widgets.process_linear

        def gated_process(params, lin, should_cancel=None):
            entered.append(1)
            # 模拟真实渲染：阻塞期间周期性让出并轮询取消（分块之间检查），
            # gate 置位或超时后正常完成
            deadline = _time.monotonic() + 5.0
            while not gate.wait(0.02):
                if should_cancel is not None and should_cancel():
                    raise core.CancelledError("已取消")
                if _time.monotonic() > deadline:
                    break
            return np.zeros(lin.shape, np.uint8)

        old_sync = ui_widgets.SYNC_MODE
        ui_widgets.process_linear = gated_process
        ui_widgets.SYNC_MODE = False
        try:
            pt.start()
            pj = core.Adjustments()
            pj.exposure = 0.1
            pt.submit(pj, ui_widgets.PREVIEW)          # rid1：占住 worker
            check("异步:任务已进入渲染", _wait_until(lambda: len(entered) >= 1))
            pt.submit(pj, ui_widgets.PREVIEW)          # rid2：入槽
            pt.submit(pj, ui_widgets.PREVIEW)          # rid3：覆盖 rid2
            gate.set()
            check("异步:合并后两帧到齐", _wait_until(lambda: len(frames) >= 2))
            check("异步:中间任务被合并（只执行 rid1 与 rid3）", frames == [1, 3])

            gate.clear()
            pt.submit(pj, ui_widgets.PREVIEW)          # rid4：占住 worker
            # 第 1 阶段 gated 已进入 2 次（rid1/rid3），必须等到第 3 次才是 rid4，
            # 否则 urgent 提交可能赶在 worker 取走 rid4 之前，rid4 被合并覆盖
            check("异步:rid4 已进入渲染", _wait_until(lambda: len(entered) >= 3))
            pt.submit(pj, ui_widgets.PREVIEW, urgent=True)   # rid5：保存类任务
            pt.submit(pj, ui_widgets.PREVIEW)          # rid6：普通任务，不得覆盖 rid5
            gate.set()
            check("异步:三帧到齐", _wait_until(lambda: len(frames) >= 5))
            check("异步:紧急任务不被普通任务覆盖", frames[2:] == [4, 5, 6])

            gate.clear()
            pt.submit(pj, ui_widgets.PREVIEW)          # rid7：阻塞且响应取消
            # 前 5 次进入 = rid1/3/4/5/6，第 6 次才是 rid7
            check("异步:rid7 已进入渲染", _wait_until(lambda: len(entered) >= 6))
            t0 = _time.monotonic()
            pt.stop()
            check("异步:stop 协作取消后线程快速退出",
                  not pt.isRunning() and _time.monotonic() - t0 < 3.0
                  and len(frames) == 5)
        finally:
            ui_widgets.process_linear = orig_process
            ui_widgets.SYNC_MODE = old_sync

        # ---- 裁剪 / 旋转 ----
        window.params.masks.append(
            core.MaskState(kind="radial", cx=0.75, cy=0.75, radius=0.2, feather=20))
        window.act_crop.trigger()
        check("裁剪模式进入", window.preview.crop_mode()
              and window.crop_bar.isVisible())
        window.preview.set_crop_mode(True, rect={"x0": 0.25, "y0": 0.25,
                                                 "x1": 0.75, "y1": 0.75})
        check("裁剪框可设置", abs(window.preview.crop_rect()["x0"] - 0.25) < 1e-6)
        window._apply_crop()
        check("裁剪后源图尺寸减半",
              window.source.shape[:2] == (IMG_H // 2, IMG_W // 2))
        m_geo = window.params.masks[0]
        check("裁剪后蒙版中心同步",
              abs(m_geo.cx - 1.0) < 0.01 and abs(m_geo.cy - 1.0) < 0.01)
        check("裁剪后退出模式", not window.preview.crop_mode()
              and not window.act_crop.isChecked())
        window._rotate90(True)
        check("右转90°尺寸互换",
              window.source.shape[:2] == (IMG_W // 2, IMG_H // 2))
        check("右转90°蒙版中心跟随",
              abs(m_geo.cx) < 0.01 and abs(m_geo.cy - 1.0) < 0.01)
        window._rotate90(False)
        check("左转抵消右转",
              window.source.shape[:2] == (IMG_H // 2, IMG_W // 2))

        window.act_crop.trigger()
        window.crop_bar.angle_row.set_value(10.0, emit=True)
        check("拉直角度写入", abs(window._crop_angle - 10.0) < 1e-6)
        window._apply_crop()
        h2, w2 = window.source.shape[:2]
        check("拉直应用成功且尺寸收缩", 0 < w2 < IMG_W // 2 and 0 < h2 < IMG_H // 2)

        window.act_crop.trigger()
        window.preview.set_crop_mode(True, rect={"x0": 0.3, "y0": 0.3,
                                                 "x1": 0.7, "y1": 0.7})
        window._on_escape()
        check("Esc 取消裁剪不改动", not window.preview.crop_mode()
              and window.source.shape[:2] == (h2, w2))

        # ---- RAW（rawpy 桩模块验证加载路径）----
        import types
        fake = types.ModuleType("rawpy")
        captured = {}

        class _FakeRaw:
            def __init__(self, path):
                self.metadata = types.SimpleNamespace(make="TestCam", model="X1")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def postprocess(self, **kwargs):
                captured.update(kwargs)
                arr = np.zeros((480, 640, 3), np.uint8)
                arr[..., 0] = 200
                arr[..., 1] = 90
                arr[..., 2] = 80
                return arr

        fake.imread = lambda path: _FakeRaw(path)
        fake.ColorSpace = types.SimpleNamespace(sRGB="sRGB")
        fake.DemosaicAlgorithm = types.SimpleNamespace(AHD="AHD")
        old_rawpy = ui_window.rawpy
        ui_window.rawpy = fake
        try:
            raw_path = os.path.join(tmp, "sample.dng")
            with open(raw_path, "wb") as fh:
                fh.write(b"\x00\x01\x02")
            window.load_image(raw_path)
            check("RAW 路径被加载", window.source is not None
                  and window.source.shape[:2] == (480, 640))
            check("RAW 解码为 RGB", int(window.source[0, 0, 0]) == 200
                  and int(window.source[0, 0, 2]) == 80)
            check("RAW 后处理使用 sRGB 输出", captured.get("output_color") == "sRGB")
            check("RAW 使用 AHD 去马赛克", captured.get("demosaic_algorithm") == "AHD")
            check("RAW 使用相机白平衡", captured.get("use_camera_wb") is True)
            check("文件过滤器包含 RAW", "RAW (" in window._image_filter())
        finally:
            ui_window.rawpy = old_rawpy

        # ---- 边界：小图 / 灰度图 ----
        small = (np.random.default_rng(1).random((60, 80, 3)) * 255).astype(np.uint8)
        sp = os.path.join(tmp, "small.png")
        cv2.imwrite(sp, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        window.load_image(sp)
        check("小图加载且不降采样", window._last_frame.shape[:2] == (60, 80))
        window.panel.basic.rows["exposure"].slider.setValue(80)
        check("小图可调色", window._last_frame.shape[:2] == (60, 80))

        gray_path = os.path.join(tmp, "gray.png")
        cv2.imwrite(gray_path, np.full((200, 300), 120, np.uint8))
        window.load_image(gray_path)
        gray = window.source
        check("灰度图加载为三通道", gray.shape == (200, 300, 3))
        window.panel.basic.rows["saturation"].slider.setValue(-100)
        check("灰度图调色不崩溃", window._last_frame.shape == (200, 300, 3))

        # ---- 无图像时保存被安全拦截 ----
        window.source = None
        check("无图像时保存被安全拦截", window._submit("full") == -1)
        window.source = gray

        closed = window.close()
        check("窗口正常关闭", closed and not window.isVisible())
    return 0


# ======================================================================

def main():
    print("== 色彩空间 ==");    test_colorspaces()
    print("== 基础调整 ==");    test_basic()
    print("== 曲线 ==");        test_curves()
    print("== 颜色分级 ==");    test_grading()
    print("== 取色器 ==");      test_picker()
    print("== 黑白模式 ==");    test_blackwhite()
    print("== 流水线 ==");      test_pipeline()
    print("== 并行一致性 ==");  test_parallel_equals_serial()
    print("== 效果调节 ==");    test_effects()
    print("== 降噪 ==");        test_denoise()
    print("== 对比度强度 ==");  test_contrast_strength()
    print("== 去朦胧不发黑 =="); test_dehaze_no_blackout()
    print("== 蒙版 ==");        test_masks()
    print("== 蒙版高级 ==");    test_mask_advanced()
    print("== 矩形取样 ==");    test_patch_copy()
    print("== 几何变换 ==");    test_geometry()
    print("== 径向形状 ==");    test_radial_shape()
    print("== 自动调色 ==");    test_auto_rule()
    print("== AI 参数解析 =="); test_ai_sanitize()
    print("== AI 蒙版 ==");     test_ai_mask()
    print("== 蒙版缓存 ==");    test_mask_cache()
    print("== GPU 一致性 ==");  test_gpu_matches_cpu()

    if RUN_UI:
        print("== UI 冒烟 ==")
        run_ui_smoke()

    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败: {FAILS}")
        return 1
    print("全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
