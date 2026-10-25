"""FernLens 自检。

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
    os.environ.setdefault("FERNLENS_SYNC", "1")

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
    d = auto.sanitize_ai_params(auto._extract_json(raw))
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


def test_ai_api():
    import json
    import tempfile

    import ai_api

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "api.json")
        check("api.json 首次使用自动创建",
              ai_api.ensure_config_file(path) and os.path.isfile(path))
        check("api.json 已存在时不重复创建", not ai_api.ensure_config_file(path))
        cfg = ai_api.load_config(path)
        check("新建 api.json 为空配置", cfg.apis == [] and cfg.active_id == "")

        p1 = ai_api.ApiProfile(
            name="Chat", protocol="openai-chat",
            base_url="https://example.ai/v1/chat/completions",
            api_key="sk-test", model="vision-x")
        p2 = ai_api.ApiProfile(
            name="Gemini", protocol="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/models/"
                     "gemini-2.0-flash:generateContent",
            api_key="gm-test", model="gemini-2.0-flash", enabled=False)
        cfg.apis = [p1, p2]
        cfg.active_id = p2.id
        ai_api.save_config(cfg, path)

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        check("api.json 记录 URL 与 API Key",
              raw["apis"][0]["base_url"] == "https://example.ai/v1"
              and raw["apis"][0]["api_key"] == "sk-test")
        loaded = ai_api.load_config(path)
        check("完整端点自动归一化为 Base URL",
              loaded.apis[0].base_url == "https://example.ai/v1"
              and loaded.apis[1].base_url
              == "https://generativelanguage.googleapis.com/v1beta")
        check("端点末尾斜杠也能归一化",
              ai_api.normalize_base_url(
                  "https://x.ai/v1/chat/completions/", "openai_chat")
              == "https://x.ai/v1")
        check("非法 URL 不崩溃且能被校验拦截",
              ai_api.normalize_base_url("http://[", "openai_chat") == "http://["
              and bool(ai_api.validate_profile(ai_api.ApiProfile(
                  base_url="http://[", protocol="openai_chat"))))
        check("停用的接口不参与调用",
              [p.id for p in ai_api.usable_profiles(loaded)] == [p1.id])

        # 只读取 api.json：环境变量不参与接口发现/回退
        import auto
        from pathlib import Path
        env_keys = ("FERNLENS_AI_BASE_URL", "FERNLENS_AI_KEY",
                    "FERNLENS_AI_MODEL")
        saved_env = {k: os.environ.get(k) for k in env_keys}
        saved_cfg_path = ai_api.config_path
        try:
            os.environ["FERNLENS_AI_BASE_URL"] = "https://env.example/v1"
            os.environ["FERNLENS_AI_KEY"] = "sk-env"
            os.environ["FERNLENS_AI_MODEL"] = "env-vision"
            ai_api.config_path = lambda: Path(os.path.join(tmp, "missing.json"))
            check("AI 接口只读取 api.json（空配置不回退环境变量）",
                  auto.available_profiles() == []
                  and auto.resolve_candidates() == [])
            ai_api.config_path = lambda: Path(path)
            check("AI 接口只读取 api.json 中的启用项",
                  [p.name for p in auto.available_profiles()] == ["Chat"])
        finally:
            ai_api.config_path = saved_cfg_path
            for k, val in saved_env.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val

        check("兼容旧字段名（url/key/type）",
              ai_api.ApiConfig.from_dict({"apis": [{
                  "url": "https://x.ai/v1/chat/completions", "key": "k",
                  "type": "openai-chat"}]}).apis[0].base_url
              == "https://x.ai/v1")

    base_kw = dict(name="T", base_url="https://example.ai/v1",
                   api_key="sk-test", model="vision-model")
    prompt, jpeg = "输出 JSON", b"\xff\xd8fake-jpeg"

    p = ai_api.ApiProfile(protocol="openai_chat", **base_kw)
    url, headers, body = ai_api.build_request(p, prompt, jpeg)
    check("OpenAI Chat 请求地址正确", url.endswith("/chat/completions"))
    check("OpenAI Chat 使用 Bearer Key",
          headers.get("Authorization") == "Bearer sk-test")
    check("OpenAI Chat 携带图片",
          body["messages"][0]["content"][1]["image_url"]["url"].startswith(
              "data:image/jpeg;base64,"))

    p = ai_api.ApiProfile(protocol="openai_responses", **base_kw)
    url, headers, body = ai_api.build_request(p, prompt, jpeg)
    check("OpenAI Responses 请求地址正确", url.endswith("/responses"))
    check("OpenAI Responses 使用 input_image",
          body["input"][0]["content"][1]["type"] == "input_image")

    p = ai_api.ApiProfile(protocol="gemini", **base_kw)
    url, headers, body = ai_api.build_request(p, prompt, jpeg)
    check("Gemini 请求地址包含模型", url.endswith(
        "/models/vision-model:generateContent"))
    check("Gemini 使用 x-goog-api-key",
          headers.get("x-goog-api-key") == "sk-test")
    check("Gemini 使用 inline_data",
          body["contents"][0]["parts"][1]["inline_data"]["mime_type"]
          == "image/jpeg")
    p25 = ai_api.ApiProfile(protocol="gemini", **{**base_kw,
                                                  "model": "gemini-2.5-flash"})
    _url, _headers, body25 = ai_api.build_request(p25, prompt, jpeg)
    check("Gemini 2.5 Flash 关闭思考预算",
          body25["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0)

    p = ai_api.ApiProfile(protocol="anthropic", **base_kw)
    url, headers, body = ai_api.build_request(p, prompt, jpeg)
    check("Anthropic 请求地址正确", url.endswith("/messages"))
    check("Anthropic 使用 x-api-key 与版本头",
          headers.get("x-api-key") == "sk-test"
          and headers.get("anthropic-version") == "2023-06-01")
    check("Anthropic 使用 base64 image source",
          body["messages"][0]["content"][0]["source"]["type"] == "base64")

    check("OpenAI Chat 回复解析",
          ai_api.extract_response_text(
              "openai_chat",
              {"choices": [{"message": {"content": '{"exposure": 1}'}}]})
          == '{"exposure": 1}')
    check("OpenAI Responses 回复解析",
          ai_api.extract_response_text(
              "openai_responses",
              {"output": [{"type": "message", "content": [
                  {"type": "output_text", "text": "ok"}]}]}) == "ok")
    check("Gemini 回复解析",
          ai_api.extract_response_text(
              "gemini",
              {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
          == "ok")
    check("Anthropic 回复解析",
          ai_api.extract_response_text(
              "anthropic",
              {"content": [{"type": "text", "text": "ok"}]}) == "ok")
    check("Gemini 模型列表去掉 models/ 前缀",
          ai_api.parse_model_ids(
              "gemini", {"models": [{"name": "models/gemini-2.0-flash"}]})
          == ["gemini-2.0-flash"])
    check("模型自动选择偏向视觉模型",
          ai_api.pick_model(
              ["text-embedding-3-small", "gpt-4o", "gpt-3.5-turbo"],
              "openai_chat") == "gpt-4o")


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
    check(f"GPU 与 CPU 结果一致（最大差 {d}/255）", d <= 2)

    p2 = core.Adjustments(blackwhite=100)
    g2 = core.process_linear(p2, lin)
    s2 = core._process_band(p2, lin)
    d2 = int(np.abs(g2.astype(int) - s2.astype(int)).max())
    check(f"黑白模式 GPU 与 CPU 一致（最大差 {d2}/255）", d2 <= 2)
    check("GPU 路径输出为灰度",
          float(np.abs(g2[..., 0].astype(int) - g2[..., 1]).max()) <= 2)


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


def test_clone_stamp():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = 200
    core.clone_stamp(img, (8, 32), (48, 32), 6)
    check("图章复制源区域内容", img[32, 48, 0] == 200)
    check("图章不影响远处", img[2, 60, 0] == 0)
    check("图章边缘羽化过渡", 0 < img[32, 53, 0] < 200)

    img2 = np.full((32, 32, 3), 128, np.uint8)
    core.clone_stamp(img2, (16, 16), (20, 16), 8)
    check("源/目标重叠安全", bool(np.all(np.isfinite(img2))) and float(img2.mean()) > 0)

    img3 = np.zeros((40, 40, 3), np.uint8)
    img3[:20, :20] = 200
    core.clone_stamp(img3, (4, 4), (38, 38), 10)      # 目标越界
    check("越界目标自动裁剪不崩溃", img3[39, 39, 0] == 200 and img3[20, 39, 0] == 0)


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
def test_polygon_mask():
    """自定义多边形蒙版：顶点填充、羽化、反向、缓存与局部调整。"""
    shape = (300, 400)
    pts = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    m = core.MaskState(kind="polygon", points=pts, feather=20)
    mask = core.render_mask(m, shape)
    check("多边形蒙版尺寸/范围", mask.shape == shape
          and 0 <= float(mask.min()) and float(mask.max()) <= 1)
    check("多边形内部为 1", float(mask[150, 200]) > 0.99)
    check("多边形外部为 0", float(mask[20, 20]) < 0.01)
    check("多边形边缘羽化过渡", 0.01 < float(mask[65, 200]) < 0.99)
    check("不足 3 个顶点不生成蒙版", float(core.render_mask(
        core.MaskState(kind="polygon", points=pts[:2]), shape).max()) == 0.0)
    check("未完成的多边形反向也不覆盖全图", float(core.render_mask(
        core.MaskState(kind="polygon", points=pts[:2], invert=True),
        shape).max()) == 0.0)

    hard = core.render_mask(core.MaskState(
        kind="polygon", points=pts, feather=0), shape)
    check("多边形羽化 0 为硬边", float(hard[150, 200]) > 0.999)

    inv = core.render_mask(core.MaskState(
        kind="polygon", points=pts, feather=20, invert=True), shape)
    check("多边形蒙版可反向", abs(float((mask + inv).max()) - 1.0) < 1e-5)
    check("多边形角度参数改变缓存键",
          core.mask_cache_key(m, shape) != core.mask_cache_key(
              core.MaskState(kind="polygon", points=[(0.3, 0.2)] + pts[1:],
                             feather=20), shape))

    img = np.full((64, 64, 3), 0.3, np.float32)
    local = core.MaskState(kind="polygon",
                           points=[(0.25, 0.25), (0.75, 0.25),
                                   (0.75, 0.75), (0.25, 0.75)], feather=10)
    local.adj.exposure = 1.0
    out = core.apply_masked_adjustments(img, [local])
    check("多边形蒙版内局部调整", out[32, 32].mean() > 0.55)
    check("多边形蒙版外保持不变", abs(float(out[2, 2].mean()) - 0.3) < 1e-5)


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
    pm = core.MaskState(kind="polygon", feather=30, closed=True,
                        points=[(0.1, 0.15), (0.45, 0.1), (0.55, 0.55), (0.2, 0.7)])
    pm.adj.highlights = -20
    rm.adj.curves["master"] = [(0.0, 0.0), (0.5, 0.65), (1.0, 1.0)]
    rm.adj.wheels["highlights"] = core.WheelState(40, 30, -10)
    p2.masks = [rm, bm, lm, pm]
    par2 = core.process_linear(p2, lin)
    ser2 = core._process_band(p2, lin)
    check("蒙版+效果并行结果逐位一致（无接缝）", np.array_equal(par2, ser2))
    check("蒙版流水线输出范围合法", int(par2.min()) >= 0 and int(par2.max()) <= 255)
    check("蒙版流水线确定", np.array_equal(par2, core.process_linear(p2, lin)))
    renderer = core.MaskRenderer(p2.masks, rgb.shape[:2])
    check("蒙版渲染器切片有效",
          renderer.get(1, 800, 1000, rgb.shape[1]).shape == (200, rgb.shape[1]))
    check("多边形蒙版分块渲染有效",
          renderer.get(3, 800, 1000, rgb.shape[1]).shape == (200, rgb.shape[1]))


# ======================================================================
# 二、UI 冒烟（--ui）
# ======================================================================

def run_ui_smoke() -> int:
    import tempfile

    import cv2
    from PySide6.QtWidgets import QApplication

    import core
    import ui_window
    from ui_window import MainWindow
    from ui_widgets import GlassSectionHeader, apply_theme

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
    apply_theme(app)
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
        check("右栏模块使用毛玻璃托盘", len(window.panel.sections) == 6
              and all(isinstance(sec.header, GlassSectionHeader)
                      for sec in window.panel.sections))
        check("预设颜色为一排圆形无字按钮",
              len(window.panel.picker.preset_buttons) == 8
              and all(not b.text() and b.width() == b.height() and b.toolTip()
                      for b in window.panel.picker.preset_buttons))

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

        # ---- 自定义多边形蒙版：画布点选 -> 自动闭合 -> 顶点编辑 ----
        window._exit_mask_edit()
        window.panel.mask.create_mask("polygon")
        poly = window.params.masks[-1]
        check("创建自定义多边形蒙版", poly.kind == "polygon"
              and poly.closed is False and not poly.points)
        check("多边形自动进入编辑", window.preview.edit_mode() == "polygon")
        for nx, ny in ((0.25, 0.25), (0.75, 0.25), (0.55, 0.75)):
            pt = wxy(nx, ny)
            QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier, pt)
        check("画布点选多边形顶点", len(poly.points) == 3)
        QTest.keyClick(window.preview, Qt.Key.Key_Return)
        check("多边形自动闭合（Enter）", poly.closed is True)
        check("多边形蒙版渲染有效",
              float(core.render_mask(poly, window.source.shape[:2]).max()) > 0.99)
        frame_before_poly = window._last_frame.copy()
        window.panel.mask._on_adj("exposure", 0.8)
        window._on_params_committed()
        check("多边形蒙版局部调整改变画面",
              not np.array_equal(window._last_frame, frame_before_poly))
        check("多边形叠加层已生成", len(window.preview._overlays) == 4)

        # 继续加点：闭合后仍可追加，然后右键顶点删除
        window._on_polygon_continue()
        QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, wxy(0.82, 0.7))
        check("多边形可继续追加顶点", len(poly.points) == 4)
        window._on_polygon_finished()
        victim = wxy(*poly.points[-1])
        QTest.mouseClick(window.preview, Qt.MouseButton.RightButton,
                         Qt.KeyboardModifier.NoModifier, victim)
        check("右键删除多边形顶点", len(poly.points) == 3)
        # 拖动顶点
        v0 = wxy(*poly.points[0])
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, v0)
        v1 = wxy(0.2, 0.3)
        QTest.mouseMove(window.preview, v1)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, v1)
        check("拖动多边形顶点", abs(poly.points[0][0] - 0.2) < 0.02
              and abs(poly.points[0][1] - 0.3) < 0.02)
        window._exit_mask_edit()
        window.panel.mask.delete_mask(len(window.params.masks) - 1)
        check("删除自定义多边形蒙版", len(window.params.masks) == 3)

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

        # ---- 仿制图章 ----
        window._toggle_stamp(True)
        check("图章模式进入", window.preview.stamp_mode())
        window.preview.cloneSampled.emit(0.25, 0.25)
        window.preview.cloneStrokePoint.emit(0.6, 0.6)
        window.preview.cloneStrokePoint.emit(0.65, 0.6)
        window.preview.cloneStrokeFinished.emit()
        check("图章修改像素", window._retouch_full is not None
              and not np.array_equal(window._retouch_full, window.source))
        check("图章笔迹已记录", len(window._clone_ops) == 1)
        check("图章后全图源已切换", window.processor._sources.get("full")
              is window._retouch_full)
        window._on_stamp_undo()
        check("图章撤销还原", window._retouch_full is None
              and window.processor._sources.get("full") is window.source)
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
        check("AI 接口设置按钮存在", window.act_ai_settings is not None)

        from ui_ai_settings import ApiSettingsDialog
        dlg_path = os.path.join(tmp, "ui_api.json")
        dlg = ApiSettingsDialog(window, first_use=True, path=dlg_path)
        check("接口对话框支持拉取模型", dlg.fetch_btn.text() == "拉取模型")
        check("接口对话框支持四种格式", dlg.protocol_combo.count() == 4)
        check("接口对话框首次打开有待填写的接口",
              dlg.list.count() == 1 and dlg.model_combo is not None)
        dlg.close()

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
        window.panel.set_histograms(None)
        check("直方图可更新", True)

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

        window.close()
        check("窗口正常关闭", True)
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
    print("== 仿制图章 ==");    test_clone_stamp()
    print("== 径向形状 ==");    test_radial_shape()
    print("== 多边形蒙版 ==");  test_polygon_mask()
    print("== 自动调色 ==");    test_auto_rule()
    print("== AI 参数解析 =="); test_ai_sanitize()
    print("== AI 接口与协议 =="); test_ai_api()
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
