"""FernLens 自动调色：规则版 + AI 版。

- 「自动」：纯本地规则（灰世界白平衡 + 直方图自动色阶），瞬时、可预期；
- 「AI 调色」：把压缩后的图像（最长边 1280、JPEG 90，过大再压）与统计数据
  发给视觉大模型，由它返回参数 JSON；提示词要求「克制、可预期」的调整，
  默认只做必要修正 + 轻微风格化，可附带用户的精修要求（intent）。
  返回值再按 strength（0..1，默认 0.6）统一缩放，避免模型给过量参数导致过度后期。

AI 蒙版不走这里：蒙版识别只保留本地像素级引擎（见 segment.py）。

端点为任意 OpenAI 兼容 chat/completions 服务，配置解析顺序：
    1. 前端「AI 设置」界面传入的 {base_url, api_key, model}
    2. 环境变量 FERNLENS_AI_BASE_URL / FERNLENS_AI_KEY / FERNLENS_AI_MODEL

两个版本都只产出参数/网格数据，渲染仍由 core 流水线完成，便于统一钳制与撤销。
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.request

import numpy as np

import core

# 应用到 Adjustments 的标量字段及其合法范围
_SCALAR_FIELDS = {
    "temperature": (-100.0, 100.0),
    "tint": (-100.0, 100.0),
    "exposure": (-5.0, 5.0),
    "contrast": (-100.0, 100.0),
    "highlights": (-100.0, 100.0),
    "shadows": (-100.0, 100.0),
    "whites": (-100.0, 100.0),
    "blacks": (-100.0, 100.0),
    "saturation": (-100.0, 100.0),
    "vibrance": (-100.0, 100.0),
    "blackwhite": (0.0, 100.0),
}

# AI 返回参数的默认强度（0=不动，1=完全按模型给的量）。0.6 让模型的
# 数值落到画面时明显更收敛，避免「一键过度后期」；用户可在 AI 设置里调。
AI_DEFAULT_STRENGTH = 0.6


def _clamp(v, lo, hi) -> float:
    return float(max(lo, min(hi, float(v))))


def _finite(v) -> bool:
    """数值字段可用性：json.loads 会接受 NaN/Infinity 字面量，必须显式排除。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


# ======================================================================
# 1. 规则版自动调色
# ======================================================================

def suggest_rule(rgb8: np.ndarray) -> dict:
    """直方图/灰世界启发式：只输出白平衡 + 影调 + 饱和度字段。

    全部保守取值并钳制，保证可预期；不碰曲线/蒙版/取色器。
    """
    small = core.downscale_max_edge(rgb8, 640)
    lin = core.rgb8_to_linear(small)
    lum = core.luminance(lin)
    srgb = core.to_srgb(lin)

    p05, p10, p50, p90, p995, p999 = np.percentile(
        lum, [0.5, 10, 50, 90, 99.5, 99.9])

    out: dict = {}

    # ---- 白平衡：灰世界（几何均值归一后反解 t/ti），只校正 70% ----
    s = srgb.reshape(-1, 3)
    mean_rgb = s.mean(axis=0)
    target = 1.0 / np.maximum(mean_rgb, 1e-4)
    tg = target / float(np.prod(target) ** (1.0 / 3.0))       # 几何均值归一
    k = 2.909 / max(float(tg[0] + tg[2] + 0.909 * tg[1]), 1e-4)
    ti = (1.0 - k * float(tg[1])) / 0.22            # t/ti ∈ [-1,1]，模型单位
    t = k * (float(tg[0]) - float(tg[2])) / 0.7
    out["temperature"] = _clamp(t * 70.0, -70, 70)  # ×100 转参数档位，再保留 70% 力度
    out["tint"] = _clamp(ti * 70.0, -70, 70)

    # ---- 曝光：线性中位数 -> 0.42 ----
    out["exposure"] = _clamp(np.log2(0.42 / max(float(p50), 1e-4)) * 0.8, -2.0, 2.0)

    # ---- 黑场/白场：雾感拉伸与高光保护 ----
    if p05 > 0.03:
        out["blacks"] = _clamp(-(p05 - 0.03) * 350, -35, 0)
    elif p05 < 0.008:
        out["blacks"] = _clamp((0.008 - p05) * 800, 0, 12)
    if p995 < 0.85:
        out["whites"] = _clamp((0.9 - p995) * 150, 0, 45)
    if p999 > 0.995:
        out["whites"] = min(out.get("whites", 0.0), -12.0)

    # ---- 高光/阴影：两端占比过大时回收/提亮 ----
    f_hi = float((lum > 0.95).mean())
    if f_hi > 0.10:
        out["highlights"] = _clamp(-(f_hi - 0.10) * 70, -40, 0)
    f_lo = float((lum < 0.05).mean())
    if f_lo > 0.30:
        out["shadows"] = _clamp((f_lo - 0.30) * 60, 0, 35)

    # ---- 对比度：p90-p10 分布过窄时拉开 ----
    spread = float(p90 - p10)
    if spread < 0.70:
        out["contrast"] = _clamp((0.70 - spread) * 60, 0, 30)

    # ---- 自然饱和度：平均彩度偏低时提升 ----
    mx = s.max(axis=1)
    mn = s.min(axis=1)
    chroma = float(((mx - mn) / (mx + mn + 1e-5)).mean())
    if chroma < 0.28:
        out["vibrance"] = _clamp((0.28 - chroma) * 120, 0, 30)

    return {k: round(v, 2) for k, v in out.items() if abs(v) > 0.4}


# ======================================================================
# 2. AI 端点配置 / 请求 / 解析
# ======================================================================

def resolve_config(cfg: dict | None = None) -> tuple | None:
    """返回 (baseUrl, apiKey, model)；未配置返回 None。

    优先级：前端「AI 设置」传入的字典 > 环境变量
    FERNLENS_AI_BASE_URL / FERNLENS_AI_KEY / FERNLENS_AI_MODEL（兼容旧 PHOTOLAB_AI_*）。
    """
    c = cfg or {}
    base = str(c.get("base_url") or "").rstrip("/")
    key = str(c.get("api_key") or "")
    model = str(c.get("model") or "")
    if not (base and key and model):
        base = (os.environ.get("FERNLENS_AI_BASE_URL") or os.environ.get("PHOTOLAB_AI_BASE_URL", "")).rstrip("/")
        key = os.environ.get("FERNLENS_AI_KEY") or os.environ.get("PHOTOLAB_AI_KEY", "")
        model = os.environ.get("FERNLENS_AI_MODEL") or os.environ.get("PHOTOLAB_AI_MODEL", "")
    if base and key and model:
        return base, key, model
    return None


def resolve_candidates() -> list:
    """旧版 UI 兼容入口：环境变量配置的单一端点（可能为空列表）。"""
    cfg = resolve_config()
    return [cfg] if cfg else []


def fetch_models(base: str, key: str, timeout: int = 8) -> list:
    """GET {base}/models（OpenAI 兼容），返回模型 id 列表；失败抛异常。"""
    req = urllib.request.Request(
        base + "/models", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def _chat(base: str, key: str, model: str, messages: list,
          timeout: int = 90, max_tokens: int = 2000,
          temperature: float = 0.3) -> str:
    """POST chat/completions，返回拼接后的文本内容。HTTP 错误带响应体消息。"""
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        detail = ""
        if hasattr(exc, "read"):                     # HTTPError 带响应体
            try:
                err = json.loads(exc.read().decode()).get("error", {})
                detail = err.get("message", "") if isinstance(err, dict) else str(err)
            except Exception:
                pass
        raise RuntimeError(f"{exc}{('：' + detail) if detail else ''}") from exc
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        # 部分实现把 content 拆成多段（{"type": "text", "text": ...}）
        content = "".join(part.get("text", "")
                          for part in content if isinstance(part, dict))
    return str(content or "")


def _no_config_error() -> RuntimeError:
    return RuntimeError("未配置 AI 模型：请在顶栏「AI 设置」中填写 URL / API Key / "
                        "模型 ID（OpenAI 兼容接口 + 视觉模型）")


def _extract_json(text: str) -> dict:
    """从模型回复中稳健地取出 JSON 对象（容忍 markdown 围栏与前后缀）。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("回复中没有 JSON 对象")
    return json.loads(text[start:end + 1])


def sanitize_ai_params(data: dict, strength: float = AI_DEFAULT_STRENGTH) -> dict:
    """把模型返回的参数钳制到合法范围，丢弃非法/无效果的字段。

    strength（0..1）对全部「幅度」类字段做统一缩放：标量直接乘，曲线向
    恒等线收缩，色轮/取色器的饱和度与明度乘，色相本身不变。这样即使模型
    仍然给了偏大的数值，落到画面上也是克制的。
    """
    if not isinstance(data, dict):
        return {}
    gain = _clamp(strength, 0.0, 1.0)
    out: dict = {}
    for name, (lo, hi) in _SCALAR_FIELDS.items():
        if name in data and _finite(data[name]):
            v = _clamp(data[name], lo, hi) * gain
            if abs(v) > 0.4:
                out[name] = round(v, 2)

    curves = data.get("curves")
    if isinstance(curves, dict):
        clean = {}
        for ch in core.CURVE_CHANNELS:
            pts = curves.get(ch)
            if not isinstance(pts, (list, tuple)) or len(pts) < 2:
                continue
            try:
                norm = [(float(x), float(y)) for x, y in pts[:6]]
                if not all(np.isfinite(x) and np.isfinite(y) for x, y in norm):
                    continue
                # 曲线向恒等线收缩：y' = x + (y - x) * gain
                scaled = [(x, x + (y - x) * gain) for x, y in norm]
                if all(abs(y - x) < 1e-4 for x, y in scaled):
                    continue          # 缩放后已退化成对角线，视同不调整
                sp = core.sanitize_points(scaled)
            except Exception:
                continue
            if [tuple(p) for p in sp] != core.default_curves()[ch]:
                clean[ch] = [tuple(p) for p in sp]
        if clean:
            out["curves"] = clean

    wheels = data.get("wheels")
    if isinstance(wheels, dict):
        clean = {}
        for name in core.GRADING_RANGES:
            w = wheels.get(name)
            if not isinstance(w, dict):
                continue
            raw = (w.get("hue", 0), w.get("saturation", 0), w.get("luminance", 0))
            if not all(_finite(v) for v in raw):     # NaN/Infinity 直接丢弃
                continue
            try:
                sat = _clamp(raw[1], -100, 100) * gain
                lum_v = _clamp(raw[2], -100, 100) * gain
                hue = float(raw[0]) % 360.0
            except Exception:
                continue
            if abs(sat) > 0.4 or abs(lum_v) > 0.4:
                clean[name] = {"hue": round(hue, 1), "saturation": round(sat, 1),
                               "luminance": round(lum_v, 1)}
        if clean:
            out["wheels"] = clean

    pickers = data.get("pickers")
    if isinstance(pickers, list):
        clean = []
        for p in pickers[:4]:
            if not isinstance(p, dict):
                continue
            raw = (p.get("hue", 0), p.get("span", 30), p.get("hue_shift", 0),
                   p.get("saturation", 0), p.get("luminance", 0))
            if not all(_finite(v) for v in raw):     # NaN/Infinity 直接丢弃
                continue
            try:
                hue = float(raw[0]) % 360.0
                span = _clamp(raw[1], 5, 180)
                shift = _clamp(raw[2], -60, 60) * gain
                sat = _clamp(raw[3], -100, 100) * gain
                lum_v = _clamp(raw[4], -100, 100) * gain
            except Exception:
                continue
            if abs(shift) > 0.4 or abs(sat) > 0.4 or abs(lum_v) > 0.4:
                clean.append({"hue": round(hue, 1), "span": round(span, 1),
                              "hue_shift": round(shift, 1), "saturation": round(sat, 1),
                              "luminance": round(lum_v, 1)})
        if clean:
            out["pickers"] = clean
    return out


def _stats_text(rgb8: np.ndarray) -> str:
    """给模型的测量数据：让数字说话，弥补视觉估计不精的问题。"""
    small = core.downscale_max_edge(rgb8, 640)
    lin = core.rgb8_to_linear(small)
    lum = core.luminance(lin)
    p = np.percentile(lum, [1, 25, 50, 75, 99])
    s = core.to_srgb(lin).reshape(-1, 3)
    mean_rgb = s.mean(axis=0)
    mx = s.max(axis=1)
    mn = s.min(axis=1)
    chroma = float(((mx - mn) / (mx + mn + 1e-5)).mean())
    return (f"线性亮度分位数 p1={p[0]:.3f} p25={p[1]:.3f} p50={p[2]:.3f} "
            f"p75={p[3]:.3f} p99={p[4]:.3f}；"
            f"sRGB 通道均值 R={mean_rgb[0]:.3f} G={mean_rgb[1]:.3f} B={mean_rgb[2]:.3f}；"
            f"平均彩度={chroma:.3f}")


def _build_messages(rgb8: np.ndarray, intent: str = "") -> list:
    data_url = _image_data_url(rgb8)
    want = intent.strip()[:600]
    if want:
        intent_block = (
            "【用户要求（最高优先级，必须明显体现）】\n"
            + want + "\n\n"
        )
    else:
        intent_block = (
            "【用户要求】\n未指定。请自己判断题材、光线与色彩问题，选定一个"
            "最有说服力的风格方向并做到位。\n\n"
        )
    prompt = (
        "你是资深商业修图师与调色师。请把这张照片整理得干净、自然、可信，"
        "保留原片的光线与质感，只做真正需要的那部分调整。\n\n"
        + intent_block
        + "【核心原则：克制】\n"
        "调得好不等于调得多。宁可少一分，也不要过度后期。任何一项改动都应该能"
        "说出「为什么必须改」，说不出就留 0。\n\n"
        "【幅度标尺】\n"
        "1. 常规项（白平衡/对比/高光/阴影/白色/黑色/清晰度/自然饱和度）多在 5~20，"
        "确实有问题时才到 25~35，极少超过 40。\n"
        "2. 曝光调整幅度（EV）通常不超过 ±0.5，明显欠曝/过曝时最多 ±1.0；"
        "线性亮度 p50 低于 0.12 视为明显欠曝、高于 0.55 视为明显过曝。\n"
        "3. 饱和度/自然饱和度单独调整即可，另一个留 0；清晰度/去朦胧一般不超过 20。\n"
        "4. 曲线优先用 master 的轻微 S 曲线（中点位移不超过 0.12），"
        "需要偏色时再动单通道；不要四个通道全给强曲线。\n"
        "5. 颜色分级只做点睛：单个色轮的 saturation/luminance 多在 5~20，"
        "全局色轮更小；不要同时给阴影、中间调、高光三个大额偏移。\n"
        "6. 取色器只在有明确目标色（如天空、肤色、植被）时使用，最多 2 个，"
        "hue_shift 通常不超过 25。\n"
        "7. 同一影调区间不要叠加三种以上手段（例如曲线压暗 + 阴影压暗 + 色轮压暗）。\n"
        "8. 除用户明确要求强风格外，整体观感应仍然像这张照片本身；"
        "肤色不许发脏、发绿、发橙，高光与暗部两端各留一点余量，不要压死。\n"
        "9. 只有用户明确要求，或画面确实适合黑白时，才使用 blackwhite（0..100，"
        "一般不整幅拉满）。\n"
        "10. 用户要求与「克制」冲突时，以用户要求为准，但仍按最小必要幅度去满足。\n"
        "11. 输出前自检：逐项问「这个数值会不会太猛」，把可疑项砍掉一半再输出。\n\n"
        "【可用字段】（全部可选，省略=0/不变）\n"
        '- 标量: temperature/tint (-100..100)，exposure (-5..5 EV)，'
        "contrast/highlights/shadows/whites/blacks/saturation/vibrance (-100..100)，"
        "blackwhite (0..100)\n"
        '- "curves": {"master"|"red"|"green"|"blue": [[x,y],...]}，每通道 2~5 个控制点，'
        "x∈[0,1] 严格递增、y∈[0,1]，首点 x=0、末点 x=1\n"
        '- "wheels": {"shadows"|"midtones"|"highlights"|"global": '
        '{"hue":0..360,"saturation":-100..100,"luminance":-100..100}}\n'
        '- "pickers": [{"hue":0..360,"span":5..180,"hue_shift":-60..60,'
        '"saturation":-100..100,"luminance":-100..100}]，最多 4 个，'
        "用于对特定色相做局部调整\n"
        '- "note": 一句话说明精修思路（中文）\n\n'
        "【输出】只输出一个 JSON 对象，不要 markdown、不要解释。\n\n"
        + _stats_text(rgb8)
    )
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]


def _encode_jpeg(rgb8: np.ndarray, quality: int = 88):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return ok, buf.tobytes()


# 上传给视觉模型的图像：先降采样再 JPEG 压缩；超过上限则逐次降低尺寸/质量。
_AI_IMAGE_MAX_EDGE = 1280
_AI_IMAGE_QUALITY = 90
_AI_IMAGE_MAX_BYTES = 2 * 1024 * 1024


def _image_data_url(rgb8: np.ndarray, max_edge: int = _AI_IMAGE_MAX_EDGE,
                    quality: int = _AI_IMAGE_QUALITY) -> str:
    """压缩后转 data URL：直出 JPEG，不传原图，控制请求体大小。"""
    edge, q = int(max_edge), int(quality)
    while True:
        small = core.downscale_max_edge(rgb8, edge)
        ok, buf = _encode_jpeg(small, q)
        if len(buf) <= _AI_IMAGE_MAX_BYTES or edge <= 640:
            break
        edge = max(640, int(edge * 0.75))
        q = max(72, q - 6)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def suggest_ai(rgb8: np.ndarray, config: dict | None = None,
               timeout: int = 90, intent: str = "",
               strength: float = AI_DEFAULT_STRENGTH) -> tuple:
    """AI 精修：返回 (参数字典, 说明, 使用的模型描述)。

    config 为前端「AI 设置」传入的 {base_url, api_key, model}，缺省回退
    环境变量；两者都没有时抛 RuntimeError。intent 是用户填写的精修要求
    （如「暖调黄昏电影感」），为空时由模型自行判断风格。strength 统一缩放
    模型给出的幅度（默认 0.6），越小越克制。
    """
    cfg = resolve_config(config)
    if cfg is None:
        raise _no_config_error()
    base, key, model = cfg
    messages = _build_messages(rgb8, intent)
    try:
        text = _chat(base, key, model, messages, timeout=timeout)
        raw = _extract_json(text)
    except ValueError as exc:
        raise RuntimeError(f"{model}: {exc}") from exc
    note = str(raw.get("note", "")).strip()[:100]
    params = sanitize_ai_params(raw, strength)
    if not params:
        raise RuntimeError(f"{model}: 返回参数为空")
    return params, note, f"{model}@{base}"


# ======================================================================
# 3. 本地 AI 景深立体调色（Depth Anything V2 Small ONNX）
# ======================================================================

_DEPTH_PRESETS = {
    "studio": {
        "near": {"exposure": 0.24, "temperature": 8.0, "clarity": 16.0, "highlights": 8.0, "vibrance": 8.0},
        "far": {"exposure": -0.32, "temperature": -8.0, "contrast": -8.0, "dehaze": -10.0, "saturation": -10.0},
    },
    "atmosphere": {
        "near": {"contrast": 10.0, "clarity": 15.0, "vibrance": 12.0, "exposure": 0.08},
        "far": {"dehaze": -28.0, "blacks": 14.0, "exposure": 0.16, "temperature": -12.0, "saturation": -14.0},
    },
    "warm_cool": {
        "near": {"temperature": 18.0, "highlights": 10.0, "saturation": 8.0, "exposure": 0.12},
        "far": {"temperature": -22.0, "tint": -4.0, "exposure": -0.20, "shadows": -8.0},
    },
    "cinematic": {
        "near": {"clarity": 14.0, "sharpen": 16.0, "exposure": 0.14, "contrast": 8.0},
        "far": {"exposure": -0.46, "saturation": -18.0, "contrast": -12.0, "blacks": -6.0},
    },
}


def suggest_depth_relight(rgb8: np.ndarray, preset: str = "studio",
                          strength: float = 1.0, near_bias: float = 0.0,
                          masks_only: bool = False) -> dict:
    """使用本地 Depth Anything V2 模型生成前景/远景景深蒙版并附加非破坏性立体调色参数。"""
    import segment

    key = preset if preset in _DEPTH_PRESETS else "studio"
    gain = float(np.clip(strength, 0.0, 1.5))
    near_prob, far_prob, preview_b64, pw, ph = segment.depth_masks(rgb8, near_bias=near_bias)

    def _encode_prob(prob: np.ndarray):
        u8 = (np.clip(prob, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        u8 = segment._resize_long_edge(u8, segment.MASK_LONG_EDGE)
        return core.encode_mask_png(u8.astype(np.float32) / np.float32(255.0))

    near_png, nw, nh = _encode_prob(near_prob)
    far_png, fw, fh = _encode_prob(far_prob)

    def _make_mask(name: str, ai_target: str, png: str, w: int, h: int, adj_map: dict) -> dict:
        adj = {k: round(float(v) * gain, 2) for k, v in adj_map.items()} if not masks_only else {}
        return {
            "kind": "ai",
            "name": name,
            "enabled": True,
            "invert": False,
            "feather": 0,
            "ai_target": ai_target,
            "mask_png": png,
            "mask_w": w,
            "mask_h": h,
            "adj": adj,
        }

    spec = _DEPTH_PRESETS[key]
    masks = [
        _make_mask(f"AI Depth Near ({key})", "depth_near", near_png, nw, nh, spec["near"]),
        _make_mask(f"AI Depth Far ({key})", "depth_far", far_png, fw, fh, spec["far"]),
    ]
    return {
        "preset": key,
        "model": "local(Depth-Anything-V2-Small)",
        "depth_preview_png": preview_b64,
        "preview_w": pw,
        "preview_h": ph,
        "masks": masks,
    }


# ======================================================================
# 4. AI 参考仿色（感知 Lab / 分位曲线 / 三阶色轮拟合）
# ======================================================================

def _extract_palette(rgb8: np.ndarray, k: int = 5) -> list:
    """提取图像 k 个主色调 HEX 色板。"""
    import cv2
    small = core.downscale_max_edge(rgb8, 120)
    pts = small.reshape(-1, 3).astype(np.float32)
    if len(pts) < k:
        return ["#808080"] * k
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    out = []
    for idx in order:
        r, g, b = [int(np.clip(round(c), 0, 255)) for c in centers[idx]]
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def match_reference_color(src_rgb8: np.ndarray, ref_rgb8: np.ndarray,
                          strength: float = 0.75,
                          protect_subject: bool = False) -> dict:
    """根据参考图提取影调曲线、白平衡、对比度与高光/中间调/阴影色轮参数。

    输出全部是 FernLens 原生可编辑参数（滑块 + RGB 曲线 + 颜色分级色轮），
    支持历史撤销、微调与全分辨率无损导出。
    """
    import cv2

    gain = float(np.clip(strength, 0.0, 1.0))
    src_s = core.downscale_max_edge(src_rgb8, 320)
    ref_s = core.downscale_max_edge(ref_rgb8, 320)

    src_weights = None
    if protect_subject:
        try:
            import segment
            if segment.models_status().get("depth_anything_v2", {}).get("ready"):
                depth = segment.estimate_depth(src_s)
                # 保护高景深前景（降低前景在全局色彩统计里的权重偏移，留更多余量）
                src_weights = np.clip(1.25 - depth * 0.65, 0.35, 1.0).astype(np.float32)
        except Exception:
            src_weights = None

    src_lin = core.rgb8_to_linear(src_s)
    ref_lin = core.rgb8_to_linear(ref_s)
    src_lum = core.luminance(src_lin)
    ref_lum = core.luminance(ref_lin)

    src_srgb = core.to_srgb(src_lin)
    ref_srgb = core.to_srgb(ref_lin)
    src_lab = cv2.cvtColor(src_s, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_s, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 1. 白平衡（中间调 Lab a/b 均值差 -> temperature / tint）
    def _mid_ab(lab, lum):
        mask = (lum > 0.12) & (lum < 0.85)
        if int(mask.sum()) < 50:
            mask = np.ones_like(lum, dtype=bool)
        a_mean = float(lab[..., 1][mask].mean()) - 128.0
        b_mean = float(lab[..., 2][mask].mean()) - 128.0
        return a_mean, b_mean

    sa, sb = _mid_ab(src_lab, src_lum)
    ra, rb = _mid_ab(ref_lab, ref_lum)
    wb_damp = 0.72 if src_weights is not None else 0.85
    raw_temp = _clamp((rb - sb) * 2.1 * wb_damp, -55.0, 55.0)
    raw_tint = _clamp((ra - sa) * 1.8 * wb_damp, -40.0, 40.0)

    # 2. 曝光 / 对比度 / 黑白场（sRGB 亮度分位数）
    src_y = (0.2126 * src_srgb[..., 0] + 0.7152 * src_srgb[..., 1] + 0.0722 * src_srgb[..., 2]).ravel()
    ref_y = (0.2126 * ref_srgb[..., 0] + 0.7152 * ref_srgb[..., 1] + 0.0722 * ref_srgb[..., 2]).ravel()
    sp05, sp25, sp50, sp75, sp95 = [float(v) for v in np.percentile(src_y, [5, 25, 50, 75, 95])]
    rp05, rp25, rp50, rp75, rp95 = [float(v) for v in np.percentile(ref_y, [5, 25, 50, 75, 95])]

    raw_exp = _clamp(math.log2(max(rp50, 0.04) / max(sp50, 0.04)) * 0.45, -1.1, 1.1)
    src_spread = max(sp75 - sp25, 0.05)
    ref_spread = max(rp75 - rp25, 0.05)
    raw_contrast = _clamp((ref_spread - src_spread) * 115.0, -35.0, 40.0)
    raw_blacks = _clamp((rp05 - sp05) * 180.0, -35.0, 40.0)
    raw_whites = _clamp((rp95 - sp95) * 110.0, -35.0, 35.0)

    # 3. 饱和度（Lab 色度均值比）
    src_chroma = float(np.hypot(src_lab[..., 1] - 128.0, src_lab[..., 2] - 128.0).mean())
    ref_chroma = float(np.hypot(ref_lab[..., 1] - 128.0, ref_lab[..., 2] - 128.0).mean())
    sat_ratio = (ref_chroma - src_chroma) / max(src_chroma, 8.0)
    raw_sat = _clamp(sat_ratio * 65.0 * (0.75 if protect_subject else 1.0), -45.0, 45.0)

    # 4. 三阶颜色分级色轮（阴影 / 中间调 / 高光 Lab 色彩倾向差）
    def _zone_wheel(z_lo: float, z_hi: float, sat_scale: float) -> dict:
        sm = (src_y >= z_lo) & (src_y < z_hi)
        rm = (ref_y >= z_lo) & (ref_y < z_hi)
        if int(sm.sum()) < 30 or int(rm.sum()) < 30:
            return {"hue": 0.0, "saturation": 0.0, "luminance": 0.0}
        da = float((ref_lab[..., 1].ravel()[rm].mean() - 128.0) - (src_lab[..., 1].ravel()[sm].mean() - 128.0))
        db = float((ref_lab[..., 2].ravel()[rm].mean() - 128.0) - (src_lab[..., 2].ravel()[sm].mean() - 128.0))
        # 扣除已被全局白平衡吸收的偏色
        da -= (ra - sa) * 0.55
        db -= (rb - sb) * 0.55
        mag = math.hypot(da, db)
        if mag < 1.8:
            return {"hue": 0.0, "saturation": 0.0, "luminance": 0.0}
        # Lab (a=红绿, b=黄蓝) 映射到色轮角度 (0°红, 60°黄, 120°绿, 180°青, 240°蓝, 300°品红)
        hue = (math.degrees(math.atan2(db, da)) * 0.9 + 360.0) % 360.0
        # 校正 Lab b*(90°黄)->HSV 60°黄, -90°蓝->240°蓝 的近似映射
        ang = math.degrees(math.atan2(db, da))
        hue = (ang if ang >= 0 else ang + 360.0)
        # Lab 0=红(0°), 90=黄(60°), 180=绿(120°), 270=蓝(240°)
        if hue <= 90:
            hue = hue * (60.0 / 90.0)
        elif hue <= 180:
            hue = 60.0 + (hue - 90.0) * (120.0 / 90.0)
        elif hue <= 270:
            hue = 180.0 + (hue - 180.0) * (60.0 / 90.0)
        else:
            hue = 240.0 + (hue - 270.0) * (120.0 / 90.0)
        sat = _clamp(mag * sat_scale, 0.0, 38.0)
        return {"hue": round(hue % 360.0, 1), "saturation": round(sat, 1), "luminance": 0.0}

    wheels = {
        "shadows": _zone_wheel(0.0, 0.28, 1.35),
        "midtones": _zone_wheel(0.28, 0.68, 0.95),
        "highlights": _zone_wheel(0.68, 1.01, 1.30),
    }

    # 5. RGB 分位曲线微调（捕获胶片暗部抬起与高光滚降特征）
    qs = [0, 25, 50, 75, 100]
    curves = {}
    for ci, ch in enumerate(("red", "green", "blue")):
        sq = np.percentile(src_srgb[..., ci], qs)
        rq = np.percentile(ref_srgb[..., ci], qs)
        pts = []
        for idx, (sx_v, rx_v) in enumerate(zip(sq, rq)):
            x = [0.0, 0.25, 0.5, 0.75, 1.0][idx]
            # 仅取 40% 分位残差，避免与曝光/白平衡叠加过度
            delta = float(np.clip((rx_v - sx_v) * 0.35, -0.14, 0.14))
            y = float(np.clip(x + delta, 0.0, 1.0))
            pts.append([round(x, 3), round(y, 3)])
        curves[ch] = pts

    raw_params = {
        "temperature": raw_temp,
        "tint": raw_tint,
        "exposure": raw_exp,
        "contrast": raw_contrast,
        "blacks": raw_blacks,
        "whites": raw_whites,
        "saturation": raw_sat,
        "curves": curves,
        "wheels": wheels,
    }
    params = sanitize_ai_params(raw_params, strength=gain)
    return {
        "adjustments": params,
        "ref_palette": _extract_palette(ref_s, 5),
        "src_palette": _extract_palette(src_s, 5),
        "protect_subject": bool(protect_subject),
        "note": "AI Reference Color Match",
    }
