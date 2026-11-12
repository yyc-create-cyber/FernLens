"""PhotoLab 自动调色：规则版 + AI 版。

- 「自动」：纯本地规则（灰世界白平衡 + 直方图自动色阶），瞬时、可预期；
- 「AI 调色」：把压缩后的图像（最长边 1280、JPEG 90，过大再压）与统计数据
  发给视觉大模型，由它返回参数 JSON；提示词要求「克制、可预期」的调整，
  默认只做必要修正 + 轻微风格化，可附带用户的精修要求（intent）。
  返回值再按 strength（0..1，默认 0.6）统一缩放，避免模型给过量参数导致过度后期。

AI 蒙版不走这里：蒙版识别只保留本地像素级引擎（见 segment.py）。

端点为任意 OpenAI 兼容 chat/completions 服务，配置解析顺序：
    1. 前端「AI 设置」界面传入的 {base_url, api_key, model}
    2. 环境变量 PHOTOLAB_AI_BASE_URL / PHOTOLAB_AI_KEY / PHOTOLAB_AI_MODEL

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
    PHOTOLAB_AI_BASE_URL / PHOTOLAB_AI_KEY / PHOTOLAB_AI_MODEL。
    """
    c = cfg or {}
    base = str(c.get("base_url") or "").rstrip("/")
    key = str(c.get("api_key") or "")
    model = str(c.get("model") or "")
    if not (base and key and model):
        base = os.environ.get("PHOTOLAB_AI_BASE_URL", "").rstrip("/")
        key = os.environ.get("PHOTOLAB_AI_KEY", "")
        model = os.environ.get("PHOTOLAB_AI_MODEL", "")
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
