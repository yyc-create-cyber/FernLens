"""FernLens 自动调色：规则版 + AI 版。

工具栏两个入口：
- 「自动」：纯本地规则（灰世界白平衡 + 直方图自动色阶），瞬时、可预期；
- 「AI 调色」：把降采样图与统计数据发给视觉大模型，由它返回参数 JSON。
  接口配置**只读取程序目录的 api.json**（支持 OpenAI Chat / OpenAI Responses /
  Gemini / Anthropic 四种格式，可配置多个并自动回退），不读取环境变量、
  本机其它应用配置或任何自动发现来源。

两个版本都只产出参数字典，渲染仍由 core 流水线完成，便于统一钳制与撤销。
"""
from __future__ import annotations

import json
import re

import numpy as np

import ai_api
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


def _clamp(v, lo, hi) -> float:
    return float(max(lo, min(hi, float(v))))


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
# 2. AI 版：接口配置 / 请求
# ======================================================================

def available_profiles() -> list:
    """返回 api.json 中已启用、填写了 URL 的接口（按优先顺序）。

    只读取 api.json；不会读取环境变量、本机其它应用配置或自动发现模型。
    """
    return ai_api.ordered_profiles(ai_api.load_config())


def resolve_candidates() -> list:
    """返回 (baseUrl, apiKey, model) 候选列表（仅来自 api.json）。"""
    return [(p.base_url, p.api_key, p.model) for p in available_profiles()]


def _extract_json(text: str) -> dict:
    """从模型回复中稳健地取出 JSON 对象（容忍 markdown 围栏与前后缀）。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("回复中没有 JSON 对象")
    return json.loads(text[start:end + 1])


def sanitize_ai_params(data: dict) -> dict:
    """把模型返回的参数钳制到合法范围，丢弃非法/无效果的字段。"""
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for name, (lo, hi) in _SCALAR_FIELDS.items():
        if name in data and isinstance(data[name], (int, float)):
            v = _clamp(data[name], lo, hi)
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
                sp = core.sanitize_points(norm)
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
            try:
                sat = _clamp(w.get("saturation", 0), -100, 100)
                lum_v = _clamp(w.get("luminance", 0), -100, 100)
                hue = float(w.get("hue", 0)) % 360.0
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
            try:
                hue = float(p.get("hue", 0)) % 360.0
                span = _clamp(p.get("span", 30), 5, 180)
                shift = _clamp(p.get("hue_shift", 0), -60, 60)
                sat = _clamp(p.get("saturation", 0), -100, 100)
                lum_v = _clamp(p.get("luminance", 0), -100, 100)
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


def _build_prompt_and_image(rgb8: np.ndarray) -> tuple[str, bytes]:
    """生成提示词与 JPEG 字节；具体请求体由 ai_api 按接口格式组装。"""
    small = core.downscale_max_edge(rgb8, 1024)
    _ok, buf = _encode_jpeg(small)
    prompt = (
        "你是专业照片调色师。根据图像与统计数据，输出 JSON 调整参数。\n\n"
        "可用字段（全部可选，省略=0/不变）：\n"
        '- 标量: temperature/tint (-100..100)，exposure (-5..5 EV)，'
        "contrast/highlights/shadows/whites/blacks/saturation/vibrance (-100..100)，"
        "blackwhite (0..100，仅当图像确实适合黑白)\n"
        '- "curves": {"master"|"red"|"green"|"blue": [[x,y],...]}，每通道 2~5 个控制点，'
        "x∈[0,1] 严格递增、y∈[0,1]，首点 x=0、末点 x=1\n"
        '- "wheels": {"shadows"|"midtones"|"highlights"|"global": '
        '{"hue":0..360,"saturation":-100..100,"luminance":-100..100}}\n'
        '- "pickers": [{"hue":0..360,"span":5..180,"hue_shift":-60..60,'
        '"saturation":-100..100,"luminance":-100..100}]，最多 4 个，'
        "用于对特定色相做局部调整\n"
        '- "note": 一句话说明调色思路（中文）\n\n'
        "原则：白平衡校正偏色但保留日落/灯光等有意的暖调；曝光使中位数亮度约 0.42"
        "（线性）；整体自然耐看、不过度，宁可保守；只输出 JSON，不要 markdown。\n\n"
        + _stats_text(rgb8)
    )
    return prompt, buf


def _encode_jpeg(rgb8: np.ndarray, quality: int = 88):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return ok, buf.tobytes()


def suggest_ai(rgb8: np.ndarray, timeout: int = 90) -> tuple:
    """AI 自动调色：返回 (参数字典, 说明, 使用的接口/模型描述)。

    按 api.json 中的优先顺序依次尝试；全部失败抛 RuntimeError。
    """
    prompt, jpeg = _build_prompt_and_image(rgb8)
    profiles = available_profiles()
    if not profiles:
        raise RuntimeError(
            "api.json 中还没有可用的 AI 接口。请点击工具栏「AI 接口…」"
            "填写 API Key、接口 URL 与接口格式并保存。")

    errors = []
    for profile in profiles:
        label = profile.name.strip() or ai_api.protocol_short(profile.protocol)
        model_hint = profile.model.strip() or "自动模型"
        try:
            text, model = ai_api.generate(profile, prompt, jpeg, timeout=timeout)
            raw = _extract_json(text)
            note = str(raw.get("note", "")).strip()[:100]
            params = sanitize_ai_params(raw)
            if not params:
                errors.append(f"{label}（{model}）：返回参数为空")
                continue
            return params, note, f"{label} / {model}"
        except Exception as exc:
            errors.append(f"{label}（{model_hint}）：{exc}")
    raise RuntimeError("所有 AI 接口均失败：\n" + "\n".join(errors))

