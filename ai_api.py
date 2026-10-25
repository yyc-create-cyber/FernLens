"""FernLens AI 接口配置与多协议调用。

配置文件固定为程序目录下的 ``api.json``，记录多个接口的名称、格式、URL、
API Key 与模型。所有接口信息**只从该文件读取**，不读取环境变量或本机其它
应用配置。支持四种接口：

- ``openai_chat``       OpenAI Chat Completions（/chat/completions）
- ``openai_responses``  OpenAI Responses（/responses）
- ``gemini``            Google Gemini generateContent
- ``anthropic``         Anthropic Messages（/messages）

本模块只负责配置读写、模型拉取与 HTTP 请求；调色参数解析仍在 auto.py。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "api.json"
CONFIG_VERSION = 1
DEFAULT_PROTOCOL = "openai_chat"
_TIMEOUT = 90


# ======================================================================
# 1. 异常
# ======================================================================

class AIApiError(RuntimeError):
    """AI 接口相关的可读错误。"""


class ApiConfigError(AIApiError):
    """api.json 读取/写入/格式错误。"""


class AIRequestError(AIApiError):
    """HTTP 请求失败；保留状态码与响应体摘要便于界面提示。"""

    def __init__(self, message: str, status: int | None = None,
                 body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


# ======================================================================
# 2. 配置模型与读写
# ======================================================================

_PROTOCOLS = {
    "openai_chat": {
        "label": "OpenAI Chat Completions",
        "short": "OpenAI Chat",
        "default_url": "https://api.openai.com/v1",
        "hint": "兼容 /chat/completions 的接口（OpenAI、DeepSeek、OpenRouter 等）",
    },
    "openai_responses": {
        "label": "OpenAI Responses",
        "short": "OpenAI Responses",
        "default_url": "https://api.openai.com/v1",
        "hint": "OpenAI Responses API（/responses），也兼容实现该格式的代理",
    },
    "gemini": {
        "label": "Gemini (Google AI Studio)",
        "short": "Gemini",
        "default_url": "https://generativelanguage.googleapis.com/v1beta",
        "hint": "Gemini 原生 generateContent 接口",
    },
    "anthropic": {
        "label": "Anthropic Messages",
        "short": "Anthropic",
        "default_url": "https://api.anthropic.com/v1",
        "hint": "Anthropic Messages API（/messages），也兼容 Claude 代理",
    },
}
PROTOCOL_ORDER = tuple(_PROTOCOLS)
PROTOCOLS = _PROTOCOLS

_PROTOCOL_ALIASES = {
    "openai": "openai_chat",
    "openai_chat": "openai_chat",
    "openai-chat": "openai_chat",
    "openai_chat_completions": "openai_chat",
    "chat": "openai_chat",
    "chat_completions": "openai_chat",
    "completions": "openai_chat",
    "openai_responses": "openai_responses",
    "openai-responses": "openai_responses",
    "openai_response": "openai_responses",
    "responses": "openai_responses",
    "response": "openai_responses",
    "gemini": "gemini",
    "google": "gemini",
    "google_gemini": "gemini",
    "google-ai": "gemini",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "anthropic_messages": "anthropic",
    "anthropic-messages": "anthropic",
    "messages": "anthropic",
}


def config_path() -> Path:
    """api.json 的固定路径：程序目录（打包时为可执行文件目录）。"""
    if getattr(sys, "frozen", False):          # PyInstaller 等打包场景
        return Path(sys.executable).resolve().parent / CONFIG_NAME
    return Path(__file__).resolve().parent / CONFIG_NAME


def normalize_protocol(value: str | None) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return _PROTOCOL_ALIASES.get(text, DEFAULT_PROTOCOL)


def protocol_label(protocol: str) -> str:
    return PROTOCOLS.get(normalize_protocol(protocol), PROTOCOLS[DEFAULT_PROTOCOL])["label"]


def protocol_short(protocol: str) -> str:
    return PROTOCOLS.get(normalize_protocol(protocol), PROTOCOLS[DEFAULT_PROTOCOL])["short"]


def protocol_hint(protocol: str) -> str:
    return PROTOCOLS.get(normalize_protocol(protocol), PROTOCOLS[DEFAULT_PROTOCOL])["hint"]


def default_base_url(protocol: str) -> str:
    return PROTOCOLS.get(normalize_protocol(protocol), PROTOCOLS[DEFAULT_PROTOCOL])["default_url"]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def clean_api_key(value) -> str:
    """去掉粘贴时常见的引号与 “Bearer ” 前缀。"""
    key = str(value or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        key = key[1:-1].strip()
    key = re.sub(r"^(?:authorization\s*:\s*)?bearer\s+", "", key,
                 flags=re.I).strip()
    return key


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _join(base: str, path: str) -> str:
    if not base:
        return path
    return base.rstrip("/") + "/" + path.lstrip("/")


def _append_query(url: str, params: dict) -> str:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return url
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.extend((str(k), str(v)) for k, v in params.items() if v != "")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path,
         urllib.parse.urlencode(query), parts.fragment))


def normalize_base_url(url: str | None, protocol: str = DEFAULT_PROTOCOL) -> str:
    """把用户填写的地址规整为 Base URL。

    容忍用户粘贴完整端点（/chat/completions、/responses、/messages 或
    Gemini 的 /models/xxx:generateContent），也容忍缺少协议头。查询参数
    会被丢弃；API Key 应填写在单独的输入框里。
    """
    text = str(url or "").strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    if not text:
        return ""
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text):
        # 本地服务通常没有 HTTPS 证书
        scheme = ("http://" if re.match(
            r"^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])(?:[:/]|$)",
            text, re.I) else "https://")
        text = scheme + text

    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError:
        return text
    path = (parts.path or "").rstrip("/")

    # 去掉常见完整端点后缀
    lowered = path.lower()
    for suffix in ("/chat/completions", "/responses", "/messages"):
        if lowered.endswith(suffix):
            path = path[:-len(suffix)]
            lowered = path.lower()
            break

    proto = normalize_protocol(protocol)
    if proto == "gemini":
        # https://.../v1beta/models/gemini-2.0-flash:generateContent
        path = re.sub(r"/models/[^/]+:(?:generateContent|streamGenerateContent)/?$",
                      "", path, flags=re.I)
        # 只填到 /models 也视为 Base URL
        path = re.sub(r"/models/?$", "", path, flags=re.I)

    path = path.rstrip("/") or ""

    # 常见官方地址省略版本前缀时自动补全
    host = (parts.hostname or "").lower()
    if not path:
        if proto in ("openai_chat", "openai_responses") and host == "api.openai.com":
            path = "/v1"
        elif proto == "anthropic" and host == "api.anthropic.com":
            path = "/v1"
        elif proto == "gemini" and host == "generativelanguage.googleapis.com":
            path = "/v1beta"

    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), parts.netloc, path, "", ""))


def validate_profile(profile: "ApiProfile") -> list[str]:
    """返回接口字段的中文错误列表（不检查网络）。"""
    errors = []
    base = normalize_base_url(profile.base_url, profile.protocol)
    if not base:
        errors.append("接口 URL 不能为空")
    else:
        try:
            parts = urllib.parse.urlsplit(base)
        except ValueError:
            parts = None
        if parts is None:
            errors.append("接口 URL 格式不正确")
        elif parts.scheme not in ("http", "https") or not parts.netloc:
            errors.append("接口 URL 必须是 http:// 或 https:// 地址")
        elif any(c.isspace() for c in base):
            errors.append("接口 URL 不能包含空格")
    if normalize_protocol(profile.protocol) not in PROTOCOLS:
        errors.append("接口格式不受支持")
    return errors


def default_profile_name(profile: "ApiProfile") -> str:
    base = normalize_base_url(profile.base_url, profile.protocol)
    if base:
        try:
            host = urllib.parse.urlsplit(base).hostname or ""
        except ValueError:
            host = ""
        if host:
            return host
    return protocol_short(profile.protocol)


@dataclass
class ApiProfile:
    """一个可调用的 AI 接口。"""

    id: str = field(default_factory=_new_id)
    name: str = ""
    protocol: str = DEFAULT_PROTOCOL
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = _new_id()
        self.protocol = normalize_protocol(self.protocol)
        self.base_url = normalize_base_url(self.base_url, self.protocol)
        self.name = str(self.name or "").strip()
        self.api_key = clean_api_key(self.api_key)
        self.model = str(self.model or "").strip()
        if not self.name:
            self.name = default_profile_name(self)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ApiProfile":
        name = data.get("name") or data.get("title") or ""
        protocol = data.get("protocol") or data.get("format") or data.get("type") or DEFAULT_PROTOCOL
        base = (data.get("base_url") or data.get("url") or data.get("endpoint")
                or data.get("baseUrl") or "")
        key = (data.get("api_key") or data.get("key") or data.get("token")
               or data.get("apiKey") or "")
        model = data.get("model") or data.get("model_id") or data.get("modelId") or ""
        enabled = _as_bool(data.get("enabled", True))
        return cls(
            id=str(data.get("id") or "").strip() or _new_id(),
            name=str(name or ""),
            protocol=str(protocol),
            base_url=str(base or ""),
            api_key=str(key or ""),
            model=str(model or ""),
            enabled=enabled,
        )


@dataclass
class ApiConfig:
    """api.json 的内存表示。"""

    apis: list[ApiProfile] = field(default_factory=list)
    active_id: str = ""

    def to_dict(self) -> dict:
        return {
            "version": CONFIG_VERSION,
            "active_id": self.active_id,
            "apis": [p.to_dict() for p in self.apis],
        }

    @classmethod
    def from_dict(cls, data) -> "ApiConfig":
        if isinstance(data, list):
            raw_apis = data
            active_id = ""
        elif isinstance(data, dict):
            raw_apis = (data.get("apis") or data.get("api")
                        or data.get("providers") or data.get("endpoints") or [])
            active_id = str(data.get("active_id") or data.get("active") or "")
            if isinstance(raw_apis, dict):
                raw_apis = [raw_apis]
            if not isinstance(raw_apis, list):
                raw_apis = []
            # 兼容把单个接口直接写在顶层的情况
            if not raw_apis and (data.get("base_url") or data.get("url")):
                raw_apis = [data]
        else:
            raw_apis, active_id = [], ""

        apis: list[ApiProfile] = []
        seen: set[str] = set()
        for item in raw_apis:
            if not isinstance(item, dict):
                continue
            try:
                profile = ApiProfile.from_dict(item)
            except Exception:
                continue
            if profile.id in seen:
                profile.id = _new_id()
            seen.add(profile.id)
            apis.append(profile)

        cfg = cls(apis=apis, active_id=active_id)
        return normalize_config(cfg)


def normalize_config(cfg: ApiConfig) -> ApiConfig:
    """补齐 ID/名称，并确保 active_id 指向一个存在的接口。"""
    seen: set[str] = set()
    for profile in cfg.apis:
        if not profile.id or profile.id in seen:
            profile.id = _new_id()
        seen.add(profile.id)
        profile.protocol = normalize_protocol(profile.protocol)
        profile.base_url = normalize_base_url(profile.base_url, profile.protocol)
        profile.name = str(profile.name or "").strip() or default_profile_name(profile)
        profile.api_key = clean_api_key(profile.api_key)
        profile.model = str(profile.model or "").strip()

    usable_ids = {p.id for p in cfg.apis if p.enabled and p.base_url}
    all_ids = {p.id for p in cfg.apis}
    if cfg.active_id not in usable_ids:
        cfg.active_id = next((p.id for p in cfg.apis if p.enabled and p.base_url),
                             next((p.id for p in cfg.apis if p.enabled),
                                  next(iter(all_ids), "")))
    return cfg


def load_config(path: str | Path | None = None) -> ApiConfig:
    """读取 api.json；文件不存在时返回空配置（不自动创建）。"""
    target = Path(path) if path is not None else config_path()
    if not target.exists():
        return ApiConfig()
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ApiConfigError(f"无法读取 {target}：{exc}") from exc
    if not text.strip():
        return ApiConfig()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApiConfigError(
            f"{target.name} 不是有效的 JSON（第 {exc.lineno} 行）：{exc.msg}") from exc
    return ApiConfig.from_dict(data)


def save_config(cfg: ApiConfig, path: str | Path | None = None) -> Path:
    """规范化并原子写入 api.json，返回写入路径。"""
    target = Path(path) if path is not None else config_path()
    normalize_config(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2) + "\n"
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ApiConfigError(f"无法写入 {target}：{exc}") from exc
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def ensure_config_file(path: str | Path | None = None) -> bool:
    """确保 api.json 存在；返回本次是否新建了文件。"""
    target = Path(path) if path is not None else config_path()
    if target.exists():
        return False
    save_config(ApiConfig(), target)
    return True


def usable_profiles(cfg: ApiConfig | None = None) -> list[ApiProfile]:
    """返回已启用且填写了 URL 的接口。"""
    cfg = cfg if cfg is not None else load_config()
    return [p for p in cfg.apis if p.enabled and p.base_url]


def ordered_profiles(cfg: ApiConfig | None = None) -> list[ApiProfile]:
    """优先接口排在最前，其余按列表顺序。"""
    cfg = cfg if cfg is not None else load_config()
    profiles = usable_profiles(cfg)
    active = [p for p in profiles if p.id == cfg.active_id]
    return active + [p for p in profiles if p.id != cfg.active_id]


# ======================================================================
# 3. 请求地址与请求体
# ======================================================================

def _gemini_model_path(model: str) -> str:
    model = (model or "").strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    return urllib.parse.quote(model, safe="-_.")


def endpoint_url(profile: ApiProfile, model: str = "") -> str:
    """返回实际请求 URL；Gemini 需要模型名，未给出时用占位符。"""
    base = normalize_base_url(profile.base_url, profile.protocol)
    proto = normalize_protocol(profile.protocol)
    if not base:
        return ""
    if proto == "openai_chat":
        return _join(base, "chat/completions")
    if proto == "openai_responses":
        return _join(base, "responses")
    if proto == "anthropic":
        return _join(base, "messages")
    if proto == "gemini":
        name = (model or profile.model or "").strip()
        path_name = _gemini_model_path(name) if name else "{model}"
        return _join(base, f"models/{path_name}:generateContent")
    return ""


def describe_endpoint(profile: ApiProfile, model: str = "") -> str:
    url = endpoint_url(profile, model)
    return f"POST {url}" if url else "填写接口 URL 后显示实际请求地址"


def build_request(profile: ApiProfile, prompt: str, image_jpeg: bytes,
                  model: str = "") -> tuple[str, dict, dict]:
    """构造 (URL, headers, JSON body)，不发起网络请求。"""
    proto = normalize_protocol(profile.protocol)
    used_model = (model or profile.model or "").strip()
    if not used_model:
        raise AIRequestError("未指定模型，请填写模型名或先点击“拉取模型”")
    url = endpoint_url(profile, used_model)
    if not url:
        raise AIRequestError("接口 URL 为空")

    b64 = base64.b64encode(image_jpeg).decode("ascii")
    data_url = "data:image/jpeg;base64," + b64
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "FernLens/1.0",
    }
    key = (profile.api_key or "").strip()

    if proto == "openai_chat":
        if key:
            headers["Authorization"] = "Bearer " + key
        body = {
            "model": used_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        return url, headers, body

    if proto == "openai_responses":
        if key:
            headers["Authorization"] = "Bearer " + key
        body = {
            "model": used_model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            "temperature": 0.3,
            "max_output_tokens": 4096,
        }
        return url, headers, body

    if proto == "gemini":
        if key:
            headers["x-goog-api-key"] = key
        generation = {
            "temperature": 0.3,
            "maxOutputTokens": 4096,
        }
        # Gemini 2.5 Flash 的思考过程也会占用输出预算，调色任务不需要思考
        if re.search(r"gemini-2\.5.*flash", used_model, re.I):
            generation["thinkingConfig"] = {"thinkingBudget": 0}
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ],
            }],
            "generationConfig": generation,
        }
        return url, headers, body

    # anthropic
    if key:
        headers["x-api-key"] = key
    headers["anthropic-version"] = "2023-06-01"
    body = {
        "model": used_model,
        "max_tokens": 4096,
        "temperature": 0.3,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    return url, headers, body


# ======================================================================
# 4. HTTP
# ======================================================================

def _error_from_body(raw: bytes) -> str:
    text = raw.decode("utf-8-sig", errors="replace").strip().lstrip("\ufeff")
    if not text:
        return ""
    try:
        data = json.loads(text)
    except Exception:
        data = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            message = err.get("message") or err.get("detail") or err.get("code")
            if message:
                return str(message)
        if isinstance(err, str) and err:
            return err
        if data.get("message"):
            return str(data["message"])
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[:600]


def _request_json(method: str, url: str, headers: dict | None,
                  body: dict | None, timeout: float) -> dict:
    payload = None
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload, method=method,
                                     headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = _error_from_body(exc.read())
        except Exception:
            pass
        message = f"HTTP {exc.code}"
        if detail:
            message += f"：{detail}"
        raise AIRequestError(message, status=exc.code, body=detail) from exc
    except urllib.error.URLError as exc:
        raise AIRequestError(f"网络错误：{exc.reason}") from exc
    except ValueError as exc:
        raise AIRequestError(f"接口 URL 格式不正确：{exc}") from exc
    except TimeoutError as exc:
        raise AIRequestError("请求超时") from exc
    except OSError as exc:
        raise AIRequestError(f"请求失败：{exc}") from exc
    try:
        return json.loads(raw.decode("utf-8-sig", errors="replace"))
    except json.JSONDecodeError as exc:
        snippet = raw.decode("utf-8-sig", errors="replace")[:300]
        raise AIRequestError(f"接口返回的不是 JSON：{snippet}") from exc


def _unsupported_param(message: str) -> bool:
    msg = (message or "").lower()
    return any(word in msg for word in (
        "unsupported", "not support", "does not support", "unknown parameter",
        "unknown field"))


def _post_json_adaptive(url: str, headers: dict, body: dict,
                        timeout: float) -> dict:
    """发送 POST，并针对少数不兼容参数做自动降级。"""
    for _ in range(5):
        try:
            return _request_json("POST", url, headers, body, timeout)
        except AIRequestError as exc:
            msg = (exc.body or str(exc)).lower()
            if ("temperature" in body and "temperature" in msg
                    and _unsupported_param(msg)):
                body.pop("temperature", None)
                continue
            generation = body.get("generationConfig")
            if (isinstance(generation, dict) and "temperature" in generation
                    and "temperature" in msg and _unsupported_param(msg)):
                generation.pop("temperature", None)
                continue
            if (isinstance(generation, dict) and generation.get("thinkingConfig")
                    and "thinking" in msg and _unsupported_param(msg)):
                generation.pop("thinkingConfig", None)
                continue
            if "max_tokens" in body and (
                    "max_completion_tokens" in msg or
                    ("max_tokens" in msg and _unsupported_param(msg))):
                body["max_completion_tokens"] = body.pop("max_tokens")
                continue
            if ("max_output_tokens" in body and "max_output_tokens" in msg
                    and _unsupported_param(msg)):
                body.pop("max_output_tokens", None)
                continue
            raise
    raise AIRequestError("接口不接受请求参数")


def _error_in_data(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("detail") or err)
    if isinstance(err, str) and err:
        return err
    feedback = data.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        return f"请求被拦截：{feedback['blockReason']}"
    return ""


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "output_text"):
            if isinstance(content.get(key), str):
                return content[key]
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ("text", "output_text", "input_text"):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(p for p in parts if p)
    return ""


def extract_response_text(protocol: str, data: dict) -> str:
    """按接口格式从响应 JSON 中提取模型文本。"""
    proto = normalize_protocol(protocol)
    if not isinstance(data, dict):
        raise AIRequestError("接口返回格式不正确")
    error = _error_in_data(data)
    if error:
        raise AIRequestError(error)

    if proto == "openai_chat":
        choices = data.get("choices") or []
        if not choices:
            raise AIRequestError("接口回复中没有 choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise AIRequestError(f"模型拒绝生成：{refusal.strip()}")
        text = _flatten_content(message.get("content"))
        if not text:
            text = _flatten_content(choice.get("text"))
        return text.strip()

    if proto == "openai_responses":
        direct = data.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        parts: list[str] = []
        output = data.get("output")
        if isinstance(output, str) and output.strip():
            return output.strip()
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in (None, "message", "text"):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    content = item.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") in (
                                    "output_text", "text"):
                                text = block.get("text")
                                if isinstance(text, str) and text.strip():
                                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()
        # 有些兼容代理仍返回 chat 结构
        if data.get("choices"):
            return extract_response_text("openai_chat", data)
        raise AIRequestError("接口回复中没有 output 文本")

    if proto == "gemini":
        candidates = data.get("candidates") or []
        if not candidates:
            raise AIRequestError("接口回复中没有 candidates")
        parts: list[str] = []
        thoughts: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    if part.get("thought"):
                        thoughts.append(text)
                    else:
                        parts.append(text)
        if not parts:
            parts = thoughts
        return "\n".join(parts).strip()

    # anthropic
    content = data.get("content")
    if isinstance(content, str):
        return content.strip()
    parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
    if not parts and isinstance(data.get("completion"), str):
        parts.append(data["completion"])
    return "\n".join(parts).strip()


# ======================================================================
# 5. 模型拉取与选择
# ======================================================================

_MODELS_CACHE: dict[tuple, list[str]] = {}

_EXCLUDE_MODEL = re.compile(
    r"embed|embedding|whisper|tts|dall-e|gpt-image|imagen|veo|sora|moderation|"
    r"rerank|transcrib|realtime|-live|audio|speech|image-gen|"
    r"computer-use|deep-research", re.I)
_VISION_MODEL = re.compile(
    r"vision|(?:^|[-_/])vl(?:[-_.]|$)|omni|gemini|claude|gpt-4o|"
    r"gpt-4\.1|gpt-5|qvq|internvl|llava|pixtral|minicpm", re.I)


def _model_cache_key(profile: ApiProfile) -> tuple:
    key_hash = hashlib.sha256(
        (profile.api_key or "").encode("utf-8")).hexdigest()[:16]
    return (normalize_protocol(profile.protocol),
            normalize_base_url(profile.base_url, profile.protocol), key_hash)


def parse_model_ids(protocol: str, data) -> list[str]:
    """从 /models 响应中取出模型 ID。"""
    proto = normalize_protocol(protocol)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if proto == "gemini":
            items = data.get("models") or []
        else:
            items = data.get("data") or data.get("models") or []
    else:
        items = []

    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            methods = item.get("supportedGenerationMethods")
            if (proto == "gemini" and isinstance(methods, list)
                    and "generateContent" not in methods):
                continue
            name = (item.get("id") or item.get("name") or item.get("model") or "")
        else:
            name = ""
        name = str(name or "").strip()
        if proto == "gemini" and name.startswith("models/"):
            name = name[len("models/"):]
        if not name or name in seen:
            continue
        seen.add(name)
        ids.append(name)
    return ids


def discover_models(profile: ApiProfile, timeout: float = 20,
                    use_cache: bool = True) -> list[str]:
    """调用接口的 /models 拉取可用模型列表。"""
    base = normalize_base_url(profile.base_url, profile.protocol)
    if not base:
        raise AIRequestError("接口 URL 为空，无法拉取模型")
    proto = normalize_protocol(profile.protocol)
    cache_key = _model_cache_key(profile)
    if use_cache and cache_key in _MODELS_CACHE:
        return list(_MODELS_CACHE[cache_key])

    key = (profile.api_key or "").strip()
    headers = {"Accept": "application/json", "User-Agent": "FernLens/1.0"}
    if proto in ("openai_chat", "openai_responses"):
        if key:
            headers["Authorization"] = "Bearer " + key
        url = _join(base, "models")
    elif proto == "anthropic":
        if key:
            headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
        url = _append_query(_join(base, "models"), {"limit": 1000})
    else:
        if key:
            headers["x-goog-api-key"] = key
        url = _append_query(_join(base, "models"), {"pageSize": 1000})

    try:
        data = _request_json("GET", url, headers, None, timeout)
    except AIRequestError as exc:
        # 少数 Gemini 代理只接受 URL 查询参数
        if proto == "gemini" and key and exc.status in (400, 401, 403):
            retry_url = _append_query(url, {"key": key})
            retry_headers = {k: v for k, v in headers.items()
                             if k.lower() != "x-goog-api-key"}
            data = _request_json("GET", retry_url, retry_headers, None, timeout)
        else:
            raise

    models = parse_model_ids(proto, data)
    if models:
        _MODELS_CACHE[cache_key] = list(models)
    return models


def pick_model(models: list[str], protocol: str = DEFAULT_PROTOCOL) -> str:
    """从模型列表中挑选最适合视觉调色的模型。"""
    proto = normalize_protocol(protocol)
    candidates = [m for m in models if m and not _EXCLUDE_MODEL.search(m)]
    if not candidates:
        candidates = [m for m in models if m]
    if not candidates:
        return ""

    def sort_key(name: str):
        low = name.lower()
        if "vision" in low:
            rank = 0
        elif _VISION_MODEL.search(low):
            rank = 1
        else:
            rank = 2
        # Gemini / Anthropic 的所有对话模型都可以看图，优先短名（通常更便宜、更快）
        if proto in ("gemini", "anthropic"):
            rank = min(rank, 1)
        return (rank, len(low), low)

    return min(candidates, key=sort_key)


def resolve_model(profile: ApiProfile, timeout: float = 20) -> str:
    """返回配置的模型；未配置时自动拉取并选择视觉模型。"""
    model = (profile.model or "").strip()
    if model:
        return model
    try:
        models = discover_models(profile, timeout=timeout, use_cache=True)
    except AIRequestError as exc:
        raise AIRequestError(
            f"{exc}；请点击“拉取模型”选择模型，或手动填写模型名") from exc
    model = pick_model(models, profile.protocol)
    if not model:
        raise AIRequestError("接口没有返回可用模型，请点击“拉取模型”或手动填写模型名")
    return model


# ======================================================================
# 6. 生成
# ======================================================================

def generate(profile: ApiProfile, prompt: str, image_jpeg: bytes,
             timeout: float = _TIMEOUT) -> tuple[str, str]:
    """调用一个接口生成文本，返回 (文本, 实际使用的模型)。"""
    model = resolve_model(profile, timeout=min(timeout, 20))
    url, headers, body = build_request(profile, prompt, image_jpeg, model)
    try:
        data = _post_json_adaptive(url, headers, body, timeout)
    except AIRequestError as exc:
        # Gemini 代理有时只接受 ?key= 鉴权
        if (normalize_protocol(profile.protocol) == "gemini"
                and profile.api_key and exc.status in (400, 401, 403)):
            retry_headers = {k: v for k, v in headers.items()
                             if k.lower() != "x-goog-api-key"}
            data = _post_json_adaptive(
                _append_query(url, {"key": profile.api_key}),
                retry_headers, body, timeout)
        else:
            raise
    text = extract_response_text(profile.protocol, data)
    if not text:
        raise AIRequestError("接口返回了空回复")
    return text, model
