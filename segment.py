"""FernLens 本地像素级 AI 分割：SAM 2.1 Hiera-Tiny + U²-Net + SkySeg + YOLOv8n。

全部走 ONNX Runtime（CPU），无需 PyTorch；模型权重从 hf-mirror.com 下载或
随安装包分发，位于仓库 models/ 目录。四个识别目标的策略：

- subject（主体）  : U²-Net 显著性 -> 阈值 + 最大连通域 -> 点/框提示 SAM 2 -> 像素掩码
- person（人物）   : YOLOv8n 人体检测（COCO class 0）-> 每框提示 SAM 2 -> 掩码并集
- sky（天空）      : SkySeg（U²-Net 架构天空权重）直接输出像素概率图
- background（背景）: 1 - 主体掩码

输出统一为最长边 1024 的单通道软掩码（256 级），base64 PNG 存入
MaskState.mask_png，由 core._ai_mask 渲染（羽化/反向/局部调整全部复用）。

依赖：onnxruntime（可选——缺失时本地引擎不可用，VLM 网格引擎不受影响）。
"""
from __future__ import annotations

import os
import sys
import threading
import urllib.request

import cv2
import numpy as np

import core

# ---------------------------------------------------------------- 模型注册
# (文件名, 期望字节数, HuggingFace 仓库路径)。期望大小用于完整性校验（±2% 容差）。
MODELS: dict = {
    "sam_encoder": ("sam2.1_hiera_tiny.encoder.onnx", 109_443_000,
                    "akiyamanx/sam2.1-hiera-tiny-onnx/resolve/main/sam2.1_hiera_tiny.encoder.onnx"),
    "sam_decoder": ("sam2.1_hiera_tiny.decoder.onnx", 16_590_000,
                    "akiyamanx/sam2.1-hiera-tiny-onnx/resolve/main/sam2.1_hiera_tiny.decoder.onnx"),
    "u2net": ("u2net.onnx", 176_000_000,
              "Heliosoph/u2net-onnx/resolve/main/u2net.onnx"),
    "skyseg": ("skyseg.onnx", 175_997_079,
               "JianyuanWang/skyseg/resolve/main/skyseg.onnx"),
    "yolov8n": ("yolov8n.onnx", 12_600_000,
                "salim4n/yolov8n-detect-onnx/resolve/main/yolov8n-onnx-web/yolov8n.onnx"),
    "depth_anything_v2": ("depth_anything_v2_small_q.onnx", 27_258_801,
                          "onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx"),
}

def _models_dir() -> str:
    if getattr(sys, "frozen", False):
        # PyInstaller：内置 models 落在 _MEIPASS/models，优先使用
        bundled = os.path.join(getattr(sys, "_MEIPASS", ""), "models")
        if os.path.isdir(bundled):
            return bundled
        # 未内置模型：落到 exe 同级可写目录，供按需下载（结果可持久保存）
        return os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "models")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


MODELS_DIR = _models_dir()
_HF_PRIMARY = "https://huggingface.co/"
_HF_FALLBACK = "https://hf-mirror.com/"

# 掩码输出分辨率（最长边；归一化存储，渲染时放大）
MASK_LONG_EDGE = 1024


def model_path(key: str) -> str:
    return os.path.join(MODELS_DIR, MODELS[key][0])


def models_status() -> dict:
    """各模型文件是否就绪（存在且大小在期望值 ±2% 内）。"""
    out = {}
    for key, (name, size, _) in MODELS.items():
        p = os.path.join(MODELS_DIR, name)
        ok = False
        if os.path.isfile(p):
            got = os.path.getsize(p)
            ok = abs(got - size) / max(size, 1) <= 0.02
        out[key] = {"file": name, "ready": ok,
                    "size": os.path.getsize(p) if os.path.isfile(p) else 0}
    return out


def models_ready() -> bool:
    return all(v["ready"] for v in models_status().values())


# ---------------------------------------------------------------- 下载器
_download_state: dict = {"running": False, "done": False, "error": "",
                         "current": "", "received": 0, "total": 0}
_download_lock = threading.Lock()


def download_progress() -> dict:
    with _download_lock:
        return {**_download_state, "models": models_status()}


def _download_file(url: str, part: str, size: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "FernLens/1.0"})
    got = os.path.getsize(part) if os.path.isfile(part) else 0
    if got:
        req.add_header("Range", f"bytes={got}-")   # 断点续传
    with urllib.request.urlopen(req, timeout=60) as resp, \
            open(part, "ab" if got else "wb") as f:
        total = got + int(resp.headers.get("Content-Length") or 0)
        with _download_lock:
            _download_state.update(received=got, total=total or size)
        while True:
            chunk = resp.read(1 << 18)
            if not chunk:
                break
            f.write(chunk)
            with _download_lock:
                _download_state["received"] += len(chunk)


def _download_all() -> None:
    global _download_state
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        for key, (name, size, path) in MODELS.items():
            dst = model_path(key)
            if os.path.isfile(dst) and abs(os.path.getsize(dst) - size) / max(size, 1) <= 0.02:
                continue                                   # 已就绪
            with _download_lock:
                _download_state.update(current=name, received=0, total=size, error="")
            part = dst + ".part"
            try:
                _download_file(_HF_PRIMARY + path, part, size)
            except Exception:
                if os.path.isfile(part):
                    try:
                        os.remove(part)
                    except OSError:
                        pass
                _download_file(_HF_FALLBACK + path, part, size)
            if abs(os.path.getsize(part) - size) / max(size, 1) > 0.02:
                raise RuntimeError(f"{name} 大小校验失败（期望 {size} 字节）")
            os.replace(part, dst)
        with _download_lock:
            _download_state.update(running=False, done=True, current="",
                                   error="", received=_download_state["total"])
    except Exception as exc:
        with _download_lock:
            _download_state.update(running=False, error=str(exc))


def start_download() -> dict:
    """启动后台下载线程（重复调用幂等）。"""
    with _download_lock:
        if _download_state["running"]:
            return {"started": False, "reason": "already"}
        _download_state.update(running=True, done=False, error="")
    threading.Thread(target=_download_all, daemon=True,
                     name="fernlens-model-dl").start()
    return {"started": True}


# ---------------------------------------------------------------- onnxruntime
try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    ort = None
    _HAS_ORT = False


def has_runtime() -> bool:
    return _HAS_ORT


_sessions: dict = {}
_sessions_lock = threading.Lock()


def _session(key: str):
    """惰性加载并缓存 InferenceSession（Run 本身线程安全）。"""
    if not _HAS_ORT:
        raise RuntimeError("未安装 onnxruntime，无法本地分割。请执行："
                           "pip install onnxruntime")
    with _sessions_lock:
        if key not in _sessions:
            if not models_status()[key]["ready"]:
                raise RuntimeError(f"模型 {MODELS[key][0]} 未就绪，请先在「AI 设置」下载")
            so = ort.SessionOptions()
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = max(1, os.cpu_count() or 4)
            _sessions[key] = ort.InferenceSession(
                model_path(key), so, providers=["CPUExecutionProvider"])
    return _sessions[key]


def drop_sessions() -> None:
    with _sessions_lock:
        _sessions.clear()


# ---------------------------------------------------------------- 通用工具
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _resize_long_edge(rgb8: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = rgb8.shape[:2]
    s = long_edge / float(max(h, w))
    if s >= 1.0:
        return rgb8
    return cv2.resize(rgb8, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


# ---------------------------------------------------------------- U²-Net 家族
def _u2net_run(key: str, rgb8: np.ndarray) -> np.ndarray:
    """U²-Net 家族推理（u2net 显著性 / skyseg 天空）-> 原图像素概率图。

    预处理与官方一致：320×320、/255、ImageNet 归一化、HWC->CHW；
    输出取第一个头（d0，最高分辨率）。多数导出已含 sigmoid——按值域
    检测避免双重激活。"""
    small = cv2.resize(rgb8, (320, 320), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / np.float32(255.0)
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = x.transpose(2, 0, 1)[None]                       # (1,3,320,320)
    sess = _session(key)
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    d0 = np.asarray(out[0][0, 0]) if np.asarray(out[0]).ndim == 4 \
        else np.asarray(out[0])[0]
    if float(d0.min()) < 0.0 or float(d0.max()) > 1.0:  # 未激活的 logits
        d0 = _sigmoid(d0)
    prob = np.clip(d0, 0.0, 1.0).astype(np.float32)
    return cv2.resize(prob, (rgb8.shape[1], rgb8.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- YOLOv8n
def _yolo_person_boxes(rgb8: np.ndarray, conf_thr: float = 0.35) -> list:
    """YOLOv8n 检测 person（COCO class 0），返回原图坐标框 [(x0,y0,x1,y1)]。

    ONNX 输出 (1,84,N)：前 4 行 cx/cy/w/h，第 5 行起为 80 类分数（无 objectness）。
    """
    h, w = rgb8.shape[:2]
    s = 640.0 / max(h, w)
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:rh, :rw] = cv2.resize(rgb8, (rw, rh), interpolation=cv2.INTER_LINEAR)
    x = canvas.astype(np.float32) / np.float32(255.0)
    x = x.transpose(2, 0, 1)[None]

    sess = _session("yolov8n")
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    pred = np.asarray(out[0])                            # (1,84,N)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] != 84:                              # (N,84) 布局
        pred = pred.T
    conf = pred[4]                                       # person = class 0
    keep = conf >= conf_thr
    boxes, scores = [], []
    cx, cy, bw, bh = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
    for i in range(int(keep.sum())):
        x0 = max(0.0, (cx[i] - bw[i] / 2) / s)
        y0 = max(0.0, (cy[i] - bh[i] / 2) / s)
        x1 = min(float(w), (cx[i] + bw[i] / 2) / s)
        y1 = min(float(h), (cy[i] + bh[i] / 2) / s)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        boxes.append([x0, y0, x1, y1])
        scores.append(float(conf[keep][i]))
    return _nms(boxes, scores)


def _nms(boxes: list, scores: list, iou_thr: float = 0.45) -> list:
    keep: list = []
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    suppressed = [False] * len(boxes)
    for ia in order:
        if suppressed[ia]:
            continue
        keep.append(boxes[ia])
        ax0, ay0, ax1, ay1 = boxes[ia]
        for ib in order:
            if ib == ia or suppressed[ib]:
                continue
            bx0, by0, bx1, by1 = boxes[ib]
            ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            iy = max(0.0, min(ay1, by1) - max(ay0, by0))
            inter = ix * iy
            union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
            if union > 0 and inter / union > iou_thr:
                suppressed[ib] = True
    return keep


# ---------------------------------------------------------------- SAM 2.1
# 预处理：最长边 1024 等比缩放 + 右下 pad 到 1024×1024，
# ImageNet-1k 均值方差归一化，HWC->CHW（SAM 官方约定）。
_SAM_EDGE = 1024
_SAM_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_SAM_STD = np.array([58.395, 57.12, 57.375], np.float32)


def _sam_preprocess(rgb8: np.ndarray):
    h, w = rgb8.shape[:2]
    scale = _SAM_EDGE / float(max(h, w))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.zeros((_SAM_EDGE, _SAM_EDGE, 3), np.uint8)
    canvas[:rh, :rw] = cv2.resize(rgb8, (rw, rh), interpolation=cv2.INTER_LINEAR)
    x = (canvas.astype(np.float32) - _SAM_MEAN) / _SAM_STD
    return x.transpose(2, 0, 1)[None], scale, (rw, rh)


class _Sam2:
    """SAM 2.1 Hiera-Tiny 推理（已按实测签名固化）。

    encoder: image(1,3,1024,1024) -> (hr0, hr1, image_embed) 三输出
    decoder: 上述三嵌入 + point_coords/point_labels（1024 域坐标）/mask_input/
             has_mask_input -> masks(n,3,256,256 logits) + iou(n,3)
    输出 3 个候选掩码按 iou 分数取最优；同一图的 encoder 嵌入跨提示复用。
    """

    def __init__(self):
        self._embed_cache: tuple = ()        # (id(rgb), embed 三件套, scale, 尺寸)

    def encode(self, rgb8: np.ndarray):
        cached = self._embed_cache
        if cached and cached[0] == id(rgb8) and cached[4] == rgb8.shape[:2]:
            return cached[1], (cached[2], cached[3])
        x, scale, (rw, rh) = _sam_preprocess(rgb8)
        enc = _session("sam_encoder").run(None, {"image": x})
        embeds = (enc[2], enc[0], enc[1])    # image_embed, hr0, hr1（模型声明序）
        self._embed_cache = (id(rgb8), embeds, scale, (rw, rh), rgb8.shape[:2])
        return embeds, (scale, (rw, rh))

    def predict(self, rgb8: np.ndarray, points: list, boxes: list) -> np.ndarray:
        """点/框提示（原图坐标）-> (H,W) 0..1 软掩码（原图分辨率）。

        points: [(x,y)] 正点击；boxes: [(x0,y0,x1,y1)]。每个提示独立跑
        decoder（坐标乘 scale 进 1024 域），结果取像素最大值（多实例并集）。
        """
        h, w = rgb8.shape[:2]
        embeds, (scale, (rw, rh)) = self.encode(rgb8)
        prompts = [{"pts": [pt], "box": None} for pt in points] \
            + [{"pts": [], "box": b} for b in boxes]
        result = np.zeros((h, w), np.float32)
        if not prompts:
            return result
        sess = _session("sam_decoder")
        for p in prompts:
            coords, labels = [], []
            for px, py in p["pts"]:
                coords.append([px * scale, py * scale])
                labels.append(1)
            if p["box"] is not None:
                bx0, by0, bx1, by1 = p["box"]
                coords += [[bx0 * scale, by0 * scale], [bx1 * scale, by1 * scale]]
                labels += [2, 3]                          # 框 = 左上/右下两点
            if not coords:
                continue
            feeds = {
                "image_embed": embeds[0],
                "high_res_feats_0": embeds[1],
                "high_res_feats_1": embeds[2],
                "point_coords": np.asarray([coords], np.float32),
                "point_labels": np.asarray([labels], np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), np.float32),
                "has_mask_input": np.array([0], np.float32),
            }
            masks, iou = sess.run(None, feeds)
            masks, iou = np.asarray(masks)[0], np.asarray(iou)[0]
            best = int(np.argmax(iou))                    # 3 候选按分数选优
            prob = np.clip(_sigmoid(masks[best]), 0.0, 1.0)
            ph = max(1, int(round(256 * rh / _SAM_EDGE)))  # 去掉 pad 区域
            pw = max(1, int(round(256 * rw / _SAM_EDGE)))
            prob = prob[:ph, :pw]
            result = np.maximum(result, cv2.resize(prob, (w, h),
                                                   interpolation=cv2.INTER_LINEAR))
        return result


_sam2 = _Sam2()


# ---------------------------------------------------------------- 目标策略
def _largest_component(mask01: np.ndarray):
    """最大连通域 -> (质心点列表, 外接框列表)；无有效主体返回 (None, None)。"""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        mask01.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w2, h2, area = (int(stats[i, j]) for j in range(5))
    if area < mask01.size * 0.002:                      # 过小视为无主体
        return None, None
    cx, cy = float(cent[i][0]), float(cent[i][1])
    return [(cx, cy)], [(float(x), float(y), float(x + w2), float(y + h2))]


_LABELS = {
    "subject": "主体",
    "person": "人物",
    "background": "背景",
    "sky": "天空",
    "depth_near": "近景景深",
    "depth_far": "远景景深",
}


# ---------------------------------------------------------------- Depth Anything V2
_depth_cache: tuple = ()
_depth_lock = threading.Lock()


def _img_sig(rgb8: np.ndarray) -> tuple:
    step_y = max(1, rgb8.shape[0] // 32)
    step_x = max(1, rgb8.shape[1] // 32)
    return (rgb8.shape[:2], hash(rgb8[::step_y, ::step_x].tobytes()))


def estimate_depth(rgb8: np.ndarray) -> np.ndarray:
    """Depth Anything V2 Small ONNX 深度估计 -> (H, W) float32 ∈ [0, 1]（1=最近前景, 0=最远背景）。"""
    global _depth_cache
    sig = _img_sig(rgb8)
    with _depth_lock:
        if _depth_cache and _depth_cache[0] == sig:
            return _depth_cache[1]

    h, w = rgb8.shape[:2]
    scale = 518.0 / float(max(h, w))
    rh = max(14, (round(h * scale) // 14) * 14)
    rw = max(14, (round(w * scale) // 14) * 14)
    small = cv2.resize(rgb8, (rw, rh), interpolation=cv2.INTER_CUBIC)
    x = small.astype(np.float32) / np.float32(255.0)
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = x.transpose(2, 0, 1)[None]

    sess = _session("depth_anything_v2")
    raw = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0][0], np.float32)
    lo = float(np.percentile(raw, 1.0))
    hi = float(np.percentile(raw, 99.0))
    if hi - lo < 1e-5:
        norm = np.full_like(raw, 0.5)
    else:
        norm = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    depth = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
    with _depth_lock:
        _depth_cache = (sig, depth)
    return depth


def depth_masks(rgb8: np.ndarray, near_bias: float = 0.0) -> tuple:
    """生成平滑前景/远景景深软掩码 + 缩略深度预览图（base64 PNG）。"""
    depth = estimate_depth(rgb8)
    bias = float(np.clip(near_bias, -0.35, 0.35))
    e0 = np.float32(np.clip(0.32 + bias, 0.02, 0.85))
    e1 = np.float32(np.clip(0.72 + bias, e0 + 0.08, 0.99))
    near = core.smoothstep(e0, e1, depth)
    far = np.float32(1.0) - near
    preview = _resize_long_edge((np.clip(depth, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), 360)
    preview_b64, pw, ph = core.encode_mask_png(preview.astype(np.float32) / np.float32(255.0))
    return near, far, preview_b64, pw, ph


def _to_mask_dict(target: str, prob: np.ndarray, model_name: str = "local(SAM2.1-tiny+U2Net)") -> dict:
    """概率图 -> 统一 1024 长边 base64 PNG 响应。"""
    prob = np.clip(prob, 0.0, 1.0)
    u8 = (prob * 255.0 + 0.5).astype(np.uint8)
    u8 = _resize_long_edge(u8, MASK_LONG_EDGE)
    if float(u8.max()) < 128:
        raise ValueError(f"画面中未识别到「{_LABELS[target]}」")
    b64, w, h = core.encode_mask_png(u8.astype(np.float32) / np.float32(255.0))
    return {"target": target, "mask_png": b64, "width": w, "height": h,
            "model": model_name}


def segment_local(rgb8: np.ndarray, target: str) -> dict:
    """本地像素级分割主入口。未识别到目标抛 ValueError（与 VLM 引擎语义一致）。"""
    if target not in _LABELS:
        raise ValueError(f"未知的 AI 蒙版目标：{target}")

    if target in ("subject", "background"):
        sal = _u2net_run("u2net", rgb8)
        pts, boxes = _largest_component(sal >= 0.5)
        if pts is None:
            raise ValueError("画面中未识别到「主体」")
        prob = _sam2.predict(rgb8, pts, boxes)
        if float(prob.max()) < 0.5:                     # SAM 异常时回退显著性
            prob = sal
        return _to_mask_dict(target, 1.0 - prob if target == "background" else prob)

    if target == "person":
        boxes = _yolo_person_boxes(rgb8)
        if not boxes:
            raise ValueError("画面中未识别到「人物」")
        return _to_mask_dict("person", _sam2.predict(rgb8, [], boxes))

    if target in ("depth_near", "depth_far"):
        near, far, _, _, _ = depth_masks(rgb8)
        prob = near if target == "depth_near" else far
        return _to_mask_dict(target, prob, "local(Depth-Anything-V2-Small)")

    return _to_mask_dict("sky", _u2net_run("skyseg", rgb8))
