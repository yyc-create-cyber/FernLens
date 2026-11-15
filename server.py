"""FernLens 本地服务端：把 core.py 的处理流水线暴露为 HTTP / WebSocket API。

架构：
- 参数即文档：完整 Adjustments JSON 随每次渲染请求由客户端发来，服务端对参数无状态；
  服务端只持有文档像素状态（源图 / 仿制图章工作副本 / 数据版本号）。
- REST 负责一切"变更"（打开/保存/图章/几何/自动调色），WebSocket 只回传渲染帧。
- 帧协议（二进制）：[uint32 BE rid][uint8 kind 0=preview 1=full][JPEG bytes]。
- 任务循环逐字复刻 ui_widgets.ProcessorThread 的语义：
  普通任务最新覆盖、urgent（保存）独立槽位、线性光缓存按（数据源, 版本号）失效、
  协作取消只在停服时触发。FERNLENS_SYNC=1 时在提交线程内同步执行（测试用）。
- 请求 id（rid）由客户端生成并随任务透传回帧头，客户端按 rid 丢弃过期帧；
  保存另用服务端自增的 save id 关联渲染结果与目标路径。

启动：python server.py [--port N] [--token XXX] [--no-auth]
就绪后向 stdout 打印一行：FERNLENS_SERVER_READY http://127.0.0.1:<port> token=<token>
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import itertools
import json
import os
import secrets
import socket
import struct
import sys
import threading
import traceback
from typing import Callable

import cv2
import numpy as np
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Query,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.responses import Response
from pydantic import BaseModel

from core import (Adjustments, CancelledError, GRADING_RANGES, HIST_BINS,
                  MaskAdjustments, MaskState, PickerState, WheelState,
                  apply_depth_of_field, copy_patch, crop_rotate,
                  depth_focus_preview, downscale_max_edge, hue_degrees,
                  linear_histograms, norm_rect_px, process_linear,
                  remap_mask_geometry, remap_masks_rot90, rgb8_to_linear_mt,
                  rotate90, sample_patch, stamp_brush)
import core
import auto
import segment

try:
    import rawpy
except ImportError:        # 与旧版一致：rawpy 可选，缺失只是打不开 RAW
    rawpy = None

try:
    import pyopencl  # noqa: F401  仅探测可用性
    _HAS_PYOPENCL = True
except ImportError:
    _HAS_PYOPENCL = False

# 与 ui_window.RAW_EXTENSIONS 相同（不能直接 import：那会拉起 PySide6）
RAW_EXTENSIONS = (".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf",
                  ".sr2", ".raw", ".rw2", ".raf", ".orf", ".pef", ".ptx", ".x3f",
                  ".r3d", ".fff", ".erf", ".kdc", ".dcr", ".mef", ".mos", ".mrw")

PREVIEW = "preview"
FULL = "full"
SYNC_MODE = (os.environ.get("FERNLENS_SYNC") or os.environ.get("PHOTOLAB_SYNC")) == "1"
_FRAME_JPEG_QUALITY = 90


# ======================================================================
# 1. 参数序列化（Adjustments <-> JSON dict）
# ======================================================================
_TONE_FIELDS = ("temperature", "tint", "exposure", "contrast", "highlights",
                "shadows", "whites", "blacks", "saturation", "vibrance",
                "blackwhite", "clarity", "denoise", "sharpen", "dehaze", "blend")


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def adjustments_to_dict(a: Adjustments) -> dict:
    return dataclasses.asdict(a)


def _curve_points(raw) -> list:
    """曲线控制点 -> [(x, y), ...] float 元组列表；非法形状的条目直接丢弃。"""
    pts = []
    for p in (raw or []):
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                continue
    return pts


def _wheel_from_dict(raw) -> WheelState:
    raw = raw or {}
    return WheelState(hue=_f(raw.get("hue")), saturation=_f(raw.get("saturation")),
                      luminance=_f(raw.get("luminance")))


def _picker_from_dict(raw) -> PickerState:
    raw = raw or {}
    color = raw.get("color") or (0.5, 0.5, 0.5)
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        color = (0.5, 0.5, 0.5)
    return PickerState(hue=_f(raw.get("hue")), color=tuple(_f(c) for c in color),
                       span=_f(raw.get("span"), 30.0), hue_shift=_f(raw.get("hue_shift")),
                       saturation=_f(raw.get("saturation")),
                       luminance=_f(raw.get("luminance")),
                       enabled=bool(raw.get("enabled", True)))


def _mask_adjustments_from_dict(raw) -> MaskAdjustments:
    raw = raw or {}
    adj = MaskAdjustments()
    for name in ("exposure", "contrast", "highlights", "shadows", "whites", "blacks",
                 "temperature", "tint", "saturation", "vibrance",
                 "clarity", "denoise", "sharpen", "dehaze"):
        if name in raw:
            setattr(adj, name, _f(raw[name]))
    if "curves" in raw:
        adj.curves = {ch: _curve_points(raw["curves"].get(ch))
                      for ch in ("master", "red", "green", "blue")}
    if "blend" in raw:
        adj.blend = _f(raw["blend"], 50.0)
    if "wheels" in raw:
        adj.wheels = {r: _wheel_from_dict(raw["wheels"].get(r)) for r in GRADING_RANGES}
    if "pickers" in raw:
        adj.pickers = [_picker_from_dict(p) for p in (raw["pickers"] or [])]
    return adj


def _mask_from_dict(raw) -> MaskState:
    raw = raw or {}
    defaults = MaskState()
    m = MaskState(kind=str(raw.get("kind") or defaults.kind),
                  name=str(raw.get("name") or ""),
                  enabled=bool(raw.get("enabled", True)),
                  invert=bool(raw.get("invert", False)),
                  feather=_f(raw.get("feather"), defaults.feather),
                  size=_f(raw.get("size"), defaults.size))
    for name in ("x1", "y1", "x2", "y2", "cx", "cy", "radius", "aspect", "angle"):
        if name in raw:
            setattr(m, name, _f(raw[name], getattr(defaults, name)))
    m.strokes = [[(float(p[0]), float(p[1])) for p in stroke
                  if isinstance(p, (list, tuple)) and len(p) == 2]
                 for stroke in (raw.get("strokes") or [])]
    m.points = [(float(p[0]), float(p[1])) for p in (raw.get("points") or [])
                if isinstance(p, (list, tuple)) and len(p) == 2]
    # AI 蒙版：本地像素掩码（base64 PNG）；旧文档里的网格（'0'/'1' 字符串）仍可读
    if m.kind == "ai":
        try:
            gw = max(0, min(int(raw.get("grid_w") or 0), 512))
            gh = max(0, min(int(raw.get("grid_h") or 0), 512))
        except (TypeError, ValueError):
            gw = gh = 0
        rows = ["".join(ch for ch in str(r) if ch in "01")[:gw]
                for r in (raw.get("grid") or [])][:gh] if (gw and gh) else []
        rows = [r.ljust(gw, "0") for r in rows]
        while len(rows) < gh:
            rows.append("0" * gw)
        m.ai_target = str(raw.get("ai_target") or "")[:32]
        m.grid = rows
        m.grid_w = gw
        m.grid_h = gh
        png = str(raw.get("mask_png") or "")
        if png and len(png) <= 4_000_000:               # ~3MB 掩码上限
            try:
                mw = max(0, min(int(raw.get("mask_w") or 0), 4096))
                mh = max(0, min(int(raw.get("mask_h") or 0), 4096))
            except (TypeError, ValueError):
                mw = mh = 0
            if core.decode_mask_png(png).size and mw and mh:
                m.mask_png, m.mask_w, m.mask_h = png, mw, mh
    m.adj = _mask_adjustments_from_dict(raw.get("adj"))
    return m


def adjustments_from_dict(d: dict) -> Adjustments:
    """JSON dict -> Adjustments。缺省字段取默认值（向前兼容），非法值忽略。"""
    d = d or {}
    defaults = Adjustments()
    a = Adjustments()
    for name in _TONE_FIELDS:
        if name in d:
            setattr(a, name, _f(d[name], getattr(defaults, name)))
    if "curves" in d:
        a.curves = {ch: _curve_points(d["curves"].get(ch))
                    for ch in ("master", "red", "green", "blue")}
    if "wheels" in d:
        a.wheels = {r: _wheel_from_dict(d["wheels"].get(r)) for r in GRADING_RANGES}
    if "pickers" in d:
        a.pickers = [_picker_from_dict(p) for p in (d["pickers"] or [])]
    if "masks" in d:
        a.masks = [_mask_from_dict(m) for m in (d["masks"] or [])]
    return a


def masks_from_dict(raw) -> list:
    return [_mask_from_dict(m) for m in (raw or [])]


def masks_to_dict(masks) -> list:
    return [dataclasses.asdict(m) for m in masks]


# ======================================================================
# 2. 文档状态（源图 / 图章工作副本 / 几何），对照 ui_window 相应逻辑
# ======================================================================
class Document:
    """当前打开的一幅图像及其像素级修改。参数不在服务端，见模块 docstring。"""

    def __init__(self, name: str = ""):
        self.name = name                    # 展示名（文件名，去扩展名）
        self.source: np.ndarray | None = None       # 全图 uint8 RGB（含已生效修改）
        self.preview_src: np.ndarray | None = None  # 全图降采样
        self.retouch_full: np.ndarray | None = None     # 图章工作副本
        self.retouch_preview: np.ndarray | None = None
        self.clone_ops: list = []                   # [(src_rect_norm, dst_rect_norm, feather_pct)]
        self.stamp_src_rect: tuple | None = None
        self.dof_params: dict | None = None         # {focus, aperture, dof_range, bokeh_highlight}
        self.dof_full: np.ndarray | None = None
        self.dof_preview: np.ndarray | None = None
        self._dof_depth_full: np.ndarray | None = None
        self._dof_depth_preview: np.ndarray | None = None
        self.serial = 0                             # 数据版本号，随像素修改自增

    # ---- 打开 ----
    def open(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in RAW_EXTENSIONS:
            rgb = self._load_raw(path)
        else:
            try:
                data = np.fromfile(path, dtype=np.uint8)     # 兼容中文路径
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                raise ValueError(f"无法解码图像：{path}")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.name = os.path.splitext(os.path.basename(path))[0] or self.name
        self._set_source_array(rgb)

    def open_bytes(self, data: bytes, ext: str = "", name: str = ""):
        """浏览器模式：直接从上传字节打开（RAW 需带扩展名以识别格式）。"""
        self.name = name or self.name
        if ext.lower() in RAW_EXTENSIONS:
            self._set_source_array(self._load_raw_bytes(data))
            return
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图像数据")
        self._set_source_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _load_raw_bytes(data: bytes) -> np.ndarray:
        if rawpy is None:
            raise RuntimeError("未安装 rawpy，无法打开 RAW。请先执行：pip install rawpy")
        try:
            import io
            with rawpy.imread(io.BytesIO(data)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=8,
                    no_auto_bright=True,
                    user_flip=0,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                )
        except Exception as exc:
            raise ValueError(f"RAW 解码失败：{exc}") from exc
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr

    def _clear_dof_state(self, drop_depth: bool = True):
        self.dof_params = None
        self.dof_full = None
        self.dof_preview = None
        if drop_depth:
            self._dof_depth_full = None
            self._dof_depth_preview = None

    def _set_source_array(self, rgb: np.ndarray):
        self.source = np.ascontiguousarray(rgb)
        self.preview_src = downscale_max_edge(self.source, 1200)
        # 新图像：丢弃图章与景深修改（与 ui_window.load_image 一致）
        self.retouch_full = None
        self.retouch_preview = None
        self.clone_ops.clear()
        self.stamp_src_rect = None
        self._clear_dof_state(drop_depth=True)
        self.serial += 1

    @staticmethod
    def _load_raw(path: str) -> np.ndarray:
        if rawpy is None:
            raise RuntimeError("未安装 rawpy，无法打开 RAW。请先执行：pip install rawpy")
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,                 # 相机白平衡
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=8,
                    no_auto_bright=True,                # 不做自动提亮，保留原始曝光
                    user_flip=0,                        # 不做旋转（EXIF 方向见 README）
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                )
        except Exception as exc:
            raise ValueError(f"RAW 解码失败：{path}\n{exc}") from exc
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr

    # ---- 图章与 AI 景深虚化（图章/景深均作用于渲染流水线之前的基底像素）----
    @property
    def _base_full(self) -> np.ndarray | None:
        return self.retouch_full if self.retouch_full is not None else self.source

    @property
    def _base_preview(self) -> np.ndarray | None:
        return self.retouch_preview if self.retouch_preview is not None else self.preview_src

    @property
    def active_full(self) -> np.ndarray | None:
        if self.dof_full is not None:
            return self.dof_full
        return self._base_full

    @property
    def active_preview(self) -> np.ndarray | None:
        if self.dof_preview is not None:
            return self.dof_preview
        return self._base_preview

    def _ensure_retouch(self):
        if self.retouch_full is None:
            self.retouch_full = self.source.copy()
            self.retouch_preview = (self.preview_src.copy()
                                    if self.preview_src is not None else None)

    def _run_clone_op(self, op) -> None:
        if len(op) == 3:
            src_r, dst_r, feather_pct = op
            fe = float(np.clip(feather_pct, 0.0, 100.0)) / 100.0
            h, w = self.retouch_full.shape[:2]
            copy_patch(self.retouch_full, norm_rect_px(src_r, h, w),
                       norm_rect_px(dst_r, h, w), fe)
            if self.retouch_preview is not None:
                ph, pw = self.retouch_preview.shape[:2]
                copy_patch(self.retouch_preview, norm_rect_px(src_r, ph, pw),
                           norm_rect_px(dst_r, ph, pw), fe)
            return
        _, src_start, pts, radius_norm, feather_pct, opacity_pct = op
        fe = float(np.clip(feather_pct, 0.0, 100.0)) / 100.0
        opp = float(np.clip(opacity_pct, 0.0, 100.0)) / 100.0
        stamp_brush(self.retouch_full, src_start, pts, radius_norm, fe, opp)
        if self.retouch_preview is not None:
            stamp_brush(self.retouch_preview, src_start, pts, radius_norm, fe, opp)

    def apply_stamp(self, src_rect, dst_rect, feather_pct: float) -> int:
        if self.source is None:
            raise RuntimeError("未打开图像")
        self._ensure_retouch()
        src_r = tuple(float(v) for v in src_rect)
        dst_r = tuple(float(v) for v in dst_rect)
        op = (src_r, dst_r, float(feather_pct))
        self._run_clone_op(op)
        self.clone_ops.append(op)
        if self.dof_params is not None:
            self._render_dof()
        self.serial += 1
        return self.serial

    def apply_stamp_brush(self, src_start, points, radius_norm: float,
                          feather_pct: float, opacity_pct: float = 100.0) -> int:
        if self.source is None:
            raise RuntimeError("未打开图像")
        if not points or len(src_start) < 2:
            raise RuntimeError("图章笔迹参数无效")
        self._ensure_retouch()
        src_pt = (float(src_start[0]), float(src_start[1]))
        pts = [(float(p[0]), float(p[1])) for p in points if len(p) >= 2]
        if not pts:
            raise RuntimeError("图章笔迹为空")
        op = ("brush", src_pt, pts, float(radius_norm), float(feather_pct), float(opacity_pct))
        self._run_clone_op(op)
        self.clone_ops.append(op)
        if self.dof_params is not None:
            self._render_dof()
        self.serial += 1
        return self.serial

    def _replay_stamps(self) -> int:
        """由原始源图重放全部覆盖操作（撤销/清除后调用）。"""
        self.retouch_full = None
        self.retouch_preview = None
        if self.clone_ops and self.source is not None:
            self._ensure_retouch()
            for op in self.clone_ops:
                self._run_clone_op(op)
        if self.dof_params is not None and self.source is not None:
            self._render_dof()
        self.serial += 1
        return self.serial

    def undo_stamp(self) -> int:
        if self.clone_ops:
            self.clone_ops.pop()
        return self._replay_stamps()

    def clear_stamps(self) -> int:
        self.clone_ops.clear()
        return self._replay_stamps()

    def bake_stamps(self) -> bool:
        """几何变换前把图章历史并入源图并清空（坐标已无法重放）。"""
        had = bool(self.clone_ops) or self.retouch_full is not None
        self.clone_ops.clear()
        self.retouch_full = None
        self.retouch_preview = None
        self.stamp_src_rect = None
        return had

    # ---- AI 调节景深（基于 HuggingFace Depth Anything V2 深度估计）----
    def _ensure_dof_depth(self) -> tuple:
        if self.source is None:
            raise RuntimeError("未打开图像")
        if self._dof_depth_full is None:
            base = self._base_full
            self._dof_depth_full = segment.estimate_depth(base)
            pv = self._base_preview
            if pv is not None:
                ph, pw = pv.shape[:2]
                if (ph, pw) == self._dof_depth_full.shape[:2]:
                    self._dof_depth_preview = self._dof_depth_full
                else:
                    self._dof_depth_preview = cv2.resize(
                        self._dof_depth_full, (pw, ph), interpolation=cv2.INTER_LINEAR)
            else:
                self._dof_depth_preview = None
        return self._dof_depth_full, self._dof_depth_preview

    def _render_dof(self) -> None:
        if self.dof_params is None:
            self.dof_full = None
            self.dof_preview = None
            return
        d_full, d_prev = self._ensure_dof_depth()
        p = self.dof_params
        if float(p["aperture"]) <= 0.5:
            self.dof_full = None
            self.dof_preview = None
            return
        base_full = self._base_full
        base_prev = self._base_preview
        self.dof_full = apply_depth_of_field(
            base_full, d_full,
            focus=p["focus"], aperture=p["aperture"],
            dof_range=p["dof_range"], bokeh_highlight=p["bokeh_highlight"],
        )
        if base_prev is not None and d_prev is not None:
            self.dof_preview = apply_depth_of_field(
                base_prev, d_prev,
                focus=p["focus"], aperture=p["aperture"],
                dof_range=p["dof_range"], bokeh_highlight=p["bokeh_highlight"],
            )
        else:
            self.dof_preview = None

    def pick_dof_focus(self, nx: float, ny: float, dof_range: float = 20.0) -> dict:
        d_full, d_prev = self._ensure_dof_depth()
        h, w = d_full.shape[:2]
        cx = int(np.clip(round(float(nx) * max(w - 1, 1)), 0, w - 1))
        cy = int(np.clip(round(float(ny) * max(h - 1, 1)), 0, h - 1))
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        focus = float(np.median(d_full[y0:y1, x0:x1]))
        vis_d = d_prev if d_prev is not None else d_full
        return {
            "focus": round(focus, 4),
            "depth_preview_png": depth_focus_preview(vis_d, focus, dof_range),
        }

    def apply_dof(self, focus: float, aperture: float, dof_range: float = 20.0,
                  bokeh_highlight: float = 25.0) -> dict:
        d_full, d_prev = self._ensure_dof_depth()
        fc = float(np.clip(focus, 0.0, 1.0))
        ap = float(np.clip(aperture, 0.0, 100.0))
        dr = float(np.clip(dof_range, 0.0, 80.0))
        bh = float(np.clip(bokeh_highlight, 0.0, 100.0))
        vis_d = d_prev if d_prev is not None else d_full
        preview_png = depth_focus_preview(vis_d, fc, dr)
        if ap <= 0.5:
            self._clear_dof_state(drop_depth=False)
            self.serial += 1
            return {
                "serial": self.serial,
                "active": False,
                "focus": fc,
                "aperture": 0.0,
                "dof_range": dr,
                "bokeh_highlight": bh,
                "depth_preview_png": preview_png,
            }
        self.dof_params = {
            "focus": fc,
            "aperture": ap,
            "dof_range": dr,
            "bokeh_highlight": bh,
        }
        self._render_dof()
        self.serial += 1
        return {
            "serial": self.serial,
            "active": True,
            **self.dof_params,
            "depth_preview_png": preview_png,
        }

    def clear_dof(self) -> int:
        had = self.dof_full is not None or self.dof_params is not None
        self._clear_dof_state(drop_depth=False)
        if had:
            self.serial += 1
        return self.serial

    # ---- 几何变换（对照 ui_window._apply_crop / _rotate90 / _commit_geometry）----
    def _commit_geometry(self, new_source: np.ndarray):
        self.source = np.ascontiguousarray(new_source)
        self.preview_src = downscale_max_edge(self.source, 1200)
        self._clear_dof_state(drop_depth=True)
        self.serial += 1

    def crop(self, rect_norm, angle_deg: float, masks) -> tuple:
        """rect_norm=(x0,y0,x1,y1) 归一化。返回 (new_masks, baked)。"""
        if self.source is None:
            raise RuntimeError("未打开图像")
        base = self.active_full
        rect_t = (float(rect_norm[0]), float(rect_norm[1]),
                  float(rect_norm[2]), float(rect_norm[3]))
        if abs(angle_deg) < 1e-9 and rect_t == (0.0, 0.0, 1.0, 1.0):
            new_src = np.ascontiguousarray(base)
            map_point = lambda nx, ny: (nx, ny)         # noqa: E731
            new_shape = new_src.shape[:2]
        else:
            new_src, map_point, new_shape = crop_rotate(base, rect_t, angle_deg)
        remap_mask_geometry(masks, map_point, base.shape[:2], new_shape,
                            angle_delta=-angle_deg)
        baked = self.bake_stamps()
        self._commit_geometry(new_src)
        return masks, baked

    def rotate(self, clockwise: bool, masks) -> tuple:
        if self.source is None:
            raise RuntimeError("未打开图像")
        base = self.active_full
        k = 3 if clockwise else 1
        new_src = rotate90(base, k)
        remap_masks_rot90(masks, k, base.shape[:2])
        self.bake_stamps()
        self._commit_geometry(new_src)
        return masks

    # ---- 派生数据 ----
    def image_at(self, x: int, y: int, patch: int = 5) -> dict:
        src = self.active_full
        color = sample_patch(src, int(x), int(y), patch)
        return {"r": float(color[0]), "g": float(color[1]), "b": float(color[2]),
                "hue": float(hue_degrees(tuple(float(c) for c in color)))}


class Library:
    """多图导入：每张图一个独立 Document（源图/图章/几何互不影响）。

    打开（/api/open、/api/open-upload）一律入列并选中新图；切换由
    /api/library/select 完成。全局 doc 始终指向当前条目的 Document。
    """

    def __init__(self):
        self.entries: list[dict] = []       # [{id, name, doc}]
        self._ids = itertools.count(1)

    def add(self, d: Document) -> int:
        eid = next(self._ids)
        self.entries.append({"id": eid, "name": d.name or f"图像 {eid}", "doc": d})
        return eid

    def get(self, eid: int) -> dict | None:
        for e in self.entries:
            if e["id"] == eid:
                return e
        return None

    def remove(self, eid: int) -> int | None:
        """删除条目，返回新的当前 id（库空返回 None）。"""
        idx = next((i for i, e in enumerate(self.entries) if e["id"] == eid), -1)
        if idx < 0:
            return None
        self.entries.pop(idx)
        if not self.entries:
            return None
        nxt = min(idx, len(self.entries) - 1)      # 优先取后一张，越界回最后
        return self.entries[nxt]["id"]

    def clear(self) -> None:
        self.entries.clear()

    def listing(self, current_id: int | None) -> list:
        return [{"id": e["id"], "name": e["name"],
                 "width": e["doc"].source.shape[1] if e["doc"].source is not None else 0,
                 "height": e["doc"].source.shape[0] if e["doc"].source is not None else 0,
                 "serial": e["doc"].serial,
                 "current": e["id"] == current_id}
                for e in self.entries]


# ======================================================================
# 3. 渲染任务循环——复刻 ui_widgets.ProcessorThread 语义（Qt 信号 -> 回调）
# ======================================================================
class RenderLoop(threading.Thread):
    """on_frame(arr, rid, is_full) 中的 rid 是任务提交时指定的 client_rid
    （未指定则为内部自增 id）；客户端据此丢弃过期帧。"""

    def __init__(self, on_frame: Callable, on_busy: Callable, on_error: Callable):
        super().__init__(daemon=True, name="fernlens-render")
        self.on_frame = on_frame
        self.on_busy = on_busy
        self.on_error = on_error
        self._cond = threading.Condition()
        self._job = None                 # 普通任务槽 (rid, kind, params, client_rid)，最新覆盖
        self._urgent = None              # 紧急任务槽（保存），不被普通任务覆盖
        self._sources: dict = {}         # kind -> uint8 RGB（私有副本）
        self._src_gen: dict = {}         # kind -> 数据版本号
        # (kind, edge) -> (gen, 线性光数组)：同一张图会按客户端需要的长边分级缓存
        # （拖动用低分辨率、停手用显示分辨率、保存用原图），避免每次都重做降采样与转换
        self._lin_cache: dict = {}
        self._lin_order: list = []       # 缓存键的插入顺序（超限时淘汰最旧）
        self._stopped = False
        self._req_counter = 0
        if not SYNC_MODE:
            self.start()

    # ------------------------------------------------------------ 控制
    def stop(self):
        with self._cond:
            self._stopped = True
            self._job = None
            self._urgent = None
            self._cond.notify_all()

    def set_source(self, kind: str, rgb8: np.ndarray | None):
        """设置数据源（拷贝一份，调用方之后原地修改不会撕裂渲染）；None 清除。"""
        with self._cond:
            if rgb8 is None:
                self._sources.pop(kind, None)
                self._src_gen.pop(kind, None)
                self._drop_lin(kind)
                return
            rgb8 = np.array(rgb8, copy=True)
            gen = self._src_gen.get(kind, 0) + 1
            self._sources[kind] = rgb8
            self._src_gen[kind] = gen
            self._drop_lin(kind)

    # ------------------------------------------------------------ 分级缓存
    def _drop_lin(self, kind: str | None = None, keep: int = 3):
        """清掉缓存：kind 为 None 时按 LRU 只保留最近 keep 级，避免大图吃满内存。"""
        if kind is None:
            keys = [k for k in self._lin_order if k in self._lin_cache]
            for k in keys[:-keep] if keep > 0 else keys:
                self._lin_cache.pop(k, None)
            self._lin_order = [k for k in self._lin_order if k in self._lin_cache]
            return
        for k in [k for k in self._lin_cache if k[0] == kind]:
            self._lin_cache.pop(k, None)
        self._lin_order = [k for k in self._lin_order if k in self._lin_cache]

    def submit(self, params: Adjustments, kind: str, urgent: bool = False,
               client_rid: int | None = None, edge: int = 0) -> int:
        """提交一次处理请求，返回内部请求 id。

        urgent 任务（保存）放入独立槽位：即使随后有普通任务提交也保证执行。
        edge > 0 时按该长边降采样渲染（屏幕只需要这么多像素，省时省内存）。
        """
        with self._cond:
            self._req_counter += 1
            rid = self._req_counter
            job = (rid, kind, params.copy(),
                   self._req_counter if client_rid is None else int(client_rid),
                   max(0, int(edge or 0)))
            if not SYNC_MODE:
                if urgent:
                    self._urgent = job
                else:
                    self._job = job
                self._cond.notify()
        if SYNC_MODE:
            self._execute(job)
        return rid

    # ------------------------------------------------------------ 线程体
    def run(self):
        while True:
            with self._cond:
                while self._urgent is None and self._job is None and not self._stopped:
                    self._cond.wait(0.2)
                if self._stopped:
                    return                 # 已请求停止：丢弃排队任务直接退出
                if self._urgent is not None:
                    job, self._urgent = self._urgent, None
                else:
                    job, self._job = self._job, None
            self._execute(job)

    def _execute(self, job):
        rid, kind, params, client_rid, edge = job
        with self._cond:
            src = self._sources.get(kind)
            gen = self._src_gen.get(kind, -1)
        if src is None:
            return
        # edge 比原图还大就没必要放大：当 0 处理
        long_edge = max(src.shape[0], src.shape[1])
        if edge >= long_edge:
            edge = 0
        key = (kind, edge)
        self.on_busy(True)
        try:
            with self._cond:
                cached = self._lin_cache.get(key)
            if cached is not None and cached[0] == gen:
                lin = cached[1]
            else:
                small = downscale_max_edge(src, edge) if edge else src
                lin = rgb8_to_linear_mt(small)
                with self._cond:           # 期间数据源被更换则丢弃本次缓存
                    if self._src_gen.get(kind, -1) == gen:
                        self._lin_cache[key] = (gen, lin)
                        self._lin_order = [k for k in self._lin_order if k != key] + [key]
                        self._drop_lin(None, keep=3)
            out = process_linear(params, lin, should_cancel=lambda: self._stopped)
        except CancelledError:             # 停止请求：静默丢弃半成品
            self.on_busy(False)
            return
        except Exception:
            traceback.print_exc()
            self.on_busy(False)
            try:
                self.on_error(client_rid, kind, traceback.format_exc(limit=5))
            except Exception:
                pass
            return
        self.on_busy(False)
        self.on_frame(out, client_rid, kind == FULL)


# ======================================================================
# 4. FastAPI 应用
# ======================================================================
app = FastAPI(title="FernLens Server", docs_url=None, redoc_url=None)

_state_lock = threading.Lock()
library = Library()                     # 多图图库；打开一律入列并选中
doc = Document()                        # 当前文档（图库条目之一；空库时为空文档）
loop: RenderLoop | None = None
_token = ""
_clients: set = set()
_asyncio_loop: asyncio.AbstractEventLoop | None = None
_pending_saves: dict = {}      # save_id -> path
_save_ids = itertools.count(1)
_last_frames: dict = {}        # kind -> uint8 RGB（直方图数据来源）


def _current_id() -> int | None:
    for e in library.entries:
        if e["doc"] is doc:
            return e["id"]
    return None


def _add_and_select(d: Document) -> int:
    """新图入列并设为当前（打开路径）。"""
    global doc
    with _state_lock:
        eid = library.add(d)
        doc = d
    _sync_sources()
    return eid


# ------------------------------------------------------------ 鉴权
async def _auth(token: str = Query(default=""),
                x_fernlens_token: str = Header(default="")):
    if _token and not secrets.compare_digest(token or "", _token) \
            and not secrets.compare_digest(x_fernlens_token or "", _token):
        raise HTTPException(status_code=401, detail="bad token")


# ------------------------------------------------------------ WebSocket 广播
def _schedule(coro):
    if _asyncio_loop is not None:
        _asyncio_loop.call_soon_threadsafe(_asyncio_loop.create_task, coro)


async def _send_to_clients(text: str | None = None, data: bytes | None = None):
    dead = []
    for ws in list(_clients):
        try:
            if text is not None:
                await ws.send_text(text)
            else:
                await ws.send_bytes(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def _broadcast_json(obj: dict):
    if _clients:
        _schedule(_send_to_clients(text=json.dumps(obj)))


def _broadcast_frame(arr: np.ndarray, rid: int, is_full: bool):
    _last_frames[FULL if is_full else PREVIEW] = arr
    if not _clients:
        return
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _FRAME_JPEG_QUALITY])
    if not ok:
        return
    payload = struct.pack(">IB", rid, 1 if is_full else 0) + buf.tobytes()
    _schedule(_send_to_clients(data=payload))


def _sync_sources():
    """把文档当前像素状态同步给渲染循环（两个数据源都换）。"""
    loop.set_source(PREVIEW, doc.active_preview)
    loop.set_source(FULL, doc.active_full)


def _on_frame(arr, rid, is_full):
    _broadcast_frame(arr, rid, is_full)
    with _state_lock:
        save_path = _pending_saves.pop(rid, None)
    if save_path is not None and is_full:
        ok, err = _write_image(arr, save_path)
        _broadcast_json({"type": "saved", "rid": rid, "path": save_path,
                         "ok": ok, "error": err})


def _on_busy(busy: bool):
    _broadcast_json({"type": "busy", "busy": bool(busy)})


def _on_error(rid, kind, message: str):
    _broadcast_json({"type": "error", "rid": rid, "kind": kind, "message": message})


def _encode_image(arr: np.ndarray, ext: str, quality: int = 95) -> tuple:
    """uint8 RGB -> (ok, 编码缓冲)。与 ui_window._write_image 相同的格式分支。"""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if ext in (".jpg", ".jpeg"):
        return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ext == ".png":
        return cv2.imencode(".png", bgr)
    if ext == ".webp":
        return cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if ext in (".tif", ".tiff"):
        return cv2.imencode(".tif", bgr)
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _write_image(arr: np.ndarray, path: str) -> tuple:
    """与 ui_window._write_image 相同的格式与参数。返回 (ok, error)。"""
    try:
        ok, buf = _encode_image(arr, os.path.splitext(path)[1].lower())
        if not ok:
            return False, f"编码失败（不支持的格式？）：{path}"
        buf.tofile(path)               # numpy 写入，支持中文路径
        return True, ""
    except Exception as exc:
        traceback.print_exc()
        return False, f"写入失败：{path}\n{exc}"


# ------------------------------------------------------------ 请求模型
class OpenReq(BaseModel):
    path: str


class SaveReq(BaseModel):
    path: str
    adjustments: dict = {}


class RenderReq(BaseModel):
    adjustments: dict = {}
    format: str = "png"           # png | jpg | webp | tif
    quality: int = 95


class StampReq(BaseModel):
    src_rect: list                # 归一化 (x0, y0, x1, y1)
    dst_rect: list
    feather: float = 20.0         # 0..100


class StampBrushReq(BaseModel):
    src_start: list               # 归一化 [sx, sy]
    points: list                  # 归一化 [[x, y], ...]
    radius: float = 0.04          # 相对图像短边半径
    feather: float = 35.0         # 0..100
    opacity: float = 100.0        # 0..100


class DepthRelightReq(BaseModel):
    preset: str = "studio"        # studio | atmosphere | warm_cool | cinematic
    strength: float = 1.0         # 0..1.5
    near_bias: float = 0.0        # -0.35..0.35
    masks_only: bool = False


class DepthDofPickReq(BaseModel):
    x: float                      # 归一化 0..1
    y: float
    dof_range: float = 20.0


class DepthDofReq(BaseModel):
    focus: float = 0.75           # 对焦深度 0..1（1=最近，0=最远）
    aperture: float = 45.0        # 光圈虚化强度 0..100
    dof_range: float = 20.0       # 清晰景深宽度 0..80
    bokeh_highlight: float = 25.0 # 焦外高光散景增强 0..100


class CropReq(BaseModel):
    rect: list                    # 归一化 (x0, y0, x1, y1)
    angle: float = 0.0
    masks: list = []


class RotateReq(BaseModel):
    clockwise: bool = True
    masks: list = []


class PickReq(BaseModel):
    x: int                        # 全图像素坐标
    y: int
    patch: int = 5


class AiConfig(BaseModel):
    """前端「AI 设置」传入的 OpenAI 兼容端点配置。"""
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    def dict_plain(self) -> dict:
        return {"base_url": self.base_url, "api_key": self.api_key,
                "model": self.model}


class AiTestReq(BaseModel):
    config: AiConfig = AiConfig()


class AiMaskReq(BaseModel):
    target: str                   # subject | person | background | sky
    engine: str = "local"         # 只保留本地像素级（SAM2 + U²-Net）；字段保留兼容旧前端


# ------------------------------------------------------------ 路由
@app.get("/api/meta")
def api_meta(_=Depends(_auth)):
    src = doc.source
    return {"app": "FernLens", "version": "1.0.0-web",
            "rawpy": rawpy is not None, "gpu": _HAS_PYOPENCL,
            "width": int(src.shape[1]) if src is not None else 0,
            "height": int(src.shape[0]) if src is not None else 0,
            "hist_bins": HIST_BINS, "preview_max_edge": 1200,
            "dof": doc.dof_params}


@app.get("/api/defaults")
def api_defaults(_=Depends(_auth)):
    return {"adjustments": adjustments_to_dict(Adjustments())}


@app.post("/api/open")
def api_open(req: OpenReq, _=Depends(_auth)):
    d = Document()
    try:
        d.open(req.path)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    eid = _add_and_select(d)
    h, w = doc.source.shape[:2]
    return {"id": eid, "name": doc.name, "width": int(w), "height": int(h),
            "serial": doc.serial, "library": len(library.entries)}


@app.post("/api/open-upload")
async def api_open_upload(file: UploadFile = File(...), _=Depends(_auth)):
    """浏览器模式：上传文件打开（文件名扩展名用于识别 RAW）。入列并选中。"""
    data = await file.read()
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1]
    d = Document()
    try:
        with _state_lock:
            d.open_bytes(data, ext, name=os.path.splitext(os.path.basename(filename))[0])
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    eid = _add_and_select(d)
    h, w = doc.source.shape[:2]
    return {"id": eid, "name": doc.name, "width": int(w), "height": int(h),
            "serial": doc.serial, "library": len(library.entries)}


@app.post("/api/render")
def api_render(req: RenderReq, _=Depends(_auth)):
    """浏览器模式：同步渲染整幅并返回图像字节（下载保存用）。

    Electron 走 /api/save（urgent 槽位 + 服务端直写）；本端点仅在用户点击下载时调用，
    单用户场景下阻塞一次全图渲染可接受。
    """
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    fmt = str(req.format or "png").lower()
    ext = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png",
           "webp": ".webp", "tif": ".tif", "tiff": ".tif"}.get(fmt)
    if ext is None:
        raise HTTPException(status_code=400, detail=f"不支持的格式：{fmt}")
    with _state_lock:
        src = np.array(doc.active_full, copy=True)     # 快照，渲染期间允许换图
    try:
        params = adjustments_from_dict(req.adjustments)
        out = process_linear(params, rgb8_to_linear_mt(src))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"渲染失败：{exc}")
    ok, buf = _encode_image(out, ext, req.quality)
    if not ok:
        raise HTTPException(status_code=500, detail="图像编码失败")
    media = {".png": "image/png", ".jpg": "image/jpeg",
             ".webp": "image/webp", ".tif": "image/tiff"}[ext]
    return Response(content=buf.tobytes(), media_type=media)


@app.post("/api/save")
def api_save(req: SaveReq, _=Depends(_auth)):
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    path = req.path
    if not os.path.splitext(path)[1]:
        path += ".jpg"
    params = adjustments_from_dict(req.adjustments)
    save_id = next(_save_ids)
    with _state_lock:                  # 先登记再提交：同步模式下回调立即触发
        _pending_saves[save_id] = path
    loop.submit(params, FULL, urgent=True, client_rid=save_id)
    with _state_lock:
        saved = save_id not in _pending_saves
    return {"rid": save_id, "saved": saved, "path": path}


@app.post("/api/stamp")
def api_stamp(req: StampReq, _=Depends(_auth)):
    try:
        with _state_lock:
            serial = doc.apply_stamp(req.src_rect, req.dst_rect, req.feather)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _sync_sources()
    return {"serial": serial}


@app.post("/api/stamp/brush")
def api_stamp_brush(req: StampBrushReq, _=Depends(_auth)):
    try:
        with _state_lock:
            serial = doc.apply_stamp_brush(
                req.src_start, req.points, req.radius, req.feather, req.opacity)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _sync_sources()
    return {"serial": serial}


@app.post("/api/stamp/undo")
def api_stamp_undo(_=Depends(_auth)):
    with _state_lock:
        serial = doc.undo_stamp()
    _sync_sources()
    return {"serial": serial}


@app.post("/api/stamp/clear")
def api_stamp_clear(_=Depends(_auth)):
    with _state_lock:
        serial = doc.clear_stamps()
    _sync_sources()
    return {"serial": serial}


@app.post("/api/geometry/crop")
def api_crop(req: CropReq, _=Depends(_auth)):
    try:
        with _state_lock:
            masks = masks_from_dict(req.masks)
            masks, baked = doc.crop(req.rect, req.angle, masks)
            h, w = doc.source.shape[:2]
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"裁剪失败：{exc}")
    _sync_sources()
    return {"masks": masks_to_dict(masks), "baked": baked,
            "width": int(w), "height": int(h)}


@app.post("/api/geometry/rotate")
def api_rotate(req: RotateReq, _=Depends(_auth)):
    try:
        with _state_lock:
            masks = masks_from_dict(req.masks)
            masks = doc.rotate(req.clockwise, masks)
            h, w = doc.source.shape[:2]
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _sync_sources()
    return {"masks": masks_to_dict(masks), "width": int(w), "height": int(h)}


@app.post("/api/pick")
def api_pick(req: PickReq, _=Depends(_auth)):
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    with _state_lock:
        return doc.image_at(req.x, req.y, req.patch)


# ------------------------------------------------------------ 多图图库
class SelectReq(BaseModel):
    id: int


@app.get("/api/library")
def api_library(_=Depends(_auth)):
    return {"images": library.listing(_current_id())}


@app.post("/api/library/select")
def api_library_select(req: SelectReq, _=Depends(_auth)):
    global doc
    e = library.get(req.id)
    if e is None or e["doc"].source is None:
        raise HTTPException(status_code=400, detail="图像不在图库中")
    with _state_lock:
        doc = e["doc"]
    _sync_sources()
    h, w = doc.source.shape[:2]
    return {"id": req.id, "name": doc.name, "width": int(w), "height": int(h),
            "serial": doc.serial, "dof": doc.dof_params}


@app.post("/api/library/remove")
def api_library_remove(req: SelectReq, _=Depends(_auth)):
    """删除一张图；删除当前图时自动切到相邻一张（库空则回到空文档）。"""
    global doc
    with _state_lock:
        e = library.get(req.id)
        if e is None:
            raise HTTPException(status_code=400, detail="图像不在图库中")
        was_current = e["doc"] is doc
        nid = library.remove(req.id)
        if was_current:
            if nid is None:
                doc = Document()
            else:
                doc = library.get(nid)["doc"]
    if was_current:
        _sync_sources()
    return {"images": library.listing(_current_id())}


@app.get("/api/library/{image_id}/thumb")
def api_library_thumb(image_id: int, _=Depends(_auth)):
    """胶片条缩略图（最长边 240 JPEG）。前端以 ?v=serial 作为缓存失效。"""
    e = library.get(image_id)
    if e is None or e["doc"].source is None:
        raise HTTPException(status_code=404, detail="图像不在图库中")
    small = downscale_max_edge(e["doc"].active_full, 240)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise HTTPException(status_code=500, detail="缩略图编码失败")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/library/{image_id}/original")
def api_library_original(image_id: int, _=Depends(_auth)):
    """当前源图 JPEG（对比原图用——路径打开/Electron 模式前端拿不到 File）。"""
    e = library.get(image_id)
    if e is None or e["doc"].source is None:
        raise HTTPException(status_code=404, detail="图像不在图库中")
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(e["doc"].source, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="原图编码失败")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/auto")
def api_auto(req: dict, _=Depends(_auth)):
    mode = str(req.get("mode") or "rule")
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    if mode == "rule":
        d = auto.suggest_rule(doc.source)
        return {"adjustments": d or {}, "note": "", "model": "rule"}
    config = _ai_config_from(req.get("config"))
    intent = str(req.get("intent") or "").strip()[:600]
    # 强度：0..1，缺省用 auto 里的默认值（0.6）——模型数值再大也落到克制范围
    try:
        strength = float(req.get("strength"))
    except (TypeError, ValueError):
        strength = auto.AI_DEFAULT_STRENGTH
    strength = max(0.0, min(1.0, strength))
    try:
        d, note, model = auto.suggest_ai(  # 阻塞型，线程池内执行
            doc.source, config, intent=intent, strength=strength)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"adjustments": d or {}, "note": note, "model": model}


def _ai_config_from(raw) -> dict | None:
    """请求体里的 config 字段 -> auto.resolve_config 接受的 dict（可缺省）。"""
    if isinstance(raw, dict):
        return {"base_url": raw.get("base_url") or "",
                "api_key": raw.get("api_key") or "",
                "model": raw.get("model") or ""}
    return None


@app.post("/api/ai/test")
def api_ai_test(req: AiTestReq, _=Depends(_auth)):
    """测试 AI 端点连通性：GET {base}/models，返回可用模型 id 列表。"""
    c = req.config
    base = c.base_url.strip().rstrip("/")
    if not (base and c.api_key.strip()):
        raise HTTPException(status_code=400, detail="请填写 URL 与 API Key")
    try:
        models = auto.fetch_models(base, c.api_key.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"连接失败：{exc}")
    return {"ok": True, "models": models}


@app.post("/api/mask/ai")
def api_mask_ai(req: AiMaskReq, _=Depends(_auth)):
    """AI 蒙版：识别主体/人物/背景/天空，返回掩码数据（前端存入 MaskState）。

    只走本地像素级分割（SAM 2.1 + U²-Net）：图像不出本机、免费、无需配置端点。
    未识别到目标映射 400，运行环境/模型问题映射 501/502。
    """
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    with _state_lock:                       # 快照，识别期间允许换图
        src = np.array(doc.source, copy=True)
    try:
        return segment.segment_local(src, req.target)
    except ValueError as exc:                   # 未知目标 / 画面中没有该目标
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:          # 未装 onnxruntime / 模型未下载
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ------------------------------------------------------------ 本地分割模型
@app.get("/api/ai/models")
def api_ai_models(_=Depends(_auth)):
    """本地像素级引擎状态：onnxruntime 可用性 + 各模型文件就绪情况。"""
    return {"runtime": segment.has_runtime(), "ready": segment.models_ready(),
            "models": segment.models_status()}


@app.post("/api/ai/models/download")
def api_ai_models_download(_=Depends(_auth)):
    """启动（或继续）后台下载全部本地模型；进度轮询 /api/ai/models/progress。"""
    if not segment.has_runtime():
        raise HTTPException(status_code=400,
                            detail="未安装 onnxruntime：pip install onnxruntime")
    return segment.start_download()


@app.get("/api/ai/models/progress")
def api_ai_models_progress(_=Depends(_auth)):
    return segment.download_progress()


@app.post("/api/ai/depth-relight")
def api_ai_depth_relight(req: DepthRelightReq, _=Depends(_auth)):
    """AI 景深立体调色：本地 Depth Anything V2 深度图 -> 前景/远景非破坏性蒙版与调色。"""
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    with _state_lock:
        src = np.array(doc.active_full, copy=True)
    try:
        return auto.suggest_depth_relight(
            src, preset=req.preset, strength=req.strength,
            near_bias=req.near_bias, masks_only=req.masks_only)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/ai/dof/pick")
def api_ai_dof_pick(req: DepthDofPickReq, _=Depends(_auth)):
    """AI 景深对焦取点：按画布归一化坐标采样 Depth Anything V2 深度与焦平面预览。"""
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    try:
        with _state_lock:
            return doc.pick_dof_focus(req.x, req.y, dof_range=req.dof_range)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/ai/dof")
def api_ai_dof(req: DepthDofReq, _=Depends(_auth)):
    """AI 调节景深：本地 Depth Anything V2 深度估计 + 光学景深散景渲染。"""
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    try:
        with _state_lock:
            res = doc.apply_dof(
                req.focus, req.aperture,
                dof_range=req.dof_range,
                bokeh_highlight=req.bokeh_highlight,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(exc))
    _sync_sources()
    return res


@app.post("/api/ai/dof/clear")
def api_ai_dof_clear(_=Depends(_auth)):
    """清除 AI 景深虚化，恢复原始清晰图像。"""
    with _state_lock:
        serial = doc.clear_dof()
    _sync_sources()
    return {"serial": serial, "active": False}


@app.post("/api/ai/color-match")
async def api_ai_color_match(
    file: UploadFile | None = File(default=None),
    ref_id: int | None = Form(default=None),
    strength: float = Form(default=0.75),
    protect_subject: bool = Form(default=False),
    _=Depends(_auth),
):
    """AI 参考仿色：从上传参考图或图库参考图提取影调与色彩风格参数。"""
    if doc.source is None:
        raise HTTPException(status_code=400, detail="未打开图像")
    with _state_lock:
        src = np.array(doc.active_full, copy=True)
        ref_arr = None
        if ref_id is not None:
            entry = library.get(ref_id)
            if entry is None or entry["doc"].active_full is None:
                raise HTTPException(status_code=400, detail="参考图像不在图库中")
            ref_arr = np.array(entry["doc"].active_full, copy=True)
    if ref_arr is None:
        if file is None:
            raise HTTPException(status_code=400, detail="请上传参考图像或选择图库中的图像")
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="无法解码参考图像")
        ref_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return auto.match_reference_color(
            src, ref_arr, strength=strength, protect_subject=protect_subject)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/histogram")
def api_histogram(_=Depends(_auth)):
    src = _last_frames.get(FULL)
    if src is None:
        src = _last_frames.get(PREVIEW)
    if src is None:
        src = doc.preview_src
    if src is None:
        src = doc.source
    if src is None:
        return {"master": [0.0] * HIST_BINS, "red": [0.0] * HIST_BINS,
                "green": [0.0] * HIST_BINS, "blue": [0.0] * HIST_BINS}
    hist = linear_histograms(src)
    return {k: [float(v) for v in arr] for k, arr in hist.items()}


# ------------------------------------------------------------ WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: str = Query(default="")):
    if _token and not secrets.compare_digest(token or "", _token):
        await ws.close(code=4401)
        return
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except ValueError:
                continue
            if data.get("type") != "render":
                continue
            kind = data.get("kind") or PREVIEW
            if kind not in (PREVIEW, FULL):
                continue
            try:
                rid = int(data.get("rid") or 0)
            except (TypeError, ValueError):
                continue
            try:
                edge = int(data.get("edge") or 0)
            except (TypeError, ValueError):
                edge = 0
            params = adjustments_from_dict(data.get("adjustments"))
            loop.submit(params, kind, client_rid=rid, edge=edge)
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


# ------------------------------------------------------------ 静态托管（打包后 Electron 直接加载本服务）
def _dist_dir() -> str:
    if getattr(sys, "frozen", False):     # PyInstaller：datas 落在 _MEIPASS/frontend_dist
        bundled = os.path.join(getattr(sys, "_MEIPASS", ""), "frontend_dist")
        if os.path.isdir(bundled):
            return bundled
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")


_DIST_DIR = _dist_dir()
if os.path.isdir(_DIST_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="ui")


# ------------------------------------------------------------ 启动
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@app.on_event("startup")
async def _startup():
    global _asyncio_loop, loop
    _asyncio_loop = asyncio.get_event_loop()
    if loop is None:
        loop = RenderLoop(_on_frame, _on_busy, _on_error)


@app.on_event("shutdown")
async def _shutdown():
    if loop is not None:
        loop.stop()


def main():
    global _token, loop
    parser = argparse.ArgumentParser(description="FernLens 本地服务端")
    parser.add_argument("--port", type=int, default=0, help="监听端口（0=自动）")
    parser.add_argument("--token", default="", help="访问令牌（默认随机生成）")
    parser.add_argument("--no-auth", action="store_true", help="关闭令牌校验")
    args = parser.parse_args()

    _token = "" if args.no_auth else (args.token or secrets.token_hex(8))
    port = args.port or _free_port()

    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    print(f"FERNLENS_SERVER_READY http://127.0.0.1:{port} token={_token}", flush=True)
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
