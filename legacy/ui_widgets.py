"""FernLens UI：主题、通用控件与后台处理线程。

内容分区：
  1. 深色主题  2. 滑块行  3. 可折叠分区  4. 图像预览
  5. 曲线编辑器  6. 色轮  7. 后台处理线程
"""
from __future__ import annotations

import os
import math
import threading
import traceback

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (QColor, QConicalGradient, QImage, QMouseEvent,
                           QPainter, QPainterPath, QPen, QPixmap)
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QVBoxLayout, QWidget)

from core import (CURVE_CHANNELS, CancelledError, LUT_SIZE, build_lut,
                  rgb8_to_linear_mt, sanitize_points, process_linear)

# ======================================================================
# 1. 深色主题（Lightroom 风格）
# ======================================================================

BG = "#1b1b1d"          # 窗口背景
PANEL = "#232326"       # 面板背景
CONTROL = "#2c2c31"     # 控件背景
BORDER = "#3a3a41"      # 边框
TEXT = "#d8d8dc"        # 主文字
TEXT_DIM = "#8b8b93"    # 次要文字
ACCENT = "#4c9be8"      # 强调色
ACCENT_DARK = "#2f6db3"
SLIDER_FILL = "#5a5a63"

STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
                 "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Segoe UI", sans-serif;
    font-size: 12px;
}}
QToolBar {{
    background-color: {PANEL};
    border-bottom: 1px solid {BORDER};
    spacing: 6px;
    padding: 4px 6px;
}}
QToolBar QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: 4px;
    padding: 5px 10px; color: {TEXT};
}}
QToolBar QToolButton:hover {{ background-color: {CONTROL}; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background-color: {ACCENT_DARK}; border-color: {ACCENT};
}}
QToolBar QToolButton:disabled {{ color: {TEXT_DIM}; }}

QScrollArea {{ border: none; background-color: {BG}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {SLIDER_FILL}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #6f6f7a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {SLIDER_FILL}; border-radius: 5px; min-width: 30px; }}

QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {SLIDER_FILL}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #dcdce2; border: 1px solid #8f8f9a; width: 12px;
    margin: -6px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; border-color: {ACCENT}; }}

QPushButton {{
    background-color: {CONTROL}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 5px 10px; color: {TEXT};
}}
QPushButton:hover {{ background-color: #34343a; border-color: #4a4a52; }}
QPushButton:pressed {{ background-color: {ACCENT_DARK}; }}
QPushButton:checked {{ background-color: {ACCENT_DARK}; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; }}

QLabel {{ background: transparent; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#value {{ color: {TEXT_DIM}; qproperty-alignment: AlignRight; }}
QLabel#sectionTitle {{ color: {TEXT}; font-weight: bold; font-size: 12px; }}
QLabel#groupLabel {{ color: {TEXT_DIM}; font-size: 11px; }}

QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 10px;
             padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT}; }}

QStatusBar {{ background-color: {PANEL}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QMenu {{ background-color: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background-color: {ACCENT_DARK}; }}
QFileDialog {{ background-color: {BG}; }}
QToolTip {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; padding: 3px; }}
"""


def apply_dark_theme(app):
    """应用 Fusion 风格 + 深色调色板 + QSS。"""
    from PySide6.QtGui import QPalette
    app.setStyle("Fusion")
    palette = QPalette()
    A = QPalette.ColorGroup.Active
    D = QPalette.ColorGroup.Disabled
    colors = (
        (A, QPalette.Window, QColor(BG)),
        (A, QPalette.WindowText, QColor(TEXT)),
        (A, QPalette.Base, QColor(CONTROL)),
        (A, QPalette.AlternateBase, QColor(PANEL)),
        (A, QPalette.ToolTipBase, QColor(PANEL)),
        (A, QPalette.ToolTipText, QColor(TEXT)),
        (A, QPalette.Text, QColor(TEXT)),
        (A, QPalette.Button, QColor(CONTROL)),
        (A, QPalette.ButtonText, QColor(TEXT)),
        (A, QPalette.BrightText, QColor("#ffffff")),
        (A, QPalette.Highlight, QColor(ACCENT_DARK)),
        (A, QPalette.HighlightedText, QColor("#ffffff")),
        (D, QPalette.Text, QColor(TEXT_DIM)),
        (D, QPalette.ButtonText, QColor(TEXT_DIM)),
        (D, QPalette.WindowText, QColor(TEXT_DIM)),
    )
    for group, role, color in colors:
        palette.setColor(group, role, color)
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


# ======================================================================
# 2. 滑块行：标签 + 滑块 + 数值（双击标签复位、滚轮微调）
# ======================================================================


class SliderRow(QWidget):
    valueChanged = Signal(float)
    sliderReleased = Signal()

    def __init__(self, label: str, minimum: float = -100.0, maximum: float = 100.0,
                 value: float = 0.0, decimals: int = 0, default_value: float | None = None,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self._decimals = decimals
        self._mult = 10 ** decimals
        self._default = value if default_value is None else default_value
        self._blocked = False

        self.label = QLabel(label)
        self.label.setObjectName("groupLabel")
        self.label.setFixedWidth(72)
        self.label.setToolTip("双击复位")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(minimum * self._mult)), int(round(maximum * self._mult)))
        self.slider.setSingleStep(self._mult)
        self.slider.setPageStep(10 * self._mult)
        self.slider.setValue(int(round(value * self._mult)))

        self.value_label = QLabel()
        self.value_label.setObjectName("value")
        self.value_label.setFixedWidth(52)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        hl.addWidget(self.label)
        hl.addWidget(self.slider, 1)
        hl.addWidget(self.value_label)

        self._fmt = f"{{:.{decimals}f}}" if decimals else "{:.0f}"
        self._suffix = suffix
        self._refresh_text()
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self.sliderReleased)
        self.slider.installEventFilter(self)
        self.value_label.installEventFilter(self)

    @property
    def value(self) -> float:
        return self.slider.value() / self._mult

    def set_value(self, value: float, emit: bool = False):
        self._blocked = True
        self.slider.setValue(int(round(value * self._mult)))
        self._blocked = False
        self._refresh_text()
        if emit:
            self.valueChanged.emit(self.value)

    def set_default(self, value: float):
        self._default = value

    def reset_to_default(self):
        self.set_value(self._default)
        self.valueChanged.emit(self.value)
        self.sliderReleased.emit()

    def _refresh_text(self):
        self.value_label.setText(self._fmt.format(self.value) + self._suffix)

    def _on_slider(self, _int_value):
        if self._blocked:
            return
        self._refresh_text()
        self.valueChanged.emit(self.value)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonDblClick and obj is self.label:
            self.reset_to_default()
            return True
        if event.type() == QEvent.Type.Wheel and obj in (self.slider, self.value_label):
            step = self.slider.singleStep()
            delta = event.angleDelta().y()
            if delta:
                self.slider.setValue(self.slider.value() + (step if delta > 0 else -step))
                return True
        return super().eventFilter(obj, event)


# ======================================================================
# 3. 可折叠分区
# ======================================================================


class CollapsibleSection(QWidget):
    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        self.header = QPushButton()
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setStyleSheet("text-align: left; padding: 7px 8px; font-weight: bold;")
        self._refresh_text()
        self.header.toggled.connect(self._on_toggled)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(content)
        self.content = content
        self.content.setVisible(expanded)

    def _refresh_text(self):
        self.header.setText(("▾  " if self._expanded else "▸  ") + self._title)

    def _on_toggled(self, checked: bool):
        self._expanded = checked
        self.content.setVisible(checked)
        self._refresh_text()
        self.toggled.emit(checked)

    def set_expanded(self, expanded: bool):
        self.header.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self._expanded


# ======================================================================
# 4. 图像预览（深灰底、缩放、平移、取色点击）
# ======================================================================

_CANVAS_BG = QColor(26, 26, 28)


def ndarray_to_qpixmap(rgb8: np.ndarray) -> QPixmap:
    """uint8 RGB (H,W,3) -> QPixmap（数据深拷贝，生命周期安全）。"""
    h, w = rgb8.shape[:2]
    img = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def mask_to_qimage(mask: np.ndarray, selected: bool = True) -> QImage:
    """(H,W) float32 蒙版 -> RGBA QImage（选中为红色高亮，否则为灰色）。"""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    m8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.float32)
    if selected:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 235, 70, 70
        rgba[..., 3] = (m8 * 0.55).astype(np.uint8)
    else:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 160, 160, 170
        rgba[..., 3] = (m8 * 0.22).astype(np.uint8)
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return img.copy()


class PreviewWidget(QWidget):
    colorPicked = Signal(int, int)     # 控件坐标（鼠标位置）
    zoomChanged = Signal(float)        # 缩放倍数（相对“适合窗口”）

    # ---- 蒙版编辑 ----
    linearHandleMoved = Signal(int, float, float, float, float)  # 手柄, x1, y1, x2, y2
    radialCenterMoved = Signal(float, float)        # 归一化 x, y
    radialSizeMoved = Signal(int, float, float)          # 轴 0=长轴 1=短轴, 半径, 长宽比
    radialAngleMoved = Signal(float)                     # 角度（度）
    brushStrokePoint = Signal(float, float)         # 归一化 x, y
    brushStrokeFinished = Signal()

    # ---- 仿制图章（矩形取样覆盖）----
    cloneSourcePicked = Signal(tuple)              # 源矩形（归一化 x0,y0,x1,y1）
    cloneApplyRequested = Signal(tuple)            # 目标矩形（归一化）

    # ---- 裁剪 ----
    cropRectChanged = Signal(dict)                  # 归一化 x0,y0,x1,y1（旋转视图坐标）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._frame_size = (0, 0)       # 当前显示帧尺寸
        self._full_size = (0, 0)        # 原图尺寸（用于 1:1）
        self._zoom = 1.0                # 相对适合窗口的倍数
        self._fit_scale = 1.0
        self._pan = QPointF(0.0, 0.0)   # 相对居中的平移量
        self._pick_mode = False
        self._edit_mode = "none"        # 'none' | 'linear' | 'radial' | 'brush'
        self._edit_info: dict = {}      # 当前编辑蒙版的几何
        self._overlays: list = []       # 蒙版叠加 QImage 列表
        self._drag_handle: int | None = None   # 线性手柄索引 / None
        self._drag_line = False
        self._radial_drag_start: dict | None = None   # 径向拖拽起始状态
        self._hover_handle: str | None = None         # 悬停的手柄（高亮 + 光标）
        self._brush_cursor: QPointF | None = None
        self._brush_drawing = False
        self._panning = False
        self._pan_last = QPointF()
        self._placeholder = True
        self._stamp_mode = False
        self._stamp_src: tuple | None = None       # 已选源矩形（归一化）
        self._stamp_rubber = None                  # 拖拽橡皮筋 ('src'|'dst', 起, 当前)
        self._crop_mode = False
        self._crop_rect: dict | None = None        # x0,y0,x1,y1 归一化（旋转视图坐标）
        self._crop_aspect: float | None = None     # 锁定宽高比 w/h（像素意义）；None=自由
        self._crop_angle = 0.0                     # 拉直角度（度，OpenCV 约定正=逆时针）
        self._crop_max = (0.0, 0.0, 1.0, 1.0)      # 归一化最大有效矩形（拉直后收缩）
        self._crop_drag = None                     # (mode, 起始 rect, 起始归一化位置)
        self._crop_hover = None

    # ------------------------------------------------------------ 接口
    def set_frame(self, rgb8: np.ndarray | None):
        if rgb8 is None:
            self._pixmap = None
            self._placeholder = True
            self._frame_size = (0, 0)
            self.update()
            return
        self._pixmap = ndarray_to_qpixmap(rgb8)
        self._placeholder = False
        self._frame_size = (rgb8.shape[1], rgb8.shape[0])
        self._recompute_layout()
        self.update()

    def set_full_size(self, w: int, h: int):
        self._full_size = (w, h)

    def set_pick_mode(self, on: bool):
        self._pick_mode = on
        if on and self._edit_mode != "none":
            self._edit_mode = "none"
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------ 仿制图章
    def set_stamp_mode(self, on: bool):
        self._stamp_mode = on
        self._stamp_rubber = None
        if on:
            self._pick_mode = False
            self._edit_mode = "none"
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    def stamp_mode(self) -> bool:
        return self._stamp_mode

    def set_stamp_source(self, rect):
        """记录/清除已选源矩形（归一化坐标），用于常驻叠加层显示。"""
        self._stamp_src = tuple(rect) if rect is not None else None
        self.update()

    def stamp_source(self):
        return self._stamp_src

    # ------------------------------------------------------------ 蒙版编辑
    def set_edit_mode(self, mode: str, info: dict | None = None):
        """进入/退出蒙版编辑模式；info 携带当前几何（归一化坐标）。"""
        self._edit_mode = mode
        self._edit_info = info or {}
        self._drag_handle = None
        self._drag_line = False
        self._radial_drag_start = None
        self._hover_handle = None
        if mode != "none" and self._pick_mode:
            self._pick_mode = False
        self.update()

    def edit_mode(self) -> str:
        return self._edit_mode

    def set_overlays(self, overlays: list):
        """设置蒙版叠加图（QImage 列表，与原图同尺寸，带 alpha）。"""
        self._overlays = overlays
        self.update()

    # ------------------------------------------------------------ 裁剪
    def crop_mode(self) -> bool:
        return self._crop_mode

    def set_crop_mode(self, on: bool, rect: dict | None = None):
        """进入/退出裁剪模式；rect 为初始框（归一化，缺省四周留 5%）。"""
        self._crop_mode = bool(on) and not self._placeholder
        self._crop_drag = None
        self._crop_hover = None
        if self._crop_mode:
            if self._edit_mode != "none":
                self.set_edit_mode("none")
            self._pick_mode = False
            self._stamp_mode = False
            self._stamp_rubber = None
            if rect is None:
                rect = {"x0": 0.05, "y0": 0.05, "x1": 0.95, "y1": 0.95}
            self._crop_rect = self._clamp_crop_rect(dict(rect))
            self._emit_crop_rect()
        self.update()

    def crop_rect(self) -> dict | None:
        if self._crop_mode and self._crop_rect:
            return dict(self._crop_rect)
        return None

    def set_crop_aspect(self, aspect):
        """锁定宽高比（像素 w/h）；None 为自由。按当前框中心重新适配。"""
        self._crop_aspect = None if aspect is None else max(float(aspect), 0.05)
        if self._crop_rect:
            self._crop_rect = self._refit_aspect(self._crop_rect)
            self._emit_crop_rect()
        self.update()

    def set_crop_angle(self, angle: float):
        """拉直角度（仅影响显示与提交；框约束由 set_crop_max 提供）。"""
        self._crop_angle = float(angle)
        self.update()

    def set_crop_max(self, max_rect):
        """归一化最大有效矩形：拉直后画布四角会超出原图，框必须收缩到有效区。"""
        self._crop_max = tuple(max_rect)
        if self._crop_rect:
            self._crop_rect = self._clamp_crop_rect(self._crop_rect)
            self._emit_crop_rect()
        self.update()

    def _emit_crop_rect(self):
        self.cropRectChanged.emit(dict(self._crop_rect))

    def _crop_min_sizes(self) -> tuple:
        fw, fh = self._frame_size
        return max(24.0 / max(fw, 1), 0.01), max(24.0 / max(fh, 1), 0.01)

    def _clamp_crop_rect(self, r: dict) -> dict:
        mx0, my0, mx1, my1 = self._crop_max
        lo_x, hi_x = max(0.0, mx0), min(1.0, mx1)
        lo_y, hi_y = max(0.0, my0), min(1.0, my1)
        mw, mh = self._crop_min_sizes()
        x0 = float(np.clip(r["x0"], lo_x, max(lo_x, hi_x - mw)))
        x1 = float(np.clip(r["x1"], x0 + mw, hi_x))
        y0 = float(np.clip(r["y0"], lo_y, max(lo_y, hi_y - mh)))
        y1 = float(np.clip(r["y1"], y0 + mh, hi_y))
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    def _refit_aspect(self, r: dict) -> dict:
        """在最大有效区内，以 r 的中心按锁定比例取最大框（比例按像素计）。"""
        if self._crop_aspect is None:
            return self._clamp_crop_rect(r)
        fw, fh = self._frame_size
        mx0, my0, mx1, my1 = self._crop_max
        bw_px, bh_px = max((mx1 - mx0) * fw, 1.0), max((my1 - my0) * fh, 1.0)
        a = self._crop_aspect
        if bw_px / bh_px > a:
            h_px, w_px = bh_px, bh_px * a
        else:
            w_px, h_px = bw_px, bw_px / a
        wn, hn = w_px / max(fw, 1), h_px / max(fh, 1)
        cx, cy = (r["x0"] + r["x1"]) / 2.0, (r["y0"] + r["y1"]) / 2.0
        return self._clamp_crop_rect(
            {"x0": cx - wn / 2, "y0": cy - hn / 2, "x1": cx + wn / 2, "y1": cy + hn / 2})

    def _crop_hit(self, pos: QPointF):
        """手柄/框内/框外命中：'nw'..'e' / 'in' / None。"""
        if not self._crop_rect or self._placeholder:
            return None
        r = self._crop_rect
        fw, fh = self._frame_size
        fx, fy = self.map_to_frame(pos.x(), pos.y())
        tol = 12.0 / max(self._scale(), 1e-6)
        x0p, x1p, y0p, y1p = r["x0"] * fw, r["x1"] * fw, r["y0"] * fh, r["y1"] * fh
        near_l, near_r = abs(fx - x0p) <= tol, abs(fx - x1p) <= tol
        near_t, near_b = abs(fy - y0p) <= tol, abs(fy - y1p) <= tol
        in_x = x0p - tol <= fx <= x1p + tol
        in_y = y0p - tol <= fy <= y1p + tol
        if near_l and near_t:
            return "nw"
        if near_r and near_t:
            return "ne"
        if near_l and near_b:
            return "sw"
        if near_r and near_b:
            return "se"
        if near_t and in_x:
            return "n"
        if near_b and in_x:
            return "s"
        if near_l and in_y:
            return "w"
        if near_r and in_y:
            return "e"
        if x0p < fx < x1p and y0p < fy < y1p:
            return "in"
        return None

    @staticmethod
    def _crop_cursor(hit):
        return {
            "nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
            "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
            "e": Qt.CursorShape.SizeHorCursor, "w": Qt.CursorShape.SizeHorCursor,
            "in": Qt.CursorShape.SizeAllCursor,
        }.get(hit, Qt.CursorShape.CrossCursor)

    def _apply_crop_drag(self, mode: str, r0: dict, p0: tuple, p1: tuple) -> dict:
        """把拖拽增量应用到起始框 r0，返回新框（含比例锁定与边界约束）。"""
        fw, fh = self._frame_size
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        r = {"x0": r0["x0"], "y0": r0["y0"], "x1": r0["x1"], "y1": r0["y1"]}

        if mode == "move":
            w, h = r["x1"] - r["x0"], r["y1"] - r["y0"]
            mx0, my0, mx1, my1 = self._crop_max
            cx = float(np.clip((r["x0"] + r["x1"]) / 2.0 + dx,
                               max(0.0, mx0) + w / 2.0, min(1.0, mx1) - w / 2.0))
            cy = float(np.clip((r["y0"] + r["y1"]) / 2.0 + dy,
                               max(0.0, my0) + h / 2.0, min(1.0, my1) - h / 2.0))
            return self._clamp_crop_rect(
                {"x0": cx - w / 2, "y0": cy - h / 2, "x1": cx + w / 2, "y1": cy + h / 2})

        corner = mode in ("nw", "ne", "sw", "se")
        if "w" in mode:
            r["x0"] += dx
        if "e" in mode:
            r["x1"] += dx
        if "n" in mode:
            r["y0"] += dy
        if "s" in mode:
            r["y1"] += dy

        a = self._crop_aspect
        if a is not None:
            # 像素比例：宽高分别按帧宽/帧高换算
            w_px = max((r["x1"] - r["x0"]) * fw, 1.0)
            if corner:
                ax = r0["x1"] if "w" in mode else r0["x0"]     # 锚点=对角
                ay = r0["y1"] if "n" in mode else r0["y0"]
                h_px = w_px / a
                if "n" in mode:
                    r["y1"], r["y0"] = ay, ay - h_px / fh
                else:
                    r["y0"], r["y1"] = ay, ay + h_px / fh
            elif mode in ("n", "s"):
                cx = (r0["x0"] + r0["x1"]) / 2.0
                h_px = max((r["y1"] - r["y0"]) * fh, 1.0)
                wn = h_px * a / fw
                r["x0"], r["x1"] = cx - wn / 2, cx + wn / 2
            else:                                                # e / w
                cy = (r0["y0"] + r0["y1"]) / 2.0
                hn = w_px / a / fh
                r["y0"], r["y1"] = cy - hn / 2, cy + hn / 2
        return self._clamp_crop_rect(r)

    def frame_point_to_widget(self, nx: float, ny: float) -> QPointF:
        """归一化图像坐标 -> 控件坐标。"""
        off = self._offset()
        s = self._scale()
        return QPointF(off.x() + nx * self._frame_size[0] * s,
                       off.y() + ny * self._frame_size[1] * s)

    def widget_to_norm(self, x: float, y: float) -> tuple:
        """控件坐标 -> 归一化图像坐标。"""
        fx, fy = self.map_to_frame(x, y)
        fw, fh = self._frame_size
        return fx / max(fw, 1), fy / max(fh, 1)

    def zoom_fit(self):
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._recompute_layout()
        self.zoomChanged.emit(self._zoom)
        self.update()

    def zoom_one_to_one(self):
        """按原图像素 1:1 显示。"""
        if not self._frame_size[0] or not self._full_size[0]:
            return
        frame_w = self._frame_size[0]
        full_w = self._full_size[0]
        self._zoom = max(0.02, (full_w / frame_w) / max(self._fit_scale, 1e-6))
        self._pan = QPointF(0.0, 0.0)
        self._recompute_layout()
        self.zoomChanged.emit(self._zoom)
        self.update()

    def zoom_at(self, factor: float, anchor: QPointF | None = None):
        if self._placeholder:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        fx = (anchor.x() - self._offset().x()) / max(self._scale(), 1e-6)
        fy = (anchor.y() - self._offset().y()) / max(self._scale(), 1e-6)
        self._zoom = float(np.clip(self._zoom * factor, 0.02, 40.0))
        self._recompute_layout()
        # 让锚点下的图像内容保持不动
        center = QPointF((self.width() - self._frame_size[0] * self._scale()) / 2.0,
                         (self.height() - self._frame_size[1] * self._scale()) / 2.0)
        self._pan = QPointF(anchor.x() - fx * self._scale() - center.x(),
                            anchor.y() - fy * self._scale() - center.y())
        self._recompute_layout()
        self.zoomChanged.emit(self._zoom)
        self.update()

    def map_to_frame(self, x: float, y: float) -> tuple:
        """控件坐标 -> 当前显示帧内的像素坐标（浮点）。"""
        off = self._offset()
        s = max(self._scale(), 1e-6)
        return ((x - off.x()) / s, (y - off.y()) / s)

    @property
    def zoom(self) -> float:
        return self._zoom

    def current_frame_size(self) -> tuple:
        return self._frame_size

    # ------------------------------------------------------------ 几何
    def _scale(self) -> float:
        return self._fit_scale * self._zoom

    def _offset(self) -> QPointF:
        fw, fh = self._frame_size
        s = self._scale()
        cx = (self.width() - fw * s) / 2.0 + self._pan.x()
        cy = (self.height() - fh * s) / 2.0 + self._pan.y()
        return QPointF(cx, cy)

    def _recompute_layout(self):
        fw, fh = self._frame_size
        if fw == 0 or fh == 0 or self.width() <= 0 or self.height() <= 0:
            return
        self._fit_scale = min(self.width() / fw, self.height() / fh)

    # ------------------------------------------------------------ 事件
    def resizeEvent(self, event):
        self._recompute_layout()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), _CANVAS_BG)
        if self._placeholder or self._pixmap is None:
            p.setPen(QColor(139, 139, 147))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "打开图像开始调色\n(Ctrl+O)")
            p.end()
            return
        off = self._offset()
        s = self._scale()
        target = QRectF(off.x(), off.y(), self._frame_size[0] * s, self._frame_size[1] * s)
        crop_on = self._crop_mode and not self._placeholder
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.save()
        if crop_on and abs(self._crop_angle) > 1e-9:
            # 与 core.rotate_image 的 warpAffine 一致：OpenCV 正角=逆时针，
            # Qt painter 正角=顺时针（y 向下），故取负
            c = target.center()
            p.translate(c)
            p.rotate(-self._crop_angle)
            p.translate(-c)
        p.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        p.restore()
        p.setPen(QColor(58, 58, 65))
        p.drawRect(target)

        # 蒙版叠加着色
        if not crop_on:
            for overlay in self._overlays:
                if overlay is not None:
                    p.drawImage(target, overlay)

            # 蒙版编辑辅助线
            self._paint_edit_overlay(p)
            self._paint_stamp_overlay(p)
        if crop_on:
            self._paint_crop_overlay(p)
        p.end()

    def _paint_crop_overlay(self, p: QPainter):
        """裁剪框：框外压暗、三分网格、8 个手柄。"""
        r = self._crop_rect
        if r is None:
            return
        a = self.frame_point_to_widget(r["x0"], r["y0"])
        b = self.frame_point_to_widget(r["x1"], r["y1"])
        rect = QRectF(a, b).normalized()

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 130))
        p.drawRect(QRectF(0.0, 0.0, self.width(), max(rect.top(), 0.0)))
        p.drawRect(QRectF(0.0, rect.bottom(), self.width(),
                          max(self.height() - rect.bottom(), 0.0)))
        p.drawRect(QRectF(0.0, rect.top(), max(rect.left(), 0.0), rect.height()))
        p.drawRect(QRectF(rect.right(), rect.top(),
                          max(self.width() - rect.right(), 0.0), rect.height()))

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(245, 245, 250), 1.4))
        p.drawRect(rect)
        p.setPen(QPen(QColor(255, 255, 255, 110), 1.0))
        for i in (1, 2):
            x = rect.left() + rect.width() * i / 3.0
            y = rect.top() + rect.height() * i / 3.0
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        names = (("nw", "n", "ne"), ("w", None, "e"), ("sw", "s", "se"))
        xs = (rect.left(), rect.center().x(), rect.right())
        ys = (rect.top(), rect.center().y(), rect.bottom())
        p.setPen(QPen(QColor(30, 30, 34), 1.0))
        p.setBrush(QColor(245, 245, 250))
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                nm = names[i][j]
                if nm is None:
                    continue
                hot = self._crop_hover == nm or self._crop_drag is not None \
                    and self._crop_drag[0] == nm
                rr = 4.5 if hot else 3.5
                p.drawRect(QRectF(x - rr, y - rr, 2 * rr, 2 * rr))

    def _paint_stamp_overlay(self, p: QPainter):
        """图章模式：已选源矩形常驻显示（青色虚线 + 中心十字），拖拽橡皮筋实时预览。"""
        if not self._stamp_mode or self._placeholder:
            return
        if self._stamp_src is not None:
            a = self.frame_point_to_widget(self._stamp_src[0], self._stamp_src[1])
            b = self.frame_point_to_widget(self._stamp_src[2], self._stamp_src[3])
            rect = QRectF(a, b).normalized()
            p.setPen(QPen(QColor("#4cd7e8"), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(QColor(76, 215, 232, 24))
            p.drawRect(rect)
            p.setBrush(Qt.BrushStyle.NoBrush)
            c = rect.center()
            p.drawLine(QPointF(c.x() - 7, c.y()), QPointF(c.x() + 7, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 7), QPointF(c.x(), c.y() + 7))
            p.setPen(QPen(QColor("#4cd7e8"), 1.0))
            p.drawText(rect.bottomLeft() + QPointF(3, -3), "源")
        if self._stamp_rubber is not None:
            kind, s0, c0 = self._stamp_rubber
            a = self.frame_point_to_widget(s0[0], s0[1])
            b = self.frame_point_to_widget(c0[0], c0[1])
            color = QColor("#4cd7e8") if kind == "src" else QColor(245, 245, 250)
            p.setPen(QPen(color, 1.2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(a, b).normalized())

    def _paint_edit_overlay(self, p: QPainter):
        if self._edit_mode == "none" or self._placeholder:
            return
        pen = QPen(QColor("#ffffff"), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        if self._edit_mode == "linear":
            a = self.frame_point_to_widget(self._edit_info.get("x1", 0.0),
                                           self._edit_info.get("y1", 0.0))
            b = self.frame_point_to_widget(self._edit_info.get("x2", 1.0),
                                           self._edit_info.get("y2", 0.0))
            p.drawLine(a, b)
            for pt in (a, b):
                p.setBrush(QColor("#ffffff"))
                p.drawEllipse(pt, 6.0, 6.0)
        if self._edit_mode == "radial":
            c, rx, ry, ang = self._radial_geometry()
            p.save()
            p.translate(c)
            p.rotate(math.degrees(ang))
            p.drawEllipse(QPointF(0.0, 0.0), rx, ry)
            p.restore()
            # 中心点
            p.setBrush(QColor("#ffffff"))
            p.drawEllipse(c, 6.0, 6.0)
            # 长轴手柄 / 短轴手柄 / 旋转手柄（悬停时放大高亮）
            hx, hy = self._radial_handles()
            for key, pt in (("size:0", hx), ("size:1", hy)):
                hot = self._hover_handle == key or self._drag_handle == 10 + int(key[-1])
                p.setBrush(QColor("#4c9be8") if not hot else QColor("#8fc4ff"))
                p.drawEllipse(pt, 6.5 if hot else 5.5, 6.5 if hot else 5.5)
            rot = QPointF(c.x() + 1.28 * rx * math.cos(ang),
                          c.y() + 1.28 * rx * math.sin(ang))
            hot_rot = self._hover_handle == "rotate" or self._drag_handle == 20
            p.setPen(QPen(QColor("#e8a33d"), 1.5))
            p.drawLine(hx, rot)
            p.setBrush(QColor("#f0be62") if hot_rot else QColor("#e8a33d"))
            p.drawEllipse(rot, 6.0 if hot_rot else 5.0, 6.0 if hot_rot else 5.0)
        elif self._edit_mode == "brush" and self._brush_cursor is not None:
            short = max(min(self._frame_size), 1)
            r = self._scale() * short * (self._edit_info.get("size", 12.0) / 100.0) * 0.5
            p.drawEllipse(self._brush_cursor, r, r)

    def _radial_geometry(self):
        """返回 (中心, rx, ry, 角度弧度)（控件像素单位）。"""
        info = self._edit_info
        fw, fh = self._frame_size
        short = max(min(fw, fh), 1)
        scale = self._scale()
        R = max(float(info.get("radius", 0.3)), 0.01) * short
        asp = max(float(info.get("aspect", 1.0)), 0.05)
        ang = math.radians(float(info.get("angle", 0.0)))
        c = self.frame_point_to_widget(info.get("cx", 0.5), info.get("cy", 0.5))
        return c, R * asp * scale, (R / asp) * scale, ang

    def _radial_handles(self):
        """长轴手柄与短轴手柄的控件坐标。"""
        c, rx, ry, ang = self._radial_geometry()
        hx = QPointF(c.x() + rx * math.cos(ang), c.y() + rx * math.sin(ang))
        hy = QPointF(c.x() - ry * math.sin(ang), c.y() + ry * math.cos(ang))
        return hx, hy

    def mask_handles_info(self) -> dict:
        """当前径向蒙版各手柄的归一化坐标（供测试与调试）。"""
        if self._edit_mode != "radial":
            return {}
        c, rx, ry, ang = self._radial_geometry()
        s = max(self._scale(), 1e-6)
        fw, fh = self._frame_size
        off = self._offset()

        def to_norm(x, y):
            return ((x - off.x()) / s / max(fw, 1), (y - off.y()) / s / max(fh, 1))

        hx = (c.x() + rx * math.cos(ang), c.y() + rx * math.sin(ang))
        hy = (c.x() - ry * math.sin(ang), c.y() + ry * math.cos(ang))
        rot = (c.x() + 1.28 * rx * math.cos(ang), c.y() + 1.28 * rx * math.sin(ang))
        return {"center": to_norm(c.x(), c.y()), "hx": to_norm(*hx),
                "hy": to_norm(*hy), "rot": to_norm(*rot)}

    def _radial_hit(self, pos: QPointF) -> str | None:
        c, rx, ry, ang = self._radial_geometry()
        hx, hy = self._radial_handles()
        if (QPointF(hx) - pos).manhattanLength() <= 12.0:
            return "size:0"
        if (QPointF(hy) - pos).manhattanLength() <= 12.0:
            return "size:1"
        rot = QPointF(c.x() + 1.28 * rx * math.cos(ang),
                      c.y() + 1.28 * rx * math.sin(ang))
        if (QPointF(rot) - pos).manhattanLength() <= 12.0:
            return "rotate"
        if (QPointF(c) - pos).manhattanLength() <= 14.0:
            return "center"
        return None

    # ------------------------------------------------------------ 事件
    def wheelEvent(self, event):
        if self._placeholder:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_at(1.0015 ** delta, QPointF(event.position()))

    def _linear_hit(self, pos: QPointF):
        """命中线性蒙版的手柄/连线，返回 'handle:i' / 'line' / None。"""
        info = self._edit_info
        a = self.frame_point_to_widget(info.get("x1", 0.0), info.get("y1", 0.0))
        b = self.frame_point_to_widget(info.get("x2", 1.0), info.get("y2", 0.0))
        for i, pt in ((0, a), (1, b)):
            if (QPointF(pt) - pos).manhattanLength() <= 12.0:
                return f"handle:{i}"
        # 点到线段距离
        ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-6:
            return None
        t = ((pos.x() - ax) * dx + (pos.y() - ay) * dy) / L2
        t = max(0.0, min(1.0, t))
        dist = math.hypot(pos.x() - (ax + t * dx), pos.y() - (ay + t * dy))
        return "line" if dist <= 10.0 else None

    def mousePressEvent(self, event):
        pos = event.position()
        if self._crop_mode and not self._placeholder:
            if event.button() == Qt.MouseButton.LeftButton:
                hit = self._crop_hit(pos)
                start = self.widget_to_norm(pos.x(), pos.y())
                if hit is None:
                    # 框外按下：拖出一个新框
                    r = self._clamp_crop_rect(
                        {"x0": start[0], "y0": start[1],
                         "x1": start[0], "y1": start[1]})
                    self._crop_rect = r
                    self._crop_drag = ("se", dict(r), start)
                    self._emit_crop_rect()
                else:
                    self._crop_drag = ("move" if hit == "in" else hit,
                                       dict(self._crop_rect), start)
            elif event.button() == Qt.MouseButton.MiddleButton:
                self._start_pan(pos)
            return
        if self._pick_mode and event.button() == Qt.MouseButton.LeftButton:
            if not self._placeholder:
                self.colorPicked.emit(int(pos.x()), int(pos.y()))
            return
        if event.button() != Qt.MouseButton.LeftButton:
            if event.button() in (Qt.MouseButton.MiddleButton,):
                self._start_pan(pos)
            return

        if self._stamp_mode:
            if not self._placeholder and event.button() == Qt.MouseButton.LeftButton:
                # 尚未选源或按住 Alt -> 选源矩形；否则拖出目标矩形
                kind = "src" if (self._stamp_src is None
                                 or event.modifiers() & Qt.KeyboardModifier.AltModifier) \
                    else "dst"
                start = self.widget_to_norm(pos.x(), pos.y())
                self._stamp_rubber = (kind, start, start)
                self.update()
            return

        if self._edit_mode == "linear":
            hit = self._linear_hit(pos)
            if hit and hit.startswith("handle:"):
                self._drag_handle = int(hit.split(":")[1])
            elif hit == "line":
                self._drag_line = True
                self._pan_last = pos
            else:
                self._start_pan(pos)
            return
        if self._edit_mode == "radial":
            hit = self._radial_hit(pos)
            if hit == "center":
                self._drag_handle = 0
            elif hit in ("size:0", "size:1"):
                self._drag_handle = 10 + int(hit.split(":")[1])
                self._begin_radial_size_drag(pos)
            elif hit == "rotate":
                self._drag_handle = 20
            else:
                self._start_pan(pos)
            return
        if self._edit_mode == "brush":
            if not self._placeholder:
                self._brush_cursor = pos
                self._brush_drawing = True
                self.brushStrokePoint.emit(*self.widget_to_norm(pos.x(), pos.y()))
            return
        self._start_pan(pos)

    def _begin_radial_size_drag(self, pos: QPointF):
        """记录大小手柄拖拽的起始比例，之后按比例缩放（手感接近 LR：拖慢、可预期）。"""
        info = self._edit_info
        fw, fh = self._frame_size
        short = max(min(fw, fh), 1)
        ang = math.radians(float(info.get("angle", 0.0)))
        cx, cy = info.get("cx", 0.5), info.get("cy", 0.5)
        dx = ((pos.x() - self._offset().x()) / max(self._scale(), 1e-6)) / max(fw, 1) - cx
        dy = ((pos.y() - self._offset().y()) / max(self._scale(), 1e-6)) / max(fh, 1) - cy
        local_x = (dx * fw * math.cos(ang) + dy * fh * math.sin(ang)) / short
        local_y = (-dx * fw * math.sin(ang) + dy * fh * math.cos(ang)) / short
        asp = max(float(info.get("aspect", 1.0)), 0.05)
        radius = max(float(info.get("radius", 0.3)), 0.01)
        self._radial_drag_start = {
            "radius": radius, "aspect": asp,
            "rx": radius * asp, "ry": radius / asp,
            "lx": abs(local_x), "ly": abs(local_y),
        }

    def _radial_drag(self, nx: float, ny: float):
        """径向蒙版拖拽：中心 / 长轴 / 短轴 / 旋转。"""
        info = self._edit_info
        handle = self._drag_handle
        if handle == 0:
            info["cx"], info["cy"] = nx, ny
            self.radialCenterMoved.emit(nx, ny)
            return

        fw, fh = self._frame_size
        short = max(min(fw, fh), 1)
        ang = math.radians(float(info.get("angle", 0.0)))
        cx, cy = info.get("cx", 0.5), info.get("cy", 0.5)
        dx = (nx - cx) * fw
        dy = (ny - cy) * fh
        dist = math.hypot(dx, dy)

        if handle == 20:                                  # 旋转手柄
            deg = math.degrees(math.atan2(dy, dx)) % 360.0
            info["angle"] = deg
            self.radialAngleMoved.emit(deg)
            return

        # 局部坐标（沿椭圆轴分解），换算成“短边归一化”半径
        local_x = (dx * math.cos(ang) + dy * math.sin(ang)) / short
        local_y = (-dx * math.sin(ang) + dy * math.cos(ang)) / short

        start = self._radial_drag_start
        if handle == 10:                                  # 长轴手柄
            if start and start["lx"] > 1e-3:
                # 成比例缩放：拖拽距离 / 起始距离 = 缩放倍数（手感缓慢、可预期）
                scale = abs(local_x) / start["lx"]
                rx_new = float(np.clip(start["rx"] * scale, 0.02, 2.0))
                ry = start["ry"]
            else:
                rx_new, ry = max(abs(local_x), 0.02), start["ry"] if start else 0.3
            asp_new = float(np.clip(rx_new / max(ry, 1e-3), 0.2, 5.0))
            info["aspect"] = asp_new
            info["radius"] = float(np.sqrt(max(rx_new * ry, 1e-6)))
        elif handle == 11:                                # 短轴手柄
            if start and start["ly"] > 1e-3:
                scale = abs(local_y) / start["ly"]
                ry_new = float(np.clip(start["ry"] * scale, 0.02, 2.0))
                rx = start["rx"]
            else:
                ry_new, rx = max(abs(local_y), 0.02), start["rx"] if start else 0.3
            asp_new = float(np.clip(max(rx, 1e-3) / ry_new, 0.2, 5.0))
            info["aspect"] = asp_new
            info["radius"] = float(np.sqrt(max(rx * ry_new, 1e-6)))
        else:
            return
        self.radialSizeMoved.emit(handle - 10, info["radius"], info["aspect"])
        self._edit_info = info

    def _update_hover(self, pos: QPointF):
        """悬停检测：命中手柄时高亮并切换光标（像 LR 那样先悬停再拖）。"""
        if self._edit_mode == "radial":
            hit = self._radial_hit(pos)
        elif self._edit_mode == "linear":
            hit = self._linear_hit(pos)
        else:
            hit = None
        if hit != self._hover_handle:
            self._hover_handle = hit
            self.update()
        if hit in ("size:0", "size:1"):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif hit == "rotate":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif hit in ("center", "handle:0", "handle:1", "line"):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _start_pan(self, pos: QPointF):
        self._panning = True
        self._pan_last = pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._crop_mode and not self._placeholder:
            if self._crop_drag is not None and \
                    event.buttons() & Qt.MouseButton.LeftButton:
                mode, r0, p0 = self._crop_drag
                p1 = self.widget_to_norm(pos.x(), pos.y())
                self._crop_rect = self._apply_crop_drag(mode, r0, p0, p1)
                self._emit_crop_rect()
            else:
                hit = self._crop_hit(pos)
                if hit != self._crop_hover:
                    self._crop_hover = hit
                    self.setCursor(self._crop_cursor(hit))
            self.update()
            return
        if self._stamp_mode:
            if self._stamp_rubber is not None and \
                    event.buttons() & Qt.MouseButton.LeftButton:
                kind, start, _ = self._stamp_rubber
                self._stamp_rubber = (kind, start,
                                      self.widget_to_norm(pos.x(), pos.y()))
            self.update()
            return
        if self._edit_mode == "brush":
            if not self._placeholder:
                self._brush_cursor = pos
                if self._brush_drawing:
                    self.brushStrokePoint.emit(*self.widget_to_norm(pos.x(), pos.y()))
                self.update()
        elif self._edit_mode != "none" and not self._placeholder and self._drag_handle is None:
            self._update_hover(pos)
        if self._drag_handle is not None:
            nx, ny = self.widget_to_norm(pos.x(), pos.y())
            if self._edit_mode == "linear":
                info = self._edit_info
                if self._drag_handle == 0:
                    info["x1"], info["y1"] = nx, ny
                elif self._drag_handle == 1:
                    info["x2"], info["y2"] = nx, ny
                self.linearHandleMoved.emit(self._drag_handle, info["x1"], info["y1"],
                                            info["x2"], info["y2"])
            elif self._edit_mode == "radial":
                self._radial_drag(nx, ny)
            return
        if self._drag_line:
            delta = pos - self._pan_last
            self._pan_last = pos
            fx, fy = delta.x() / max(self._scale(), 1e-6), delta.y() / max(self._scale(), 1e-6)
            fw, fh = self._frame_size
            info = self._edit_info
            dx, dy = fx / max(fw, 1), fy / max(fh, 1)
            info["x1"] = float(np.clip(info.get("x1", 0.0) + dx, 0.0, 1.0))
            info["y1"] = float(np.clip(info.get("y1", 0.0) + dy, 0.0, 1.0))
            info["x2"] = float(np.clip(info.get("x2", 1.0) + dx, 0.0, 1.0))
            info["y2"] = float(np.clip(info.get("y2", 0.0) + dy, 0.0, 1.0))
            self.linearHandleMoved.emit(2, info["x1"], info["y1"], info["x2"], info["y2"])
            self.update()
            return
        if self._panning:
            delta = pos - self._pan_last
            self._pan_last = pos
            self._pan += delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._crop_mode:
                self._crop_drag = None
                if self._crop_rect:
                    self._emit_crop_rect()
            if self._stamp_mode and self._stamp_rubber is not None:
                kind, start, cur = self._stamp_rubber
                self._stamp_rubber = None
                r = (min(start[0], cur[0]), min(start[1], cur[1]),
                     max(start[0], cur[0]), max(start[1], cur[1]))
                fw, fh = self._frame_size
                s = max(self._scale(), 1e-6)
                if (r[2] - r[0]) * fw * s > 6.0 and (r[3] - r[1]) * fh * s > 6.0:
                    if kind == "src":
                        self.cloneSourcePicked.emit(r)
                    else:
                        self.cloneApplyRequested.emit(r)
                self.update()
            if self._edit_mode == "brush" and self._brush_drawing:
                self._brush_drawing = False
                self.brushStrokeFinished.emit()
            self._drag_handle = None
            self._drag_line = False
            self._radial_drag_start = None
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self._pick_mode
                           else Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        if self._crop_mode:
            return                          # 裁剪模式下双击不触发“适合窗口”
        if not self._placeholder:
            self.zoom_fit()

    def leaveEvent(self, event):
        if self._hover_handle is not None:
            self._hover_handle = None
            self.update()
        if self._crop_mode and self._crop_hover is not None:
            self._crop_hover = None
            self.update()


# ======================================================================
# 5. 曲线编辑器（主通道 + R/G/B，单调三次插值）
# ======================================================================

CHANNEL_LABELS = {"master": "RGB", "red": "红", "green": "绿", "blue": "蓝"}
CHANNEL_COLORS = {
    "master": QColor(225, 225, 232),
    "red": QColor(226, 92, 92),
    "green": QColor(96, 200, 120),
    "blue": QColor(96, 146, 255),
}
_GRID = QColor(44, 44, 48)
_BORDER = QColor(58, 58, 65)
_DIAG = QColor(70, 70, 78)
_POINT_FILL = QColor(30, 30, 34)
_SELECT_EDGE = QColor("#4c9be8")
_HIT_RADIUS = 9.0
_MARGIN = 12.0


class CurveEditor(QWidget):
    pointsChanged = Signal(str, list)     # (通道, 控制点) 拖动中
    pointsCommitted = Signal(str, list)   # (通道, 控制点) 松手

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setMouseTracking(True)
        self._channel = "master"
        self._points = {ch: [(0.0, 0.0), (1.0, 1.0)] for ch in CURVE_CHANNELS}
        self._hist: dict | None = None
        self._drag_index: int | None = None
        self._selected: int | None = None

    # ------------------------------------------------------------ 数据接口
    def set_channel(self, channel: str):
        if channel in CURVE_CHANNELS:
            self._channel = channel
            self.update()

    def current_channel(self) -> str:
        return self._channel

    def set_points(self, channel: str, points):
        self._points[channel] = [tuple(p) for p in sanitize_points(points)]
        self.update()

    def get_points(self, channel: str):
        return [tuple(p) for p in self._points[channel]]

    def set_histograms(self, hist: dict | None):
        self._hist = hist
        self.update()

    # ------------------------------------------------------------ 坐标变换
    def _plot_rect(self) -> QRectF:
        return QRectF(_MARGIN, _MARGIN,
                      max(self.width() - 2 * _MARGIN, 1.0),
                      max(self.height() - 2 * _MARGIN, 1.0))

    def _to_widget(self, x: float, y: float) -> QPointF:
        r = self._plot_rect()
        return QPointF(r.left() + x * r.width(), r.top() + (1.0 - y) * r.height())

    def _to_image(self, px: float, py: float) -> tuple:
        r = self._plot_rect()
        x = (px - r.left()) / max(r.width(), 1e-6)
        y = 1.0 - (py - r.top()) / max(r.height(), 1e-6)
        return x, y

    # ------------------------------------------------------------ 绘制
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self._plot_rect()

        p.fillRect(self.rect(), QColor(18, 18, 20))
        p.setPen(QPen(_BORDER, 1))
        p.drawRect(r)

        # 网格 + 对角线
        p.setPen(QPen(_GRID, 1))
        for i in range(1, 4):
            x = r.left() + r.width() * i / 4.0
            y = r.top() + r.height() * i / 4.0
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(_DIAG, 1, Qt.PenStyle.DashLine))
        p.drawLine(self._to_widget(0.02, 0.02), self._to_widget(0.98, 0.98))

        # 直方图背景
        if self._hist:
            for ch in ("red", "green", "blue"):
                data = self._hist.get(ch)
                if data is not None:
                    self._draw_histogram(p, r, data, CHANNEL_COLORS[ch], 70)
            data = self._hist.get("master")
            if data is not None:
                self._draw_histogram(p, r, data, QColor(200, 200, 205), 50)

        # 其它通道曲线（淡色上下文）+ 当前通道曲线
        for ch in CURVE_CHANNELS:
            if ch != self._channel and len(self._points[ch]) >= 2:
                self._draw_curve(p, self._points[ch], CHANNEL_COLORS[ch], 60, 1.0)
        self._draw_curve(p, self._points[self._channel], CHANNEL_COLORS[self._channel], 255, 2.0)

        # 控制点
        for i, (x, y) in enumerate(self._points[self._channel]):
            c = self._to_widget(x, y)
            selected = i in (self._selected, self._drag_index)
            p.setPen(QPen(_SELECT_EDGE if selected else CHANNEL_COLORS[self._channel], 1.5))
            p.setBrush(_SELECT_EDGE if selected else _POINT_FILL)
            p.drawEllipse(c, 5.5 if selected else 4.2, 5.5 if selected else 4.2)
        p.end()

    def _draw_histogram(self, p: QPainter, r: QRectF, data: np.ndarray, color: QColor, alpha: int):
        n = len(data)
        if n == 0 or data.max() <= 0:
            return
        path = QPainterPath(QPointF(r.left(), r.bottom()))
        for i, v in enumerate(data):
            x = r.left() + r.width() * (i / (n - 1.0))
            y = r.bottom() - float(v) * r.height() * 0.92
            path.lineTo(x, y)
        path.lineTo(r.right(), r.bottom())
        path.closeSubpath()
        c = QColor(color)
        c.setAlpha(alpha)
        p.fillPath(path, c)

    def _draw_curve(self, p: QPainter, points, color: QColor, alpha: int, width: float):
        lut = build_lut(points, LUT_SIZE)
        c = QColor(color)
        c.setAlpha(alpha)
        p.setPen(QPen(c, width))
        path = QPainterPath()
        step = max(1, LUT_SIZE // 220)
        pts = [self._to_widget(i / (LUT_SIZE - 1.0), float(lut[i]))
               for i in range(0, LUT_SIZE, step)]
        pts.append(self._to_widget(1.0, float(lut[-1])))
        path.moveTo(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        p.drawPath(path)

    # ------------------------------------------------------------ 交互
    def _hit_test(self, pos: QPointF) -> int | None:
        best, best_d = None, _HIT_RADIUS
        for i, (x, y) in enumerate(self._points[self._channel]):
            d = (self._to_widget(x, y) - pos).manhattanLength()
            if d < best_d:
                best, best_d = i, d
        return best

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_test(event.position())
            if idx is not None:
                self._drag_index = idx
                self._selected = idx
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            idx = self._hit_test(event.position())
            pts = self._points[self._channel]
            if idx is not None and 0 < idx < len(pts) - 1:
                del pts[idx]
                self._selected = None
                self.pointsChanged.emit(self._channel, [tuple(q) for q in pts])
                self.pointsCommitted.emit(self._channel, [tuple(q) for q in pts])
                self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._plot_rect().contains(event.position()):
            return
        x, y = self._to_image(event.position().x(), event.position().y())
        pts = list(self._points[self._channel])
        if any(abs(x - px) < 1 / 512.0 for px, _ in pts):
            return
        new_pt = (float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0)))
        pts.append(new_pt)
        pts = sorted(pts, key=lambda q: q[0])
        self._selected = next(i for i, q in enumerate(pts) if q == new_pt)
        self._points[self._channel] = pts
        self.pointsChanged.emit(self._channel, [tuple(q) for q in pts])
        self.pointsCommitted.emit(self._channel, [tuple(q) for q in pts])
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_index is None:
            return
        pts = list(self._points[self._channel])
        i = self._drag_index
        x, y = self._to_image(event.position().x(), event.position().y())
        y = float(np.clip(y, 0.0, 1.0))
        gap = 1 / 512.0
        lo = 0.0 if i == 0 else pts[i - 1][0] + gap
        hi = 1.0 if i == len(pts) - 1 else pts[i + 1][0] - gap
        if i == 0 or i == len(pts) - 1:
            x = float(i)                   # 端点钉在 x=0/1，与 sanitize_points 生效曲线一致
        else:
            x = float(np.clip(x, lo, hi))
        pts[i] = (x, y)
        self._points[self._channel] = pts
        self.pointsChanged.emit(self._channel, [tuple(q) for q in pts])
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag_index is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_index = None
            self.pointsCommitted.emit(self._channel,
                                      [tuple(q) for q in self._points[self._channel]])
            self.update()


# ======================================================================
# 6. 色轮（外环色相 + 内盘拖动）
# ======================================================================

_DISC_R = 40.0
_RING_W = 9.0


class ColorWheel(QWidget):
    hueSatChanged = Signal(float, float)   # 色相(度), 饱和度 0..1
    wheelReleased = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        size = int((_DISC_R + _RING_W) * 2 + 6)
        self.setFixedSize(size, size)
        self._hue = 0.0
        self._sat = 0.0
        self._dragging = False

    def hue_sat(self) -> tuple:
        return self._hue, self._sat

    def set_hue_sat(self, hue: float, sat: float):
        self._hue = float(hue) % 360.0
        self._sat = float(min(max(sat, 0.0), 1.0))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        ring_r = _DISC_R + _RING_W / 2.0

        grad = QConicalGradient(center, 0.0)
        for i in range(13):
            grad.setColorAt((i / 12.0) % 1.0, QColor.fromHsvF((i / 12.0) % 1.0, 1.0, 1.0))
        pen = QPen()
        pen.setWidth(_RING_W)
        pen.setBrush(grad)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center, ring_r, ring_r)

        p.setPen(QPen(QColor(58, 58, 65), 1))
        p.setBrush(QColor(38, 38, 42))
        p.drawEllipse(center, _DISC_R, _DISC_R)

        h_rad = math.radians(self._hue)
        r = self._sat * (_DISC_R - 3.0)
        # 与 QConicalGradient 的环一致：色相 0° 在 3 点钟方向、90° 在 12 点钟方向
        # （Qt 屏幕坐标 y 向下，故 sin 分量取负，否则拖点与色环上下镜像）
        dot = QPointF(center.x() + r * math.cos(h_rad),
                      center.y() - r * math.sin(h_rad))
        p.setPen(QPen(QColor(240, 240, 245), 1.5))
        p.setBrush(QColor.fromHsvF(self._hue / 360.0, self._sat, 1.0))
        p.drawEllipse(dot, 5.0, 5.0)
        p.end()

    def _update_from_pos(self, pos: QPointF):
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        dx, dy = pos.x() - center.x(), pos.y() - center.y()
        # 屏幕坐标 y 向下，取 -dy 与色环渐变方向一致（见 paintEvent）
        self._hue = math.degrees(math.atan2(-dy, dx)) % 360.0
        self._sat = min(math.hypot(dx, dy) / (_DISC_R - 3.0), 1.0)
        self.update()
        self.hueSatChanged.emit(self._hue, self._sat)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_from_pos(event.position())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_from_pos(event.position())

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.wheelReleased.emit()


# ======================================================================
# 7. 后台处理线程（请求合并 + 线性光缓存）
# ======================================================================

PREVIEW = "preview"
FULL = "full"

# 供无界面测试使用：同步执行
SYNC_MODE = (os.environ.get("FERNLENS_SYNC") or os.environ.get("PHOTOLAB_SYNC")) == "1"


class ProcessorThread(QThread):
    """线程体：循环等待任务 -> 处理 -> 发信号。

    - 普通任务任何时刻只保留最新一次：连续拖动滑块时中间的帧被覆盖丢弃；
      urgent=True 的任务（如保存）走独立槽位，永不被普通任务覆盖。
    - 预览（最长边 1200px）与全图两种数据源，线性光结果按数据源缓存；
      数据源在 set_source 时拷贝，GUI 侧对图章工作副本的原地修改不会撕裂渲染。
    - 结果带请求 id 回到主线程，过期帧由接收方丢弃。
    - stop() 触发协作取消：正在渲染的任务在分块之间尽快中止，保证关窗时
      线程能及时退出（大图全图渲染可能长达数秒，硬等会超时）。
    """

    frameReady = Signal(np.ndarray, int, bool)     # (uint8 RGB, 请求 id, 是否全图)
    busyChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._job = None                 # 普通任务槽：(req_id, kind, params)，最新覆盖
        self._urgent = None              # 紧急任务槽（保存等），不被普通任务覆盖
        self._sources: dict = {}         # kind -> uint8 RGB 原图（私有副本）
        self._src_gen: dict = {}         # kind -> 数据版本号
        self._lin_cache: dict = {}       # kind -> (gen, 线性光数组)
        self._stopped = False
        self._req_counter = 0

    # ------------------------------------------------------------ 控制
    def stop(self):
        with self._cond:
            self._stopped = True
            self._job = None
            self._urgent = None
            self._cond.notify_all()
        if self.isRunning():
            self.wait(5000)

    def set_source(self, kind: str, rgb8: np.ndarray):
        """设置数据源（打开新图像时调用），并让线性缓存失效。

        传入的数组会拷贝一份：调用方之后原地修改（仿制图章工作副本）
        不会与正在进行的渲染并发读写同一块内存。
        """
        rgb8 = np.array(rgb8, copy=True)
        gen = self._src_gen.get(kind, 0) + 1
        with self._cond:
            self._sources[kind] = rgb8
            self._src_gen[kind] = gen
            self._lin_cache.pop(kind, None)

    def submit(self, params, kind: str, urgent: bool = False) -> int:
        """提交一次处理请求，返回请求 id。

        urgent 任务（保存）放入独立槽位：即使随后有普通任务提交也保证执行。
        """
        with self._cond:
            self._req_counter += 1
            rid = self._req_counter
            job = (rid, kind, params.copy())
            if not SYNC_MODE:
                if urgent:
                    self._urgent = job
                else:
                    self._job = job
                self._cond.notify()
        if SYNC_MODE:
            self._execute(job)
        return rid

    def pending_request(self) -> int:
        with self._cond:
            return self._req_counter

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
        rid, kind, params = job
        with self._cond:
            src = self._sources.get(kind)
            gen = self._src_gen.get(kind, -1)
            cached = self._lin_cache.get(kind)
        if src is None:
            return
        self.busyChanged.emit(True)
        try:
            if cached is not None and cached[0] == gen:
                lin = cached[1]
            else:
                lin = rgb8_to_linear_mt(src)
                with self._cond:           # 期间数据源被更换则丢弃本次缓存
                    if self._src_gen.get(kind, -1) == gen:
                        self._lin_cache[kind] = (gen, lin)
            out = process_linear(params, lin,
                                 should_cancel=lambda: self._stopped)
        except CancelledError:             # 停止请求：静默丢弃半成品
            self.busyChanged.emit(False)
            return
        except Exception:                      # 单个任务失败不应中断线程
            traceback.print_exc()
            self.busyChanged.emit(False)
            return
        self.busyChanged.emit(False)
        self.frameReady.emit(out, rid, kind == FULL)
