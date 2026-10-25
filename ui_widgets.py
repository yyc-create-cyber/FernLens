"""FernLens UI：主题、通用控件与后台处理线程。

内容分区：
  1. 淡蓝灰主题  2. 滑块行  3. 可折叠毛玻璃分区  4. 图像预览
  5. 曲线编辑器  6. 色轮  7. 后台处理线程
"""
from __future__ import annotations

import os
import math
import threading
import traceback

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (QColor, QConicalGradient, QImage, QLinearGradient,
                           QMouseEvent, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QVBoxLayout, QWidget)

from core import (CURVE_CHANNELS, LUT_SIZE, build_lut, rgb8_to_linear_mt,
                  sanitize_points, process_linear)

# ======================================================================
# 1. 淡蓝灰主题（白色毛玻璃托盘 + 浅蓝磨砂标题栏）
# ======================================================================

BG = "#dbe6f0"          # 窗口背景：淡蓝灰
PANEL = "#eaf2f8"       # 工具栏/面板背景
CONTROL = "#f8fbfe"     # 控件背景
BORDER = "#b6c9d9"      # 边框
TEXT = "#263b4a"        # 主文字
TEXT_DIM = "#678093"    # 次要文字
ACCENT = "#4a9ed6"      # 强调色
ACCENT_DARK = "#2f7fb3"
SLIDER_FILL = "#c7d8e6"  # 更浅的滑块轨道
SLIDER_FILLED = "#8dc7ea"  # 滑块已填充部分（浅蓝）

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
                 "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Segoe UI", sans-serif;
    font-size: 12px;
}}
QMainWindow, QDialog, QSplitter {{ background-color: {BG}; }}
QToolBar {{
    background-color: {PANEL};
    border-bottom: 1px solid {BORDER};
    spacing: 6px;
    padding: 4px 6px;
}}
QToolBar QToolButton {{
    background: rgba(255, 255, 255, 105); border: 1px solid transparent;
    border-radius: 5px; padding: 5px 10px; color: {TEXT};
}}
QToolBar QToolButton:hover {{ background-color: rgba(255, 255, 255, 205); }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background-color: #c4e4f7; border-color: {ACCENT}; color: #173a52;
}}
QToolBar QToolButton:disabled {{ color: {TEXT_DIM}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #9fb9cb; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #7fa3bb; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #9fb9cb; border-radius: 5px; min-width: 30px; }}

QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 5px; background: {SLIDER_FILL}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {SLIDER_FILLED}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #ffffff; border: 1px solid #91b6ce; width: 13px;
    margin: -6px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; border: 2px solid {ACCENT}; }}

QPushButton {{
    background-color: {CONTROL}; border: 1px solid {BORDER}; border-radius: 5px;
    padding: 5px 10px; color: {TEXT};
}}
QPushButton:hover {{ background-color: #eef8fe; border-color: #94bed8; }}
QPushButton:pressed {{ background-color: #cfe8f7; border-color: {ACCENT}; }}
QPushButton:checked {{ background-color: #c9e6f7; border-color: {ACCENT}; color: #173a52; }}
QPushButton:disabled {{ color: #9aacba; background-color: rgba(255, 255, 255, 75); }}

QLineEdit, QComboBox {{
    background-color: #ffffff; border: 1px solid {BORDER}; border-radius: 5px;
    padding: 4px 6px; color: {TEXT}; selection-background-color: #b8ddf5;
    selection-color: #173a52;
}}
QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {TEXT_DIM}; background: #eef3f7; }}
QComboBox QAbstractItemView {{
    background-color: #fbfdff; border: 1px solid {BORDER};
    selection-background-color: #c4e4f7; selection-color: #173a52; color: {TEXT};
}}
QListWidget {{ background-color: rgba(255, 255, 255, 135); border: 1px solid {BORDER};
               border-radius: 5px; }}
QListWidget::item {{ padding: 6px 8px; border-radius: 3px; }}
QListWidget::item:selected {{ background-color: #c4e4f7; color: #173a52; }}
QCheckBox {{ background: transparent; spacing: 6px; }}
QDialog {{ background-color: {BG}; }}

QLabel {{ background: transparent; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#value {{ color: {TEXT_DIM}; qproperty-alignment: AlignRight; }}
QLabel#sectionTitle {{ color: #2d526b; font-weight: bold; font-size: 12px; }}
QLabel#groupLabel {{ color: {TEXT_DIM}; font-size: 11px; }}

QGroupBox {{ border: 1px solid rgba(163, 190, 210, 120); border-radius: 8px;
             margin-top: 10px; padding: 8px 6px 6px 6px;
             background-color: rgba(255, 255, 255, 72); }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {ACCENT_DARK}; }}

QStatusBar {{ background-color: {PANEL}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QMenu {{ background-color: #fbfdff; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background-color: #c4e4f7; color: #173a52; }}
QFileDialog {{ background-color: {BG}; }}
QToolTip {{
    background-color: #fbfdff; color: {TEXT}; border: 1px solid #9fb8ca;
    padding: 3px 5px; border-radius: 4px;
}}
"""


def apply_theme(app):
    """应用 Fusion 风格 + 淡蓝灰色调色板 + QSS。"""
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
# 3. 可折叠分区（白色毛玻璃托盘 + 浅蓝磨砂标题条）
# ======================================================================

_FROST_TEX: QPixmap | None = None


def _frost_texture() -> QPixmap:
    """生成可平铺的细微噪点，模拟毛玻璃的颗粒质感。"""
    global _FROST_TEX
    if _FROST_TEX is None:
        rng = np.random.default_rng(17)
        size = 64
        noise = rng.integers(0, 26, (size, size), dtype=np.uint8)
        rgba = np.dstack([np.full((size, size, 3), 255, np.uint8), noise])
        img = QImage(rgba.data, size, size, 4 * size,
                     QImage.Format.Format_RGBA8888).copy()
        _FROST_TEX = QPixmap.fromImage(img)
    return _FROST_TEX


def _top_rounded_path(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(rect.left(), rect.bottom())
    path.lineTo(rect.left(), rect.top() + radius)
    path.quadTo(rect.left(), rect.top(), rect.left() + radius, rect.top())
    path.lineTo(rect.right() - radius, rect.top())
    path.quadTo(rect.right(), rect.top(), rect.right(), rect.top() + radius)
    path.lineTo(rect.right(), rect.bottom())
    path.closeSubpath()
    return path


class GlassSectionHeader(QPushButton):
    """浅蓝色半透明磨砂标题条（自绘渐变、高光与细噪点）。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("background: transparent; border: none; padding: 0;")

    def set_title(self, title: str):
        self._title = title
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = _top_rounded_path(rect, 12.0)

        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        if self.underMouse():
            grad.setColorAt(0.0, QColor(242, 251, 255, 238))
            grad.setColorAt(1.0, QColor(167, 214, 246, 224))
        else:
            grad.setColorAt(0.0, QColor(236, 248, 255, 228))
            grad.setColorAt(1.0, QColor(181, 220, 247, 210))
        p.fillPath(path, grad)

        # 毛玻璃颗粒 + 顶部高光
        p.save()
        p.setClipPath(path)
        p.setOpacity(0.42)
        p.drawTiledPixmap(self.rect(), _frost_texture())
        p.setOpacity(1.0)
        highlight = QLinearGradient(rect.topLeft(), QPointF(rect.left(), rect.top() + 12))
        highlight.setColorAt(0.0, QColor(255, 255, 255, 175))
        highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(QRectF(rect.left(), rect.top(), rect.width(), 12), highlight)
        p.restore()

        p.setPen(QPen(QColor(255, 255, 255, 170), 1.0))
        p.drawPath(path)
        p.setPen(QPen(QColor(120, 170, 205, 82), 1.0))
        p.drawLine(QPointF(rect.left() + 10, rect.bottom()),
                   QPointF(rect.right() - 10, rect.bottom()))

        font = p.font()
        font.setBold(True)
        font.setPixelSize(12)
        p.setFont(font)
        p.setPen(QColor("#23465f"))
        arrow = "▾  " if self.isChecked() else "▸  "
        p.drawText(rect.adjusted(14, 0, -12, 0),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   arrow + self._title)
        p.end()


class CollapsibleSection(QWidget):
    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        self.header = GlassSectionHeader(title)
        self.header.setChecked(expanded)
        self.header.toggled.connect(self._on_toggled)

        body = QWidget()
        body.setObjectName("glassSectionBody")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        body.setStyleSheet("QWidget#glassSectionBody { background: transparent; }")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(10, 8, 10, 10)
        body_lay.setSpacing(0)
        body_lay.addWidget(content)
        self.body = body

        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 3, 5, 8)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(body)
        self.content = content
        self.content.setVisible(expanded)
        body.setVisible(expanded)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        card = QRectF(self.rect()).adjusted(5.0, 3.0, -5.0, -8.0)
        radius = 13.0

        # 柔和投影（多层半透明圆角矩形，避免 QGraphicsEffect 的滚动开销）
        p.setPen(Qt.PenStyle.NoPen)
        for spread, alpha in ((5.0, 8), (3.0, 13), (1.5, 20)):
            p.setBrush(QColor(69, 107, 137, alpha))
            p.drawRoundedRect(card.adjusted(-spread * 0.3, spread * 0.45,
                                            spread * 0.3, spread),
                              radius + spread * 0.3, radius + spread * 0.3)

        grad = QLinearGradient(card.topLeft(), card.bottomRight())
        grad.setColorAt(0.0, QColor(255, 255, 255, 218))
        grad.setColorAt(0.55, QColor(252, 254, 255, 176))
        grad.setColorAt(1.0, QColor(242, 249, 255, 148))
        p.setBrush(grad)
        p.setPen(QPen(QColor(255, 255, 255, 205), 1.2))
        p.drawRoundedRect(card.adjusted(0.6, 0.6, -0.6, -0.6), radius, radius)

        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(card, radius, radius)
        p.setClipPath(clip)
        p.setOpacity(0.30)
        p.drawTiledPixmap(card.toRect(), _frost_texture())
        p.setOpacity(1.0)
        sheen = QLinearGradient(card.topLeft(), QPointF(card.right(), card.bottom()))
        sheen.setColorAt(0.0, QColor(255, 255, 255, 86))
        sheen.setColorAt(0.42, QColor(255, 255, 255, 12))
        sheen.setColorAt(1.0, QColor(199, 228, 248, 34))
        p.fillPath(clip, sheen)
        p.restore()

        p.setPen(QPen(QColor(165, 191, 211, 95), 1.0))
        p.drawRoundedRect(card, radius, radius)
        p.end()

    def _refresh_text(self):
        self.header.set_title(self._title)
        self.header.update()

    def _on_toggled(self, checked: bool):
        self._expanded = checked
        self.content.setVisible(checked)
        self.body.setVisible(checked)
        self._refresh_text()
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().invalidate()
            parent.layout().activate()
        self.toggled.emit(checked)

    def set_expanded(self, expanded: bool):
        self.header.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self._expanded


# ======================================================================
# 4. 图像预览（淡蓝灰底、缩放、平移、取色点击）
# ======================================================================

_CANVAS_BG = QColor(184, 199, 212)      # 淡蓝灰画布底色


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
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 226, 62, 78
        rgba[..., 3] = (m8 * 0.58).astype(np.uint8)
    else:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 64, 99, 128
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

    # ---- 自定义多边形蒙版 ----
    polygonPointAdded = Signal(float, float)        # 新增顶点（归一化 x, y）
    polygonVertexMoved = Signal(int, float, float)  # 顶点索引, 归一化 x, y
    polygonVertexDeleted = Signal(int)              # 顶点索引
    polygonFinished = Signal()                      # 完成多边形（至少 3 点）

    # ---- 仿制图章 ----
    cloneSampled = Signal(float, float)             # Alt+点击取样（归一化 x, y）
    cloneStrokePoint = Signal(float, float)         # 涂抹目标点（归一化 x, y）
    cloneStrokeFinished = Signal()

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
        self._polygon_cursor: QPointF | None = None   # 多边形选点时的橡皮筋末端
        self._polygon_drag_index: int | None = None
        self._hover_vertex: int | None = None
        self._panning = False
        self._pan_last = QPointF()
        self._placeholder = True
        self._stamp_mode = False
        self._stamp_drawing = False
        self._stamp_size = 12.0
        self._clone_marker: QPointF | None = None      # 取样源点（控件坐标）

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
        self._stamp_drawing = False
        if on:
            self._pick_mode = False
            self._edit_mode = "none"
        elif not self._pick_mode:
            pass
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    def stamp_mode(self) -> bool:
        return self._stamp_mode

    def set_stamp_info(self, size: float):
        self._stamp_size = float(size)
        if self._stamp_mode:
            self.update()

    def set_clone_marker(self, marker):
        """设置/清除取样源点（归一化坐标 tuple 或 None）。"""
        if marker is None:
            self._clone_marker = None
        else:
            self._clone_marker = self.frame_point_to_widget(marker[0], marker[1])
        self.update()

    # ------------------------------------------------------------ 蒙版编辑
    def set_edit_mode(self, mode: str, info: dict | None = None):
        """进入/退出蒙版编辑模式；info 携带当前几何（归一化坐标）。"""
        self._edit_mode = mode
        self._edit_info = info or {}
        self._drag_handle = None
        self._drag_line = False
        self._radial_drag_start = None
        self._hover_handle = None
        self._polygon_cursor = None
        self._polygon_drag_index = None
        self._hover_vertex = None
        if mode != "none" and self._pick_mode:
            self._pick_mode = False
        if mode == "polygon":
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def edit_mode(self) -> str:
        return self._edit_mode

    def set_overlays(self, overlays: list):
        """设置蒙版叠加图（QImage 列表，与原图同尺寸，带 alpha）。"""
        self._overlays = overlays
        self.update()

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
            p.setPen(QColor(81, 106, 126))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "打开图像开始调色\n(Ctrl+O)")
            p.end()
            return
        off = self._offset()
        s = self._scale()
        target = QRectF(off.x(), off.y(), self._frame_size[0] * s, self._frame_size[1] * s)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        p.setPen(QColor(129, 151, 170))
        p.drawRect(target)

        # 蒙版叠加着色
        for overlay in self._overlays:
            if overlay is not None:
                p.drawImage(target, overlay)

        # 蒙版编辑辅助线
        self._paint_edit_overlay(p)
        self._paint_stamp_overlay(p)
        p.end()

    def _paint_stamp_overlay(self, p: QPainter):
        """图章模式：取样源点十字标记 + 笔刷圈。"""
        if not self._stamp_mode or self._placeholder:
            return
        if self._clone_marker is not None:
            p.setPen(QPen(QColor("#0d7f95"), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._clone_marker, 8.0, 8.0)
            p.drawLine(QPointF(self._clone_marker.x() - 13, self._clone_marker.y()),
                       QPointF(self._clone_marker.x() + 13, self._clone_marker.y()))
            p.drawLine(QPointF(self._clone_marker.x(), self._clone_marker.y() - 13),
                       QPointF(self._clone_marker.x(), self._clone_marker.y() + 13))
        if self._brush_cursor is not None:
            short = max(min(self._frame_size), 1)
            r = self._scale() * short * (self._stamp_size / 100.0) * 0.5
            p.setPen(QPen(QColor("#0d7f95"), 1.0))
            p.drawEllipse(self._brush_cursor, r, r)

    def _paint_edit_overlay(self, p: QPainter):
        if self._edit_mode == "none" or self._placeholder:
            return
        pen = QPen(QColor("#1d6f9d"), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        if self._edit_mode == "polygon":
            info = self._edit_info
            pts = [self.frame_point_to_widget(float(x), float(y))
                   for x, y in info.get("points", [])]
            closed = bool(info.get("closed", False))
            if len(pts) >= 2:
                poly = QPolygonF(pts)
                p.setBrush(Qt.BrushStyle.NoBrush)
                if closed and len(pts) >= 3:
                    p.drawPolygon(poly)
                else:
                    p.drawPolyline(poly)
            if not closed and self._polygon_cursor is not None and pts:
                p.setPen(QPen(QColor(47, 143, 196), 1.3, Qt.PenStyle.DashLine))
                p.drawLine(pts[-1], self._polygon_cursor)
                if len(pts) >= 2:
                    p.setPen(QPen(QColor(47, 143, 196, 125), 1.0,
                                  Qt.PenStyle.DotLine))
                    p.drawLine(self._polygon_cursor, pts[0])
            for i, pt in enumerate(pts):
                hot = (i == self._hover_vertex or i == self._polygon_drag_index)
                first = (i == 0 and not closed and len(pts) >= 3)
                r = 7.0 if hot else (6.4 if first else 5.0)
                p.setPen(QPen(QColor("#ffffff"), 1.6))
                p.setBrush(QColor("#2f8fc9") if (hot or first)
                           else QColor("#ffffff"))
                p.drawEllipse(pt, r, r)
        elif self._edit_mode == "linear":
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

    def _polygon_hit(self, pos: QPointF) -> int | None:
        """命中多边形顶点时返回索引。"""
        for i, (x, y) in enumerate(self._edit_info.get("points", [])):
            pt = self.frame_point_to_widget(float(x), float(y))
            if (QPointF(pt) - pos).manhattanLength() <= 12.0:
                return i
        return None

    def _polygon_close_hit(self, pos: QPointF) -> bool:
        pts = self._edit_info.get("points", [])
        if len(pts) < 3:
            return False
        first = self.frame_point_to_widget(float(pts[0][0]), float(pts[0][1]))
        return (QPointF(first) - pos).manhattanLength() <= 13.0

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
        if self._pick_mode and event.button() == Qt.MouseButton.LeftButton:
            if not self._placeholder:
                self.colorPicked.emit(int(pos.x()), int(pos.y()))
            return
        if (self._edit_mode == "polygon" and not self._placeholder
                and event.button() == Qt.MouseButton.RightButton):
            closed = bool(self._edit_info.get("closed", False))
            hit = self._polygon_hit(pos)
            if not closed:
                self.polygonFinished.emit()
            elif hit is not None:
                self.polygonVertexDeleted.emit(hit)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            if event.button() in (Qt.MouseButton.MiddleButton,):
                self._start_pan(pos)
            return

        if self._stamp_mode:
            if not self._placeholder:
                if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                    self.cloneSampled.emit(*self.widget_to_norm(pos.x(), pos.y()))
                else:
                    self._stamp_drawing = True
                    self.cloneStrokePoint.emit(*self.widget_to_norm(pos.x(), pos.y()))
            return

        if self._edit_mode == "polygon":
            if self._placeholder:
                return
            self.setFocus()
            closed = bool(self._edit_info.get("closed", False))
            if not closed:
                if self._polygon_close_hit(pos):
                    self.polygonFinished.emit()
                else:
                    self.polygonPointAdded.emit(*self.widget_to_norm(pos.x(), pos.y()))
            else:
                idx = self._polygon_hit(pos)
                if idx is not None:
                    self._polygon_drag_index = idx
                    self._hover_vertex = idx
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self._start_pan(pos)
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
        if self._stamp_mode:
            self._brush_cursor = pos
            if self._stamp_drawing and not self._placeholder:
                self.cloneStrokePoint.emit(*self.widget_to_norm(pos.x(), pos.y()))
            self.update()
            if not self._panning:
                return
        if self._edit_mode == "polygon":
            if not self._placeholder:
                self._polygon_cursor = pos
                if self._polygon_drag_index is not None:
                    nx, ny = self.widget_to_norm(pos.x(), pos.y())
                    info = self._edit_info
                    points = info.get("points", [])
                    i = self._polygon_drag_index
                    if 0 <= i < len(points):
                        points[i] = (float(nx), float(ny))
                        self.polygonVertexMoved.emit(i, float(nx), float(ny))
                    self.update()
                    return
                closed = bool(self._edit_info.get("closed", False))
                if not self._panning:
                    if closed:
                        hit = self._polygon_hit(pos)
                        if hit != self._hover_vertex:
                            self._hover_vertex = hit
                            self.update()
                        self.setCursor(Qt.CursorShape.SizeAllCursor if hit is not None
                                       else Qt.CursorShape.OpenHandCursor)
                    else:
                        self.setCursor(Qt.CursorShape.CrossCursor)
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
            if self._stamp_mode and self._stamp_drawing:
                self._stamp_drawing = False
                self.cloneStrokeFinished.emit()
            if self._edit_mode == "brush" and self._brush_drawing:
                self._brush_drawing = False
                self.brushStrokeFinished.emit()
            self._drag_handle = None
            self._drag_line = False
            self._radial_drag_start = None
            self._polygon_drag_index = None
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            if self._pick_mode or self._stamp_mode:
                cursor = Qt.CursorShape.CrossCursor
            elif (self._edit_mode == "polygon"
                  and not bool(self._edit_info.get("closed", False))):
                cursor = Qt.CursorShape.CrossCursor
            elif self._edit_mode == "polygon":
                cursor = Qt.CursorShape.OpenHandCursor
            else:
                cursor = Qt.CursorShape.ArrowCursor
            self.setCursor(cursor)

    def mouseDoubleClickEvent(self, event):
        if (self._edit_mode == "polygon" and
                not bool(self._edit_info.get("closed", False))):
            self.polygonFinished.emit()
            return
        if not self._placeholder:
            self.zoom_fit()

    def keyPressEvent(self, event):
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and self._edit_mode == "polygon"
                and not bool(self._edit_info.get("closed", False))):
            self.polygonFinished.emit()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event):
        if self._hover_handle is not None:
            self._hover_handle = None
            self.update()
        if self._hover_vertex is not None:
            self._hover_vertex = None
            self.update()
        if self._polygon_cursor is not None:
            self._polygon_cursor = None
            self.update()


# ======================================================================
# 5. 曲线编辑器（主通道 + R/G/B，单调三次插值）
# ======================================================================

CHANNEL_LABELS = {"master": "RGB", "red": "红", "green": "绿", "blue": "蓝"}
CHANNEL_COLORS = {
    "master": QColor(47, 72, 90),
    "red": QColor(196, 63, 71),
    "green": QColor(47, 137, 89),
    "blue": QColor(52, 108, 199),
}
_GRID = QColor(202, 219, 232)
_BORDER = QColor(154, 178, 197)
_DIAG = QColor(172, 195, 212)
_POINT_FILL = QColor(255, 255, 255)
_SELECT_EDGE = QColor("#2f8fc9")
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

        p.fillRect(self.rect(), QColor(250, 253, 255, 175))
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
                self._draw_histogram(p, r, data, QColor(80, 108, 128), 55)

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
        pts[i] = (float(np.clip(x, lo, hi)), y)
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

        p.setPen(QPen(QColor(150, 176, 195), 1))
        p.setBrush(QColor(244, 250, 255))
        p.drawEllipse(center, _DISC_R, _DISC_R)

        h_rad = math.radians(self._hue)
        r = self._sat * (_DISC_R - 3.0)
        dot = QPointF(center.x() + r * math.cos(h_rad),
                      center.y() + r * math.sin(h_rad))
        p.setPen(QPen(QColor(67, 92, 111), 1.4))
        p.setBrush(QColor.fromHsvF(self._hue / 360.0, self._sat, 0.96))
        p.drawEllipse(dot, 5.0, 5.0)
        p.end()

    def _update_from_pos(self, pos: QPointF):
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        dx, dy = pos.x() - center.x(), pos.y() - center.y()
        self._hue = math.degrees(math.atan2(dy, dx)) % 360.0
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
SYNC_MODE = os.environ.get("FERNLENS_SYNC") == "1"


class ProcessorThread(QThread):
    """线程体：循环等待任务 -> 处理 -> 发信号。

    - 任何时刻只保留最新一次任务：连续拖动滑块时中间的帧被覆盖丢弃
    - 预览（最长边 1200px）与全图两种数据源，线性光结果按数据源缓存
    - 结果带请求 id 回到主线程，过期帧由接收方丢弃
    """

    frameReady = Signal(np.ndarray, int, bool)     # (uint8 RGB, 请求 id, 是否全图)
    busyChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._job = None                 # (req_id, kind, params)
        self._sources: dict = {}         # kind -> uint8 RGB 原图
        self._src_gen: dict = {}         # kind -> 数据版本号
        self._lin_cache: dict = {}       # kind -> (gen, 线性光数组)
        self._stopped = False
        self._req_counter = 0

    # ------------------------------------------------------------ 控制
    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self.isRunning():
            self.wait(5000)

    def set_source(self, kind: str, rgb8: np.ndarray):
        """设置数据源（打开新图像时调用），并让线性缓存失效。"""
        gen = self._src_gen.get(kind, 0) + 1
        with self._cond:
            self._sources[kind] = rgb8
            self._src_gen[kind] = gen
            self._lin_cache.pop(kind, None)

    def submit(self, params, kind: str) -> int:
        """提交一次处理请求，返回请求 id。"""
        with self._cond:
            self._req_counter += 1
            rid = self._req_counter
            job = (rid, kind, params.copy())
            if not SYNC_MODE:
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
                while self._job is None and not self._stopped:
                    self._cond.wait(0.2)
                if self._job is None:
                    if self._stopped:
                        return
                    continue
                job = self._job
                self._job = None
            self._execute(job)

    def _execute(self, job):
        rid, kind, params = job
        src = self._sources.get(kind)
        if src is None:
            return
        self.busyChanged.emit(True)
        try:
            gen = self._src_gen.get(kind, -1)
            cached = self._lin_cache.get(kind)
            if cached is not None and cached[0] == gen:
                lin = cached[1]
            else:
                lin = rgb8_to_linear_mt(src)
                self._lin_cache[kind] = (gen, lin)
            out = process_linear(params, lin)
        except Exception:                      # 单个任务失败不应中断线程
            traceback.print_exc()
            self.busyChanged.emit(False)
            return
        self.busyChanged.emit(False)
        self.frameReady.emit(out, rid, kind == FULL)
