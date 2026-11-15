"""FernLens UI：调整面板与主窗口。

内容分区：
  1. 基础面板    2. 曲线面板   3. 颜色分级面板   4. 取色器面板
  4b. 蒙版面板   5. 右侧总面板  6. 主窗口（工具栏/状态栏/调度/取色/蒙版/保存）
"""
from __future__ import annotations

import os
import threading
import time
import traceback

import cv2
import numpy as np

import core
import auto
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QSplitter, QToolBar, QVBoxLayout,
                               QWidget)

try:                                    # RAW 支持（可选依赖）
    import rawpy
except ImportError:                     # pragma: no cover
    rawpy = None

from core import (PREVIEW_MAX_EDGE, Adjustments, PickerState, downscale_max_edge,
                  hue_degrees, linear_histograms, mask_cache_key, render_mask,
                  sample_patch)
from ui_widgets import (CHANNEL_LABELS, CollapsibleSection, ColorWheel,
                        CurveEditor, FULL, PREVIEW, PreviewWidget,
                        ProcessorThread, SliderRow, mask_to_qimage)

RANGE_LABELS = {"shadows": "阴影", "midtones": "中间调",
                "highlights": "高光", "global": "全局"}

# rawpy 可解码的常见 RAW 格式
RAW_EXTENSIONS = (".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf",
                  ".sr2", ".raw", ".rw2", ".raf", ".orf", ".pef", ".ptx", ".x3f",
                  ".r3d", ".fff", ".erf", ".kdc", ".dcr", ".mef", ".mos", ".mrw")


# ======================================================================
# 1. 基础面板
# ======================================================================

class BasicPanel(QWidget):
    changed = Signal()
    committed = Signal()

    ROWS = (
        ("基础", (
            ("曝光", "exposure", -5.0, 5.0, 2),
            ("对比度", "contrast", -100, 100, 0),
            ("高光", "highlights", -100, 100, 0),
            ("阴影", "shadows", -100, 100, 0),
            ("白色", "whites", -100, 100, 0),
            ("黑色", "blacks", -100, 100, 0),
        )),
        ("颜色", (
            ("色温", "temperature", -100, 100, 0),
            ("色调", "tint", -100, 100, 0),
            ("饱和度", "saturation", -100, 100, 0),
            ("自然饱和度", "vibrance", -100, 100, 0),
        )),
        ("效果", (
            ("清晰度", "clarity", -100, 100, 0),
            ("降噪", "denoise", 0, 100, 0),
            ("锐度", "sharpen", 0, 100, 0),
            ("去朦胧", "dehaze", -100, 100, 0),
        )),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._blocked = False
        self.rows: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(8)
        for group_title, rows in self.ROWS:
            box = QGroupBox(group_title)
            lay = QVBoxLayout(box)
            lay.setSpacing(6)
            for label, name, lo, hi, decimals in rows:
                suffix = " EV" if name == "exposure" else ""
                row = SliderRow(label, lo, hi, 0.0, decimals=decimals,
                                suffix=suffix, default_value=0.0)
                row.valueChanged.connect(lambda v, n=name: self._on_changed(n, v))
                row.sliderReleased.connect(self.committed)
                self.rows[name] = row
                lay.addWidget(row)
            outer.addWidget(box)
        outer.addStretch(1)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        self._blocked = True
        for name, row in self.rows.items():
            row.set_value(getattr(self._params, name))
        self._blocked = False

    def _on_changed(self, name: str, value: float):
        if self._blocked or self._params is None:
            return
        setattr(self._params, name, value)
        self.changed.emit()


# ======================================================================
# 2. 曲线面板
# ======================================================================

class CurvePanel(QWidget):
    changed = Signal()
    committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._blocked = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        ch_row = QHBoxLayout()
        self._channel_buttons = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for ch in ("master", "red", "green", "blue"):
            btn = QPushButton(CHANNEL_LABELS[ch])
            btn.setCheckable(True)
            btn.setChecked(ch == "master")
            btn.clicked.connect(lambda _=False, c=ch: self._select_channel(c))
            self._channel_buttons[ch] = btn
            self._group.addButton(btn)
            ch_row.addWidget(btn)
        reset_btn = QPushButton("重置")
        reset_btn.setToolTip("重置当前通道曲线")
        reset_btn.clicked.connect(self._reset_current)
        ch_row.addWidget(reset_btn)
        reset_all_btn = QPushButton("全部重置")
        reset_all_btn.setToolTip("重置全部通道曲线")
        reset_all_btn.clicked.connect(self._reset_all)
        ch_row.addWidget(reset_all_btn)
        lay.addLayout(ch_row)

        self.editor = CurveEditor()
        self.editor.pointsChanged.connect(self._on_points_changed)
        self.editor.pointsCommitted.connect(self._on_points_committed)
        lay.addWidget(self.editor, 1)

        hint = QLabel("双击曲线添加控制点，右键删除，拖动调整")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        self._blocked = True
        from core import default_curves
        for ch in ("master", "red", "green", "blue"):
            self.editor.set_points(ch, self._params.curves.get(ch)
                                   or default_curves()[ch])
        self._blocked = False

    def _select_channel(self, channel: str):
        self.editor.set_channel(channel)
        for ch, btn in self._channel_buttons.items():
            btn.setChecked(ch == channel)

    def set_histograms(self, hist):
        self.editor.set_histograms(hist)

    def _on_points_changed(self, channel: str, points):
        if self._params is None or self._blocked:
            return
        self._params.curves[channel] = [tuple(p) for p in points]
        self.changed.emit()

    def _on_points_committed(self, channel: str, points):
        if self._params is None:
            return
        self._params.curves[channel] = [tuple(p) for p in points]
        self.committed.emit()

    def _reset_current(self):
        self._apply_reset(self.editor.current_channel())

    def _reset_all(self):
        self._apply_reset("all")

    def _apply_reset(self, channel: str):
        if self._params is None:
            return
        from core import default_curves
        channels = ("master", "red", "green", "blue") if channel == "all" else (channel,)
        for ch in channels:
            self._params.curves[ch] = [(0.0, 0.0), (1.0, 1.0)]
            self.editor.set_points(ch, self._params.curves[ch])
        self.changed.emit()
        self.committed.emit()


# ======================================================================
# 3. 颜色分级面板
# ======================================================================

class GradingPanel(QWidget):
    changed = Signal()
    committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._blocked = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.wheels: dict = {}
        self.sat_rows: dict = {}
        self.lum_rows: dict = {}
        positions = {"shadows": (0, 0), "midtones": (0, 1),
                     "highlights": (1, 0), "global": (1, 1)}
        for name in ("shadows", "midtones", "highlights", "global"):
            cell = QVBoxLayout()
            cell.setSpacing(4)
            caption = QLabel(RANGE_LABELS[name])
            caption.setObjectName("sectionTitle")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.addWidget(caption)
            wheel = ColorWheel()
            wheel.hueSatChanged.connect(lambda h, s, n=name: self._on_wheel(n, h, s))
            wheel.wheelReleased.connect(self.committed)
            cell.addWidget(wheel, 0, Qt.AlignmentFlag.AlignCenter)
            sat_row = SliderRow("饱和度", -100, 100, 0)
            sat_row.valueChanged.connect(lambda v, n=name: self._on_sat(n, v))
            sat_row.sliderReleased.connect(self.committed)
            lum_row = SliderRow("亮度", -100, 100, 0)
            lum_row.valueChanged.connect(lambda v, n=name: self._on_lum(n, v))
            lum_row.sliderReleased.connect(self.committed)
            wrapper = QWidget()
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(4)
            wl.addLayout(cell)
            wl.addWidget(sat_row)
            wl.addWidget(lum_row)
            grid.addWidget(wrapper, *positions[name])
            self.wheels[name] = wheel
            self.sat_rows[name] = sat_row
            self.lum_rows[name] = lum_row
        lay.addLayout(grid)

        box = QGroupBox("混合")
        bl = QVBoxLayout(box)
        self.blend_row = SliderRow("混合", 0, 100, 50)
        self.blend_row.valueChanged.connect(self._on_blend)
        self.blend_row.sliderReleased.connect(self.committed)
        bl.addWidget(self.blend_row)
        lay.addWidget(box)
        lay.addStretch(1)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        self._blocked = True
        for name in ("shadows", "midtones", "highlights", "global"):
            w = self._params.wheels[name]
            self.sat_rows[name].set_value(w.saturation)
            self.lum_rows[name].set_value(w.luminance)
            self._refresh_wheel(name)
        self.blend_row.set_value(self._params.blend)
        self._blocked = False

    def _refresh_wheel(self, name: str):
        w = self._params.wheels[name]
        negative = w.saturation < 0
        self.wheels[name].set_hue_sat(w.hue + (180.0 if negative else 0.0),
                                      abs(w.saturation) / 100.0)

    def _on_wheel(self, name: str, hue: float, sat: float):
        if self._blocked or self._params is None:
            return
        w = self._params.wheels[name]
        w.hue = hue
        w.saturation = sat * 100.0
        self.sat_rows[name].set_value(w.saturation)
        self.changed.emit()

    def _on_sat(self, name: str, value: float):
        if self._blocked or self._params is None:
            return
        self._params.wheels[name].saturation = value
        self._refresh_wheel(name)
        self.changed.emit()

    def _on_lum(self, name: str, value: float):
        if self._blocked or self._params is None:
            return
        self._params.wheels[name].luminance = value
        self.changed.emit()

    def _on_blend(self, value: float):
        if self._blocked or self._params is None:
            return
        self._params.blend = value
        self.changed.emit()


# ======================================================================
# 4. 取色器面板
# ======================================================================

class _PickerRow(QFrame):
    def __init__(self, picker: PickerState, index: int, panel: "PickerPanel"):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.index = index
        self.panel = panel

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        head = QHBoxLayout()
        self.swatch = QPushButton()
        self.swatch.setFixedSize(26, 26)
        self.swatch.setToolTip("选中此取色点")
        self.swatch.clicked.connect(lambda: panel.select_row(index))
        self.info = QLabel()
        self.info.setObjectName("groupLabel")
        repick = QPushButton("重新取色")
        repick.setToolTip("然后在图像上单击以重新拾取颜色")
        repick.clicked.connect(lambda: panel.arm_repick(index))
        delete = QPushButton("✕")
        delete.setFixedWidth(30)
        delete.setToolTip("删除此取色点")
        delete.clicked.connect(lambda: panel.delete_picker(index))
        head.addWidget(self.swatch)
        head.addWidget(self.info, 1)
        head.addWidget(repick)
        head.addWidget(delete)
        lay.addLayout(head)

        self.span_row = SliderRow("影响范围", 5, 180, picker.span,
                                  default_value=30.0, suffix="°")
        self.span_row.valueChanged.connect(lambda v: self._set("span", v))
        self.span_row.sliderReleased.connect(panel.committed)
        # 色相偏移限制在 ±60°，即可令"绿"偏黄或偏青、"红"偏橙或偏品红……
        self.hue_row = SliderRow("色相偏移", -60, 60, picker.hue_shift,
                                 default_value=0.0, suffix="°")
        self.hue_row.valueChanged.connect(lambda v: self._set("hue_shift", v))
        self.hue_row.sliderReleased.connect(panel.committed)
        self.sat_row = SliderRow("饱和度", -100, 100, picker.saturation)
        self.sat_row.valueChanged.connect(lambda v: self._set("saturation", v))
        self.sat_row.sliderReleased.connect(panel.committed)
        self.lum_row = SliderRow("明亮度", -100, 100, picker.luminance)
        self.lum_row.valueChanged.connect(lambda v: self._set("luminance", v))
        self.lum_row.sliderReleased.connect(panel.committed)
        for r in (self.span_row, self.hue_row, self.sat_row, self.lum_row):
            lay.addWidget(r)

        self.update_from(picker)

    def _set(self, attr: str, value: float):
        picker = self.panel._params.pickers[self.index]
        setattr(picker, attr, value)
        self.panel.changed.emit()

    def update_from(self, picker: PickerState):
        r, g, b = [int(c * 255) for c in picker.color]
        self.swatch.setStyleSheet(f"background-color: rgb({r},{g},{b});")
        self.info.setText(f"色相 {picker.hue:.0f}°  范围 ±{picker.span:.0f}°")
        self.span_row.set_value(picker.span)
        self.hue_row.set_value(picker.hue_shift)
        self.sat_row.set_value(picker.saturation)
        self.lum_row.set_value(picker.luminance)
        selected = self.index == self.panel._selected
        self.setStyleSheet(
            "QFrame { border: 1px solid #3a3a41; border-radius: 6px; background: #232326; }"
            + ("QFrame { border: 1px solid #4c9be8; }" if selected else ""))


class PickerPanel(QWidget):
    changed = Signal()
    committed = Signal()
    pickRequested = Signal(object)     # 'new' 或 取色点索引
    pickCancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._selected: int | None = None
        self._target = None             # None | 'new' | int
        self._rows: list = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self.new_btn = QPushButton("＋ 从图像取色")
        self.new_btn.setCheckable(True)
        self.new_btn.clicked.connect(self.arm_new)
        top.addWidget(self.new_btn)
        self.hint = QLabel("在图像上单击拾取颜色")
        self.hint.setObjectName("dim")
        top.addWidget(self.hint, 1)
        lay.addLayout(top)

        # 预设颜色（每个可单独微调色相，如绿 -> 黄/青）
        preset_box = QGroupBox("预设颜色")
        preset_lay = QGridLayout(preset_box)
        preset_lay.setSpacing(4)
        self.preset_buttons = []
        for i, (name, hue) in enumerate(core.HS_PRESETS):
            btn = QPushButton(name)
            r, g, b = [int(c * 255) for c in core.preset_color(hue)]
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: #101012;")
            btn.setToolTip(f"新建「{name}」取色点（色相 {hue:.0f}°，可再微调色相）")
            btn.clicked.connect(lambda _=False, h=hue: self.new_preset(h))
            preset_lay.addWidget(btn, i // 4, i % 4)
            self.preset_buttons.append(btn)
        lay.addWidget(preset_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(6)
        self.list_layout.addStretch(1)
        scroll.setWidget(self.list_widget)
        lay.addWidget(scroll, 1)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        if len(self._params.pickers) != len(self._rows):
            self._rebuild()
        else:
            for row in self._rows:
                row.update_from(self._params.pickers[row.index])

    def _rebuild(self):
        while self.list_layout.count() > 1:      # 保留末尾 stretch
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rows = []
        for i, p in enumerate(self._params.pickers):
            row = _PickerRow(p, i, self)
            self.list_layout.insertWidget(self.list_layout.count() - 1, row)
            self._rows.append(row)

    # ------------------------------------------------------------ 交互
    def arm_new(self):
        self._target = "new"
        self.new_btn.setChecked(True)
        self.hint.setText("在图像上单击拾取颜色…")
        self.pickRequested.emit("new")

    def new_preset(self, hue_deg: float):
        """由预设色相直接创建取色点（无需在图像上取色）。"""
        if self._params is None:
            return
        state = PickerState(hue=hue_deg, color=core.preset_color(hue_deg),
                            span=30.0, saturation=0.0)
        self._params.pickers.append(state)
        self._selected = len(self._params.pickers) - 1
        self.sync()
        self.committed.emit()

    def arm_repick(self, index: int):
        self._target = index
        self._selected = index
        self.new_btn.setChecked(False)
        self.hint.setText(f"在图像上单击以更新取色点 {index + 1}…")
        self.pickRequested.emit(index)

    def cancel_pick(self):
        self._target = None
        self.new_btn.setChecked(False)
        self.hint.setText("在图像上单击拾取颜色")
        self.pickCancelled.emit()

    def select_row(self, index: int):
        self._selected = index
        self.sync()

    def delete_picker(self, index: int):
        if self._params is None or not (0 <= index < len(self._params.pickers)):
            return
        del self._params.pickers[index]
        if self._selected is not None and self._selected >= len(self._params.pickers):
            self._selected = None
        self.sync()
        self.committed.emit()

    def on_picked(self, color, hue_deg: float):
        """图像上完成取色后由主窗口调用，color 为 sRGB 三元组。"""
        if self._params is None:
            return
        if self._target == "new" or self._target is None:
            state = PickerState(hue=hue_deg, color=tuple(color), span=30.0,
                                saturation=25.0)
            self._params.pickers.append(state)
            self._selected = len(self._params.pickers) - 1
        else:
            p = self._params.pickers[int(self._target)]
            p.hue = hue_deg
            p.color = tuple(color)
        self.cancel_pick()
        self.sync()
        self.committed.emit()


# ======================================================================
# 4b. 蒙版面板（线性渐变 / 径向渐变 / 画笔）
# ======================================================================

MASK_LOCAL_ROWS = (
    ("曝光", "exposure", -5.0, 5.0, 2, " EV"),
    ("对比度", "contrast", -100, 100, 0, ""),
    ("高光", "highlights", -100, 100, 0, ""),
    ("阴影", "shadows", -100, 100, 0, ""),
    ("白色", "whites", -100, 100, 0, ""),
    ("黑色", "blacks", -100, 100, 0, ""),
    ("色温", "temperature", -100, 100, 0, ""),
    ("色调", "tint", -100, 100, 0, ""),
    ("饱和度", "saturation", -100, 100, 0, ""),
    ("清晰度", "clarity", -100, 100, 0, ""),
    ("降噪", "denoise", 0, 100, 0, ""),
    ("锐度", "sharpen", 0, 100, 0, ""),
    ("去朦胧", "dehaze", -100, 100, 0, ""),
)
MASK_KIND_LABELS = {"linear": "线性渐变", "radial": "径向渐变", "brush": "画笔"}


class _MaskRow(QFrame):
    def __init__(self, mask: "core.MaskState", index: int, panel: "MaskPanel"):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.index = index
        self.panel = panel

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 5, 6, 5)
        lay.setSpacing(6)

        self.pick_btn = QPushButton()
        self.pick_btn.setFixedWidth(92)
        self.pick_btn.setCheckable(True)
        self.pick_btn.clicked.connect(lambda: panel.select_mask(index))
        lay.addWidget(self.pick_btn)

        self.edit_btn = QPushButton("编辑")
        self.edit_btn.setFixedWidth(48)
        self.edit_btn.setToolTip("在图像上拖动编辑该蒙版（Esc 退出）")
        self.edit_btn.clicked.connect(lambda: panel.request_edit(index))
        lay.addWidget(self.edit_btn)

        self.update_from(mask, False, False)

    def update_from(self, mask: "core.MaskState", selected: bool, editing: bool):
        self.pick_btn.setText(MASK_KIND_LABELS.get(mask.kind, mask.kind) +
                              ("（反）" if mask.invert else ""))
        self.pick_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 6px; }"
            + ("QPushButton:checked { background-color: #2f6db3; border-color: #4c9be8; }"
               if selected else ""))
        self.pick_btn.setChecked(selected)
        self.edit_btn.setText("完成" if editing else "编辑")
        self.edit_btn.setChecked(editing)
        self.setStyleSheet(
            "QFrame { border: 1px solid #3a3a41; border-radius: 6px; background: #232326; }"
            + ("QFrame { border-color: #4c9be8; }" if selected else ""))


class MaskPanel(QWidget):
    changed = Signal()
    committed = Signal()
    overlayRefresh = Signal()             # 几何/羽化/反向变化 -> 重绘叠加层
    selectionChanged = Signal(int)
    editModeRequested = Signal(object)    # 蒙版索引 或 None（退出）
    brushCleared = Signal()
    brushUndone = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params = None
        self._selected: int | None = None
        self._editing: int | None = None
        self._blocked = False
        self._rows: list = []
        self._param_rows: dict = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        new_row = QHBoxLayout()
        for kind, label in (("linear", "线性渐变"), ("radial", "径向渐变"), ("brush", "画笔")):
            btn = QPushButton("＋ " + label)
            btn.setToolTip({"linear": "从一端到另一端的渐变蒙版",
                            "radial": "椭圆形渐变蒙版",
                            "brush": "用画笔涂抹出蒙版"}[kind])
            btn.clicked.connect(lambda _=False, k=kind: self.create_mask(k))
            new_row.addWidget(btn)
        lay.addLayout(new_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(4)
        self.list_layout.addStretch(1)
        scroll.setWidget(self.list_widget)
        lay.addWidget(scroll)

        # ---- 参数区（随选中蒙版动态构建）----
        self.params_box = QWidget()
        self.params_layout = QVBoxLayout(self.params_box)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(6)
        lay.addWidget(self.params_box)

        self.hint = QLabel("创建蒙版后可在此调整参数与局部调整")
        self.hint.setObjectName("dim")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        lay.addStretch(1)

    # ------------------------------------------------------------ 数据
    def set_params(self, params):
        self._params = params
        self.sync()

    def selected_index(self):
        return self._selected

    def sync(self):
        if self._params is None:
            return
        if self._selected is not None and self._selected >= len(self._params.masks):
            self._selected = len(self._params.masks) - 1 if self._params.masks else None
            self._editing = None
        if len(self._params.masks) != len(self._rows):
            self._rebuild_rows()
        else:
            for i, row in enumerate(self._rows):
                row.update_from(self._params.masks[i], i == self._selected,
                                i == self._editing)
        self._rebuild_params(want_focus=False)

    def _rebuild_rows(self):
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rows = []
        for i, m in enumerate(self._params.masks):
            row = _MaskRow(m, i, self)
            self.list_layout.insertWidget(self.list_layout.count() - 1, row)
            self._rows.append(row)

    # ------------------------------------------------------------ 面板操作
    def create_mask(self, kind: str):
        if self._params is None:
            return
        m = core.MaskState(kind=kind, name=MASK_KIND_LABELS[kind])
        if kind == "linear":
            m.x1, m.y1, m.x2, m.y2 = 0.3, 0.7, 0.7, 0.3
        elif kind == "radial":
            m.cx, m.cy, m.radius, m.aspect = 0.5, 0.5, 0.30, 1.0
        else:
            m.size, m.feather = 12.0, 25.0
        self._params.masks.append(m)
        self._selected = len(self._params.masks) - 1
        self.sync()
        self.selectionChanged.emit(self._selected)
        self.request_edit(self._selected)
        self.committed.emit()

    def select_mask(self, index: int):
        self._selected = index
        self.sync()
        self.selectionChanged.emit(index)

    def request_edit(self, index: int):
        self._editing = index
        self._selected = index
        self.sync()
        self.selectionChanged.emit(index)
        self.editModeRequested.emit(index)

    def exit_edit(self):
        if self._editing is None:
            return
        self._editing = None
        self.sync()
        self.editModeRequested.emit(None)

    def delete_mask(self, index: int):
        if self._params is None or not (0 <= index < len(self._params.masks)):
            return
        if self._editing == index:
            self._editing = None
            self.editModeRequested.emit(None)
        elif self._editing is not None and self._editing > index:
            self._editing -= 1
        del self._params.masks[index]
        if self._selected is not None and self._selected >= len(self._params.masks):
            self._selected = None
        self.sync()
        self.selectionChanged.emit(self._selected if self._selected is not None else -1)
        self.committed.emit()

    # ------------------------------------------------------------ 参数区
    def _current_mask(self):
        if self._params is None or self._selected is None:
            return None
        if self._selected >= len(self._params.masks):
            return None
        return self._params.masks[self._selected]

    def _clear_params(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._param_rows = {}
        self._geo_rows = []

    def _add_row(self, label, lo, hi, value, decimals, suffix, default,
                 on_change, on_release=None, geo=None):
        row = SliderRow(label, lo, hi, value, decimals=decimals, suffix=suffix,
                        default_value=default)
        row.valueChanged.connect(on_change)
        row.sliderReleased.connect(on_release or self.committed)
        self.params_layout.addWidget(row)
        if geo is not None:
            self._geo_rows.append((geo[0], geo[1], row))
        return row

    def _rebuild_params(self, want_focus=False):
        self._clear_params()
        m = self._current_mask()
        if m is None:
            self.hint.setText("创建蒙版后可在此调整参数与局部调整")
            return
        self.hint.setText(f"{MASK_KIND_LABELS.get(m.kind, m.kind)}蒙版："
                          "在图像上拖动编辑；Esc 退出编辑")

        head = QHBoxLayout()
        invert = QPushButton("反向")
        invert.setCheckable(True)
        invert.setChecked(m.invert)
        invert.setToolTip("反向：作用于蒙版之外的区域")
        invert.clicked.connect(lambda checked: self._set_attr("invert", checked))
        head.addWidget(invert)
        delete = QPushButton("删除")
        delete.setToolTip("删除该蒙版")
        delete.clicked.connect(lambda: self.delete_mask(self._selected))
        head.addWidget(delete)
        head.addStretch(1)
        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addLayout(head)
        self.params_layout.addWidget(holder)

        self._add_row("羽化", 0, 100, m.feather, 0, "", 30.0,
                      self._on_feather)

        if m.kind == "brush":
            self._add_row("笔刷大小", 1, 100, m.size, 0, "", 12.0, self._on_size)
            btns = QHBoxLayout()
            undo = QPushButton("撤销一笔")
            undo.clicked.connect(lambda: self.brushUndone.emit())
            clear = QPushButton("清除笔迹")
            clear.clicked.connect(lambda: self.brushCleared.emit())
            btns.addWidget(undo)
            btns.addWidget(clear)
            btns.addStretch(1)
            hb = QWidget()
            hbl = QVBoxLayout(hb)
            hbl.setContentsMargins(0, 0, 0, 0)
            hbl.addLayout(btns)
            self.params_layout.addWidget(hb)
        elif m.kind == "radial":
            self._add_row("半径", 5, 100, m.radius * 100, 0, "%", 30.0, self._on_radius,
                          geo=("radius", 100.0))
            self._add_row("长宽比", 20, 300, m.aspect * 100, 0, "%", 100.0, self._on_aspect,
                          geo=("aspect", 100.0))
            self._add_row("旋转", 0, 360, m.angle, 0, "°", 0.0, self._on_angle,
                          geo=("angle", 1.0))
        else:
            self._add_row("起点 X", 0, 100, m.x1 * 100, 0, "%", 30.0,
                          lambda v: self._on_linear(0, v), geo=("x1", 100.0))
            self._add_row("起点 Y", 0, 100, m.y1 * 100, 0, "%", 70.0,
                          lambda v: self._on_linear(1, v), geo=("y1", 100.0))
            self._add_row("终点 X", 0, 100, m.x2 * 100, 0, "%", 70.0,
                          lambda v: self._on_linear(2, v), geo=("x2", 100.0))
            self._add_row("终点 Y", 0, 100, m.y2 * 100, 0, "%", 30.0,
                          lambda v: self._on_linear(3, v), geo=("y2", 100.0))

        box = QGroupBox("局部调整")
        gl = QVBoxLayout(box)
        gl.setSpacing(6)
        for label, name, lo, hi, decimals, suffix in MASK_LOCAL_ROWS:
            row = SliderRow(label, lo, hi, getattr(m.adj, name), decimals=decimals,
                            suffix=suffix, default_value=0.0)
            row.valueChanged.connect(lambda v, n=name: self._on_adj(n, v))
            row.sliderReleased.connect(self.committed)
            gl.addWidget(row)
            self._param_rows[name] = row
        self.params_layout.addWidget(box)

    # ------------------------------------------------------------ 事件
    def _set_attr(self, attr, value):
        m = self._current_mask()
        if m is None:
            return
        setattr(m, attr, value)
        self.overlayRefresh.emit()
        self.sync()
        self.committed.emit()

    def _on_feather(self, v: float):
        m = self._current_mask()
        if m is None:
            return
        m.feather = v
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_size(self, v: float):
        m = self._current_mask()
        if m is None or m.kind != "brush":
            return
        m.size = v
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_radius(self, v: float):
        m = self._current_mask()
        if m is None:
            return
        m.radius = v / 100.0
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_aspect(self, v: float):
        m = self._current_mask()
        if m is None:
            return
        m.aspect = v / 100.0
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_angle(self, v: float):
        m = self._current_mask()
        if m is None or m.kind != "radial":
            return
        m.angle = float(v)
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_linear(self, index: int, v: float):
        m = self._current_mask()
        if m is None or m.kind != "linear":
            return
        attr = ("x1", "y1", "x2", "y2")[index]
        setattr(m, attr, v / 100.0)
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_adj(self, name: str, v: float):
        m = self._current_mask()
        if m is None:
            return
        setattr(m.adj, name, v)
        self.changed.emit()

    def sync_values(self):
        """外部（画布拖拽）修改了几何后，只刷新数值不回填控件。"""
        self._rebuild_params()

    def update_geometry_sliders(self):
        """画布拖拽蒙版时，只更新几何滑块（避免重建导致拖动中断）。"""
        m = self._current_mask()
        if m is None or not self._geo_rows:
            return
        self._blocked = True
        for attr, mult, row in self._geo_rows:
            row.set_value(getattr(m, attr) * mult)
        self._blocked = False


# ======================================================================
# 4c. 仿制图章面板
# ======================================================================

class StampPanel(QWidget):
    featherChanged = Signal(float)       # 边缘羽化 0..100
    newSourceRequested = Signal()
    undoRequested = Signal()
    clearRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        hint = QLabel("第一步：在图像上拖出矩形，选取源区域（青色框）。\n"
                      "第二步：在目标处拖出矩形，源区域内容即缩放覆盖过去。\n"
                      "Alt+拖动或点「重选源区」重新取样；Esc 退出。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.feather_row = SliderRow("边缘羽化", 0, 100, 20, default_value=20.0,
                                     suffix="%")
        self.feather_row.valueChanged.connect(self.featherChanged)
        lay.addWidget(self.feather_row)

        btns = QHBoxLayout()
        resrc = QPushButton("重选源区")
        resrc.setToolTip("清除当前源矩形，下一次拖动重新选取")
        resrc.clicked.connect(self.newSourceRequested)
        undo = QPushButton("撤销一次")
        undo.clicked.connect(self.undoRequested)
        clear = QPushButton("清除全部")
        clear.clicked.connect(self.clearRequested)
        btns.addWidget(resrc)
        btns.addWidget(undo)
        btns.addWidget(clear)
        btns.addStretch(1)
        lay.addLayout(btns)
        lay.addStretch(1)


# ======================================================================
# 4d. 裁剪工具条（比例 / 拉直 / 应用）
# ======================================================================

class CropBar(QWidget):
    """裁剪模式下显示在预览下方：比例预设、拉直角度、输出尺寸、应用/取消。"""

    applyRequested = Signal()
    cancelRequested = Signal()
    aspectChanged = Signal(object)      # None（自由）或 float 宽/高
    angleChanged = Signal(float)

    ASPECTS = (("自由", None), ("原始", "orig"), ("1:1", 1.0), ("3:2", 1.5),
               ("2:3", 2.0 / 3.0), ("4:3", 4.0 / 3.0), ("3:4", 0.75),
               ("16:9", 16.0 / 9.0), ("9:16", 9.0 / 16.0))

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)
        self._orig_aspect: float | None = None

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for label, val in self.ASPECTS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(val is None)
            btn.setToolTip("裁剪比例" + ("" if val is None else f" {label}"))
            btn.clicked.connect(lambda _=False, v=val: self._emit_aspect(v))
            self._group.addButton(btn)
            lay.addWidget(btn)

        self.angle_row = SliderRow("拉直", -45, 45, 0.0, decimals=1,
                                    suffix="°", default_value=0.0)
        self.angle_row.valueChanged.connect(self.angleChanged.emit)
        lay.addWidget(self.angle_row, 1)

        self.size_label = QLabel("")
        self.size_label.setObjectName("dim")
        self.size_label.setMinimumWidth(96)
        lay.addWidget(self.size_label)

        cancel = QPushButton("取消")
        cancel.clicked.connect(self.cancelRequested)
        lay.addWidget(cancel)
        apply_btn = QPushButton("应用")
        apply_btn.setToolTip("应用裁剪与拉直（Enter）")
        apply_btn.clicked.connect(self.applyRequested)
        lay.addWidget(apply_btn)

    def _emit_aspect(self, val):
        if val == "orig":
            val = self._orig_aspect
        self.aspectChanged.emit(val)

    def set_original_aspect(self, aspect):
        self._orig_aspect = float(aspect) if aspect else None

    def reset(self):
        """进入裁剪模式时复位（自由比例、角度 0）。"""
        self._group.buttons()[0].setChecked(True)
        self.angle_row.set_value(0.0)
        self.size_label.setText("")

    def set_output_size(self, w: int, h: int):
        self.size_label.setText(f"{w} × {h}")


# ======================================================================
# 5. 右侧总面板（可折叠分区）
# ======================================================================

class AdjustmentPanel(QWidget):
    changed = Signal()
    committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.basic = BasicPanel()
        self.curve = CurvePanel()
        self.grading = GradingPanel()
        self.picker = PickerPanel()
        self.mask = MaskPanel()
        self.stamp = StampPanel()
        self._global = None
        self._mask_adj = None

        for p in (self.basic, self.curve, self.grading, self.picker):
            p.changed.connect(self.changed)
            p.committed.connect(self.committed)
        self.mask.changed.connect(self.changed)
        self.mask.committed.connect(self.committed)

        titles = {BasicPanel: "基础", CurvePanel: "曲线",
                  GradingPanel: "颜色分级", PickerPanel: "取色器",
                  MaskPanel: "蒙版", StampPanel: "仿制图章"}
        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)
        for p in (self.basic, self.curve, self.grading, self.picker,
                  self.mask, self.stamp):
            cl.addWidget(CollapsibleSection(titles[type(p)], p, expanded=True))
        cl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_params(self, params: Adjustments):
        self._global = params
        self._mask_adj = None
        for p in (self.basic, self.curve, self.grading, self.picker, self.mask):
            p.set_params(params)

    def set_mask_target(self, mask_adj):
        """选中蒙版时，曲线/色轮/取色器三个分区改编辑该蒙版的局部参数。"""
        self._mask_adj = mask_adj
        target = mask_adj if mask_adj is not None else self._global
        if target is None:
            return
        for p in (self.curve, self.grading, self.picker):
            p.set_params(target)

    def sync(self):
        for p in (self.basic, self.curve, self.grading, self.picker, self.mask):
            p.sync()

    def set_histograms(self, hist):
        self.curve.set_histograms(hist)


# ======================================================================
# 6. 主窗口
# ======================================================================

class MainWindow(QMainWindow):
    aiDone = Signal(dict, str, str, int)    # (参数字典, 说明, 模型描述, 发起时的图像序号)
    aiError = Signal(str)

    # 蒙版叠加层缓存上限：拖动几何时每个鼠标事件都会产生新缓存项，无上限会吃掉数百 MB
    _OVERLAY_CACHE_MAX = 16

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FernLens — 照片调色")
        self.resize(1460, 920)
        self.setAcceptDrops(True)

        # ---------------- 状态 ----------------
        self.params = Adjustments()
        self.source: np.ndarray | None = None        # 全图 uint8 RGB
        self.preview_src: np.ndarray | None = None   # 降采样图 uint8 RGB
        self._newest_rid = 0
        self._comparing = False
        self._last_frame: np.ndarray | None = None
        self._save_path: str | None = None
        self._save_rid = -1
        self._last_hist_time = 0.0
        self._last_overlay_time = 0.0
        self._source_serial = 0        # 换图自增，用于丢弃迟到的 AI 结果

        # ---------------- 后台处理线程 ----------------
        self.processor = ProcessorThread()
        self.processor.frameReady.connect(self._on_frame_ready)
        self.processor.busyChanged.connect(self._on_busy_changed)
        self.processor.start()

        # ---------------- 界面 ----------------
        self.preview = PreviewWidget()
        self.preview.colorPicked.connect(self._on_image_picked)
        self.preview.zoomChanged.connect(self._update_status)

        self.panel = AdjustmentPanel()
        self.panel.set_params(self.params)
        self.panel.changed.connect(self._on_params_changed)
        self.panel.committed.connect(self._on_params_committed)
        self.panel.picker.pickRequested.connect(self._on_pick_requested)
        self.panel.picker.pickCancelled.connect(self._on_pick_cancelled)
        self.aiDone.connect(self._on_ai_done)
        self.aiError.connect(self._on_ai_error)

        # ---- 蒙版 ----
        self._mask_sel: int | None = None
        self._mask_edit: int | None = None
        self._overlay_cache: dict = {}
        self._stroke_closed = False     # 画笔：上一笔是否已结束
        self.panel.mask.selectionChanged.connect(self._on_mask_selected)
        self.panel.mask.editModeRequested.connect(self._on_mask_edit_requested)
        self.panel.mask.overlayRefresh.connect(self._refresh_mask_overlays)
        self.panel.mask.brushCleared.connect(self._on_brush_cleared)
        self.panel.mask.brushUndone.connect(self._on_brush_undo)
        self.preview.linearHandleMoved.connect(self._on_linear_handle)
        self.preview.radialCenterMoved.connect(self._on_radial_center)
        self.preview.radialSizeMoved.connect(self._on_radial_size)
        self.preview.radialAngleMoved.connect(self._on_radial_angle)
        self.preview.brushStrokePoint.connect(self._on_brush_point)
        self.preview.brushStrokeFinished.connect(self._on_brush_finished)

        # ---- 仿制图章（矩形取样覆盖）----
        self._stamp_src_rect = None    # 已选源矩形（归一化 x0,y0,x1,y1）
        self._stamp_feather = 20.0     # 边缘羽化 0..100
        self._clone_ops: list = []     # 已提交覆盖 (源矩形, 目标矩形, 羽化%)
        self._retouch_full: np.ndarray | None = None
        self._retouch_preview: np.ndarray | None = None
        self.preview.cloneSourcePicked.connect(self._on_clone_source)
        self.preview.cloneApplyRequested.connect(self._on_clone_apply)
        self.panel.stamp.featherChanged.connect(self._on_stamp_feather)
        self.panel.stamp.newSourceRequested.connect(self._on_stamp_new_source)
        self.panel.stamp.undoRequested.connect(self._on_stamp_undo)
        self.panel.stamp.clearRequested.connect(self._on_stamp_clear)

        # ---- 裁剪 / 旋转 ----
        self._crop_angle = 0.0
        self.crop_bar = CropBar()
        self.crop_bar.setVisible(False)
        self.crop_bar.applyRequested.connect(self._apply_crop)
        self.crop_bar.cancelRequested.connect(self._cancel_crop)
        self.crop_bar.aspectChanged.connect(self.preview.set_crop_aspect)
        self.crop_bar.angleChanged.connect(self._on_crop_angle)
        self.preview.cropRectChanged.connect(self._on_crop_rect_changed)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(self.preview, 1)
        lv.addWidget(self.crop_bar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1040, 380])
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self._build_statusbar()
        self._build_shortcuts()

        # 空闲后自动处理全图
        self._full_timer = QTimer(self)
        self._full_timer.setSingleShot(True)
        self._full_timer.setInterval(400)
        self._full_timer.timeout.connect(lambda: self._submit(FULL))

        self._update_status()

    # ------------------------------------------------------------ 工具栏
    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(self.iconSize())
        self.addToolBar(tb)

        act_open = QAction("打开", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_image)
        tb.addAction(act_open)

        act_save = QAction("保存", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_image)
        tb.addAction(act_save)

        act_reset = QAction("重置", self)
        act_reset.setToolTip("重置所有调整 (Ctrl+R)")
        act_reset.triggered.connect(self.reset_adjustments)
        tb.addAction(act_reset)

        tb.addSeparator()

        self.act_compare = QAction("对比原图", self)
        self.act_compare.setCheckable(True)
        self.act_compare.setShortcut("\\")
        self.act_compare.triggered.connect(self._toggle_compare)
        tb.addAction(self.act_compare)

        tb.addSeparator()

        act_fit = QAction("适合窗口", self)
        act_fit.triggered.connect(self.preview.zoom_fit)
        tb.addAction(act_fit)

        act_1to1 = QAction("1:1", self)
        act_1to1.triggered.connect(self.preview.zoom_one_to_one)
        tb.addAction(act_1to1)

        act_in = QAction("放大", self)
        act_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        act_in.triggered.connect(lambda: self.preview.zoom_at(1.25))
        tb.addAction(act_in)

        act_out = QAction("缩小", self)
        act_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        act_out.triggered.connect(lambda: self.preview.zoom_at(0.8))
        tb.addAction(act_out)

        tb.addSeparator()

        self.act_pick = QAction("取色器", self)
        self.act_pick.setCheckable(True)
        self.act_pick.setToolTip("在图像上单击拾取目标颜色")
        self.act_pick.triggered.connect(self._toggle_pick_mode)
        tb.addAction(self.act_pick)

        tb.addSeparator()

        self.act_bw = QAction("黑白", self)
        self.act_bw.setCheckable(True)
        self.act_bw.setToolTip("黑白模式：按亮度转为灰度（再点一次恢复彩色）")
        self.act_bw.triggered.connect(self._toggle_blackwhite)
        tb.addAction(self.act_bw)

        tb.addSeparator()

        self.act_stamp = QAction("仿制图章", self)
        self.act_stamp.setCheckable(True)
        self.act_stamp.setToolTip("仿制图章：Alt+点击取样源点，左键涂抹复制（Esc 退出）")
        self.act_stamp.triggered.connect(self._toggle_stamp)
        tb.addAction(self.act_stamp)

        tb.addSeparator()

        self.act_crop = QAction("裁剪", self)
        self.act_crop.setCheckable(True)
        self.act_crop.setToolTip("裁剪/拉直：拖动框与手柄调整，选比例、拉直角度，Enter 应用，Esc 取消")
        self.act_crop.triggered.connect(self._toggle_crop)
        tb.addAction(self.act_crop)

        act_rotl = QAction("左转90°", self)
        act_rotl.setToolTip("逆时针旋转 90°（立即应用）")
        act_rotl.triggered.connect(lambda: self._rotate90(False))
        tb.addAction(act_rotl)

        act_rotr = QAction("右转90°", self)
        act_rotr.setToolTip("顺时针旋转 90°（立即应用）")
        act_rotr.triggered.connect(lambda: self._rotate90(True))
        tb.addAction(act_rotr)

        tb.addSeparator()

        act_auto = QAction("自动", self)
        act_auto.setToolTip("规则版自动调色：灰世界白平衡 + 直方图自动色阶（本地瞬时）")
        act_auto.triggered.connect(self._auto_rule)
        tb.addAction(act_auto)

        self.act_ai = QAction("AI 调色", self)
        self.act_ai.setToolTip("由视觉大模型分析图像并生成调色参数（需联网）")
        self.act_ai.triggered.connect(self._auto_ai)
        tb.addAction(self.act_ai)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.status_image = QLabel("未打开图像")
        self.status_zoom = QLabel("")
        self.status_proc = QLabel("")
        sb.addWidget(self.status_image, 1)
        sb.addPermanentWidget(self.status_zoom)
        sb.addPermanentWidget(self.status_proc)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+R"), self, self.reset_adjustments)
        QShortcut(QKeySequence("0"), self, self.preview.zoom_fit)
        QShortcut(QKeySequence("1"), self, self.preview.zoom_one_to_one)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected_picker)
        QShortcut(QKeySequence("Escape"), self, self._on_escape)
        QShortcut(QKeySequence("Return"), self, self._crop_enter)
        QShortcut(QKeySequence("Enter"), self, self._crop_enter)

    def _crop_enter(self):
        if self.preview.crop_mode():
            self._apply_crop()

    def _delete_selected_picker(self):
        idx = self.panel.picker._selected
        if idx is not None:
            self.panel.picker.delete_picker(idx)

    # ------------------------------------------------------------ 图像
    def _image_filter(self) -> str:
        parts = ["常见图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"]
        if rawpy is not None:
            parts.append("RAW (" + " ".join("*" + e for e in RAW_EXTENSIONS) + ")")
        parts.append("所有文件 (*.*)")
        return ";;".join(parts)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", self._image_filter())
        if path:
            self.load_image(path)

    def _load_raw(self, path: str):
        """用 rawpy(LibRaw) 解码 RAW，返回 uint8 RGB（失败返回 None）。"""
        if rawpy is None:
            QMessageBox.information(self, "无法打开 RAW",
                                    "未安装 rawpy，请先执行：pip install rawpy")
            return None
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
                meta = f"{getattr(raw.metadata, 'make', '') or ''} " \
                       f"{getattr(raw.metadata, 'model', '') or ''}".strip()
        except Exception as exc:
            QMessageBox.warning(self, "RAW 解码失败", f"{path}\n\n{exc}")
            return None
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if meta:
            self.statusBar().showMessage(f"RAW：{meta}  {arr.shape[1]}×{arr.shape[0]}", 6000)
        return arr

    def load_image(self, path: str):
        if self.preview.crop_mode() or self.act_crop.isChecked():
            self.act_crop.setChecked(False)
            self._close_crop_ui()
        ext = os.path.splitext(path)[1].lower()
        if ext in RAW_EXTENSIONS:
            rgb = self._load_raw(path)
            if rgb is None:
                return
        else:
            try:
                data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)   # 兼容中文路径
            except Exception:
                img = None
            if img is None:
                QMessageBox.warning(self, "打开失败", f"无法解码图像：\n{path}")
                return
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.source = np.ascontiguousarray(rgb)
        self.preview_src = downscale_max_edge(self.source, PREVIEW_MAX_EDGE)
        self._source_serial += 1

        # 新图像：丢弃图章修改
        self._stamp_src_rect = None
        self._clone_ops.clear()
        self._retouch_full = None
        self._retouch_preview = None
        self.preview.set_stamp_source(None)

        self.processor.set_source(PREVIEW, self.preview_src)
        self.processor.set_source(FULL, self.source)
        self.preview.set_full_size(self.source.shape[1], self.source.shape[0])
        self.preview.zoom_fit()

        self._overlay_cache.clear()
        self._newest_rid = self.processor.pending_request()
        self._on_params_changed(force_full=True)
        self._refresh_mask_overlays()
        self._update_status()

    def _toggle_blackwhite(self, checked: bool):
        self.params.blackwhite = 100.0 if checked else 0.0
        self._on_params_changed(force_full=True)

    # ------------------------------------------------------------ 自动调色
    _TONE_FIELDS = ("temperature", "tint", "exposure", "contrast", "highlights",
                    "shadows", "whites", "blacks", "saturation", "vibrance",
                    "blackwhite")

    def _apply_adjustment_dict(self, d: dict):
        """把参数字典（auto.suggest_* 的输出）套到当前参数上并重渲染。"""
        for name in self._TONE_FIELDS:
            if name in d:
                setattr(self.params, name, float(d[name]))
        for ch, pts in (d.get("curves") or {}).items():
            self.params.curves[ch] = [tuple(p) for p in pts]
        for name, w in (d.get("wheels") or {}).items():
            self.params.wheels[name] = core.WheelState(**w)
        if "pickers" in d:
            self.params.pickers = [
                core.PickerState(hue=p["hue"], color=core.preset_color(p["hue"]),
                                 span=p["span"], hue_shift=p["hue_shift"],
                                 saturation=p["saturation"], luminance=p["luminance"])
                for p in d["pickers"]]
        self.panel.set_params(self.params)
        self._sync_toolbar_state()
        self._on_params_committed()

    def _auto_rule(self):
        if self.source is None:
            QMessageBox.information(self, "自动调色", "请先打开一幅图像。")
            return
        d = auto.suggest_rule(self.source)
        self._apply_adjustment_dict(d)
        self.statusBar().showMessage(
            "已应用规则自动调色" if d else "已分析：图像无需调整", 5000)

    def _auto_ai(self):
        if self.source is None:
            QMessageBox.information(self, "AI 调色", "请先打开一幅图像。")
            return
        self.act_ai.setEnabled(False)
        self.status_proc.setText("AI 分析中…")
        src = self.source
        serial = self._source_serial

        def work():
            try:
                if not auto.resolve_candidates():
                    self.aiError.emit(
                        "未配置 AI 模型。\n\n"
                        "请设置环境变量 FERNLENS_AI_BASE_URL / FERNLENS_AI_KEY / "
                        "FERNLENS_AI_MODEL\n（OpenAI 兼容接口 + 视觉模型）。")
                    return
                params, note, desc = auto.suggest_ai(src)
            except Exception as exc:
                self.aiError.emit(str(exc))
                return
            self.aiDone.emit(params, note, desc, serial)

        self._ai_thread = threading.Thread(target=work, daemon=True)
        self._ai_thread.start()

    def _on_ai_done(self, params: dict, note: str, desc: str, serial: int):
        try:
            self.act_ai.setEnabled(True)
            self.status_proc.setText("")
            if serial != self._source_serial:
                # 分析期间已换图：旧图的结果不能套到新图上
                self.statusBar().showMessage("图像已更换，AI 调色结果已忽略", 4000)
                return
            if self.source is None:
                return
            self._apply_adjustment_dict(params)
            msg = f"AI 调色完成（{desc.split('@', 1)[0]}）"
            if note:
                msg += f"：{note}"
            self.statusBar().showMessage(msg, 8000)
        except RuntimeError:               # 窗口已销毁（AI 返回太晚）
            pass

    def _on_ai_error(self, message: str):
        try:
            self.act_ai.setEnabled(True)
            self.status_proc.setText("")
            QMessageBox.warning(self, "AI 调色失败", message[:800])
        except RuntimeError:
            pass

    def reset_adjustments(self):
        self.params = Adjustments()
        self._mask_sel = None
        self._mask_edit = None
        self._overlay_cache.clear()
        self.preview.set_edit_mode("none")
        self.panel.set_params(self.params)
        self._sync_toolbar_state()
        self._on_params_changed(force_full=True)
        self._refresh_mask_overlays()
        self.statusBar().showMessage("已重置全部调整", 1500)

    def _sync_toolbar_state(self):
        """让工具栏开关与当前参数保持一致。"""
        self.act_bw.setChecked(self.params.blackwhite > 0)
        self.act_compare.setChecked(self._comparing)
        self.act_pick.setChecked(self.panel.picker._target is not None)

    def save_image(self):
        if self.source is None:
            QMessageBox.information(self, "保存", "请先打开一幅图像。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "output.jpg",
            "JPEG (*.jpg);;PNG (*.png);;WebP (*.webp);;TIFF (*.tif)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".jpg"
        self._save_path = path
        # 保存走 urgent 独立槽位（不会被随后的参数更新覆盖），并停掉空闲
        # 定时器，防止它在保存任务之后又提交一个普通全图任务造成状态混乱
        self._full_timer.stop()
        self._save_rid = self.processor.submit(self.params, FULL, urgent=True)
        self.status_proc.setText("正在渲染全图…")

    def _write_image(self, arr: np.ndarray, path: str):
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".jpg", ".jpeg"):
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == ".png":
                ok, buf = cv2.imencode(".png", bgr)
            elif ext == ".webp":
                ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, 95])
            elif ext in (".tif", ".tiff"):
                ok, buf = cv2.imencode(".tif", bgr)
            else:
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(path)          # numpy 写入，支持中文路径
                self.statusBar().showMessage(f"已保存：{path}", 4000)
            else:
                QMessageBox.warning(self, "保存失败", f"编码失败（不支持的格式？）：\n{path}")
        except Exception:
            traceback.print_exc()
            QMessageBox.warning(self, "保存失败", f"写入失败：\n{path}")

    # ------------------------------------------------------------ 调度
    def _submit(self, kind: str) -> int:
        if self.source is None:
            return -1
        return self.processor.submit(self.params, kind)

    def _on_params_changed(self, _value=None, force_full: bool = False):
        """参数变化：先刷新降采样预览，短暂空闲后处理全图。"""
        if self.source is None:
            return
        self._submit(PREVIEW)
        if force_full:
            self._full_timer.stop()
            self._submit(FULL)
        else:
            self._full_timer.start()

    def _on_params_committed(self):
        if self.source is None:
            return
        self._full_timer.stop()
        self._submit(FULL)

    def _on_busy_changed(self, busy: bool):
        self.status_proc.setText("⏳ 处理中…" if busy else "")

    def _on_frame_ready(self, arr: np.ndarray, rid: int, is_full: bool):
        if rid < self._newest_rid:
            return                                # 过期结果（已被更新的请求取代）
        self._newest_rid = rid
        self._last_frame = arr

        if self._comparing:
            if self.preview_src is not None:
                self.preview.set_frame(self.preview_src)
        else:
            self.preview.set_frame(arr)

        now = time.monotonic()
        if not self._comparing and now - self._last_hist_time > 0.3:
            self._last_hist_time = now
            try:
                self.panel.set_histograms(linear_histograms(arr))
            except Exception:
                traceback.print_exc()

        if self._save_path is not None and is_full and rid == self._save_rid:
            path, self._save_path = self._save_path, None
            self._write_image(arr, path)

        self._update_status()

    # ------------------------------------------------------------ 取色器
    def _toggle_pick_mode(self, checked: bool):
        if checked:
            self.panel.picker.arm_new()
        else:
            self.panel.picker.cancel_pick()
        self.act_pick.setChecked(self.panel.picker._target is not None)

    def _on_pick_requested(self, _target):
        self.preview.set_pick_mode(True)
        self.act_pick.setChecked(True)

    def _on_pick_cancelled(self):
        self.preview.set_pick_mode(False)
        self.act_pick.setChecked(False)

    def _on_image_picked(self, x: int, y: int):
        if self.source is None:
            return
        frame_w = self.preview.current_frame_size()[0]
        if frame_w <= 0:
            return
        fx, fy = self.preview.map_to_frame(x, y)
        scale = self.source.shape[1] / frame_w
        px = int(round(fx * scale))
        py = int(round(fy * scale))
        patch = max(3, int(round(self.source.shape[0] * 0.01)))
        color = sample_patch(self.source, px, py, patch=patch)
        self.panel.picker.on_picked(color, hue_degrees(color))
        self._on_params_changed()

    # ------------------------------------------------------------ 蒙版
    def _current_mask(self):
        if self._mask_sel is None:
            return None
        if not (0 <= self._mask_sel < len(self.params.masks)):
            return None
        return self.params.masks[self._mask_sel]

    def _on_mask_selected(self, index: int):
        self._mask_sel = self._clamp_mask_index(index)
        adj = self._current_mask()
        self.panel.set_mask_target(adj.adj if adj is not None else None)
        self._refresh_mask_overlays()

    @staticmethod
    def _clamp_mask_index(index):
        if index is None or index < 0:
            return None
        return index

    def _on_mask_edit_requested(self, target):
        if target is None:
            self._exit_mask_edit()
            return
        self._mask_sel = int(target)
        self._mask_edit = int(target)
        m = self._current_mask()
        if m is None:
            return
        if self.preview._pick_mode:
            self.panel.picker.cancel_pick()
        self.preview.set_edit_mode(m.kind, self._mask_geometry(m))
        self._refresh_mask_overlays()
        self._update_status()

    @staticmethod
    def _mask_geometry(m) -> dict:
        if m.kind == "linear":
            return {"x1": m.x1, "y1": m.y1, "x2": m.x2, "y2": m.y2}
        elif m.kind == "radial":
            return {"cx": m.cx, "cy": m.cy, "radius": m.radius,
                    "aspect": m.aspect, "angle": m.angle}
        return {"size": m.size}

    def _exit_mask_edit(self):
        if self._mask_edit is None:
            return
        self._mask_edit = None
        self.preview.set_edit_mode("none")
        self.panel.mask.exit_edit()

    # ------------------------------------------------------------ 仿制图章
    def _toggle_stamp(self, checked: bool):
        if checked:
            if self.preview._pick_mode:
                self.panel.picker.cancel_pick()
            self._exit_mask_edit()
            self.statusBar().showMessage(
                "仿制图章：先拖出源矩形，再在目标处拖矩形覆盖；Alt+拖动重选源，Esc 退出", 6000)
        self.preview.set_stamp_mode(checked)

    def _on_escape(self):
        if self.preview.crop_mode():
            self._cancel_crop()
        elif self.preview.stamp_mode():
            self.act_stamp.setChecked(False)
            self._toggle_stamp(False)
        else:
            self._exit_mask_edit()

    # ------------------------------------------------------------ 裁剪 / 旋转
    def _toggle_crop(self, checked: bool):
        if checked:
            if self.source is None:
                QMessageBox.information(self, "裁剪", "请先打开一幅图像。")
                self.act_crop.setChecked(False)
                return
            if self._comparing:
                self.act_compare.setChecked(False)
                self._toggle_compare(False)
            if self.preview._pick_mode:
                self.panel.picker.cancel_pick()
            if self.preview.stamp_mode():
                self.act_stamp.setChecked(False)
                self._toggle_stamp(False)
            self._exit_mask_edit()
            self._crop_angle = 0.0
            self.crop_bar.set_original_aspect(self.source.shape[1]
                                              / self.source.shape[0])
            self.crop_bar.reset()
            self.preview.set_crop_mode(True)
            self._update_crop_max()
            self.crop_bar.setVisible(True)
            self.statusBar().showMessage(
                "裁剪：拖动框/手柄调整，可锁定比例、拉直；Enter 应用，Esc 取消", 6000)
        else:
            self._close_crop_ui()

    def _close_crop_ui(self):
        self.preview.set_crop_mode(False)
        self.crop_bar.setVisible(False)

    def _cancel_crop(self):
        self.act_crop.setChecked(False)
        self._close_crop_ui()
        self.statusBar().showMessage("已取消裁剪", 2000)

    def _update_crop_max(self):
        """按当前拉直角度计算裁剪框的最大有效区（归一化），传给预览。"""
        h, w = self.source.shape[:2]
        x0, y0, x1, y1 = core.fit_rect_in_rotation((0, 0, w, h), (h, w),
                                                    self._crop_angle)
        self.preview.set_crop_max((x0 / w, y0 / h, x1 / w, y1 / h))

    def _on_crop_angle(self, angle: float):
        self._crop_angle = float(angle)
        self._update_crop_max()

    def _on_crop_rect_changed(self, rect: dict):
        if self.source is None:
            return
        h, w = self.source.shape[:2]
        self.crop_bar.set_output_size(
            round((rect["x1"] - rect["x0"]) * w),
            round((rect["y1"] - rect["y0"]) * h))

    def _clear_stamps(self) -> bool:
        """几何变换后图章坐标无法重放，把已生效的修改并入源图并清空历史。"""
        had = bool(self._clone_ops) or self._retouch_full is not None
        self._clone_ops.clear()
        self._retouch_full = None
        self._retouch_preview = None
        self._stamp_src_rect = None
        self.preview.set_stamp_source(None)
        return had

    def _commit_geometry(self, new_source: np.ndarray):
        """几何变换落地：重建各级缓存与预览（调整参数保留，蒙版已同步）。"""
        self.source = np.ascontiguousarray(new_source)
        self.preview_src = downscale_max_edge(self.source, PREVIEW_MAX_EDGE)
        self.processor.set_source(PREVIEW, self.preview_src)
        self.processor.set_source(FULL, self.source)
        self.preview.set_full_size(self.source.shape[1], self.source.shape[0])
        self.preview.zoom_fit()
        self._overlay_cache.clear()
        self._on_params_changed(force_full=True)
        self._refresh_mask_overlays()
        self._update_status()

    def _apply_crop(self):
        rect = self.preview.crop_rect()
        if rect is None or self.source is None:
            return
        angle = self._crop_angle
        rect_t = (rect["x0"], rect["y0"], rect["x1"], rect["y1"])
        base = self._retouch_full if self._retouch_full is not None else self.source
        if abs(angle) < 1e-9 and rect_t == (0.0, 0.0, 1.0, 1.0):
            new_src = np.ascontiguousarray(base)
            map_point = lambda nx, ny: (nx, ny)
            new_shape = new_src.shape[:2]
        else:
            new_src, map_point, new_shape = core.crop_rotate(base, rect_t, angle)
        core.remap_mask_geometry(self.params.masks, map_point,
                                 base.shape[:2], new_shape, angle_delta=-angle)
        baked = self._clear_stamps()
        self.act_crop.setChecked(False)
        self._close_crop_ui()
        self._commit_geometry(new_src)
        msg = "已应用裁剪" if abs(angle) < 1e-9 else f"已应用拉直（{angle:+.1f}°）与裁剪"
        if baked:
            msg += "；图章修改已一并合并"
        self.statusBar().showMessage(msg, 5000)

    def _rotate90(self, clockwise: bool):
        if self.source is None:
            QMessageBox.information(self, "旋转", "请先打开一幅图像。")
            return
        base = self._retouch_full if self._retouch_full is not None else self.source
        k = 3 if clockwise else 1
        new_src = core.rotate90(base, k)
        core.remap_masks_rot90(self.params.masks, k, base.shape[:2])
        self._clear_stamps()
        self._commit_geometry(new_src)
        self.statusBar().showMessage("已右转 90°" if clockwise else "已左转 90°", 2500)

    def _stamp_feather_frac(self) -> float:
        return float(np.clip(self._stamp_feather, 0.0, 100.0)) / 100.0

    def _ensure_retouch(self):
        """首次图章修改时创建工作副本（全图 + 预览）。"""
        if self._retouch_full is None:
            self._retouch_full = self.source.copy()
            self._retouch_preview = (self.preview_src.copy()
                                     if self.preview_src is not None else None)

    def _on_stamp_feather(self, v: float):
        self._stamp_feather = float(v)

    def _on_stamp_new_source(self):
        self._stamp_src_rect = None
        self.preview.set_stamp_source(None)
        self.statusBar().showMessage("请在图像上拖出新的源矩形", 4000)

    def _on_clone_source(self, rect):
        self._stamp_src_rect = tuple(rect)
        self.preview.set_stamp_source(rect)
        self.statusBar().showMessage(
            "已选源区域：在目标处拖出矩形覆盖（Alt+拖动或「重选源区」可重选）", 5000)

    def _on_clone_apply(self, rect):
        if self.source is None:
            return
        if self._stamp_src_rect is None:
            self.statusBar().showMessage("请先在图像上拖出源矩形", 3000)
            return
        self._ensure_retouch()
        feather = self._stamp_feather_frac()
        src_r = self._stamp_src_rect
        h, w = self._retouch_full.shape[:2]
        core.copy_patch(self._retouch_full, core.norm_rect_px(src_r, h, w),
                        core.norm_rect_px(rect, h, w), feather)
        if self._retouch_preview is not None:
            ph, pw = self._retouch_preview.shape[:2]
            core.copy_patch(self._retouch_preview,
                            core.norm_rect_px(src_r, ph, pw),
                            core.norm_rect_px(rect, ph, pw), feather)
        self._clone_ops.append((tuple(src_r), tuple(rect), float(self._stamp_feather)))
        self.processor.set_source(PREVIEW, self._retouch_preview)
        self.processor.set_source(FULL, self._retouch_full)
        self._submit(PREVIEW)          # 先给一帧降采样反馈，全图随后
        self._on_params_committed()

    def _replay_stamps(self):
        """由原始源图重放全部覆盖操作（撤销/清除后调用）。"""
        self._retouch_full = None
        self._retouch_preview = None
        if self._clone_ops and self.source is not None:
            self._ensure_retouch()
            for src_r, dst_r, feather_pct in self._clone_ops:
                fe = float(np.clip(feather_pct, 0.0, 100.0)) / 100.0
                h, w = self._retouch_full.shape[:2]
                core.copy_patch(self._retouch_full,
                                core.norm_rect_px(src_r, h, w),
                                core.norm_rect_px(dst_r, h, w), fe)
                if self._retouch_preview is not None:
                    ph, pw = self._retouch_preview.shape[:2]
                    core.copy_patch(self._retouch_preview,
                                    core.norm_rect_px(src_r, ph, pw),
                                    core.norm_rect_px(dst_r, ph, pw), fe)
        self.processor.set_source(PREVIEW, self._retouch_preview
                                  if self._retouch_preview is not None else self.preview_src)
        self.processor.set_source(FULL, self._retouch_full
                                  if self._retouch_full is not None else self.source)
        self._submit(PREVIEW)
        self._on_params_committed()

    def _on_stamp_undo(self):
        if self._clone_ops:
            self._clone_ops.pop()
        self._replay_stamps()

    def _on_stamp_clear(self):
        if not self._clone_ops and self._retouch_full is None:
            return
        self._clone_ops.clear()
        self._replay_stamps()
        self.statusBar().showMessage("已清除全部图章修改", 3000)

    def _on_linear_handle(self, handle: int, x1: float, y1: float,
                          x2: float, y2: float):
        m = self._current_mask()
        if m is None or m.kind != "linear" or self._mask_edit is None:
            return
        m.x1, m.y1, m.x2, m.y2 = x1, y1, x2, y2
        self._after_mask_geometry_change(m)

    def _on_radial_center(self, nx: float, ny: float):
        m = self._current_mask()
        if m is None or m.kind != "radial" or self._mask_edit is None:
            return
        m.cx, m.cy = nx, ny
        self._after_mask_geometry_change(m)

    def _on_radial_size(self, axis: int, radius: float, aspect: float):
        m = self._current_mask()
        if m is None or m.kind != "radial" or self._mask_edit is None:
            return
        m.radius = radius
        m.aspect = aspect
        self._after_mask_geometry_change(m)

    def _on_radial_angle(self, deg: float):
        m = self._current_mask()
        if m is None or m.kind != "radial" or self._mask_edit is None:
            return
        m.angle = deg % 360.0
        self._after_mask_geometry_change(m)

    def _on_brush_point(self, nx: float, ny: float):
        m = self._current_mask()
        if m is None or m.kind != "brush" or self._mask_edit is None:
            return
        if not m.strokes or self._stroke_closed:
            m.strokes.append([])          # 新的一笔
            self._stroke_closed = False
        m.strokes[-1].append((nx, ny))
        now = time.monotonic()
        if now - self._last_overlay_time > 0.05:
            self._last_overlay_time = now
            self._refresh_mask_overlays()
            self._on_params_changed()

    def _on_brush_finished(self):
        self._stroke_closed = True
        m = self._current_mask()
        if m is not None and m.strokes and not m.strokes[-1]:
            m.strokes.pop()
        self._refresh_mask_overlays()
        self._on_params_committed()

    def _on_brush_cleared(self):
        m = self._current_mask()
        if m is None or m.kind != "brush":
            return
        m.strokes.clear()
        self._refresh_mask_overlays()
        self._on_params_committed()

    def _on_brush_undo(self):
        m = self._current_mask()
        if m is None or m.kind != "brush" or not m.strokes:
            return
        m.strokes.pop()
        self._refresh_mask_overlays()
        self._on_params_committed()

    def _after_mask_geometry_change(self, m):
        self.preview._edit_info = self._mask_geometry(m)
        self.preview.update()
        self.panel.mask.update_geometry_sliders()
        self._refresh_mask_overlays()
        self._on_params_changed()

    def _refresh_mask_overlays(self):
        """重绘蒙版叠加层（按缓存键缓存，避免重复渲染；低分辨率渲染足够显示）。"""
        if self.source is None or not self.params.masks:
            self.preview.set_overlays([])
            return
        shape = self.source.shape[:2]
        overlays = []
        for i, m in enumerate(self.params.masks):
            key = (mask_cache_key(m, shape), i == self._mask_sel)
            img = self._overlay_cache.pop(key, None)
            if img is None:
                img = mask_to_qimage(render_mask(m, shape, max_edge=768),
                                     selected=(i == self._mask_sel))
            self._overlay_cache[key] = img    # 重新插入以刷新“最近使用”顺序
            overlays.append(img)
        while len(self._overlay_cache) > self._OVERLAY_CACHE_MAX:
            del self._overlay_cache[next(iter(self._overlay_cache))]
        self.preview.set_overlays(overlays)

    # ------------------------------------------------------------ 对比
    def _toggle_compare(self, checked: bool):
        self._comparing = checked
        if checked:
            if self.preview_src is not None:
                self.preview.set_frame(self.preview_src)
            self.status_proc.setText("对比原图")
        else:
            self.status_proc.setText("")
            if self._last_frame is not None:
                self.preview.set_frame(self._last_frame)

    # ------------------------------------------------------------ 其它
    def _update_status(self):
        if self.source is None:
            self.status_image.setText("未打开图像")
            self.status_zoom.setText("")
            return
        h, w = self.source.shape[:2]
        self.status_image.setText(f"{w} × {h} px")
        if self._last_frame is not None:
            fh, fw = self._last_frame.shape[:2]
            tag = "全图" if (fw, fh) == (w, h) else f"预览 {max(fw, fh)}px"
            self.status_proc.setText(f"{tag} {fw}×{fh}")
        self.status_zoom.setText(f"缩放 {self.preview.zoom * 100:.0f}%")
        if self._mask_edit is not None:
            self.status_image.setText(f"{w} × {h} px   ·   蒙版编辑中：拖动编辑，Esc 退出")

    def closeEvent(self, event):
        self._full_timer.stop()
        self.processor.stop()        # 协作取消：正在渲染的任务在分块之间尽快中止
        super().closeEvent(event)

    # ------------------------------------------------------------ 拖放打开图像
    @staticmethod
    def _is_loadable_url(url) -> bool:
        if not url.isLocalFile():
            return False
        ext = os.path.splitext(url.toLocalFile())[1].lower()
        return ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp") \
            or ext in RAW_EXTENSIONS

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() and any(self._is_loadable_url(u) for u in md.urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for u in event.mimeData().urls():
            if self._is_loadable_url(u):
                self.load_image(u.toLocalFile())
                event.acceptProposedAction()
                return
