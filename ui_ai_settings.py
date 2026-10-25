"""AI 接口设置对话框：增删改接口、拉取模型、写入 api.json。"""
from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

import ai_api


class ApiSettingsDialog(QDialog):
    """管理多个 AI 接口，保存时自动写入 api.json。"""

    def __init__(self, parent=None, first_use: bool = False,
                 path: str | Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("AI 接口设置" + (" — 首次使用" if first_use else ""))
        self.setMinimumSize(780, 560)
        self.resize(880, 640)
        self.setModal(True)

        self._path = Path(path) if path is not None else ai_api.config_path()
        self._first_use = bool(first_use)
        self._load_error = ""
        self._current_id = ""
        self._active_id = ""
        self.saved_config: ai_api.ApiConfig | None = None

        try:
            cfg = ai_api.load_config(self._path)
        except Exception as exc:
            self._load_error = str(exc)
            cfg = ai_api.ApiConfig()
        self._profiles = [copy.copy(p) for p in cfg.apis]
        self._active_id = cfg.active_id
        self._first_use = not self._profiles

        self._build_ui()

        if self._load_error:
            self._banner.setText(
                "⚠ 读取 api.json 失败：" + self._load_error +
                "\n可以继续编辑并保存，保存会覆盖原文件。")
            self._banner.setStyleSheet("color: #9a5b13;")
        elif self._first_use:
            self._banner.setText(
                "首次使用：请填写 API Key、接口 URL，并选择接口格式。"
                "填写后可点击“拉取模型”选择模型；保存后会自动写入 "
                f"{self._path.name}。AI 调色时图像会上传到所选接口。")
        else:
            self._banner.setText(
                "接口信息保存在 " + str(self._path) +
                "。API Key 为明文，请勿分享该文件；AI 调色时图像会上传到所选接口。")

        if not self._profiles:
            self._append_blank_profile()
        else:
            self._reload_list(self._profiles[0].id)

    # ------------------------------------------------------------ 构建界面
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 12)
        outer.setSpacing(10)

        self._banner = QLabel()
        self._banner.setObjectName("dim")
        self._banner.setWordWrap(True)
        outer.addWidget(self._banner)

        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, 1)

        # ---- 左侧：接口列表 ----
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(QLabel("接口列表"))
        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.list, 1)

        list_buttons = QHBoxLayout()
        self.add_btn = QPushButton("添加")
        self.add_btn.setToolTip("添加一个新的接口")
        self.add_btn.clicked.connect(self._add_profile)
        self.del_btn = QPushButton("删除")
        self.del_btn.setToolTip("删除当前选中的接口")
        self.del_btn.clicked.connect(self._delete_profile)
        list_buttons.addWidget(self.add_btn)
        list_buttons.addWidget(self.del_btn)
        left.addLayout(list_buttons)
        body.addLayout(left, 0)

        # ---- 右侧：接口表单 ----
        self.form_box = QGroupBox("接口信息")
        form = QFormLayout(self.form_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(8)
        body.addWidget(self.form_box, 1)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：OpenAI 官方 / 我的代理")
        self.name_edit.textChanged.connect(self._on_name_changed)
        form.addRow("名称", self.name_edit)

        self.protocol_combo = QComboBox()
        for pid in ai_api.PROTOCOL_ORDER:
            self.protocol_combo.addItem(ai_api.protocol_label(pid), pid)
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
        form.addRow("接口格式", self.protocol_combo)

        self.protocol_hint = QLabel()
        self.protocol_hint.setObjectName("dim")
        self.protocol_hint.setWordWrap(True)
        form.addRow("", self.protocol_hint)

        self.url_edit = QLineEdit()
        self.url_edit.setClearButtonEnabled(True)
        self.url_edit.textChanged.connect(self._update_endpoint_preview)
        form.addRow("接口 URL", self.url_edit)

        url_hint = QLabel("填写 Base URL（例如 https://example.ai/v1，"
                          "通常以 /v1 结尾）；粘贴完整端点也可以，"
                          "程序会自动去掉 /chat/completions 等后缀。")
        url_hint.setObjectName("dim")
        url_hint.setWordWrap(True)
        form.addRow("", url_hint)

        key_row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setClearButtonEnabled(True)
        self.key_edit.setPlaceholderText("本地服务不需要 Key 时可留空")
        self.show_key = QCheckBox("显示")
        self.show_key.toggled.connect(self._toggle_key_visible)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.show_key)
        key_wrap = QWidget()
        key_wrap.setLayout(key_row)
        key_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("API Key", key_wrap)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setMinimumWidth(260)
        self.model_combo.lineEdit().setPlaceholderText(
            "留空自动选择；也可点击“拉取模型”")
        self.model_combo.currentTextChanged.connect(self._update_endpoint_preview)
        self.fetch_btn = QPushButton("拉取模型")
        self.fetch_btn.setToolTip("使用当前 URL 与 API Key 获取可用模型列表")
        self.fetch_btn.clicked.connect(self._fetch_models)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.fetch_btn)
        model_wrap = QWidget()
        model_wrap.setLayout(model_row)
        form.addRow("模型", model_wrap)

        self.fetch_status = QLabel("")
        self.fetch_status.setObjectName("dim")
        self.fetch_status.setWordWrap(True)
        form.addRow("", self.fetch_status)

        self.enabled_check = QCheckBox("启用此接口（AI 调色时会尝试调用）")
        self.enabled_check.setChecked(True)
        self.enabled_check.toggled.connect(self._on_enabled_changed)
        form.addRow("", self.enabled_check)

        self.endpoint_label = QLabel("")
        self.endpoint_label.setObjectName("dim")
        self.endpoint_label.setWordWrap(True)
        self.endpoint_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("请求地址", self.endpoint_label)

        form.addRow(QLabel(""))
        self._set_form_enabled(False)
        self._update_protocol_hint()
        self._update_endpoint_preview()

        # ---- 底部：优先接口与操作按钮 ----
        active_row = QHBoxLayout()
        active_row.addWidget(QLabel("优先使用："))
        self.active_combo = QComboBox()
        self.active_combo.setMinimumWidth(260)
        self.active_combo.setToolTip(
            "AI 调色时优先尝试此接口；失败后按列表顺序尝试其他已启用接口")
        self.active_combo.currentIndexChanged.connect(self._on_active_changed)
        active_row.addWidget(self.active_combo)
        active_row.addStretch(1)
        outer.addLayout(active_row)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn = QPushButton("保存")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.save_btn)
        outer.addLayout(buttons)

    # ------------------------------------------------------------ 列表辅助
    def _find(self, profile_id: str) -> ai_api.ApiProfile | None:
        return next((p for p in self._profiles if p.id == profile_id), None)

    def _profile_text(self, profile: ai_api.ApiProfile) -> str:
        name = profile.name.strip() or ai_api.default_profile_name(profile)
        prefix = "★ " if profile.id == self._active_id else ""
        suffix = "" if profile.enabled else "（已停用）"
        return f"{prefix}{name}{suffix}"

    def _refresh_item(self, profile: ai_api.ApiProfile):
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == profile.id:
                item.setText(self._profile_text(profile))
                item.setToolTip(
                    f"{ai_api.protocol_label(profile.protocol)}\n"
                    f"{profile.base_url or '（未填写 URL）'}")
                break
        idx = self.active_combo.findData(profile.id)
        if idx >= 0:
            label = profile.name.strip() or ai_api.default_profile_name(profile)
            if not profile.enabled:
                label += "（已停用）"
            self.active_combo.setItemText(idx, label)

    def _refresh_list_texts(self):
        for i in range(self.list.count()):
            item = self.list.item(i)
            profile = self._find(item.data(Qt.ItemDataRole.UserRole))
            if profile is not None:
                item.setText(self._profile_text(profile))

    def _refresh_active_combo(self):
        current = self._active_id
        self.active_combo.blockSignals(True)
        self.active_combo.clear()
        if self._profiles:
            for profile in self._profiles:
                label = profile.name.strip() or ai_api.default_profile_name(profile)
                if not profile.enabled:
                    label += "（已停用）"
                self.active_combo.addItem(label, profile.id)
            idx = self.active_combo.findData(current)
            if idx < 0:
                idx = next((i for i, p in enumerate(self._profiles)
                            if p.enabled and p.base_url),
                           next((i for i, p in enumerate(self._profiles)
                                 if p.enabled), 0))
                self._active_id = self.active_combo.itemData(idx) or ""
            self.active_combo.setCurrentIndex(idx)
        else:
            self.active_combo.addItem("（无接口）", "")
        self.active_combo.blockSignals(False)

    def _reload_list(self, select_id: str | None = None):
        if select_id is None:
            select_id = self._current_id
        self.list.blockSignals(True)
        self.list.clear()
        for profile in self._profiles:
            item = QListWidgetItem(self._profile_text(profile))
            item.setData(Qt.ItemDataRole.UserRole, profile.id)
            item.setToolTip(
                f"{ai_api.protocol_label(profile.protocol)}\n"
                f"{profile.base_url or '（未填写 URL）'}")
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._refresh_active_combo()
        self._refresh_list_texts()

        ids = [p.id for p in self._profiles]
        if select_id not in ids:
            select_id = ids[0] if ids else ""
        if select_id:
            idx = ids.index(select_id)
            self.list.blockSignals(True)
            self.list.setCurrentRow(idx)
            self.list.blockSignals(False)
            self._current_id = select_id
            self._load_form(self._profiles[idx])
        else:
            self._current_id = ""
            self._clear_form()
            self._set_form_enabled(False)

    def _load_form(self, profile: ai_api.ApiProfile):
        self._set_form_enabled(True)
        blockers = (self.name_edit, self.protocol_combo, self.url_edit,
                    self.key_edit, self.model_combo, self.enabled_check)
        for widget in blockers:
            widget.blockSignals(True)
        self.name_edit.setText(profile.name)
        idx = self.protocol_combo.findData(profile.protocol)
        self.protocol_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.url_edit.setText(profile.base_url)
        self.url_edit.setPlaceholderText(ai_api.default_base_url(profile.protocol))
        self.key_edit.setText(profile.api_key)
        self.model_combo.clear()
        if profile.model:
            self.model_combo.addItem(profile.model)
            self.model_combo.setCurrentText(profile.model)
        self.enabled_check.setChecked(profile.enabled)
        for widget in blockers:
            widget.blockSignals(False)
        self.fetch_status.setText("")
        self._update_protocol_hint()
        self._update_endpoint_preview()

    def _clear_form(self):
        blockers = (self.name_edit, self.protocol_combo, self.url_edit,
                    self.key_edit, self.model_combo, self.enabled_check)
        for widget in blockers:
            widget.blockSignals(True)
        self.name_edit.clear()
        self.protocol_combo.setCurrentIndex(0)
        self.url_edit.clear()
        self.key_edit.clear()
        self.model_combo.clear()
        self.enabled_check.setChecked(True)
        for widget in blockers:
            widget.blockSignals(False)

    def _set_form_enabled(self, enabled: bool):
        for widget in (self.form_box, self.fetch_btn):
            widget.setEnabled(enabled)
        if not enabled:
            self.endpoint_label.setText("")
            self.fetch_status.setText("")

    # ------------------------------------------------------------ 表单事件
    def _on_row_changed(self, _row: int):
        item = self.list.currentItem()
        new_id = item.data(Qt.ItemDataRole.UserRole) if item else ""
        if not new_id or new_id == self._current_id:
            return
        self._sync_form_to_profile()
        profile = self._find(new_id)
        if profile is None:
            return
        self._current_id = new_id
        self._load_form(profile)

    def _on_name_changed(self, _text: str):
        profile = self._find(self._current_id)
        if profile is not None:
            profile.name = self.name_edit.text().strip()
            self._refresh_item(profile)

    def _on_enabled_changed(self, checked: bool):
        profile = self._find(self._current_id)
        if profile is not None:
            profile.enabled = checked
            self._refresh_item(profile)

    def _on_protocol_changed(self, _index: int):
        profile = self._find(self._current_id)
        if profile is None:
            return
        protocol = self.protocol_combo.currentData() or ai_api.DEFAULT_PROTOCOL
        profile.protocol = protocol
        # 输入框里的地址通常是上一格式的完整端点；切换格式时同步重新规整
        normalized = ai_api.normalize_base_url(self.url_edit.text(), protocol)
        if normalized != self.url_edit.text().strip():
            self.url_edit.setText(normalized)
        self.url_edit.setPlaceholderText(ai_api.default_base_url(protocol))
        self._update_protocol_hint()
        self._update_endpoint_preview()

    def _toggle_key_visible(self, checked: bool):
        self.key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)

    def _on_active_changed(self, _index: int):
        value = self.active_combo.currentData() or ""
        if not value:
            return
        self._active_id = value
        self._refresh_list_texts()

    def _form_profile(self, profile_id: str = "") -> ai_api.ApiProfile:
        protocol = self.protocol_combo.currentData() or ai_api.DEFAULT_PROTOCOL
        return ai_api.ApiProfile(
            id=profile_id or self._current_id or "preview",
            name=self.name_edit.text().strip(),
            protocol=protocol,
            base_url=self.url_edit.text().strip(),
            api_key=self.key_edit.text().strip(),
            model=self.model_combo.currentText().strip(),
            enabled=self.enabled_check.isChecked(),
        )

    def _sync_form_to_profile(self):
        profile = self._find(self._current_id)
        if profile is None:
            return
        protocol = self.protocol_combo.currentData() or ai_api.DEFAULT_PROTOCOL
        profile.protocol = protocol
        profile.name = self.name_edit.text().strip()
        profile.base_url = ai_api.normalize_base_url(
            self.url_edit.text(), protocol)
        profile.api_key = ai_api.clean_api_key(self.key_edit.text())
        profile.model = self.model_combo.currentText().strip()
        profile.enabled = self.enabled_check.isChecked()
        if not profile.name:
            profile.name = ai_api.default_profile_name(profile)
        self._refresh_item(profile)

    def _update_protocol_hint(self):
        protocol = self.protocol_combo.currentData() or ai_api.DEFAULT_PROTOCOL
        self.protocol_hint.setText(ai_api.protocol_hint(protocol))

    def _update_endpoint_preview(self, *_args):
        profile = self._form_profile()
        model = self.model_combo.currentText().strip()
        self.endpoint_label.setText(ai_api.describe_endpoint(profile, model))

    # ------------------------------------------------------------ 增删
    def _append_blank_profile(self):
        used = {p.name for p in self._profiles}
        n = 1
        while f"新接口 {n}" in used:
            n += 1
        profile = ai_api.ApiProfile(name=f"新接口 {n}")
        self._profiles.append(profile)
        if not self._active_id:
            self._active_id = profile.id
        self._reload_list(profile.id)

    def _add_profile(self):
        self._sync_form_to_profile()
        self._append_blank_profile()

    def _delete_profile(self):
        profile = self._find(self._current_id)
        if profile is None:
            return
        answer = QMessageBox.question(
            self, "删除接口", f"确定删除「{profile.name}」吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return

        idx = self._profiles.index(profile)
        self._profiles.remove(profile)
        if self._active_id == profile.id:
            self._active_id = self._profiles[0].id if self._profiles else ""
        next_id = ""
        if self._profiles:
            next_id = self._profiles[min(idx, len(self._profiles) - 1)].id
        self._current_id = ""
        self._reload_list(next_id)

    # ------------------------------------------------------------ 拉取模型
    def _fetch_models(self):
        self._sync_form_to_profile()
        profile = self._find(self._current_id)
        if profile is None:
            return
        if not profile.base_url:
            QMessageBox.information(self, "拉取模型", "请先填写接口 URL。")
            self.url_edit.setFocus()
            return

        self.fetch_btn.setEnabled(False)
        self.form_box.setEnabled(False)
        for widget in (self.list, self.add_btn, self.del_btn,
                       self.save_btn, self.cancel_btn):
            widget.setEnabled(False)
        self.fetch_status.setText("正在拉取模型…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            models = ai_api.discover_models(profile, timeout=20, use_cache=False)
        except Exception as exc:
            self.fetch_status.setText("")
            QApplication.restoreOverrideCursor()
            self.fetch_btn.setEnabled(True)
            QMessageBox.warning(self, "拉取模型失败", str(exc)[:1200])
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self.form_box.setEnabled(self._find(self._current_id) is not None)
            self.fetch_btn.setEnabled(True)
            for widget in (self.list, self.add_btn, self.del_btn,
                           self.save_btn, self.cancel_btn):
                widget.setEnabled(True)

        if not models:
            self.fetch_status.setText("接口未返回模型，可手动填写模型名。")
            QMessageBox.information(
                self, "拉取模型",
                "接口没有返回可用模型。\n请确认 URL、Key 与接口格式，"
                "或手动填写模型名。")
            return

        current = self.model_combo.currentText().strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if current and current not in models:
            self.model_combo.insertItem(0, current)
        if not current:
            selected = ai_api.pick_model(models, profile.protocol)
            if selected:
                self.model_combo.setCurrentText(selected)
        else:
            self.model_combo.setCurrentText(current)
        self.model_combo.blockSignals(False)

        self.fetch_status.setText(f"已拉取 {len(models)} 个模型。")
        self._update_endpoint_preview()

    # ------------------------------------------------------------ 保存
    def _save(self):
        self._sync_form_to_profile()

        errors = []
        for profile in self._profiles:
            profile.protocol = ai_api.normalize_protocol(profile.protocol)
            profile.base_url = ai_api.normalize_base_url(
                profile.base_url, profile.protocol)
            profile.name = (profile.name.strip()
                            or ai_api.default_profile_name(profile))
            for message in ai_api.validate_profile(profile):
                errors.append(f"「{profile.name}」：{message}")
        if errors:
            QMessageBox.warning(self, "请检查接口信息", "\n".join(errors))
            return
        if self._first_use and not self._profiles:
            QMessageBox.information(self, "首次使用",
                                    "请至少添加一个 AI 接口，或点击“取消”。")
            return
        if self._profiles and not any(p.enabled for p in self._profiles):
            answer = QMessageBox.question(
                self, "接口均已停用",
                "当前所有接口都已停用，AI 调色将无法使用。仍要保存吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self._load_error:
            answer = QMessageBox.question(
                self, "覆盖 api.json",
                "当前 api.json 读取失败，保存会用对话框中的配置覆盖它。继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return

        cfg = ai_api.ApiConfig(apis=self._profiles, active_id=self._active_id)
        try:
            ai_api.save_config(cfg, self._path)
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.saved_config = cfg
        self.accept()
