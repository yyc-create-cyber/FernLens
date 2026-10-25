"""FernLens 入口。

用法：
    python main.py                启动
    python main.py 图片路径       启动并打开图片
"""
from __future__ import annotations

import os
import sys


def _excepthook(exc_type, exc_value, exc_tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stderr.flush()


sys.excepthook = _excepthook


def main():
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    app.setApplicationName("FernLens")
    app.setApplicationDisplayName("FernLens")
    app.setOrganizationName("FernLens")

    from ui_widgets import apply_theme
    from ui_window import MainWindow

    apply_theme(app)

    # 首次启动创建 api.json；随后弹出接口设置，让用户填写 Key 与 URL
    import ai_api
    first_run = False
    try:
        first_run = ai_api.ensure_config_file()
    except Exception as exc:
        QMessageBox.warning(None, "无法创建 api.json", str(exc))

    window = MainWindow()
    window.show()

    if first_run:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(250, lambda: window.open_ai_settings(first_use=True))

    # 支持命令行传入图像路径
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.load_image(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
