"""FernLens 入口。

用法：
    python main.py                启动
    python main.py 图片路径       启动并打开图片
"""
from __future__ import annotations

import os
import sys

# 从 legacy/ 子目录运行：把仓库根目录加入导入路径（core/ui_* 在根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _excepthook(exc_type, exc_value, exc_tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stderr.flush()


sys.excepthook = _excepthook


def main():
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("FernLens")
    app.setOrganizationName("FernLens")

    from ui_widgets import apply_dark_theme
    from ui_window import MainWindow

    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    # 支持命令行传入图像路径
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.load_image(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
