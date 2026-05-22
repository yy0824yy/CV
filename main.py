"""
程序入口。

启动：
    python main.py

依赖：见 requirements.txt
"""
import os
import sys

# 让 main.py 能在任何路径下正常 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    # 高 DPI 适配（笔记本屏幕）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    # 设置中文友好字体
    app.setFont(QFont("Microsoft YaHei", 9))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
