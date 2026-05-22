"""
主窗口：现代深色科技风 UI。

设计要点：
    - 顶部状态条：设备指示灯 + 标题 + 录制状态 + FPS + 实时时钟
    - 中央：圆角视频区 + 右侧分组卡片面板
    - 底部：分主次的操作按钮行
    - 全局采用 GitHub Dark 风格调色板，强调色青蓝/橙红
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QKeySequence,
    QIcon, QPainter, QColor, QBrush, QPen,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QFrame, QStatusBar, QAction, QShortcut,
    QSizePolicy, QMessageBox,
)

from ui.video_thread import VideoThread, ProcessedFrame
from core.dynamic_gesture import DYNAMIC_GESTURE_CN
from data.logger import DataLogger
from config import CAMERA


# ============================================================
# 配色（GitHub Dark 风格）
# ============================================================
BG       = "#0d1117"   # 主背景
PANEL    = "#161b22"   # 卡片
PANEL2   = "#1c2128"   # 次级面板（按钮等）
BORDER   = "#30363d"   # 默认边框
BORDER_H = "#444c56"   # hover 边框
TEXT     = "#e6edf3"   # 主要文字
TEXT_DIM = "#7d8590"   # 次要文字
TEXT_LO  = "#484f58"   # 三级文字 / 占位

ACCENT   = "#58a6ff"   # 青蓝（主操作）
ACCENT_H = "#79b8ff"   # 青蓝 hover
SUCCESS  = "#3fb950"   # 绿（运行中）
WARNING  = "#d29922"   # 琥珀（过远）
DANGER   = "#f85149"   # 红（录制 / 过近）
INFO     = "#79c0ff"   # 信息蓝（深度 / FPS）
GOLD     = "#d2a8ff"   # 紫（手势数值）— 区别于 FPS
ACTION   = "#7ee787"   # 浅绿（动作）

FONT_UI  = "'Microsoft YaHei UI','Segoe UI',sans-serif"
FONT_NUM = "'Consolas','JetBrains Mono','Microsoft YaHei UI',monospace"


# ============================================================
# 动作英文 -> 中文 显示映射
# ============================================================
ACTION_CN = {
    "Standing":         "站立",
    "Raise_Left_Hand":  "举左手",
    "Raise_Right_Hand": "举右手",
    "Raise_Both_Hands": "举双手",
    "Lean_Left":        "身体左倾",
    "Lean_Right":       "身体右倾",
    "Squat":            "下蹲",
    "Bend_Over":        "弯腰",
    "No_Person":        "未检测到人",
    "Unknown":          "未知",
}


# ============================================================
# 应用图标（程序化生成）
# ============================================================
def make_app_icon(size: int = 64) -> QIcon:
    """生成一个简洁的方形 'K' 图标，用于窗口和任务栏。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    # 背景圆角矩形
    p.setBrush(QBrush(QColor(BG)))
    p.setPen(QPen(QColor(ACCENT), 2))
    radius = size * 0.18
    p.drawRoundedRect(2, 2, size - 4, size - 4, radius, radius)
    # 字母 K
    p.setPen(QColor(ACCENT))
    f = QFont("Microsoft YaHei UI", int(size * 0.5), QFont.Bold)
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignCenter, "K")
    p.end()
    return QIcon(pix)


# ============================================================
# 工具：BGR ndarray -> QPixmap
# ============================================================
def bgr_to_qpixmap(bgr: np.ndarray, target_w: int, target_h: int) -> QPixmap:
    if bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ============================================================
# 通用小组件
# ============================================================
class Card(QFrame):
    """带标题的圆角卡片容器。鼠标悬停时强调色边框。"""
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet(
            f"QFrame#card {{ background:{PANEL}; border:1px solid {BORDER};"
            f" border-radius:8px; }}"
            f"QFrame#card:hover {{ border:1px solid {ACCENT}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)

        # 标题行：左侧色条 + 标题
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)
        bar = QLabel()
        bar.setFixedSize(3, 12)
        bar.setStyleSheet(f"background:{ACCENT}; border-radius:1px;")
        ttl = QLabel(title)
        ttl.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{FONT_UI};"
            f" font-size:11px; font-weight:600; letter-spacing:1px;"
        )
        hdr.addWidget(bar)
        hdr.addWidget(ttl)
        hdr.addStretch(1)
        layout.addLayout(hdr)

        # 内容区
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 2, 0, 0)
        self._body_layout.setSpacing(4)
        layout.addWidget(self._body)

    def add_widget(self, w: QWidget):
        self._body_layout.addWidget(w)

    def add_layout(self, lay):
        self._body_layout.addLayout(lay)


class MetricLine(QWidget):
    """卡片内一行：左侧标签 + 右侧数值。"""
    def __init__(self, label: str, value: str = "--", value_color: str = TEXT):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 3, 0, 3)
        h.setSpacing(8)
        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{FONT_UI}; font-size:13px;"
        )
        self._value = QLabel(value)
        self._value_color = value_color
        self._value.setStyleSheet(self._value_style(value_color))
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self._label)
        h.addStretch(1)
        h.addWidget(self._value)

    @staticmethod
    def _value_style(color: str) -> str:
        return (f"color:{color}; font-family:{FONT_NUM};"
                f" font-size:14px; font-weight:600;")

    def set_value(self, text: str, color: Optional[str] = None):
        self._value.setText(text)
        if color is not None and color != self._value_color:
            self._value_color = color
            self._value.setStyleSheet(self._value_style(color))


class BigMetric(QWidget):
    """大号居中数值（用于距离、动作）。"""
    def __init__(self, value: str = "--", color: str = ACCENT, font_size: int = 26):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 4)
        v.setSpacing(2)
        self._value = QLabel(value)
        self._color = color
        self._size = font_size
        self._value.setStyleSheet(self._style(color, font_size))
        self._value.setAlignment(Qt.AlignCenter)
        v.addWidget(self._value)

    @staticmethod
    def _style(color: str, size: int) -> str:
        return (f"color:{color}; font-family:{FONT_NUM};"
                f" font-size:{size}px; font-weight:700;")

    def set_value(self, text: str, color: Optional[str] = None):
        self._value.setText(text)
        if color is not None and color != self._color:
            self._color = color
            self._value.setStyleSheet(self._style(color, self._size))


class StatusDot(QLabel):
    """顶栏状态指示灯：彩色小圆点。"""
    def __init__(self, color: str = TEXT_LO):
        super().__init__()
        self.setFixedSize(10, 10)
        self.set_color(color)

    def set_color(self, color: str):
        self.setStyleSheet(
            f"background:{color}; border-radius:5px;"
            f" min-width:10px; max-width:10px; min-height:10px; max-height:10px;"
        )


# ============================================================
# 进度条指标（深度有效率用）
# ============================================================
class ProgressMetric(QWidget):
    """标签 + 数值 + 进度条；颜色随比率自动变化。"""
    def __init__(self, label: str):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 4)
        v.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)
        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{FONT_UI}; font-size:13px;"
        )
        self._value = QLabel("-- %")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self._label)
        top.addStretch(1)
        top.addWidget(self._value)
        v.addLayout(top)

        # 进度条：外层 bg + 内层 fill
        self._bg = QFrame()
        self._bg.setFixedHeight(6)
        self._bg.setStyleSheet(f"background:{PANEL2}; border-radius:3px;")
        self._fill = QFrame(self._bg)
        self._fill.setGeometry(0, 0, 0, 6)
        v.addWidget(self._bg)

        self._ratio = 0.0
        self._color = INFO
        self._apply_color(INFO)

    def _apply_color(self, color: str):
        self._color = color
        self._value.setStyleSheet(
            f"color:{color}; font-family:{FONT_NUM};"
            f" font-size:13px; font-weight:600;"
        )
        self._fill.setStyleSheet(f"background:{color}; border-radius:3px;")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_fill_geom()

    def _update_fill_geom(self):
        w = max(0, int(self._bg.width() * self._ratio))
        self._fill.setGeometry(0, 0, w, 6)

    def set_ratio(self, ratio: Optional[float]):
        if ratio is None or ratio <= 0:
            self._ratio = 0.0
            self._value.setText("-- %")
            self._apply_color(TEXT_DIM)
            self._update_fill_geom()
            return
        r = max(0.0, min(1.0, float(ratio)))
        self._ratio = r
        self._value.setText(f"{r * 100:.1f} %")
        if r < 0.4:
            color = DANGER
        elif r < 0.7:
            color = WARNING
        else:
            color = INFO
        if color != self._color:
            self._apply_color(color)
        self._update_fill_geom()


# ============================================================
# 动作显示（中文大字 + 英文原值）
# ============================================================
class ActionDisplay(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 4)
        v.setSpacing(2)
        self._cn = QLabel("--")
        self._cn.setStyleSheet(
            f"color:{ACTION}; font-family:{FONT_UI};"
            f" font-size:22px; font-weight:700;"
        )
        self._cn.setAlignment(Qt.AlignCenter)
        self._en = QLabel("")
        self._en.setStyleSheet(
            f"color:{TEXT_LO}; font-family:{FONT_NUM};"
            f" font-size:11px; letter-spacing:1px;"
        )
        self._en.setAlignment(Qt.AlignCenter)
        v.addWidget(self._cn)
        v.addWidget(self._en)

    def set_value(self, name: str):
        if not name or name == "--":
            self._cn.setText("--")
            self._en.setText("")
            return
        self._cn.setText(ACTION_CN.get(name, name))
        self._en.setText(name if name not in ACTION_CN else name)


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):

    VIEW_COLOR = "color"
    VIEW_DEPTH = "depth"
    VIEW_BOTH = "both"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于 Azure Kinect DK 的人体姿态与手势交互识别系统")
        self.setWindowIcon(make_app_icon(64))
        self.resize(1440, 860)
        self._view_mode = self.VIEW_COLOR
        self._thread: Optional[VideoThread] = None
        self._last_frame: Optional[ProcessedFrame] = None
        self._logger = DataLogger(root="data/outputs")
        self._auto_csv_on_open = True
        self._build_ui()
        self._build_shortcuts()
        # 顶栏时钟
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        self._build_menubar()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(14, 12, 14, 10)
        outer.setSpacing(10)

        outer.addWidget(self._build_top_bar())
        # 中部：视频 + 右面板
        mid = QHBoxLayout()
        mid.setSpacing(12)
        outer.addLayout(mid, stretch=1)
        mid.addWidget(self._build_video_area(), stretch=4)
        mid.addWidget(self._build_right_panel(), stretch=0)
        # 底部按钮行
        outer.addWidget(self._build_bottom_bar())

        # 状态栏（胶囊状指示器）
        self.setStatusBar(QStatusBar())
        self.lbl_csv_status = QLabel("CSV  --")
        self.lbl_rec_status = QLabel("REC  --")
        self.lbl_dist_status = QLabel("距离  --")
        for lbl in (self.lbl_dist_status, self.lbl_csv_status, self.lbl_rec_status):
            lbl.setStyleSheet(self._pill_style(TEXT_DIM))
            self.statusBar().addPermanentWidget(lbl)
        self.statusBar().showMessage("就绪：请点击 [打开设备] 启动 Azure Kinect")

        # 全局样式
        self.setStyleSheet(
            f"QMainWindow {{ background:{BG}; }}"
            f"QWidget {{ color:{TEXT}; font-family:{FONT_UI}; }}"
            f"QMenuBar {{ background:{PANEL}; color:{TEXT}; border-bottom:1px solid {BORDER}; }}"
            f"QMenuBar::item {{ padding:6px 12px; background:transparent; }}"
            f"QMenuBar::item:selected {{ background:{PANEL2}; }}"
            f"QMenu {{ background:{PANEL}; color:{TEXT}; border:1px solid {BORDER}; padding:4px; }}"
            f"QMenu::item {{ padding:6px 24px; }}"
            f"QMenu::item:selected {{ background:{PANEL2}; color:{ACCENT}; }}"
            f"QStatusBar {{ background:{PANEL}; color:{TEXT_DIM}; border-top:1px solid {BORDER}; }}"
            f"QStatusBar::item {{ border:0; }}"
            f"QToolTip {{ background:{PANEL2}; color:{TEXT}; border:1px solid {BORDER};"
            f" padding:4px 8px; }}"
        )

    def _build_menubar(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("文件 (&F)")
        act_exit = QAction("退出", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        m_view = menubar.addMenu("视图 (&V)")
        self._act_view_color = QAction("仅彩色画面", self, checkable=True)
        self._act_view_depth = QAction("仅深度画面", self, checkable=True)
        self._act_view_both = QAction("彩色 + 深度 并排", self, checkable=True)
        self._act_view_color.setChecked(True)
        self._act_view_color.triggered.connect(lambda: self._set_view(self.VIEW_COLOR))
        self._act_view_depth.triggered.connect(lambda: self._set_view(self.VIEW_DEPTH))
        self._act_view_both.triggered.connect(lambda: self._set_view(self.VIEW_BOTH))
        m_view.addAction(self._act_view_color)
        m_view.addAction(self._act_view_depth)
        m_view.addAction(self._act_view_both)

        m_help = menubar.addMenu("帮助 (&H)")
        act_shortcuts = QAction("快捷键", self)
        act_shortcuts.triggered.connect(self._on_shortcuts_help)
        m_help.addAction(act_shortcuts)
        act_about = QAction("关于", self)
        act_about.triggered.connect(self._on_about)
        m_help.addAction(act_about)

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setStyleSheet(
            f"QFrame#topBar {{ background:{PANEL}; border:1px solid {BORDER};"
            f" border-radius:8px; }}"
        )
        bar.setFixedHeight(54)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 10, 16, 10)
        h.setSpacing(12)

        # 左：连接状态灯 + 标题
        self.lbl_dev_dot = StatusDot(TEXT_LO)
        h.addWidget(self.lbl_dev_dot)
        self.lbl_dev_text = QLabel("设备未连接")
        self.lbl_dev_text.setStyleSheet(
            f"color:{TEXT}; font-family:{FONT_UI}; font-size:13px; font-weight:600;"
        )
        h.addWidget(self.lbl_dev_text)
        sep1 = QFrame(); sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet(f"color:{BORDER};")
        sep1.setFixedHeight(20)
        h.addWidget(sep1)
        proj = QLabel("Kinect Pose & Gesture Suite")
        proj.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{FONT_UI}; font-size:12px; letter-spacing:1px;"
        )
        h.addWidget(proj)
        h.addStretch(1)

        # 右：录制 + FPS + 时钟
        self.lbl_rec_badge = QLabel("REC ●")
        self.lbl_rec_badge.setStyleSheet(
            f"color:{DANGER}; font-family:{FONT_NUM}; font-size:13px; font-weight:700;"
            f" padding:2px 10px; border:1px solid {DANGER}; border-radius:4px;"
        )
        self.lbl_rec_badge.setVisible(False)
        h.addWidget(self.lbl_rec_badge)

        fps_box = QHBoxLayout(); fps_box.setSpacing(6)
        fps_lbl = QLabel("FPS")
        fps_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{FONT_UI}; font-size:11px; letter-spacing:1px;"
        )
        self.lbl_fps_val = QLabel("0.0")
        self.lbl_fps_val.setStyleSheet(
            f"color:{INFO}; font-family:{FONT_NUM}; font-size:18px; font-weight:700;"
        )
        self.lbl_fps_val.setMinimumWidth(58)
        self.lbl_fps_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fps_box.addWidget(fps_lbl)
        fps_box.addWidget(self.lbl_fps_val)
        fps_w = QWidget(); fps_w.setLayout(fps_box)
        h.addWidget(fps_w)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet(f"color:{BORDER};")
        sep2.setFixedHeight(20)
        h.addWidget(sep2)

        self.lbl_clock = QLabel("--:--:--")
        self.lbl_clock.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{FONT_NUM}; font-size:14px;"
        )
        h.addWidget(self.lbl_clock)
        return bar

    def _build_video_area(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("videoWrap")
        wrap.setStyleSheet(
            f"QFrame#videoWrap {{ background:#000; border:1px solid {BORDER};"
            f" border-radius:8px; }}"
        )
        v = QVBoxLayout(wrap)
        v.setContentsMargins(2, 2, 2, 2)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setTextFormat(Qt.RichText)
        self.video_label.setText(self._placeholder_html("设备未连接"))
        self.video_label.setStyleSheet(
            f"QLabel {{ background:#000; color:{TEXT_LO};"
            f" font-family:{FONT_UI}; border-radius:6px; }}"
        )
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(720, 480)
        v.addWidget(self.video_label)
        return wrap

    @staticmethod
    def _placeholder_html(title: str, hint: str = "点击下方 [打开设备] 启动 Azure Kinect") -> str:
        return (
            f"<div style='line-height:1.6;'>"
            f"<div style='font-size:22px; color:{TEXT}; font-weight:600; margin-bottom:8px;'>"
            f"{title}</div>"
            f"<div style='font-size:13px; color:{TEXT_DIM};'>{hint}</div>"
            f"</div>"
        )

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(310)
        panel.setMaximumWidth(360)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        # 1) 手势识别（静态 + 动态）
        c1 = Card("手势识别")
        self.row_dynamic = MetricLine("动态手势", "--", TEXT_DIM)
        self.row_left_gesture = MetricLine("左手", "--", GOLD)
        self.row_right_gesture = MetricLine("右手", "--", GOLD)
        self.row_hands = MetricLine("检测到手数", "0", TEXT)
        c1.add_widget(self.row_dynamic)
        c1.add_widget(self.row_left_gesture)
        c1.add_widget(self.row_right_gesture)
        c1.add_widget(self.row_hands)
        v.addWidget(c1)

        # 2) 人体动作（中文大字 + 英文原值）
        c2 = Card("人体动作")
        self.row_action = ActionDisplay()
        c2.add_widget(self.row_action)
        v.addWidget(c2)

        # 3) 人体距离（大号 + 深度有效率进度条）
        c3 = Card("人体距离")
        self.row_distance = BigMetric("-- m", ACCENT, font_size=28)
        c3.add_widget(self.row_distance)
        self.row_depth_valid = ProgressMetric("深度有效率")
        c3.add_widget(self.row_depth_valid)
        v.addWidget(c3)

        # 4) 关节角度（2 列）
        c4 = Card("关节角度")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        self.row_left_elbow = MetricLine("左肘", "--", TEXT)
        self.row_right_elbow = MetricLine("右肘", "--", TEXT)
        self.row_left_knee = MetricLine("左膝", "--", TEXT)
        self.row_right_knee = MetricLine("右膝", "--", TEXT)
        grid.addWidget(self.row_left_elbow, 0, 0)
        grid.addWidget(self.row_right_elbow, 0, 1)
        grid.addWidget(self.row_left_knee, 1, 0)
        grid.addWidget(self.row_right_knee, 1, 1)
        c4.add_layout(grid)
        v.addWidget(c4)

        # 不加 stretch：让 4 张卡片均匀占满纵向空间
        return panel

    def _build_bottom_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("btnBar")
        bar.setStyleSheet(
            f"QFrame#btnBar {{ background:{PANEL}; border:1px solid {BORDER};"
            f" border-radius:8px; }}"
        )
        bar.setFixedHeight(56)
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(8)

        self.btn_open = self._make_btn("打开设备", primary=True)
        self.btn_pause = self._make_btn("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_snap = self._make_btn("截图")
        self.btn_snap.setEnabled(False)
        self.btn_record = self._make_btn("开始录制")
        self.btn_record.setEnabled(False)
        self.btn_export = self._make_btn("导出 CSV")
        self.btn_export.setEnabled(False)
        self.btn_close = self._make_btn("关闭设备")
        self.btn_close.setEnabled(False)

        for b in (self.btn_open, self.btn_pause, self.btn_snap,
                  self.btn_record, self.btn_export):
            h.addWidget(b)
        h.addStretch(1)
        h.addWidget(self.btn_close)

        self.btn_open.clicked.connect(self._on_open)
        self.btn_close.clicked.connect(self._on_close)
        self.btn_pause.clicked.connect(self._on_pause_toggle)
        self.btn_snap.clicked.connect(self._on_snap)
        self.btn_record.clicked.connect(self._on_record_toggle)
        self.btn_export.clicked.connect(self._on_export)
        return bar

    @staticmethod
    def _btn_style_default() -> str:
        return (
            f"QPushButton {{ background:{PANEL2}; color:{TEXT};"
            f" border:1px solid {BORDER}; padding:7px 18px;"
            f" border-radius:6px; font-family:{FONT_UI};"
            f" font-size:13px; font-weight:500; }}"
            f"QPushButton:hover {{ background:#262d36; border-color:{BORDER_H}; }}"
            f"QPushButton:pressed {{ background:#1a2027; }}"
            f"QPushButton:disabled {{ color:{TEXT_LO}; background:{PANEL};"
            f" border-color:{BORDER}; }}"
        )

    @staticmethod
    def _btn_style_primary() -> str:
        return (
            f"QPushButton {{ background:{ACCENT}; color:#0d1117;"
            f" border:0; padding:7px 20px; border-radius:6px;"
            f" font-family:{FONT_UI}; font-size:13px; font-weight:700; }}"
            f"QPushButton:hover {{ background:{ACCENT_H}; }}"
            f"QPushButton:pressed {{ background:#3a8be6; }}"
            f"QPushButton:disabled {{ color:{TEXT_LO}; background:{PANEL2}; }}"
        )

    @staticmethod
    def _btn_style_danger() -> str:
        return (
            f"QPushButton {{ background:{DANGER}; color:#ffffff;"
            f" border:0; padding:7px 20px; border-radius:6px;"
            f" font-family:{FONT_UI}; font-size:13px; font-weight:700; }}"
            f"QPushButton:hover {{ background:#ff6b62; }}"
        )

    def _make_btn(self, text: str, primary: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(self._btn_style_primary() if primary else self._btn_style_default())
        return b

    @staticmethod
    def _pill_style(color: str, bold: bool = False) -> str:
        weight = 700 if bold else 500
        return (
            f"QLabel {{ color:{color}; background:{PANEL2};"
            f" border:1px solid {BORDER}; border-radius:10px;"
            f" padding:2px 12px; margin:2px 3px;"
            f" font-family:{FONT_NUM}; font-size:12px; font-weight:{weight}; }}"
        )

    # ---------------- 顶栏时钟 ----------------
    def _tick_clock(self):
        self.lbl_clock.setText(datetime.now().strftime("%H:%M:%S"))

    # ---------------- 状态指示器 ----------------
    def _refresh_indicators(self):
        # CSV
        if self._logger.is_csv_logging():
            name = os.path.basename(self._logger._csv_path or "")
            self.lbl_csv_status.setText(f"CSV ●  {name}")
            self.lbl_csv_status.setStyleSheet(self._pill_style(SUCCESS, bold=True))
        else:
            self.lbl_csv_status.setText("CSV  --")
            self.lbl_csv_status.setStyleSheet(self._pill_style(TEXT_DIM))
        # 录制
        if self._logger.is_recording():
            name = os.path.basename(self._logger._video_path or "")
            self.lbl_rec_status.setText(f"REC ●  {name}")
            self.lbl_rec_status.setStyleSheet(self._pill_style(DANGER, bold=True))
            self.lbl_rec_badge.setVisible(True)
        else:
            self.lbl_rec_status.setText("REC  --")
            self.lbl_rec_status.setStyleSheet(self._pill_style(TEXT_DIM))
            self.lbl_rec_badge.setVisible(False)

    def _set_device_state(self, connected: bool):
        if connected:
            self.lbl_dev_dot.set_color(SUCCESS)
            self.lbl_dev_text.setText("设备已连接")
            self.lbl_dev_text.setStyleSheet(
                f"color:{SUCCESS}; font-family:{FONT_UI}; font-size:13px; font-weight:600;"
            )
        else:
            self.lbl_dev_dot.set_color(TEXT_LO)
            self.lbl_dev_text.setText("设备未连接")
            self.lbl_dev_text.setStyleSheet(
                f"color:{TEXT}; font-family:{FONT_UI}; font-size:13px; font-weight:600;"
            )

    # ---------------- 视图模式 ----------------
    def _set_view(self, mode: str):
        self._view_mode = mode
        self._act_view_color.setChecked(mode == self.VIEW_COLOR)
        self._act_view_depth.setChecked(mode == self.VIEW_DEPTH)
        self._act_view_both.setChecked(mode == self.VIEW_BOTH)
        if self._last_frame is not None:
            self._render(self._last_frame)

    # ---------------- 设备控制 ----------------
    def _on_open(self):
        if self._thread is not None and self._thread.isRunning():
            return
        self.video_label.setText(self._placeholder_html(
            "正在启动设备", "请稍候 ..."))
        self._thread = VideoThread(source=CAMERA.source)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.error.connect(self._on_error)
        self._thread.info.connect(self._on_info)
        self._thread.start()
        self.btn_open.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_snap.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_close.setEnabled(True)
        if self._auto_csv_on_open:
            try:
                self._logger.start_csv_log()
            except Exception as e:
                self.statusBar().showMessage(f"CSV 启动失败: {e}")
        self._set_device_state(True)
        self.statusBar().showMessage("正在启动设备 ...")
        self._refresh_indicators()

    def _on_close(self):
        if self._thread is not None:
            self._thread.stop()
            self._thread.wait(2000)
            self._thread = None
        rec_path = self._logger.stop_recording()
        csv_path = self._logger.stop_csv_log()
        self.btn_open.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_record.setText("开始录制")
        self.btn_record.setStyleSheet(self._btn_style_default())
        self.btn_export.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(
            self._placeholder_html("设备已关闭", "点击 [打开设备] 重新启动")
        )
        msgs = ["设备已关闭"]
        if rec_path:
            msgs.append(f"录制已保存: {os.path.basename(rec_path)}")
        if csv_path:
            msgs.append(f"CSV 已保存: {os.path.basename(csv_path)}")
        self.statusBar().showMessage("  |  ".join(msgs))
        self._set_device_state(False)
        self._refresh_indicators()
        # 复位面板
        self.row_dynamic.set_value("--", TEXT_DIM)
        self.row_left_gesture.set_value("--", GOLD)
        self.row_right_gesture.set_value("--", GOLD)
        self.row_action.set_value("--")
        self.row_distance.set_value("-- m", ACCENT)
        self.row_depth_valid.set_ratio(None)
        self.row_hands.set_value("0")
        for r in (self.row_left_elbow, self.row_right_elbow,
                  self.row_left_knee, self.row_right_knee):
            r.set_value("--")
        self.lbl_fps_val.setText("0.0")

    def _on_pause_toggle(self):
        if self._thread is None:
            return
        paused = self.btn_pause.text() == "暂停"
        self._thread.set_paused(paused)
        self.btn_pause.setText("继续" if paused else "暂停")
        self.statusBar().showMessage("已暂停" if paused else "运行中")

    def _on_snap(self):
        if self._last_frame is None:
            self.statusBar().showMessage("尚无可用画面")
            return
        try:
            paths = self._logger.save_snapshot(self._last_frame)
            short = os.path.basename(paths.get("visualized", "snapshot"))
            self.statusBar().showMessage(
                f"截图已保存: {short}（共 {len(paths)} 个文件，位于 data/outputs/snapshots/）"
            )
        except Exception as e:
            QMessageBox.warning(self, "截图失败", str(e))

    def _on_record_toggle(self):
        if self._last_frame is None:
            self.statusBar().showMessage("尚无可用画面")
            return
        if not self._logger.is_recording():
            try:
                fps = max(self._last_frame.fps, 15.0)
                path = self._logger.start_recording(self._last_frame, fps=fps)
                self.btn_record.setText("停止录制")
                self.btn_record.setStyleSheet(self._btn_style_danger())
                self.statusBar().showMessage(f"开始录制: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.warning(self, "录制失败", str(e))
        else:
            path = self._logger.stop_recording()
            self.btn_record.setText("开始录制")
            self.btn_record.setStyleSheet(self._btn_style_default())
            self.statusBar().showMessage(
                f"录制已保存: {os.path.basename(path) if path else '?'}"
            )
        self._refresh_indicators()

    def _on_export(self):
        if self._logger.is_csv_logging():
            path = self._logger.stop_csv_log()
            self.statusBar().showMessage(
                f"已停止并导出 CSV: {os.path.basename(path) if path else '?'}"
            )
            new_path = self._logger.start_csv_log()
            self.statusBar().showMessage(
                self.statusBar().currentMessage()
                + f"  |  新日志: {os.path.basename(new_path)}"
            )
        else:
            new_path = self._logger.start_csv_log()
            self.statusBar().showMessage(
                f"已开始 CSV 记录: {os.path.basename(new_path)}"
            )
        self._refresh_indicators()

    def _on_about(self):
        QMessageBox.information(
            self, "关于",
            "基于 Azure Kinect DK 的人体姿态与手势交互识别系统\n\n"
            "技术栈: Python 3.9 + OpenCV + MediaPipe + PyQt5 + pyk4a\n"
            "支持: 手势识别 / 动作识别 / 深度测距 / 实时数据记录\n"
            "扩展: 加权投票稳定化 / 全键盘快捷键"
        )

    def _on_shortcuts_help(self):
        QMessageBox.information(
            self, "快捷键说明",
            "─── 键盘快捷键 ───\n"
            "  空格      截图\n"
            "  R        切换录制\n"
            "  P        暂停 / 继续\n"
            "  V        循环切换视图（彩色 → 深度 → 并排）\n"
            "  E        导出当前 CSV\n"
            "  O        打开设备\n"
            "  C        关闭设备\n"
            "  B        虚拟绘画开关\n"
            "  X        清空画布\n"
            "  Esc      关闭程序\n"
            "\n─── 虚拟绘画手势 ───\n"
            "  伸出食指 (Number_1)  落笔画线\n"
            "  其他手势            抬笔\n"
            "  Pinch 动态手势      切换颜色\n"
            "  Grab  动态手势      清空画布"
        )

    def _build_shortcuts(self):
        def add(seq: str, slot, ctx=Qt.WindowShortcut):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(ctx)
            sc.activated.connect(slot)
            return sc

        add("Space", self._on_snap)
        add("R", self._on_record_toggle)
        add("P", self._on_pause_toggle)
        add("V", self._cycle_view)
        add("E", self._on_export)
        add("O", self._on_open)
        add("C", self._on_close)
        add("B", self._on_paint_toggle)
        add("X", self._on_paint_clear)
        add("Esc", self.close)

    # ---------- 虚拟绘画 ----------
    def _on_paint_toggle(self):
        if self._thread is None:
            self.statusBar().showMessage("请先打开设备")
            return
        enabled = self._thread.toggle_paint()
        self.statusBar().showMessage(
            f"虚拟绘画: {'已开启' if enabled else '已关闭'}"
        )

    def _on_paint_clear(self):
        if self._thread is None:
            return
        self._thread.clear_paint()
        self.statusBar().showMessage("画布已清空")

    def _cycle_view(self):
        order = [self.VIEW_COLOR, self.VIEW_DEPTH, self.VIEW_BOTH]
        try:
            i = order.index(self._view_mode)
        except ValueError:
            i = 0
        self._set_view(order[(i + 1) % len(order)])

    # ---------------- 帧处理 ----------------
    def _on_frame(self, processed: ProcessedFrame):
        self._last_frame = processed
        self._render(processed)
        self._update_panel(processed)
        if self._logger.is_csv_logging():
            self._logger.append_csv(processed)
        if self._logger.is_recording():
            self._logger.append_video_frame(processed.bgr)

    def _render(self, processed: ProcessedFrame):
        if processed is None:
            return
        bgr = processed.bgr
        depth = processed.depth_color
        if self._view_mode == self.VIEW_DEPTH and depth is not None:
            disp = depth
        elif self._view_mode == self.VIEW_BOTH and depth is not None:
            disp = np.hstack([bgr, depth])
        else:
            disp = bgr
        size = self.video_label.size()
        pix = bgr_to_qpixmap(disp, size.width(), size.height())
        self.video_label.setPixmap(pix)

    def _update_panel(self, p: ProcessedFrame):
        # 手势
        left_g = "--"
        right_g = "--"
        for hand, g in zip(p.hands, p.gestures):
            if hand.handedness == "Left":
                left_g = f"{g.name} ({g.score:.2f})"
            elif hand.handedness == "Right":
                right_g = f"{g.name} ({g.score:.2f})"
        self.row_left_gesture.set_value(left_g)
        self.row_right_gesture.set_value(right_g)

        # 动作
        if p.action is not None and p.action.valid:
            self.row_action.set_value(p.action.primary)
        else:
            self.row_action.set_value("--")

        # 距离 + 警告（胶囊状指示器统一样式）
        if p.person_distance_m is not None:
            if p.too_close:
                self.row_distance.set_value(f"{p.person_distance_m:.2f} m", DANGER)
                self.lbl_dist_status.setText(f"距离过近  {p.person_distance_m:.2f} m")
                self.lbl_dist_status.setStyleSheet(self._pill_style(DANGER, bold=True))
            elif p.too_far:
                self.row_distance.set_value(f"{p.person_distance_m:.2f} m", WARNING)
                self.lbl_dist_status.setText(f"距离过远  {p.person_distance_m:.2f} m")
                self.lbl_dist_status.setStyleSheet(self._pill_style(WARNING, bold=True))
            else:
                self.row_distance.set_value(f"{p.person_distance_m:.2f} m", ACCENT)
                self.lbl_dist_status.setText(f"距离  {p.person_distance_m:.2f} m")
                self.lbl_dist_status.setStyleSheet(self._pill_style(SUCCESS))
        else:
            self.row_distance.set_value("-- m", ACCENT)
            self.lbl_dist_status.setText("距离  --")
            self.lbl_dist_status.setStyleSheet(self._pill_style(TEXT_DIM))

        # 角度
        a = p.angles or {}
        self.row_left_elbow.set_value(f"{a.get('left_elbow', 0):.0f}°" if a else "--")
        self.row_right_elbow.set_value(f"{a.get('right_elbow', 0):.0f}°" if a else "--")
        self.row_left_knee.set_value(f"{a.get('left_knee', 0):.0f}°" if a else "--")
        self.row_right_knee.set_value(f"{a.get('right_knee', 0):.0f}°" if a else "--")

        # 深度有效率（进度条）
        self.row_depth_valid.set_ratio(p.depth_valid_ratio)

        # 动态手势（事件型，触发后 2秒内高亮显示）
        dg = p.dynamic_gesture
        if dg is not None:
            cn = DYNAMIC_GESTURE_CN.get(dg.name, dg.name)
            self.row_dynamic.set_value(f"{cn}  ({dg.name})", WARNING)
        else:
            self.row_dynamic.set_value("--", TEXT_DIM)

        # FPS / 手数
        self.row_hands.set_value(str(len(p.hands)))
        self.lbl_fps_val.setText(f"{p.fps:.1f}")

        if self.statusBar().currentMessage().startswith("正在启动"):
            self.statusBar().showMessage("设备已连接，系统运行正常")

    # ---------------- 信号回调 ----------------
    def _on_error(self, msg: str):
        self.statusBar().showMessage(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
        self._on_close()

    def _on_info(self, msg: str):
        self.statusBar().showMessage(msg)

    # ---------------- 关闭 ----------------
    def closeEvent(self, event):
        if self._thread is not None:
            self._thread.stop()
            self._thread.wait(2000)
        try:
            self._logger.close_all()
        except Exception:
            pass
        super().closeEvent(event)
