"""
虚拟空中绘画模块。

交互设计：
    - 静态手势 Number_1 (食指伸出)        → 落笔，跟随食指尖画线
    - 静态手势 ≠ Number_1                 → 抬笔
    - 动态手势 Pinch (Open_Palm → OK)     → 切换画笔颜色
    - 动态手势 Grab  (Open_Palm → Fist)   → 清空画布
    - 主程序按 B                          → 开关绘画模式

实现要点：
    - 食指尖坐标做 EMA 平滑，避免抖动
    - 最小移动阈值过滤微小抖动产生的密集顶点
    - 跳变保护：异常大跳跃时自动断笔，防止"跨屏拉线"
    - HUD 显示当前颜色、操作指引
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# 颜色调色板（BGR）+ 名称
# 经过精选：在深色摄像头画面上对比度好、视觉舒适
COLORS_BGR: List[Tuple[Tuple[int, int, int], str]] = [
    ((255, 220, 80),  "Cyan"),     # 青色（默认起手）
    ((120, 240, 100), "Lime"),     # 嫩绿
    ((200, 120, 255), "Pink"),     # 粉色
    ((60,  180, 255), "Amber"),    # 琥珀橙
    ((255, 180, 80),  "Sky"),      # 天蓝
    ((255, 255, 255), "White"),    # 纯白
]


@dataclass
class Stroke:
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int]   # BGR
    thickness: int = 4


class VirtualPaintCanvas:
    """虚拟空中绘画画布。"""

    def __init__(self,
                 smooth_alpha: float = 0.6,
                 min_step_px: float = 2.0,
                 max_jump_px: float = 140.0,
                 thickness: int = 4,
                 lift_grace_frames: int = 4):
        self.strokes: List[Stroke] = []
        self.current: Optional[Stroke] = None
        self.color_idx: int = 0
        self.thickness: int = int(thickness)
        self.enabled: bool = False

        self._alpha: float = float(smooth_alpha)
        self._min_step: float = float(min_step_px)
        self._max_jump: float = float(max_jump_px)
        self._smooth_xy: Optional[Tuple[float, float]] = None
        # 抬笔 grace 窗口：必须连续 N 帧不画才真正断笔，
        # 用以容忍 MediaPipe 单帧丢手 / 静态手势单帧抖动
        self._lift_grace: int = int(lift_grace_frames)
        self._lift_count: int = 0

    # --------------- 状态与配置 ---------------
    @property
    def color(self) -> Tuple[int, int, int]:
        return COLORS_BGR[self.color_idx][0]

    @property
    def color_name(self) -> str:
        return COLORS_BGR[self.color_idx][1]

    def set_enabled(self, b: bool) -> None:
        self.enabled = bool(b)
        if not self.enabled:
            self._end_stroke()
            self._smooth_xy = None
            self._lift_count = 0

    def toggle_enabled(self) -> bool:
        self.set_enabled(not self.enabled)
        return self.enabled

    def next_color(self) -> None:
        self.color_idx = (self.color_idx + 1) % len(COLORS_BGR)
        # 颜色变了：当前笔画结束，下一段用新色
        self._end_stroke()

    def clear(self) -> None:
        self.strokes.clear()
        self.current = None

    # --------------- 主循环 ---------------
    def update(self,
               fingertip_xy: Optional[Tuple[int, int]],
               static_gesture: str,
               dynamic_event_name: Optional[str] = None) -> None:
        """每帧调用一次。"""
        if not self.enabled:
            return

        # 动态手势：颜色切换 / 清屏
        if dynamic_event_name == "Pinch":
            self.next_color()
        elif dynamic_event_name == "Grab":
            self.clear()
            self._smooth_xy = None
            self._lift_count = 0
            return

        # 没手 → 抬笔（走 grace）
        if fingertip_xy is None:
            self._lift_count += 1
            if self._lift_count >= self._lift_grace:
                self._end_stroke()
                self._smooth_xy = None
            return

        x, y = float(fingertip_xy[0]), float(fingertip_xy[1])

        # EMA + 跳变保护
        if self._smooth_xy is None:
            self._smooth_xy = (x, y)
        else:
            sx, sy = self._smooth_xy
            jump = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if jump > self._max_jump:
                # 异常大跳跃：直接断笔并跳转到新点
                self._end_stroke()
                self._smooth_xy = (x, y)
                self._lift_count = self._lift_grace
            else:
                a = self._alpha
                self._smooth_xy = (a * x + (1 - a) * sx,
                                   a * y + (1 - a) * sy)

        # 落笔 / 抬笔（走 grace 窗口）
        if static_gesture == "Number_1":
            self._lift_count = 0
            self._add_point(self._smooth_xy)
        else:
            self._lift_count += 1
            if self._lift_count >= self._lift_grace:
                self._end_stroke()
            # grace 窗口内：保持当前笔画，等待静态手势恢复 Number_1 时继续

    # --------------- 渲染 ---------------
    def render(self, image: np.ndarray) -> np.ndarray:
        """把所有笔画绘制到图像上（就地修改）。"""
        for s in self.strokes:
            self._draw_stroke(image, s)
        if self.current is not None:
            self._draw_stroke(image, self.current)
        # 当前指尖位置画一个光标圆
        if self.enabled and self._smooth_xy is not None:
            cx, cy = int(round(self._smooth_xy[0])), int(round(self._smooth_xy[1]))
            cv2.circle(image, (cx, cy), 8, self.color, 2, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 2, self.color, -1, cv2.LINE_AA)
        if self.enabled:
            self._draw_hud(image)
        return image

    # --------------- 内部 ---------------
    def _add_point(self, xy: Tuple[float, float]) -> None:
        ix, iy = int(round(xy[0])), int(round(xy[1]))
        if self.current is None:
            self.current = Stroke(points=[(ix, iy)],
                                  color=self.color,
                                  thickness=self.thickness)
            return
        last = self.current.points[-1]
        dist = ((ix - last[0]) ** 2 + (iy - last[1]) ** 2) ** 0.5
        if dist >= self._min_step:
            self.current.points.append((ix, iy))

    def _end_stroke(self) -> None:
        if self.current is not None and len(self.current.points) >= 2:
            self.strokes.append(self.current)
        self.current = None

    @staticmethod
    def _draw_stroke(img: np.ndarray, s: Stroke) -> None:
        if len(s.points) == 0:
            return
        if len(s.points) == 1:
            cv2.circle(img, s.points[0], max(1, s.thickness // 2),
                       s.color, -1, cv2.LINE_AA)
            return
        pts = np.array(s.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=s.color,
                      thickness=s.thickness, lineType=cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        # 面板（右上角）
        x0 = w - 240
        y0 = 14
        x1 = w - 14
        y1 = y0 + 100
        # 背景半透明
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (24, 26, 31), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.rectangle(img, (x0, y0), (x1, y1), (90, 95, 105), 1, cv2.LINE_AA)
        # 当前颜色块
        cv2.rectangle(img, (x0 + 12, y0 + 12),
                      (x0 + 56, y0 + 38), self.color, -1)
        cv2.rectangle(img, (x0 + 12, y0 + 12),
                      (x0 + 56, y0 + 38), (220, 220, 220), 1)
        cv2.putText(img, f"Paint: {self.color_name}",
                    (x0 + 66, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(img, "Index up: draw   Pinch: color",
                    (x0 + 12, y0 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (200, 200, 205), 1, cv2.LINE_AA)
        cv2.putText(img, "Grab: clear     B: toggle",
                    (x0 + 12, y0 + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (200, 200, 205), 1, cv2.LINE_AA)
