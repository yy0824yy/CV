"""
后台视频采集 + 视觉处理线程。

设计要点：
    - 在 QThread 中独占摄像头读取与 MediaPipe 推理，避免阻塞主线程
    - 通过 pyqtSignal 把处理好的一帧（含识别结果）推给主线程更新 UI
    - 支持运行时控制：开始 / 暂停 / 翻转 / 切换深度视图 / 关闭

输出数据结构 ProcessedFrame：
    bgr           主显示图像（已绘制骨架/关键点/手势文字）
    depth_color   深度伪彩图（Kinect 才有，否则 None）
    raw_bgr       未叠加任何绘图的原始彩色帧（用于截图保存）
    raw_depth     未处理的深度图 mm（用于距离测量、保存）
    hands         手部检测列表
    gestures      手势识别结果列表（与 hands 同序）
    pose          姿态检测结果
    action        动作识别结果
    angles        关节角度字典
    fps           当前帧率
    person_distance_m  人体到镜头的距离（m），无法测量为 None
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.camera import create_camera, CameraBase
from core.depth_utils import colorize_depth, resize_depth_to_color
from core.hand_detector import HandDetector, HandLandmarks
from core.pose_detector import PoseDetector, PoseLandmarks, NOSE
from core.stable_recognizers import StableGestureRecognizer, StableActionRecognizer
from core.gesture_recognizer import GestureResult
from core.action_recognizer import ActionResult
from core.dynamic_gesture import DynamicGestureRecognizer, DynamicGestureResult
from core.virtual_paint import VirtualPaintCanvas
from core.geometry import compute_body_angles
from core.smoothing import LandmarkSmoother
from core.distance_estimator import DistanceEstimator, DistanceReport
from config import CAMERA, DISPLAY


@dataclass
class ProcessedFrame:
    """一帧处理结果，跨线程传递。"""
    bgr: np.ndarray                              # 已叠加可视化的主图
    depth_color: Optional[np.ndarray] = None     # 深度伪彩图（与 bgr 等高，可拼接）
    raw_bgr: Optional[np.ndarray] = None         # 未叠加绘图的彩色帧
    raw_depth: Optional[np.ndarray] = None       # 16-bit 深度图（mm）
    hands: List[HandLandmarks] = field(default_factory=list)
    gestures: List[GestureResult] = field(default_factory=list)
    pose: Optional[PoseLandmarks] = None
    action: Optional[ActionResult] = None
    angles: Dict[str, float] = field(default_factory=dict)
    fps: float = 0.0
    # 距离测量（来自 DistanceEstimator）
    person_distance_m: Optional[float] = None        # 兼容旧字段：人体代表距离
    distances_m: Dict[str, float] = field(default_factory=dict)  # 多关节距离
    depth_valid_ratio: float = 0.0
    too_close: bool = False
    too_far: bool = False
    # 动态手势事件（事件型，未触发或已淡出时为 None）
    dynamic_gesture: Optional[DynamicGestureResult] = None
    timestamp: float = 0.0


class VideoThread(QThread):
    """后台采集 + 推理线程。"""

    frame_ready = pyqtSignal(object)   # 发射 ProcessedFrame
    error = pyqtSignal(str)
    info = pyqtSignal(str)

    def __init__(
        self,
        source: str = None,
        flip: bool = None,
        enable_hand: bool = True,
        enable_pose: bool = True,
        smooth_landmarks: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._source = source or CAMERA.source
        self._flip = DISPLAY.flip_horizontal if flip is None else flip
        self._enable_hand = enable_hand
        self._enable_pose = enable_pose
        self._smooth_landmarks = smooth_landmarks
        self._running = False
        self._paused = False
        # 虚拟绘画画布（线程内引用，主线程通过下面的接口控制）
        self.paint_canvas = VirtualPaintCanvas()

    # ---------- 控制接口（主线程调用） ----------
    def stop(self):
        self._running = False

    def set_paused(self, paused: bool):
        self._paused = bool(paused)

    def set_flip(self, flip: bool):
        self._flip = bool(flip)

    def set_enable_hand(self, b: bool):
        self._enable_hand = bool(b)

    def set_enable_pose(self, b: bool):
        self._enable_pose = bool(b)

    # ---------- 虚拟绘画控制（主线程调用） ----------
    def toggle_paint(self) -> bool:
        return self.paint_canvas.toggle_enabled()

    def set_paint_enabled(self, b: bool):
        self.paint_canvas.set_enabled(b)

    def clear_paint(self):
        self.paint_canvas.clear()

    def is_paint_enabled(self) -> bool:
        return self.paint_canvas.enabled

    def get_paint_color_name(self) -> str:
        return self.paint_canvas.color_name

    # ---------- 主循环 ----------
    def run(self):
        # ---- 创建摄像头 ----
        try:
            if self._source == "kinect":
                cam: CameraBase = create_camera(
                    "kinect",
                    color_resolution=CAMERA.kinect_color_resolution,
                    depth_mode=CAMERA.kinect_depth_mode,
                    fps=CAMERA.kinect_fps,
                )
            else:
                cam = create_camera(
                    self._source,
                    width=CAMERA.opencv_width,
                    height=CAMERA.opencv_height,
                )
            cam.open()
            self.info.emit(f"摄像头已打开: {self._source}")
        except Exception as e:
            self.error.emit(f"打开摄像头失败: {e}")
            return

        # ---- 检测器与识别器 ----
        hand_det = HandDetector(max_num_hands=2, already_flipped=False) \
            if self._enable_hand else None
        pose_det = PoseDetector(model_complexity=1) if self._enable_pose else None
        gesture_rec = StableGestureRecognizer(window=7)
        action_rec = StableActionRecognizer(window=9)
        # 动态手势：左右手各一个识别器
        dyn_left = DynamicGestureRecognizer()
        dyn_right = DynamicGestureRecognizer()
        lm_smoother = LandmarkSmoother(alpha=0.7)
        dist_est = DistanceEstimator(near_thr_m=0.6, far_thr_m=3.0, ema_alpha=0.4)

        fps = 0.0
        prev_t = time.time()
        self._running = True
        try:
            while self._running:
                if self._paused:
                    self.msleep(20)
                    continue
                frame = cam.read()
                if frame is None:
                    self.msleep(5)
                    continue

                color = frame.color
                if self._flip:
                    color = cv2.flip(color, 1)
                raw_color = color.copy()
                raw_depth = None
                if frame.depth is not None:
                    raw_depth = cv2.flip(frame.depth, 1) if self._flip else frame.depth

                # ---------- 手部 + 静态手势 ----------
                hands: List[HandLandmarks] = []
                gestures: List[GestureResult] = []
                if hand_det is not None and self._enable_hand:
                    hands = hand_det.detect(color)
                    gestures = gesture_rec.recognize_all(hands)

                # ---------- 动态手势 ----------
                t_frame = time.time()
                # 先扫一遍，分别拿到左/右手（如果存在）
                left_hand = None
                right_hand = None
                left_g_name = ""
                right_g_name = ""
                for h_o, g in zip(hands, gestures):
                    if h_o.handedness == "Left":
                        left_hand = h_o
                        left_g_name = g.name
                    elif h_o.handedness == "Right":
                        right_hand = h_o
                        right_g_name = g.name
                # 每只手 each frame 只调用一次 update：有手传坐标，无手传 None
                # 捕获本帧新触发的事件（用于驱动绘画动作，避免 fade 期重复触发）
                if left_hand is not None:
                    new_left = dyn_left.update(
                        float(left_hand.landmarks_norm[0, 0]),
                        float(left_hand.landmarks_norm[0, 1]),
                        left_g_name, t_frame,
                    )
                else:
                    new_left = dyn_left.update(None, None, "", t_frame)
                if right_hand is not None:
                    new_right = dyn_right.update(
                        float(right_hand.landmarks_norm[0, 0]),
                        float(right_hand.landmarks_norm[0, 1]),
                        right_g_name, t_frame,
                    )
                else:
                    new_right = dyn_right.update(None, None, "", t_frame)
                # 本帧新触发的事件：用于绘画 / 状态切换驱动
                new_event = new_right or new_left
                # UI 显示：fade 期内仍然显示
                dynamic_event = (
                    dyn_right.get_last_event(t_frame)
                    or dyn_left.get_last_event(t_frame)
                )

                # ---------- 姿态 + 动作 + 角度 ----------
                pose: Optional[PoseLandmarks] = None
                action: Optional[ActionResult] = None
                angles: Dict[str, float] = {}
                if pose_det is not None and self._enable_pose:
                    pose = pose_det.detect(color)
                    if pose is not None and self._smooth_landmarks:
                        # 对归一化坐标做 EMA 平滑
                        smoothed = lm_smoother.update(pose.landmarks_norm)
                        pose.landmarks_norm = smoothed
                        h, w = color.shape[:2]
                        pose.landmarks_px[:, 0] = np.clip(smoothed[:, 0] * w, 0, w - 1)
                        pose.landmarks_px[:, 1] = np.clip(smoothed[:, 1] * h, 0, h - 1)
                    action = action_rec.recognize(pose)
                    if pose is not None:
                        angles = compute_body_angles(pose.landmarks_norm)

                # ---------- 距离测量（多关键点 + 平滑 + 警告） ----------
                dist_report: DistanceReport = dist_est.estimate(raw_depth, pose)
                person_distance_m = dist_report.body_distance_m

                # ---------- 可视化绘图 ----------
                vis = color  # 直接画在 color 上
                if pose is not None and pose_det is not None:
                    pose_det.draw(vis, pose)
                if hands and hand_det is not None:
                    hand_det.draw(vis, hands)

                # 在每只手腕处标注手势
                for h, g in zip(hands, gestures):
                    wx, wy = h.landmarks_px[0]
                    cv2.putText(vis, f"{h.handedness}:{g.name}",
                                (max(wx - 70, 5), wy + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 215, 255), 2)

                # 在画面顶部标注主动作
                if action is not None and action.valid:
                    cv2.putText(vis, action.primary,
                                (vis.shape[1] - 360, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0), 3)

                # 距离标注：在每个关节附近标注其距离
                if pose is not None and dist_report.distances_m:
                    from core.distance_estimator import TRACK_POINTS
                    for key, idx in TRACK_POINTS.items():
                        if key not in dist_report.distances_m:
                            continue
                        d_m = dist_report.distances_m[key]
                        px = pose.landmarks_px[idx]
                        cv2.putText(vis, f"{d_m:.2f}m",
                                    (int(px[0]) - 24, int(px[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 2)

                # 距离过近警告：整画面加红色边框 + 顶部红字
                if dist_report.too_close:
                    h_v, w_v = vis.shape[:2]
                    cv2.rectangle(vis, (0, 0), (w_v - 1, h_v - 1), (0, 0, 255), 8)
                    cv2.putText(vis, "TOO CLOSE!  Please step back.",
                                (40, h_v - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                # ---------- 虚拟绘画 ----------
                # 优先右手食指；无则左手
                paint_hand = right_hand if right_hand is not None else left_hand
                paint_static = right_g_name if right_hand is not None else left_g_name
                fingertip = None
                if paint_hand is not None:
                    px8 = paint_hand.landmarks_px[8]
                    fingertip = (int(px8[0]), int(px8[1]))
                ev_name = new_event.name if new_event is not None else None
                self.paint_canvas.update(fingertip, paint_static, ev_name)
                self.paint_canvas.render(vis)

                # ---------- 深度伪彩 ----------
                depth_color_img = None
                if raw_depth is not None:
                    depth_resized = resize_depth_to_color(raw_depth, color.shape)
                    depth_color_img = colorize_depth(
                        depth_resized, max_mm=DISPLAY.depth_max_mm
                    )

                # ---------- FPS ----------
                now = time.time()
                inst_fps = 1.0 / max(now - prev_t, 1e-6)
                fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
                prev_t = now

                processed = ProcessedFrame(
                    bgr=vis,
                    depth_color=depth_color_img,
                    raw_bgr=raw_color,
                    raw_depth=raw_depth,
                    hands=hands,
                    gestures=gestures,
                    pose=pose,
                    action=action,
                    angles=angles,
                    fps=fps,
                    person_distance_m=person_distance_m,
                    distances_m=dist_report.distances_m,
                    depth_valid_ratio=dist_report.depth_valid_ratio,
                    too_close=dist_report.too_close,
                    too_far=dist_report.too_far,
                    dynamic_gesture=dynamic_event,
                    timestamp=now,
                )
                self.frame_ready.emit(processed)
        except Exception as e:
            self.error.emit(f"采集线程异常: {e}")
        finally:
            try:
                cam.close()
            except Exception:
                pass
            if hand_det is not None:
                hand_det.close()
            if pose_det is not None:
                pose_det.close()
            self.info.emit("采集线程已停止")
