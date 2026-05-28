"""
人体姿态关键点检测器：MediaPipe Pose 封装。

输出 33 个 2D + 深度（z）关键点 + 可见度（visibility）。
与 hand_detector 保持一致的设计：纯 numpy 输出，与 MediaPipe 内部类型解耦。

注意：MediaPipe Pose 的 "LEFT/RIGHT" 是 被拍摄者本人 视角下的左右，
而不是观察者（摄像头）视角。例如 LEFT_SHOULDER 是这个人自己的左肩。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None


# ============================================================
# 33 个关键点的命名常量（来自 MediaPipe 官方文档）
# ============================================================
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# 关键点名称表（用于调试输出）
POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]


@dataclass
class PoseLandmarks:
    """单人姿态关键点数据。"""
    landmarks_norm: np.ndarray   # shape=(33, 3), float32, 归一化坐标 (x, y, z)
    landmarks_px: np.ndarray     # shape=(33, 2), int32, 像素坐标
    visibility: np.ndarray       # shape=(33,), float32, 每个点的可见度 [0,1]

    def is_visible(self, idx: int, thr: float = 0.5) -> bool:
        return float(self.visibility[idx]) >= thr

    def px(self, idx: int) -> Tuple[int, int]:
        return tuple(self.landmarks_px[idx])

    def norm(self, idx: int) -> np.ndarray:
        return self.landmarks_norm[idx]


class PoseDetector:
    """MediaPipe Pose 的轻量封装（单人）。"""

    POSE_CONNECTIONS = (
        mp.solutions.pose.POSE_CONNECTIONS if mp is not None else tuple()
    )

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True,
    ):
        """
        Args:
            model_complexity: 0=lite, 1=full, 2=heavy
            enable_segmentation: 是否输出人体分割掩码（耗时，默认关）
            smooth_landmarks: MediaPipe 内部的时序平滑（与我们 Step 7 的平滑互补）
        """
        if mp is None:
            raise ModuleNotFoundError(
                "未安装 mediapipe，无法使用 PoseDetector。"
                "请先执行 python -m pip install -r requirements.txt"
            )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self._pose.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def detect(self, bgr_image: np.ndarray) -> Optional[PoseLandmarks]:
        """在 BGR 图像上检测人体姿态。

        Returns:
            PoseLandmarks，未检测到返回 None
        """
        if bgr_image is None or bgr_image.size == 0:
            return None
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        rgb.flags.writeable = True

        if result.pose_landmarks is None:
            return None

        lms = result.pose_landmarks.landmark
        norm = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        vis = np.array([lm.visibility for lm in lms], dtype=np.float32)
        px = np.zeros((33, 2), dtype=np.int32)
        px[:, 0] = np.clip(norm[:, 0] * w, 0, w - 1).astype(np.int32)
        px[:, 1] = np.clip(norm[:, 1] * h, 0, h - 1).astype(np.int32)

        return PoseLandmarks(landmarks_norm=norm, landmarks_px=px, visibility=vis)

    def draw(
        self,
        bgr_image: np.ndarray,
        pose: Optional[PoseLandmarks],
        vis_thr: float = 0.5,
    ) -> np.ndarray:
        """绘制骨架。可见度低于阈值的连线/点用更暗颜色。"""
        if pose is None:
            return bgr_image
        # 连线
        for a, b in self.POSE_CONNECTIONS:
            va = pose.visibility[a] >= vis_thr
            vb = pose.visibility[b] >= vis_thr
            color = (0, 255, 0) if (va and vb) else (60, 100, 60)
            thick = 2 if (va and vb) else 1
            cv2.line(bgr_image, tuple(pose.landmarks_px[a]),
                     tuple(pose.landmarks_px[b]), color, thick)
        # 关键点
        for i in range(33):
            p = tuple(pose.landmarks_px[i])
            if pose.visibility[i] >= vis_thr:
                # 主要关节用红色大点突出
                if i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                         LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
                         LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
                    cv2.circle(bgr_image, p, 6, (0, 0, 255), -1)
                else:
                    cv2.circle(bgr_image, p, 3, (255, 0, 0), -1)
            else:
                cv2.circle(bgr_image, p, 2, (80, 80, 80), -1)
        return bgr_image
