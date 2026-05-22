"""
手部关键点检测器：MediaPipe Hands 封装。

输出结构化数据 HandLandmarks，与 MediaPipe 内部类型解耦，
便于下游模块（手势识别、UI、数据记录）独立开发与测试。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp


# MediaPipe Hand 21 个关键点的索引名（标准）
HAND_LANDMARK_NAMES = [
    "WRIST",            # 0
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",       # 1-4
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",                  # 5-8
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",                # 9-12
    "RING_FINGER_MCP", "RING_FINGER_PIP",
    "RING_FINGER_DIP", "RING_FINGER_TIP",                    # 13-16
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",      # 17-20
]

# 五指指尖、第二关节索引（用于手势规则）
FINGER_TIPS = [4, 8, 12, 16, 20]   # 拇指 / 食指 / 中指 / 无名 / 小指 - 指尖
FINGER_PIPS = [3, 6, 10, 14, 18]   # 对应的第二关节（PIP，拇指用 IP）
FINGER_MCPS = [2, 5, 9, 13, 17]    # 对应的掌指关节（MCP）


@dataclass
class HandLandmarks:
    """一只手的 21 关键点数据。"""
    handedness: str                                 # "Left" 或 "Right"（已考虑镜像翻转）
    score: float                                    # MediaPipe 给出的置信度
    landmarks_norm: np.ndarray                      # shape=(21, 3)，归一化坐标 (x, y, z) ∈ [0,1]
    landmarks_px: np.ndarray                        # shape=(21, 2)，像素坐标 (px, py)，int

    def tip(self, finger_idx: int) -> Tuple[int, int]:
        """获取指尖像素坐标。finger_idx: 0=拇指 ... 4=小指"""
        return tuple(self.landmarks_px[FINGER_TIPS[finger_idx]])

    def get_px(self, idx: int) -> Tuple[int, int]:
        return tuple(self.landmarks_px[idx])


class HandDetector:
    """MediaPipe Hands 的轻量封装。

    用法：
        det = HandDetector()
        hands = det.detect(bgr_image)
        det.draw(bgr_image, hands)
    """

    # MediaPipe 提供的手部连接线（用于绘图）
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
        already_flipped: bool = False,
    ):
        """
        Args:
            max_num_hands: 最大检测手数
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence: 跟踪置信度阈值
            model_complexity: 0=轻量, 1=完整
            already_flipped: 是否需要把 handedness 标签做一次额外的左右对调。
                经实测，MediaPipe 0.10.x 在我们当前的输入（已经过 cv2.flip 镜像）下，
                输出的 Left/Right 已经与用户自拍视角一致，因此默认 False（不再对调）。
                若你升级了 mediapipe 版本后发现左右手标反了，可以传 True 来翻转。
        """
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._already_flipped = already_flipped

    def close(self):
        self._hands.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def detect(self, bgr_image: np.ndarray) -> List[HandLandmarks]:
        """在 BGR 图像上检测手部关键点。

        Returns:
            HandLandmarks 列表，每只手一项。未检测到则返回空列表。
        """
        if bgr_image is None or bgr_image.size == 0:
            return []

        h, w = bgr_image.shape[:2]
        # MediaPipe 接受 RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # 标记为不可写以加速
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        rgb.flags.writeable = True

        hands: List[HandLandmarks] = []
        if not result.multi_hand_landmarks:
            return hands

        handedness_list = result.multi_handedness or []
        for i, hand_lm in enumerate(result.multi_hand_landmarks):
            # 21 个点：归一化坐标
            norm = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                dtype=np.float32,
            )
            # 像素坐标（注意 y 不要超出图像）
            px = np.zeros((21, 2), dtype=np.int32)
            px[:, 0] = np.clip(norm[:, 0] * w, 0, w - 1).astype(np.int32)
            px[:, 1] = np.clip(norm[:, 1] * h, 0, h - 1).astype(np.int32)

            # 左右手标签
            label = "Unknown"
            score = 0.0
            if i < len(handedness_list):
                cls = handedness_list[i].classification[0]
                label = cls.label
                score = float(cls.score)
                if self._already_flipped:
                    # 镜像后左右手要对调
                    label = "Left" if label == "Right" else "Right"

            hands.append(HandLandmarks(
                handedness=label,
                score=score,
                landmarks_norm=norm,
                landmarks_px=px,
            ))
        return hands

    def draw(
        self,
        bgr_image: np.ndarray,
        hands: List[HandLandmarks],
        draw_labels: bool = True,
    ) -> np.ndarray:
        """把检测结果绘制到图像上。原地修改并返回。"""
        for hand in hands:
            # 画连接线
            for a, b in self.HAND_CONNECTIONS:
                pa = tuple(hand.landmarks_px[a])
                pb = tuple(hand.landmarks_px[b])
                cv2.line(bgr_image, pa, pb, (0, 255, 0), 2)
            # 画关键点
            for idx in range(21):
                p = tuple(hand.landmarks_px[idx])
                # 指尖用红色大点，其余用蓝色小点
                if idx in FINGER_TIPS:
                    cv2.circle(bgr_image, p, 6, (0, 0, 255), -1)
                else:
                    cv2.circle(bgr_image, p, 3, (255, 0, 0), -1)

            # 标注左右手
            if draw_labels:
                wrist = tuple(hand.landmarks_px[0])
                txt = f"{hand.handedness} ({hand.score:.2f})"
                color = (0, 200, 255) if hand.handedness == "Right" else (255, 200, 0)
                cv2.putText(bgr_image, txt,
                            (wrist[0] - 30, wrist[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return bgr_image
