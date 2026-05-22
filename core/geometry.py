"""
几何工具：关节角度、距离、向量夹角等。

所有函数都接受 numpy 数组（2D 或 3D 坐标），不依赖项目其他模块，
便于单元测试。
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np


# ============================================================
# 基本几何
# ============================================================
def euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """两点欧式距离。"""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def angle_3points(a, b, c) -> float:
    """计算 A-B-C 在 B 点处的夹角（度）。

    Args:
        a, b, c: 2D 或 3D 坐标，类型为 array-like
    Returns:
        角度，范围 [0, 180]；若任意向量为零向量返回 0
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-6:
        return 0.0
    cos = float(np.dot(ba, bc) / n)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def angle_2vectors(v1, v2) -> float:
    """两个向量的夹角（度）。"""
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / n)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


# ============================================================
# 人体常用关节角度（封装好，调用方只传 PoseLandmarks 即可）
# ============================================================
def compute_body_angles(landmarks_2d: np.ndarray) -> dict:
    """根据 33 关键点的 2D 坐标计算常用关节角度。

    Args:
        landmarks_2d: shape=(33, 2) 或 (33, 3)（仅取前两维）

    Returns:
        dict:
            left_elbow, right_elbow      # 肘关节角度
            left_knee,  right_knee       # 膝关节角度
            left_shoulder, right_shoulder# 肩关节角度（肘-肩-髋）
            shoulder_tilt                # 两肩连线相对水平的偏角（°，正=左肩低）
            torso_tilt                   # 躯干相对垂直方向的前/后倾角
    """
    # 关键点索引（与 pose_detector 中常量一致，避免循环 import）
    LS, RS = 11, 12
    LE, RE = 13, 14
    LW, RW = 15, 16
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28

    p = np.asarray(landmarks_2d, dtype=np.float32)
    # 仅使用 (x, y)
    p = p[:, :2]

    angles = {}
    # 肘关节
    angles["left_elbow"] = angle_3points(p[LS], p[LE], p[LW])
    angles["right_elbow"] = angle_3points(p[RS], p[RE], p[RW])
    # 膝关节
    angles["left_knee"] = angle_3points(p[LH], p[LK], p[LA])
    angles["right_knee"] = angle_3points(p[RH], p[RK], p[RA])
    # 肩关节（肘-肩-髋）
    angles["left_shoulder"] = angle_3points(p[LE], p[LS], p[LH])
    angles["right_shoulder"] = angle_3points(p[RE], p[RS], p[RH])

    # 肩线倾角（正：左肩低；反映镜像视角下用户感觉的左/右倾）
    dy = p[RS][1] - p[LS][1]
    dx = p[RS][0] - p[LS][0]
    shoulder_angle = math.degrees(math.atan2(dy, dx))
    tilt = (shoulder_angle + 180) % 180
    if tilt > 90:
        tilt -= 180
    angles["shoulder_tilt"] = float(tilt)

    # 躯干前/后倾（肩中点 → 髋中点 与 竖直方向夹角）
    shoulder_mid = (p[LS] + p[RS]) * 0.5
    hip_mid = (p[LH] + p[RH]) * 0.5
    torso = shoulder_mid - hip_mid
    angles["torso_tilt"] = angle_2vectors(torso, np.array([0, -1], dtype=np.float32))

    return angles
