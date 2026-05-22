"""
深度图相关工具：伪彩色可视化、点距查询、深度有效性过滤等。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def colorize_depth(depth_mm: np.ndarray, max_mm: int = 4000) -> np.ndarray:
    """将 16 位深度图（单位毫米）转换为 BGR 伪彩色图（JET）。

    Args:
        depth_mm: shape=(H, W), dtype=uint16 的深度图
        max_mm: 可视化的最大距离（mm），超过此距离裁剪为 max_mm

    Returns:
        shape=(H, W, 3), dtype=uint8 的 BGR 伪彩色图
    """
    if depth_mm is None:
        return None
    d = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    d8 = (d / max_mm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    # 0 值（无效深度）设为黑色
    color[depth_mm == 0] = 0
    return color


def get_distance_mm(
    depth_mm: np.ndarray,
    x: int,
    y: int,
    window: int = 5,
) -> Optional[float]:
    """查询深度图上 (x, y) 点的距离（毫米），取 window x window 邻域的有效中值。

    用中值而不是单点值，是为了抵抗深度图边缘噪声和无效值（0）。

    Args:
        depth_mm: 深度图，单位毫米
        x, y: 像素坐标
        window: 邻域窗口大小

    Returns:
        距离（mm），若无有效值返回 None
    """
    if depth_mm is None:
        return None
    h, w = depth_mm.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None

    half = window // 2
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def resize_depth_to_color(
    depth_mm: np.ndarray,
    color_shape: Tuple[int, int],
) -> np.ndarray:
    """把深度图缩放到与彩色图相同的高宽。

    用最近邻插值，避免边缘像素插值产生伪深度值。
    """
    h, w = color_shape[:2]
    return cv2.resize(depth_mm, (w, h), interpolation=cv2.INTER_NEAREST)
