"""
全局参数配置。所有可调阈值集中在这里，方便实验调参与报告记录。
"""
from dataclasses import dataclass, field
from typing import Tuple


# ============================================================
# 摄像头配置
# ============================================================
@dataclass
class CameraConfig:
    # 采集源："kinect" 表示 Azure Kinect DK，"opencv:0" 表示 OpenCV 0 号设备
    source: str = "kinect"

    # OpenCV 摄像头的分辨率（仅在 source 以 opencv: 开头时生效）
    opencv_width: int = 1280
    opencv_height: int = 720

    # Kinect 配置
    kinect_color_resolution: str = "720P"   # 720P / 1080P / 1440P / 2160P
    kinect_depth_mode: str = "NFOV_UNBINNED"  # NFOV / WFOV
    kinect_fps: int = 30


# ============================================================
# 显示配置
# ============================================================
@dataclass
class DisplayConfig:
    show_fps: bool = True
    flip_horizontal: bool = True    # 镜像翻转，符合"自拍"习惯
    depth_max_mm: int = 4000        # 深度伪彩可视化的最大距离（mm）


CAMERA = CameraConfig()
DISPLAY = DisplayConfig()
