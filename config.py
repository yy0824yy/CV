"""
全局参数配置。所有可调阈值集中在这里，方便实验调参与报告记录。
"""
from dataclasses import dataclass


# ============================================================
# 摄像头配置
# ============================================================
@dataclass
class CameraConfig:
    # 采集源："kinect" 表示 Azure Kinect DK，"opencv:0" 表示 OpenCV 0 号设备
    source: str = "kinect"
    fallback_source: str = "opencv:0"   # Kinect 打开失败时的普通摄像头兜底
    enable_fallback: bool = True

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


@dataclass
class SystemConfig:
    auto_csv_on_open: bool = True   # 打开设备后自动开始 CSV 日志
    # 手势识别模式："rule" 使用规则法；"ml" 使用 models/gesture_rf.joblib
    gesture_recognizer_mode: str = "rule"
    gesture_model_path: str = "models/gesture_rf.joblib"


CAMERA = CameraConfig()
DISPLAY = DisplayConfig()
SYSTEM = SystemConfig()
