"""
摄像头抽象层。

设计目标：
- 主程序只依赖 CameraBase 接口，底层是 OpenCV 还是 Azure Kinect 透明切换。
- 统一返回 (color_bgr, depth_mm) 二元组；depth_mm 在普通摄像头下为 None。
- 通过工厂函数 create_camera(source) 选择具体实现。

source 格式：
    "kinect"       -> Azure Kinect DK
    "opencv:0"     -> OpenCV 0 号设备（笔记本默认摄像头）
    "opencv:1"     -> OpenCV 1 号设备
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


def _pick(mapping: Dict, key, name: str):
    """带清晰错误信息的配置映射选择。"""
    if key not in mapping:
        supported = ", ".join(str(k) for k in mapping.keys())
        raise ValueError(f"无效的 {name}: {key}。支持: {supported}")
    return mapping[key]


# ============================================================
# 抽象基类
# ============================================================
@dataclass
class Frame:
    """一帧采集结果。"""
    color: np.ndarray                   # BGR 彩色图，shape=(H, W, 3), dtype=uint8
    depth: Optional[np.ndarray] = None  # 深度图（毫米），shape=(H, W), dtype=uint16；非 Kinect 设备为 None
    timestamp: float = 0.0              # 时间戳（秒）


class CameraBase(ABC):
    """所有摄像头实现的基类。"""

    @abstractmethod
    def open(self) -> None:
        """打开设备。失败时抛异常。"""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """读一帧。读取失败返回 None。"""

    @abstractmethod
    def close(self) -> None:
        """关闭设备。"""

    @abstractmethod
    def is_opened(self) -> bool:
        ...

    def has_depth(self) -> bool:
        """是否提供深度图。默认 False，Kinect 重写为 True。"""
        return False

    # 支持 with 语法
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# OpenCV 普通摄像头实现
# ============================================================
class OpenCVCamera(CameraBase):
    """基于 cv2.VideoCapture 的普通 USB / 笔记本摄像头。"""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        # Windows 上加 CAP_DSHOW，启动更快且兼容性更好
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 OpenCV 摄像头 (index={self.index})")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        import time
        return Frame(color=frame, depth=None, timestamp=time.time())

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


# ============================================================
# Azure Kinect 实现
# ============================================================
class KinectCamera(CameraBase):
    """基于 pyk4a 的 Azure Kinect DK 摄像头。

    同步输出 BGR 彩色图 + 深度图（毫米）。
    """

    def __init__(
        self,
        color_resolution: str = "720P",
        depth_mode: str = "NFOV_UNBINNED",
        fps: int = 30,
    ):
        self.color_resolution = color_resolution
        self.depth_mode = depth_mode
        self.fps = fps
        self._k4a = None  # 延迟导入 pyk4a，避免没装 SDK 时影响 OpenCV 模式

    def open(self) -> None:
        # 延迟导入：未装 pyk4a 时，用户仍可使用 OpenCVCamera
        from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS

        color_map = {
            "720P": ColorResolution.RES_720P,
            "1080P": ColorResolution.RES_1080P,
            "1440P": ColorResolution.RES_1440P,
            "2160P": ColorResolution.RES_2160P,
        }
        depth_map = {
            "NFOV_UNBINNED": DepthMode.NFOV_UNBINNED,
            "NFOV_2X2BINNED": DepthMode.NFOV_2X2BINNED,
            "WFOV_UNBINNED": DepthMode.WFOV_UNBINNED,
            "WFOV_2X2BINNED": DepthMode.WFOV_2X2BINNED,
        }
        fps_map = {5: FPS.FPS_5, 15: FPS.FPS_15, 30: FPS.FPS_30}

        color_resolution = _pick(
            color_map, self.color_resolution, "Kinect 彩色分辨率"
        )
        depth_mode = _pick(depth_map, self.depth_mode, "Kinect 深度模式")
        camera_fps = _pick(fps_map, self.fps, "Kinect FPS")

        cfg = Config(
            color_resolution=color_resolution,
            depth_mode=depth_mode,
            camera_fps=camera_fps,
            synchronized_images_only=True,  # 仅输出 RGB 与 Depth 都齐的帧
        )
        self._k4a = PyK4A(cfg)
        self._k4a.start()

    def read(self) -> Optional[Frame]:
        if self._k4a is None:
            return None
        try:
            cap = self._k4a.get_capture()
        except Exception:
            return None
        if cap.color is None:
            return None

        # pyk4a 默认返回 BGRA -> 转 BGR
        color_bgr = cv2.cvtColor(cap.color, cv2.COLOR_BGRA2BGR)
        # 关键：使用 transformed_depth，把深度对齐到彩色相机视角，
        # 这样深度图与彩色图分辨率一致、像素一一对应。否则用 cap.depth
        # 直接在彩色坐标上采样会取到错误位置（取到背景距离）。
        depth = cap.transformed_depth  # 可能为 None（少数帧）
        import time
        return Frame(color=color_bgr, depth=depth, timestamp=time.time())

    def close(self) -> None:
        if self._k4a is not None:
            try:
                self._k4a.stop()
            except Exception:
                pass
            self._k4a = None

    def is_opened(self) -> bool:
        return self._k4a is not None

    def has_depth(self) -> bool:
        return True


# ============================================================
# 工厂函数
# ============================================================
def create_camera(source: str = "kinect", **kwargs) -> CameraBase:
    """根据 source 字符串创建对应摄像头实例。

    Args:
        source: "kinect" 或 "opencv:<index>"，如 "opencv:0"
        **kwargs: 透传给具体实现的构造参数

    Returns:
        CameraBase 子类实例（尚未 open）
    """
    if not source:
        raise ValueError("摄像头源不能为空。支持: 'kinect' 或 'opencv:<index>'")
    s = source.lower().strip()
    if s == "kinect":
        return KinectCamera(**kwargs)
    if s.startswith("opencv:"):
        try:
            idx = int(s.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"无效的 OpenCV 设备号: {source}")
        return OpenCVCamera(index=idx, **kwargs)
    raise ValueError(
        f"未知的摄像头源: {source}。支持: 'kinect' 或 'opencv:<index>'"
    )
