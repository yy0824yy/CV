# 基于 Azure Kinect DK 的人体姿态与手势交互识别系统

计算机视觉课程设计项目。基于 Azure Kinect DK + MediaPipe + PyQt5，实现人体姿态识别、手势识别、深度测距、UI 交互、数据记录与实验分析。

## 环境要求

- Windows 10/11
- Python 3.9（conda 环境推荐）
- Azure Kinect Sensor SDK v1.4.1（已安装）
- Azure Kinect DK 硬件设备

## 安装步骤

```powershell
# 1. 创建 conda 环境
conda create -n cv_kinect python=3.9 -y
conda activate cv_kinect

# 2. 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
CV/
├── main.py                 # PyQt5 主程序入口
├── requirements.txt        # Python 依赖
├── config.py               # 全局参数配置
├── core/                   # 核心算法模块
│   ├── camera.py           # 摄像头抽象层（OpenCV / Kinect）
│   ├── hand_detector.py    # 手部 21 关键点检测
│   ├── pose_detector.py    # 人体姿态检测
│   ├── gesture_recognizer.py
│   ├── action_recognizer.py
│   ├── geometry.py         # 角度 / 距离 / 几何工具
│   ├── smoothing.py        # 关键点平滑 + 帧投票
│   └── depth_utils.py      # 深度图处理
├── ui/                     # PyQt5 界面
│   ├── main_window.py
│   ├── video_thread.py
│   └── widgets.py
├── data/                   # 数据记录
│   ├── logger.py
│   └── outputs/            # 截图 / 视频 / CSV / JSON
├── experiments/            # 实验脚本
│   ├── run_accuracy.py
│   ├── run_distance.py
│   └── plot_results.py
├── tools/                  # 阶段性验证脚本（开发期间用）
│   ├── test_opencv_camera.py
│   ├── test_kinect.py
│   └── ...
└── docs/                   # 报告素材
```

## 开发进度

- [x] Step 1: 环境与 SDK 配置
- [ ] Step 2: 摄像头抽象层 + FPS
- [ ] Step 3: MediaPipe Hands
- [ ] Step 4: 手势识别
- [ ] Step 5: MediaPipe Pose
- [ ] Step 6: 动作识别
- [ ] Step 7: 关节角度 + 平滑
- [ ] Step 8: 连续帧投票
- [ ] Step 9: PyQt5 主窗口
- [ ] Step 10: 数据记录
- [ ] Step 11: Kinect 深度图 + 距离测量
- [ ] Step 12: 实验脚本
- [ ] Step 13: 报告与答辩素材
