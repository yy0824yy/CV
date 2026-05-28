# 基于 Azure Kinect DK 的人体姿态与手势交互识别系统

计算机视觉课程设计项目。系统基于 Azure Kinect DK、MediaPipe、OpenCV 与 PyQt5，实现实时人体姿态检测、静态/动态手势识别、深度测距、虚拟绘画、数据记录和实验分析。

## 功能概览

- 摄像头抽象层：支持 Azure Kinect DK 与普通 OpenCV 摄像头切换。
- 手势识别：MediaPipe Hands 21 点检测，支持规则式静态手势和状态切换型动态手势。
- 姿态识别：MediaPipe Pose 33 点检测，支持举手、下蹲、弯腰、左右倾等动作识别。
- 深度测距：使用 Kinect 对齐深度图进行人体关键点距离测量。
- 稳定化处理：EMA 平滑、关键点平滑、多数投票和加权投票。
- PyQt5 实时界面：彩色/深度/并排视图、截图、录制、CSV 日志、状态面板。
- 实验脚本：带标签数据采集、规则方法与机器学习分类器对比、鲁棒性和学习曲线分析。

## 环境要求

- Windows 10/11
- Python 3.9 推荐
- Azure Kinect Sensor SDK v1.4.1
- Azure Kinect DK 硬件设备
- 普通摄像头可作为备用输入，但深度测距功能需要 Kinect

## 安装

```powershell
conda create -n cv_kinect python=3.9 -y
conda activate cv_kinect
pip install -r requirements.txt
```

如只使用普通摄像头演示，可在 `config.py` 中将 `CAMERA.source` 改为 `opencv:0`。默认配置会在 Kinect 打开失败时尝试切换到 `opencv:0`。

## 运行主程序

```powershell
python main.py
```

常用快捷键：

- `Space`：截图
- `R`：开始/停止录制
- `P`：暂停/继续
- `V`：切换彩色、深度、并排视图
- `E`：导出当前 CSV 并开始新日志
- `O`：打开设备
- `C`：关闭设备
- `B`：虚拟绘画开关
- `X`：清空画布

## 数据采集

```powershell
python experiments\collect_labeled.py --mode gesture --frames 80 --prepare 4
python experiments\collect_labeled.py --mode action --frames 80 --prepare 4
python experiments\collect_labeled.py --mode distance --frames 60 --distances 1.0,1.5,2.0,2.5,3.0
```

带标签数据会输出到：

```text
data/outputs/labeled/
```

## 实验复现

```powershell
python experiments\train_gesture_classifier.py
python experiments\analyze.py
```

图表会输出到：

```text
data/outputs/figures/
```

`train_gesture_classifier.py` 会额外执行两类更严格的评估：

- 按采集文件留一验证：检查模型跨采集 session 的泛化能力。
- 按时间前后段划分：避免连续帧被随机打散造成结果虚高。

训练完成后会保存实时系统可加载的 RandomForest 模型：

```text
models/gesture_rf.joblib
```

如需在主程序中使用机器学习手势识别，把 `config.py` 中的
`SYSTEM.gesture_recognizer_mode` 改为 `"ml"`。如果模型文件不存在或加载失败，
程序会自动回退到规则法。

## 输出目录

```text
data/outputs/
├── logs/       # CSV 实时日志
├── labeled/    # 带标签采集数据
├── figures/    # 实验图表
├── snapshots/  # 截图、深度图、关键点 JSON
└── videos/     # 录制视频
```

所有输出路径都按项目根目录解析，即使从其他工作目录启动程序，也会写入本项目的 `data/outputs/`。

## 项目结构

```text
CV/
├── main.py
├── config.py
├── requirements.txt
├── core/
│   ├── camera.py
│   ├── hand_detector.py
│   ├── pose_detector.py
│   ├── gesture_recognizer.py
│   ├── dynamic_gesture.py
│   ├── action_recognizer.py
│   ├── distance_estimator.py
│   ├── smoothing.py
│   └── virtual_paint.py
├── ui/
│   ├── main_window.py
│   └── video_thread.py
├── data/
│   └── logger.py
├── experiments/
│   ├── collect_labeled.py
│   ├── train_gesture_classifier.py
│   └── analyze.py
├── models/
├── tests/
└── tools/
```

## 常见问题

1. Kinect 打不开  
   检查 Azure Kinect Sensor SDK、USB 连接、电源和设备占用。默认配置会尝试回退到 `opencv:0`，但深度测距不可用。

2. 没有深度画面  
   普通摄像头没有深度图；只有 Kinect 模式下才有深度伪彩和距离测量。

3. 实验脚本找不到数据  
   先运行 `experiments\collect_labeled.py` 采集带标签数据，再运行分析脚本。

4. 中文图表字体异常  
   Windows 下建议使用 Microsoft YaHei。若图表中文显示异常，可检查 Matplotlib 字体配置。
