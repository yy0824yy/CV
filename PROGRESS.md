# 项目进度总结

> 基于 Azure Kinect DK 的人体姿态与手势交互识别系统  
> 更新日期：2026-05-22

---

## 一、项目概览

| 项 | 内容 |
|---|---|
| **目标** | 基于 Azure Kinect DK 的实时人体姿态与手势识别系统，支持距离测量、数据记录与可视化 |
| **技术栈** | Python 3.9 · MediaPipe · OpenCV · pyk4a · PyQt5 · scikit-learn · NumPy · Matplotlib |
| **运行环境** | Windows + Conda 环境 `cv_kinect` |
| **代码结构** | `core/` 算法模块 · `ui/` 界面 · `data/` 输出与日志 · `experiments/` 实验脚本 · `tools/` 工具 |

---

## 二、已完成功能（按模块）

### 1. 摄像头抽象层（`core/camera.py`）
- ✅ 统一接口 `CameraBase`，支持 OpenCV 摄像头与 Azure Kinect DK 切换
- ✅ Kinect 使用 `pyk4a`，提供彩色 + 深度对齐图（`transformed_depth`）
- ✅ 配置项：分辨率、FPS、深度模式（NFOV / WFOV）

### 2. 手部检测与静态手势（`core/hand_detector.py` · `core/gesture_recognizer.py`）
- ✅ MediaPipe Hands 检测 21 个关键点
- ✅ **基于规则的静态手势识别**：7 类
  - `Number_1` `Number_2` `Number_3` `Open_Palm` `Fist` `OK` `Thumbs_Up`
- ✅ 用 `palm_size` 归一化阈值，让手离镜头远近不影响判别
- ✅ 修复 OK 手势与 Number_3 的混淆 bug

### 3. 人体姿态与动作识别（`core/pose_detector.py` · `core/action_recognizer.py`）
- ✅ MediaPipe Pose 检测 33 个关键点
- ✅ **基于角度阈值的动作识别**：8 类
  - `Standing` `Raise_Left_Hand` `Raise_Right_Hand` `Raise_Both_Hands`
  - `Lean_Left` `Lean_Right` `Squat` `Bend_Over`
- ✅ 关节角度计算（左右肘、膝、肩、躯干倾角等）

### 4. 距离测量（`core/distance_estimator.py` · `core/depth_utils.py`）
- ✅ 多关键点距离测量（鼻、肩、腕、髋）
- ✅ 中值滤波 + EMA 平滑
- ✅ 距离过近 / 过远警告
- ✅ 深度有效率统计（用于 UI 进度条显示）

### 5. 稳定化层（`core/smoothing.py` · `core/stable_recognizers.py`）
- ✅ `EMASmoother`：连续值指数滑动平均
- ✅ `MajorityVoter`：离散标签多数投票
- ✅ `WeightedVoter`：带置信度的加权投票（手势用）
- ✅ `LandmarkSmoother`：关键点坐标 EMA 平滑

### 6. 动态手势识别（`core/dynamic_gesture.py`）⭐ 新增
- ✅ **状态切换型识别**：4 种动态手势
  - `Grab` 抓取   ：Open_Palm → Fist
  - `Release` 释放：Fist → Open_Palm
  - `Pinch` 捏合 ：Open_Palm → OK
  - `Point` 指向 ：Open_Palm → Number_1
- ✅ 0.6 秒滑动窗口检测 + 1 秒冷却 + 抑制连发
- ✅ 实测全部一发即中

### 7. 数据记录（`data/logger.py`）
- ✅ 17 列 CSV 实时日志（手势/动作/距离/角度/FPS）
- ✅ 视频录制（叠加可视化标注）
- ✅ 截图保存（彩色 + 深度伪彩 + 原始深度 npy + JSON 元数据）
- ✅ 自动按时间戳命名

### 8. 用户界面（`ui/main_window.py` · `ui/video_thread.py`）⭐ 全新设计
- ✅ **现代深色科技风**（GitHub Dark 配色）
- ✅ **顶部状态条**：设备指示灯 + 项目名 + 录制徽章 + FPS + 实时时钟
- ✅ **中央视频区**：圆角带边框，支持纯彩色 / 纯深度 / 并排三种视图
- ✅ **右侧分组卡片**：
  - 手势识别（含动态手势高亮显示）
  - 人体动作（中文大字 + 英文 ID）
  - 人体距离（大号数字 + 深度有效率进度条）
  - 关节角度（2×2 网格）
- ✅ **底部按钮行**：打开/关闭设备、暂停、截图、录制、导出 CSV
- ✅ **状态栏胶囊指示器**：CSV / REC / 距离实时状态
- ✅ **快捷键**：空格截图、R 录制、P 暂停、V 切换视图、E 导出 CSV、O/C 开关
- ✅ **应用图标**：程序化生成（黑底蓝边 K 字）
- ✅ 卡片悬停高亮、距离警告变色、占位卡片等视觉细节

### 9. 标注数据采集（`experiments/collect_labeled.py`）
- ✅ 命令行交互式采集，支持 3 种模式
  - `--mode gesture`：7 类手势
  - `--mode action`：8 类动作
  - `--mode distance`：5 个距离档（标准距离测量）
- ✅ **保存原始 21 个手部 landmarks 坐标**（用于 ML 训练）
- ✅ 倒计时提示 + 实时画面标注

---

## 三、实验与评估

### 已完成实验

#### A. 静态手势：规则方法 vs 机器学习对比 ⭐ 报告核心实验

**数据集**：560 样本（7 类 × 80 帧），单受试者条件下采集

**5-fold 交叉验证结果**

| 方法 | 准确率 | Macro-F1 |
|---|---|---|
| Rule-based（几何规则） | 69.46% | 0.601 |
| RandomForest | **100.00%** | 1.000 |
| MLP（128-64） | **100.00%** | 1.000 |
| SVM (RBF) | **100.00%** | 1.000 |

**输出图表**
- `fig_classifier_comparison.png` 准确率对比柱状图
- `fig_classifier_randomforest_cm.png` RF 混淆矩阵
- `fig_classifier_mlp_cm.png` MLP 混淆矩阵
- `fig_classifier_svm_rbf_cm.png` SVM 混淆矩阵

#### B. 噪声鲁棒性实验

在测试集 landmarks 上施加不同强度高斯噪声 σ ∈ {0, 0.005, 0.01, 0.02, 0.03, 0.05}：

| σ | Rule | RF | MLP | SVM |
|---|---|---|---|---|
| 0.000 | 71.4% | 100.0% | 100.0% | 100.0% |
| 0.010 | 80.8% | 99.6% | 98.4% | 99.8% |
| 0.020 | 70.8% | **93.1%** | 87.7% | 89.5% |
| 0.030 | 60.1% | **88.9%** | 74.4% | 65.9% |
| 0.050 | 42.5% | **71.4%** | 50.8% | 28.6% |

**关键结论**
- RandomForest 在严重噪声下仍保持 71.4%，鲁棒性最强
- SVM 在高噪声下崩溃（28.6%），最脆弱
- 规则方法在 σ=0.01 时反而上升（小扰动起到正则化效果）

**输出**：`fig_classifier_noise_robustness.png`

#### C. 学习曲线实验（含噪测试集 σ=0.02）

| N/类 | RandomForest | MLP | SVM (RBF) |
|---|---|---|---|
| 2 | 87.6%±2.2 | 83.5%±1.7 | 80.2%±6.0 |
| 5 | 89.7% | 83.7% | 81.3% |
| 10 | 87.3% | 84.6% | 87.1% |
| 20 | 91.1% | 87.0% | 86.2% |
| 40 | 93.5% | 85.9% | 87.9% |
| 60 | **95.3%** | 86.7% | 84.9% |

**关键结论**
- RF 单调上升，未饱和，体现持续学习能力
- MLP 在 N=10 后早早饱和（受网络容量限制）
- SVM 在 N=40 取得峰值后回落（核宽度需随数据量调整）

**输出**：`fig_classifier_learning_curve.png`

---

## 四、关键代码文件

```
e:\大三\CV/
├── main.py                           # 程序入口
├── config.py                         # 全局配置
├── core/
│   ├── camera.py                     # 摄像头抽象（OpenCV / Kinect）
│   ├── hand_detector.py              # 手部 21 点检测
│   ├── pose_detector.py              # 人体 33 点检测
│   ├── gesture_recognizer.py         # 规则手势识别
│   ├── action_recognizer.py          # 规则动作识别
│   ├── dynamic_gesture.py            # 动态手势（状态切换型）⭐
│   ├── distance_estimator.py         # 距离测量（多关键点 + 平滑）
│   ├── depth_utils.py                # 深度图工具
│   ├── geometry.py                   # 关节角度计算
│   ├── smoothing.py                  # EMA / 投票稳定化
│   └── stable_recognizers.py         # 稳定包装器
├── ui/
│   ├── main_window.py                # 主窗口（深色科技风）⭐
│   └── video_thread.py               # 后台采集 + 推理线程
├── data/
│   ├── logger.py                     # CSV / 视频 / 截图记录
│   └── outputs/                      # 输出根目录
│       ├── labeled/                  # 标注数据 CSV
│       ├── figures/                  # 实验图表
│       ├── snapshots/                # 截图
│       ├── videos/                   # 录像
│       └── logs/                     # 运行 CSV 日志
├── experiments/
│   ├── collect_labeled.py            # 带标注数据采集
│   ├── train_gesture_classifier.py   # ML 训练 + 噪声 + 学习曲线 ⭐
│   └── analyze.py                    # 整体准确率分析
└── tools/
    └── test_body_tracking.py         # K4ABT 测试脚本
```

---

## 五、技术亮点（写报告用）

1. **模块化分层设计**：核心算法、UI、数据记录、实验解耦，便于复用与替换
2. **从规则到学习的方法演进**：先用几何规则法理解手势特征空间，再用 ML 方法解决规则的局限性
3. **多维度评估体系**：基础准确率 + 鲁棒性 + 样本效率三维度评估，体现工程严谨性
4. **状态切换型动态手势**：复合静态手势构造高级语义事件，避免直接的运动检测带来的不稳定
5. **稳定化策略**：EMA 平滑（连续值）+ 加权投票（离散标签）双重保障实时输出稳定
6. **自适应阈值**：所有距离阈值用 `palm_size` 归一化，让识别不受手离镜头远近影响

---

## 六、待完成事项

| 优先级 | 任务 | 预计耗时 |
|---|---|---|
| 🔥 高 | 动作数据采集（80 帧 × 8 类） | 10 分钟 |
| 🔥 高 | 距离数据采集（60 帧 × 5 个距离档） | 10 分钟 |
| 🔥 高 | 重跑 `analyze.py` 输出动作/距离图 | 2 分钟 |
| ⭐ 中 | 录制完整演示视频（2 分钟） | 10 分钟 |
| ⭐ 中 | 撰写课程报告（建议 30-40 页） | 6-10 小时 |
| ⭐ 中 | 制作答辩 PPT（建议 15-20 页） | 3-5 小时 |
| 可选 | K4ABT 3D 骨架对比实验 | 1-2 小时 |
| 可选 | 跌倒检测 / 手势 PPT 控制等交互 demo | 1-2 小时 |

---

## 七、报告大纲建议

```
1. 引言
   1.1 研究背景与意义
   1.2 国内外研究现状
   1.3 主要贡献

2. 相关技术
   2.1 Azure Kinect DK 硬件介绍
   2.2 MediaPipe 关键点检测
   2.3 深度学习 vs 规则方法

3. 系统设计
   3.1 整体架构
   3.2 模块化分层
   3.3 摄像头抽象层
   3.4 数据流设计

4. 核心算法
   4.1 手部检测与静态手势识别（规则法）
   4.2 静态手势的机器学习方法 ★
   4.3 动态手势识别（状态切换型）★
   4.4 人体姿态与动作识别
   4.5 距离测量与平滑
   4.6 稳定化策略（投票 + EMA）

5. 实验与评估 ★
   5.1 数据采集协议
   5.2 静态手势：规则 vs ML 对比
   5.3 噪声鲁棒性分析
   5.4 学习曲线分析
   5.5 动作识别准确率
   5.6 距离测量误差
   5.7 系统实时性能（FPS）

6. 系统实现
   6.1 用户界面设计
   6.2 数据记录与可视化
   6.3 用户体验细节

7. 结论与展望
   7.1 工作总结
   7.2 局限性
   7.3 未来工作（多人识别、3D 骨架、跨用户泛化）
```

---

## 八、运行说明

### 环境激活
```powershell
conda activate cv_kinect
```

### 主程序
```powershell
python main.py
```

### 数据采集
```powershell
python experiments\collect_labeled.py --mode gesture --frames 80 --prepare 4
python experiments\collect_labeled.py --mode action  --frames 80 --prepare 4
python experiments\collect_labeled.py --mode distance --frames 60 --distances 1.0,1.5,2.0,2.5,3.0
```

### 实验
```powershell
python experiments\train_gesture_classifier.py    # ML 对比 + 鲁棒性 + 学习曲线
python experiments\analyze.py                     # 整体准确率分析
```
