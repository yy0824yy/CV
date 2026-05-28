# 系统总览（供报告写作 / GPT 输入用）

> **目的**：把整个项目的**架构、模块、技术、实验、设计决策**完整盘清。
> 每一节标题与建议的报告章节对齐；每一节内列出关键术语和实现细节，
> 便于 GPT 据此扩写为正式报告文字。

---

## 0. 项目身份

| 项 | 内容 |
|---|---|
| **项目名称（建议）** | 基于 Azure Kinect DK 的多模态人体感知与交互系统 —— 以居家陪护应用为例 |
| **英文 / 简称** | Kinect Pose & Gesture Suite (KPGS) |
| **课程性质** | 计算机视觉课程设计 |
| **核心定位** | 双层架构：**通用人体感知与交互平台** + **居家陪护应用案例** |
| **代码仓库** | `e:\大三\CV`（Windows 本地）；GitHub：yy0824yy/CV |
| **开发周期** | 2026-05 ~ 2026-06 |

### 一句话定位
> 系统结合 Kinect 深度摄像头、MediaPipe 姿态/手部识别、传统机器学习手势分类器、
> 以及大语言模型（LLM）语义理解层，构建了一个能够实时感知、识别、理解人体行为，
> 并基于此提供应用闭环（如跌倒告警、活动日报、远程访问）的多模态视觉交互系统。

---

## 1. 系统总体架构

### 1.1 分层架构（强调）

```
┌──────────────────────────────────────────────────────────────────┐
│  应用层 (Application)                                              │
│   · 跌倒告警闭环   · 活动日志 + AI 日报   · 距离异常提醒          │
│   · 虚拟绘画 / 留言   · Web 远程访问（家属端）                     │
└──────────────────────────────────────────────────────────────────┘
                              ↑
┌──────────────────────────────────────────────────────────────────┐
│  理解层 (Understanding)  —— 由 LLM 驱动                            │
│   · 实时场景解读   · 异常事件文案生成   · 长时段活动总结日报      │
│   · 多厂商抽象（硅基流动 / DeepSeek / 智谱 / 通义）                │
└──────────────────────────────────────────────────────────────────┘
                              ↑
┌──────────────────────────────────────────────────────────────────┐
│  识别层 (Recognition)                                              │
│   · 静态手势 (规则 + ML)   · 动态手势 (4 类)   · 姿态/动作 (8 类) │
│   · 距离测量 (深度+关节)   · 跌倒检测 (时序状态机)                │
│   · 关节角度 / 平滑滤波 / 投票稳定                                │
└──────────────────────────────────────────────────────────────────┘
                              ↑
┌──────────────────────────────────────────────────────────────────┐
│  感知层 (Perception)                                               │
│   · Azure Kinect DK：BGRA 1080p + 深度图（NFOV 2x2 binned）       │
│   · OpenCV 后备相机自动 fallback                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流（线程视角）

```
[Kinect 设备] →  read()  →  [VideoThread]
                                ↓
                ┌──────────────────────────────────┐
                │ 同帧串行处理：手部 → 姿态 →     │
                │ 静/动手势 → 动作 → 距离 →       │
                │ 跌倒 → 虚拟绘画 → 可视化绘图    │
                └──────────────────────────────────┘
                                ↓
                ┌────────────┐    ┌─────────────────┐
                │ Qt 信号    │ →  │ MainWindow      │
                │ frame_ready│    │ - 渲染视频区     │
                └────────────┘    │ - 更新右侧面板   │
                       ↓          │ - 写 CSV / 视频  │
                JPEG 编码 ↓       │ - 接收跌倒事件   │
                ┌────────────┐    │ - 触发 LLM       │
                │ web_state  │    └─────────────────┘
                │（共享）    │            ↓
                └────────────┘    ┌─────────────────┐
                       ↓          │ LLMWorker       │
                ┌────────────┐    │（QThread 异步） │
                │ Flask 线程 │    │ → 流式回 chunk  │
                │ MJPEG /SSE │    │ → push 到 Web   │
                └────────────┘    └─────────────────┘
                       ↓
                浏览器（家属端）
```

### 1.3 进程内并发模型

| 线程 | 职责 |
|---|---|
| **Qt 主线程** | UI 渲染、事件循环、所有 LLM 触发逻辑 |
| **VideoThread** (QThread) | 设备读取 + 全部 CV 推理 + JPEG 推流 |
| **LLMWorker** (QThread) | LLM 流式调用，UI 解读 / 跌倒告警 / 日报各一个 |
| **WebServerThread** (Python Thread, daemon) | Flask werkzeug 服务器，监听 8765 端口 |
| **Flask 内部线程池** | 每个 HTTP 客户端一个 worker（threaded=True） |

---

## 2. 技术栈一览

| 类别 | 技术 / 库 | 版本 | 用途 |
|---|---|---|---|
| 编程语言 | Python | 3.10.11 | 全栈 |
| 操作系统 | Windows 10/11 | — | Kinect SDK 限制 |
| GUI | PyQt5 | 5.15.x | 主窗口 |
| 计算机视觉 | OpenCV | 4.x | 图像处理、深度伪彩、绘制 |
| 视觉 SDK | MediaPipe | 0.10+ | Hands / Pose 关键点检测 |
| 设备 SDK | pykinect_azure | latest | Azure Kinect DK 控制 |
| 数值计算 | NumPy | 2.x | 全部张量操作 |
| 机器学习 | scikit-learn | 1.x | RandomForest / MLP / SVM 手势分类 |
| 模型持久化 | joblib | 1.x | 保存 `.joblib` 模型 |
| 时序 ML（待做） | PyTorch | 2.x | LSTM 动作识别 |
| LLM 集成 | openai (官方 SDK) | 2.x | OpenAI 兼容协议调用 |
| LLM 服务商 | 硅基流动 | — | DeepSeek-V4-Flash 模型 |
| Web 框架 | Flask | 3.1 | HTTP / SSE / MJPEG |
| 数据格式 | CSV / JSON / JPEG / MP4 | — | 日志与媒体 |
| 实验图表 | matplotlib + seaborn | — | 混淆矩阵 / 学习曲线 |
| 版本控制 | Git + GitHub | — | 协作 |

---

## 3. 感知层（Perception Layer）

### 3.1 Azure Kinect DK 摄像头

- **设备**：Microsoft Azure Kinect DK
- **传感器组**：1MP ToF 深度相机 + 12MP RGB + 7 麦克风阵列 + IMU
- **本系统使用模式**：
  - 彩色：BGRA 1080p / 30fps（实际程序内 fps ≈ 12-15，受 CV 推理拖累）
  - 深度：NFOV 2×2 binned（512×512，最远 ≈ 5.46m）
- **封装**：`core/camera.py` 抽象 `CameraBase` 基类，子类 `KinectCamera` / `OpenCVCamera`
- **自动 fallback**：`config.CAMERA.enable_fallback=True` 时，Kinect 打开失败自动切到普通 USB 摄像头
- **彩色-深度对齐**：使用 Kinect SDK 的 `transformation.color_image_to_depth_camera()` 把深度配准到彩色画面坐标系

### 3.2 数据预处理

- **镜像翻转**（自拍习惯）：默认开启 `DISPLAY.flip_horizontal`
- **深度有效率**：每帧统计 `depth > 0` 像素比例，作为深度可用性指标显示在 UI

---

## 4. 识别层（Recognition Layer）

### 4.1 静态手势识别（7 类，双路径）

#### 类别（与训练数据对齐）
| 英文标签 | 中文 |
|---|---|
| `Open_Palm` | 张开手掌 |
| `Fist` | 握拳 |
| `OK` | OK 手势 |
| `Like` | 点赞 |
| `Number_1` | 伸出食指 |
| `Victory` | 比 V |
| `Three` | 三指 |

#### 路径 A：规则法（默认）
- **文件**：`core/gesture_recognizer.py`
- **依赖**：MediaPipe Hands 21 关键点
- **核心算法**：
  1. 计算每根手指 5 个关节的弯曲度（指尖 / 中节 / 根节 与手掌中心的距离比）
  2. 得到二值"手指开合"向量 `(thumb, index, middle, ring, pinky)`
  3. 对照预设的指型表查 7 类标签
  4. 失败时返回 `Unknown`

#### 路径 B：机器学习法
- **文件**：`core/ml_gesture_recognizer.py`
- **训练脚本**：`experiments/train_gesture_classifier.py`
- **特征工程**：
  1. 取 21×3 = 63 维 landmarks
  2. 减去 wrist 坐标（位移不变性）
  3. 除以手掌最大距离（尺度不变性）
  4. flatten 成 (63,) 向量
- **训练样本**：560 条，7 类（采集脚本 `tools/collect_labeled.py`）
- **对比模型**：RandomForest / MLP / SVM(RBF)
- **保存格式**：`models/gesture_rf.joblib`，dict 包含 `model`, `model_name`, `classes`, `train_samples`, `feature_version`
- **运行时切换**：`config.SYSTEM.gesture_recognizer_mode = 'rule' | 'ml'`

#### 时序稳定（双路径共用）
- **文件**：`core/stable_recognizers.py` 中的 `StableGestureRecognizer`
- **机制**：每只手（Left / Right）维护独立的 **WeightedVoter**
  - 窗口长度 7 帧
  - 每帧投票携带置信度权重
  - 取窗口内权重最高的标签作为输出
- **效果**：单帧抖动（如 OK 偶尔识别成 Number_3）被多数票压制

### 4.2 动态手势识别（4 类，状态切换型）

- **文件**：`core/dynamic_gesture.py`
- **类别**：
  | 标签 | 状态机转换 | 含义 |
  |---|---|---|
  | `Grab` | 张开手掌 → 握拳 | 抓取 |
  | `Release` | 握拳 → 张开手掌 | 释放 |
  | `Pinch` | 张开手掌 → OK | 捏合（用于切色） |
  | `Point` | 张开手掌 → 食指 | 指向 |
- **设计要点**：
  - 不依赖运动特征（之前尝试用速度阈值不稳）
  - 直接监听 **静态手势 A → B 的状态转换**
  - 转换在 0.4 秒内完成才认为是有效动态手势
  - 每次触发后 1 秒冷却避免连发

### 4.3 姿态识别（MediaPipe Pose 33 关键点）

- **文件**：`core/pose_detector.py`
- **MediaPipe Pose Lite/Full**：可选 `model_complexity=1`（默认）
- **输出**：
  - `landmarks_norm`: shape=(33, 3) 归一化坐标
  - `landmarks_px`: shape=(33, 2) 像素坐标
  - `visibility`: shape=(33,) 每点可见度 [0,1]
- **关键点常量**：`NOSE`, `LEFT/RIGHT_SHOULDER`, `LEFT/RIGHT_HIP`, `LEFT/RIGHT_ELBOW`, ...
- **平滑**：`core/smoothing.py` 的 `LandmarkSmoother`（EMA, α=0.7）抑制抖动

### 4.4 动作识别（8 类）

- **文件**：`core/action_recognizer.py`
- **类别**：
  | 标签 | 中文 | 触发条件（核心） |
  |---|---|---|
  | `Standing` | 站立 | 默认（无其他匹配） |
  | `Raise_Left_Hand` | 举左手 | 左手腕 y 高于头部 + 视野内 |
  | `Raise_Right_Hand` | 举右手 | 右手腕 y 高于头部 |
  | `Raise_Both_Hands` | 举双手 | 两手都满足 |
  | `Lean_Left` | 向左倾 | 双肩 tilt 角度 > 阈值，左肩低 |
  | `Lean_Right` | 向右倾 | 同上，右肩低 |
  | `Squat` | 下蹲 | 髋部 y 下降比例 > 30% |
  | `Bend_Forward` | 弯腰 | 躯干向量前倾角 > 阈值 |
- **设计原则**：
  - 阈值基于**身体比例**（而非像素），距离镜头远近不影响判定
  - 关键点 visibility 过低时拒绝判定
- **优先级表**：`ACTION_PRIORITY` 决定多标签时的 `primary` 选择
- **稳定**：`StableActionRecognizer` 包一层 9 帧 MajorityVoter

### 4.5 距离测量（深度 + 多关节融合）

- **文件**：`core/distance_estimator.py`
- **方法**：
  1. 把 Pose 13 个核心关节（鼻子 / 双肩 / 双肘 / 双腕 / 双髋 / 双膝 / 双踝）的像素坐标
  2. 每个关节取一个 5×5 的 ROI 在深度图中
  3. 取 ROI 内**非零值的中位数**（鲁棒过滤）
  4. 多关节加权得到 `body_distance_m`（躯干关节权重高）
  5. 时序 EMA 平滑，α=0.4
- **报警**：
  - 过近：< 0.6 m → 红色边框 + "TOO CLOSE!"
  - 过远：> 3.0 m → 黄色提示

### 4.6 关节角度

- **文件**：`core/geometry.py` 的 `compute_body_angles()`
- **计算**：左右肘、左右膝四个角度（向量内积法）
- **用途**：UI 显示 + LLM prompt 中加入

### 4.7 跌倒检测（NEW · 应用层关键算法）

- **文件**：`core/fall_detector.py`
- **3 个互补特征**：
  1. **躯干倾角** `torso_angle_deg`：髋部→肩部向量与垂直方向的夹角
     - 站立 ≈ 0°，平躺 ≈ 90°，阈值 55°
  2. **头部下降幅度** `head_drop`：当前鼻子 y 减去最近 1 秒内最小 y
     - 阈值 0.20（归一化坐标）
  3. **倒地确认时长** `confirmed_duration_sec`：满足倒地特征的累计时间
     - 必须 ≥ 0.8 秒才触发告警（防误判）
- **状态机**：`NORMAL → SUSPECT → FALLEN → COOLDOWN`
- **冷却期**：触发后 20 秒内不重复告警
- **去抖**：关键点不可见时清空累计器（不允许"挂着过去状态"）
- **单帧时间复杂度**：O(1)（只看常数个点 + 维护一个 deque）

### 4.8 虚拟绘画

- **文件**：`core/virtual_paint.py`
- **核心**：以**右手食指（landmark 8）**作为画笔尖
  - 静态手势 = `Number_1` 时落笔
  - 其他手势抬笔
- **抖动抑制**：
  - 4 帧 grace 窗口（瞬时手势抖动不断笔）
  - EMA 平滑指尖坐标（α=0.6）
  - 跳变保护（位移 > 140px 视为异常，自动断笔）
- **6 色调色板**（精选）：青/绿/粉/琥珀/天蓝/白
- **动态手势触发**：
  - `Pinch`：切换下一个颜色
  - `Grab`：清空画布
- **快捷键**：`B` 开关、`X` 清空

---

## 5. 理解层（Understanding Layer · LLM）

### 5.1 LLM 客户端封装

- **文件**：`core/llm_client.py`
- **支持厂商**（按优先级自动选）：
  | 厂商 | 模型 | 协议 | 价格 |
  |---|---|---|---|
  | 硅基流动 | DeepSeek-V4-Flash（实际使用） | OpenAI 兼容 | 付费 |
  | 智谱 | glm-4-flash | OpenAI 兼容 | 免费 |
  | DeepSeek 官方 | deepseek-chat | OpenAI 兼容 | 付费但常 503 |
  | 通义千问 | qwen-turbo | OpenAI 兼容 | 免费试用 |
- **自动选厂商**：检测环境变量优先级 `SILICONFLOW_API_KEY` > `ZHIPU_API_KEY` > ...
- **特性**：
  - 流式输出（chunk 生成器）
  - 503 / 网络错误**指数退避自动重试** 3 次
  - `available` / `last_error` 状态查询
- **接入设备 SDK**：`openai` Python 包（兼容协议无需厂商单独 SDK）

### 5.2 异步 LLM Worker（不阻塞 UI）

- **文件**：`ui/llm_worker.py`
- **核心**：`LLMWorker(QThread)`
  - `chunk(str)` 信号：流式片段（用于实时显示）
  - `done(str)` 信号：完整结果
  - `failed(str)` 信号：错误
- **设计要点**：每次调用新建一个 Worker 实例（一次性），简化生命周期

### 5.3 Prompt 工程

- **文件**：`core/llm_understand.py`

#### 5.3.1 通用系统人设（SYSTEM_PROMPT_BASE）
> 你是一个智能居家陪护视觉系统的语义解读助手...
> 输出要求：1) 中文回答 2) 80-160 字 3) 4 个层次（行为/状态/意图/建议）
> 4) 不罗列原始数据 5) 信息不足时使用"可能/似乎"

#### 5.3.2 三大 Prompt 模板

| 函数 | 用途 | 输入 | 输出长度 |
|---|---|---|---|
| `build_scene_prompt(frame)` | 实时场景解读 | ProcessedFrame 序列化 | 80-160 字 |
| `build_fall_alert_prompt(ctx)` | 跌倒告警文案 | 跌倒特征 + 上下文 | 80-120 字 |
| `summarize_history(events, ...)` | 活动日报 | 事件列表 + 时长 | 4-6 句 |

#### 5.3.3 特征序列化
- `_frame_to_dict(frame)` 把 ProcessedFrame 转成结构化字典
- 字段：动作 / 双手手势 / 动态手势 / 距离 / 关节角度 / 异常状态
- 中文映射表：`ACTION_CN`, `GESTURE_CN`, `DYNAMIC_CN`

### 5.4 活动日志（应用层支撑模块）

- **文件**：`core/activity_log.py`
- **核心类**：`ActivityRecorder`
  - 每帧调 `update(processed)`
  - 内部去重：连续多帧相同状态合并为一个事件
  - 时间窗口聚合：`recent_events(since_seconds=300)`
- **5 种事件类型**：
  - `action_change`（动作切换，需持续 1 秒以上才记录）
  - `dynamic_gesture`（动态手势触发）
  - `fall`（跌倒事件）
  - `distance_warning`（距离过近 / 过远进入或解除）
  - `session_start`（会话开始）
- **设计要点**：
  - 状态记忆：`_last_action`, `_pending_action`, `_pending_since`
  - 防抖：候选动作必须持续 ≥ 1 秒才确认切换

---

## 6. 应用层（Application Cases · 居家陪护场景）

### 6.1 跌倒告警闭环 ⭐

完整流程：
```
跌倒检测器 触发
    ↓
落盘保存：data/outputs/alerts/fall_<ts>.jpg + .txt
    ↓
UI 红色横幅 + 可视化红框
    ↓
LLMWorker 异步调用，build_fall_alert_prompt
    ↓
流式输出告警通知文案
    ↓
回写到 .txt 文件
    ↓
推送到 Web 端（家属手机弹出红色卡片）
```

**输出例子**：
> "用户于 20:42:03 发生疑似跌倒事件，躯干倾角 78°，已倒地约 1.2 秒，距离镜头 1.5 米。当前情况建议立即前往现场查看，必要时请联系医护或家属协助处理。"

### 6.2 活动日志 + AI 日报 ⭐

- 用户连续活动一段时间 → 按 `D` → AI 总结
- 默认窗口：最近 5 分钟
- LLM 输出 4-6 句：
  > 总览 / 重点事件 / 节奏 / 关怀建议

### 6.3 距离 / 异常状态实时提示

- 过近过远 → 画面警告 + 状态栏提示
- 加入活动日志 → 后续日报会提到

### 6.4 虚拟绘画（家属交互场景）

- 报告中可定位为"空中留言通道"
- 实测可写"O / S / Hello"等基本字符
- 6 色切换、清屏、抬笔均靠手势完成

### 6.5 Web 远程访问 ⭐ NEW

- **文件**：`services/web_state.py` + `services/web_server.py`
- **后端**：Flask 3.1 + 守护线程
- **暴露接口**：
  | 路径 | 类型 | 用途 |
  |---|---|---|
  | `/` | HTML | 单页应用首页（嵌入式） |
  | `/stream` | MJPEG | 实时视频流（`<img src>` 直接用） |
  | `/api/snapshot.jpg` | JPEG | 单帧（移动端低带宽 fallback） |
  | `/api/events` | SSE | 跌倒告警 / AI 解读 / 日报 推送 |
  | `/api/recent` | JSON | 历史事件回放（最近 1 小时） |
  | `/api/status` | JSON | 系统状态 |
- **前端**：纯 HTML + JS，无依赖
  - 深色科技风（与桌面端一致）
  - 移动端响应式（< 760px 单列）
  - SSE 自动重连
  - 视频流断流自动 fallback 到 snapshot
- **共享状态**：`WebState` 单例 + 锁 + 条件变量
  - VideoThread 推 JPEG（限速 ~10 fps，cv2.imencode quality=75）
  - 主窗口推事件（`_push_web_event`）
- **应用闭环**：家属在外用手机看到老人画面 + 实时告警 + AI 解读

---

## 7. 工程实现细节

### 7.1 UI（PyQt5 现代深色科技风）

- **文件**：`ui/main_window.py`（~1300 行）
- **配色**：GitHub Dark 风格调色板
  - 背景 `#0d1117`，卡片 `#161b22`
  - 强调色青蓝 `#58a6ff`，告警红 `#f85149`
- **结构**：
  - 顶部状态条：设备指示灯 + 标题 + 录制状态 + FPS + 时钟
  - 中央：圆角视频区 + 右侧 5 张卡片
  - 底部：分组操作按钮
  - 菜单：文件 / 视图 / 帮助
- **5 张右侧卡片**：
  1. 手势识别（左右手 + 动态）
  2. 人体动作（中文大字）
  3. 人体距离（大号数值 + 深度有效率进度条）
  4. 关节角度（4 个）
  5. **AI 解读（流式输出 + 解读/日报/清空 三按钮）** ← 新加
- **快捷键**：
  - `O / C` 开关设备 · `P` 暂停 · `V` 切视图
  - `空格` 截图 · `R` 录制 · `E` 导出 CSV
  - `B / X` 虚拟绘画开关 / 清空
  - **`Q` AI 解读** · **`D` AI 日报**（新加）

### 7.2 数据日志（DataLogger）

- **文件**：`data/logger.py`
- **三种输出**：
  - **CSV 日志**：每帧识别结果记录到 `data/outputs/logs/log_<ts>.csv`
    - 列：时间、FPS、左右手手势 + 置信度、动作、距离、跌倒状态、跌倒特征 ...
    - 每次打开设备自动开始（`SYSTEM.auto_csv_on_open=True`）
  - **视频录制**：`data/outputs/videos/record_<ts>.mp4`，带可视化叠加
  - **截图快照**：`data/outputs/snapshots/`
    - 同时输出 vis（带标注）/ 原始 BGR / 16-bit 深度 PNG / depth_raw.npy / data.json
- **跌倒专用输出**：`data/outputs/alerts/fall_<ts>.jpg + .txt`

### 7.3 配置管理（中心化）

- **文件**：`config.py`
- 三个 dataclass：`CameraConfig`, `DisplayConfig`, `SystemConfig`
- 关键参数：
  - `kinect_color_resolution / depth_mode / fps`
  - `flip_horizontal`, `depth_max_mm`
  - `auto_csv_on_open`, `gesture_recognizer_mode`, `gesture_model_path`

### 7.4 多线程通信

| 线程 → 线程 | 通信方式 |
|---|---|
| VideoThread → MainWindow | `Qt pyqtSignal(ProcessedFrame)` |
| LLMWorker → MainWindow | 三个信号：chunk / done / failed |
| MainWindow → VideoThread | 直接调方法（虚拟绘画控制） |
| VideoThread → Flask | 通过 `WebState` 单例（共享内存 + 锁） |
| MainWindow → Flask | 同上，通过 `WebState.push_event` |

---

## 8. 实验与评估（Experiments & Evaluation）

### 8.1 手势识别 ML 三分类器对比 ✅ 已完成

- **脚本**：`experiments/train_gesture_classifier.py`
- **数据集**：`data/outputs/labeled/gesture_20260522_181458.csv`，560 样本，7 类
- **特征**：63 维（21×3 减 wrist 归一化）
- **对比**：RandomForest / MLP / SVM(RBF)
- **图表**（在 `data/outputs/figures/`）：
  - `fig_classifier_comparison.png` —— 三模型准确率柱状图
  - `fig_classifier_<model>_cm.png` —— 各模型混淆矩阵
  - `fig_classifier_learning_curve.png` —— 学习曲线（样本量 vs 准确率）
  - `fig_classifier_noise_robustness.png` —— 加高斯噪声后的鲁棒性
- **典型结果**（实验数据请用最新 json 校对）：RandomForest 略胜，~92-95% 测试准确率

### 8.2 系统其他实验（部分待数据采集）

| 实验 | 状态 | 输出图 |
|---|---|---|
| 静态手势识别准确率 | ✅ | `fig_gesture_accuracy_bar`, `fig_gesture_confusion` |
| 动作识别准确率 | ⏳ 待采新数据 | （文件已生成但旧） |
| 距离 vs 准确率 | ⏳ 待采新数据 | （文件已生成但旧） |
| 距离测量误差直方图 | ⏳ 待采新数据 | `fig_distance_histogram` |
| 角度平滑前后对比 | ✅ | `fig_angle_smoothness` |
| FPS 直方图 | ✅ | `fig_fps_histogram` |

### 8.3 待做实验

- **跨用户泛化**：让队友录数据测当前模型 → 暴露过拟合
- **LSTM 时序动作识别**：与单帧规则法对比（待做完整版）

---

## 9. 关键设计决策（写在报告"讨论"章节）

### 9.1 为什么选 MediaPipe 而非 K4ABT？

| 项 | MediaPipe Pose | Azure K4ABT |
|---|---|---|
| 输入 | RGB 单帧 | 深度 + IR |
| 输出 | 33 关键点（2D 归一化 + 可见度） | 32 关键点（3D mm 坐标） |
| 精度 | 良好（适合大部分场景） | 高（深度直接得 3D） |
| 平台 | 全平台（CPU 也能跑） | Windows 限制 + 推荐 NVIDIA GPU |
| 模型大小 | < 10 MB | 约 200 MB |
| 社区 | 极其活跃 | 较小众 |
| 本系统取舍 | ✅ 选用 | 留作未来工作 |

> 报告里写一段："本系统采用 MediaPipe Pose..."

### 9.2 为什么 LLM 选硅基流动而非 DeepSeek 官方？

- 官方 API 经常 503，影响演示稳定性
- 硅基流动提供同款 DeepSeek-V3/V4 模型，但服务器集群独立，稳定性高得多
- 价格相近，新用户送 14 元额度

### 9.3 为什么手势识别保留规则法 + ML 双路径？

- 规则法：物理几何为基础，鲁棒、可解释
- ML 法：可被数据驱动迭代，泛化潜力高但依赖数据量
- 实测发现：当前 ML 模型在训练集分布外（新用户、新光照）反而不如规则法
- 系统提供运行时切换 (`gesture_recognizer_mode`)，便于消融实验

### 9.4 为什么用 LLM 做"理解层"而不是规则模板？

- 模板能做但僵硬："用户做出 OK 手势，距离 1.5 米"
- LLM 能整合多模态：动作 + 手势 + 距离 + 角度 → 一段连贯的场景描述
- LLM 还能做异常告警的人话化、活动日报的总结生成
- 对应当下"AIGC + 视觉"的研究/工程趋势

### 9.5 多线程架构选型

- 设备读取 + CV 推理放 QThread：避免阻塞 UI
- LLM 调用单独 QThread：网络延迟不卡帧
- Flask 守护线程：与 PyQt 完全解耦
- 状态共享用单例 + 锁，不引入消息队列以保持简单

---

## 10. 项目目录结构

```
e:/大三/CV/
├── main.py                  # 程序入口
├── config.py                # 配置中心
├── requirements.txt
├── README.md
├── PROGRESS.md              # 项目进度文档
│
├── core/                    # 算法核心
│   ├── camera.py
│   ├── hand_detector.py
│   ├── pose_detector.py
│   ├── gesture_recognizer.py
│   ├── ml_gesture_recognizer.py
│   ├── stable_recognizers.py
│   ├── action_recognizer.py
│   ├── dynamic_gesture.py
│   ├── distance_estimator.py
│   ├── fall_detector.py     ← 应用层
│   ├── virtual_paint.py
│   ├── activity_log.py      ← 应用层
│   ├── geometry.py
│   ├── smoothing.py
│   ├── depth_utils.py
│   ├── llm_client.py        ← 理解层
│   └── llm_understand.py    ← 理解层
│
├── ui/                      # PyQt5 UI
│   ├── main_window.py
│   ├── video_thread.py
│   └── llm_worker.py        ← 理解层
│
├── services/                # Web 服务（应用层）
│   ├── web_state.py         ← NEW
│   └── web_server.py        ← NEW
│
├── data/                    # 数据日志
│   ├── logger.py
│   └── outputs/
│       ├── logs/             # CSV
│       ├── videos/           # MP4
│       ├── snapshots/        # 截图
│       ├── alerts/           # 跌倒告警
│       ├── labeled/          # 训练数据
│       └── figures/          # 实验图
│
├── experiments/             # 训练 / 分析脚本
│   ├── train_gesture_classifier.py
│   └── analyze.py
│
├── tests/                   # 单元测试
│   └── test_gesture_trigger.py
│
├── tools/                   # 实用工具
│   ├── collect_labeled.py    # 数据采集 (静态手势)
│   ├── collect_sequence.py   # ⏳ 待写：LSTM 时序数据
│   └── test_llm.py           # LLM 连通性测试
│
└── models/
    └── gesture_rf.joblib    # 训练好的手势模型
```

**代码规模（粗估）**：
- 核心算法 `core/`: ~3000 行
- UI `ui/`: ~1500 行
- 服务 `services/`: ~400 行
- 实验 `experiments/`: ~600 行
- 总计 Python ~6000 行

---

## 11. 演示脚本（用于答辩 + 演示视频）

```
[① 开场 10s] 全身入画，UI 各模块满屏。
[② 静态手势 20s] 7 类手势依次秀，右上"动态手势"格保持空。
[③ 动态手势 25s] 张手→握拳(Grab) → 张手(Release)
                  → OK(Pinch) → 张手 → 食指(Point)
[④ 动作识别 20s] 8 类动作：站立/抬左手/右手/双手/左倾/右倾/下蹲/弯腰
[⑤ 距离测量 15s] 走位演示距离动态变化（1m → 3m → 0.5m 触发警告）
[⑥ 虚拟绘画 30s] 写"Hello"，Pinch 换色继续，Grab 清空再画笑脸
[⑦ AI 实时解读 20s] 摆个有趣姿势，按 Q，AI 流式描述
[⑧ 跌倒告警 30s] 假装慢慢倒下 → 红框 + 红字 + AI 告警文案
[⑨ AI 日报 25s] 按 D，AI 总结过去 3 分钟做了什么
[⑩ Web 远程 25s] 启动远程，掏出手机，扫码连接，显示同样画面 + 推送告警
```

**总长 ~3 分 30 秒**。

---

## 12. 报告章节建议（与本文档对应）

| 报告章节 | 对应本文档 | 备注 |
|---|---|---|
| 1. 引言 | 0 + 5.4 (LLM 必要性) | 老龄化 + 多模态需求 |
| 2. 相关工作 | 9.1 (K4ABT) + 9.4 (LLM) | 简短综述 |
| 3. 系统架构 | 1 | 含两图 |
| 4. 感知层 | 3 | Kinect + 数据流 |
| 5. 识别层 | 4.1 - 4.6 | 7 个子节 |
| 6. 跌倒检测算法 | 4.7 | **独立一章，详细写算法** |
| 7. 理解层（LLM） | 5 | **新章节，亮点** |
| 8. 应用层 | 6 | **新章节，包含 Web 远程** |
| 9. 工程实现 | 7 | UI / 日志 / 配置 / 多线程 |
| 10. 实验与评估 | 8 | 含混淆矩阵等图 |
| 11. 讨论 | 9 | 设计决策 |
| 12. 总结与展望 | 8.3 | LSTM 实做 + K4ABT 留白 |

---

## 13. 待补 / 待做（自我提醒）

- [ ] 动作 / 距离数据采集（回宿舍）
- [ ] LSTM 时序动作识别完整实现
- [ ] 课程报告（30+ 页）
- [ ] 答辩 PPT（15-20 页）
- [ ] 演示视频录制
- [ ] 跨用户泛化测试

---

## 14. 给 GPT 的写作提示

如果把本文档喂给 GPT 让它代写报告章节，建议：

1. **分章节让 GPT 写**：一次只让 GPT 写一章，避免输出超长
2. **明确目标长度**：例如"写第 6 章跌倒检测，约 4-5 页 1500 字"
3. **指定语气**：学术正式语 / 但避免堆砌空话
4. **引用规范**：让 GPT 留出 `[?]` 占位，你后期补 IEEE / GB7714 引用
5. **术语统一**：把第 0 节的"项目身份"和第 4-6 节的中文术语作为统一词表
6. **必须保留具体数字和文件名**：如 "560 样本"、"`core/fall_detector.py`"、"55° 阈值"
7. **不要让 GPT 编造没做过的实验**：明确告知"已做：X / Y / Z；未做：A / B"
8. **图表占位**：让 GPT 在合适处写 `（见图 5：xxx 混淆矩阵）`，你之后插图

工作流总评：
- ✅ 这种"详细 outline → GPT 扩写 → Overleaf 排版"方式效率很高
- ⚠️ 但 GPT 写的初稿**必须人工通读修改**，特别是技术细节（GPT 经常写错）
- ⚠️ Overleaf 模板挑选学校/课程允许的（建议用 IEEE 或学校官方模板）
- 💡 写完一章先粘到 Overleaf 看排版效果，避免最后一次性大返工
