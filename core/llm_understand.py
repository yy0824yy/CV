"""
LLM 语义理解层。

把视觉识别结果（手势 / 动作 / 距离 / 历史）转成自然语言描述。

主要任务：
    - describe_scene(frame)        -> 实时场景一句话解读
    - summarize_history(records)   -> 一段时间内的活动总结
    - explain_alert(alert)         -> 异常告警的自然语言描述

设计原则：
    - 把结构化识别结果"翻译"成 prompt（不要让模型瞎猜）
    - 给 LLM 必要的上下文（距离 / 手势含义 / 已知动作清单）
    - 控制输出长度（实时场景要短，日报可以长）
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from core.llm_client import LLMClient
    from ui.video_thread import ProcessedFrame


# 动作英文 -> 中文
ACTION_CN = {
    "Standing":         "站立",
    "Raise_Left_Hand":  "举左手",
    "Raise_Right_Hand": "举右手",
    "Raise_Both_Hands": "举双手",
    "Lean_Left":        "向左倾",
    "Lean_Right":       "向右倾",
    "Squat":            "下蹲",
    "Bend_Forward":     "前倾/弯腰",
    "No_Person":        "未检测到人",
    "Unknown":          "无法判定",
}

# 静态手势英文 -> 中文
GESTURE_CN = {
    "Open_Palm": "张开手掌",
    "Fist":      "握拳",
    "OK":        "OK 手势",
    "Like":      "点赞",
    "Number_1":  "伸出食指",
    "Victory":   "比 V",
    "Three":     "三指",
    "Unknown":   "未识别",
    "":          "无",
    "No_Hand":   "无",
}

# 动态手势英文 -> 中文
DYNAMIC_CN = {
    "Grab":    "抓取（张手→握拳）",
    "Release": "释放（握拳→张手）",
    "Pinch":   "捏合（张手→OK）",
    "Point":   "指向（张手→食指）",
}


# ============================================================
# Prompt 模板
# ============================================================

# 通用系统人设
SYSTEM_PROMPT_BASE = (
    "你是一个智能居家陪护视觉系统的语义解读助手。"
    "用户基于 Azure Kinect DK 摄像头建立了人体感知系统，"
    "可以实时识别手势、人体动作、距离、关节角度等结构化信息。"
    "你的任务是把这些冰冷的识别结果，翻译成有温度、有洞察的中文场景解读。\n\n"
    "输出要求：\n"
    "1) 中文回答，平实自然，不使用 Markdown 标题、列表符号或代码块；\n"
    "2) 整体长度约 80-160 字，可以包含 3-4 个层次的内容：\n"
    "   ① 用户当前在做什么（基于动作 + 手势的行为描述）；\n"
    "   ② 用户的状态/姿态如何（是否放松、紧张、专注、疲惫等可能的状态）；\n"
    "   ③ 可能的意图或情境推测（结合距离、动态手势综合判断）；\n"
    "   ④ 一句简短的陪护建议或友好回应（语气温和，不夸张）。\n"
    "3) 不要罗列原始数据，要把数据『翻译成场景』；\n"
    "4) 信息不足时使用『可能 / 似乎 / 看起来』等词做可能性推测；\n"
    "5) 行文连贯，避免机械地分点回答，写成一段流畅的话。"
)


# ============================================================
# 工具：把 ProcessedFrame 序列化成 prompt 输入
# ============================================================

def _format_distance(d: Optional[float]) -> str:
    if d is None:
        return "未知"
    return f"{d:.2f} 米"


def _frame_to_dict(frame: "ProcessedFrame") -> dict:
    """从 ProcessedFrame 提取结构化信息。"""
    info: dict = {
        "person_distance_m": frame.person_distance_m,
        "fps": round(frame.fps, 1) if frame.fps else 0.0,
        "too_close": frame.too_close,
        "too_far": frame.too_far,
    }

    # 手部
    hands_info = []
    for h, g in zip(frame.hands, frame.gestures):
        hands_info.append({
            "side": "左手" if h.handedness == "Left" else "右手",
            "gesture": GESTURE_CN.get(g.name, g.name),
        })
    info["hands"] = hands_info

    # 动作
    if frame.action is not None and frame.action.valid:
        info["action"] = ACTION_CN.get(
            frame.action.primary, frame.action.primary)
    else:
        info["action"] = None

    # 动态手势事件（如果近期发生）
    if frame.dynamic_gesture is not None:
        info["dynamic_gesture"] = DYNAMIC_CN.get(
            frame.dynamic_gesture.name, frame.dynamic_gesture.name)
    else:
        info["dynamic_gesture"] = None

    # 关节角度（精简）
    if frame.angles:
        info["angles_deg"] = {
            k: round(v, 1) for k, v in frame.angles.items()
        }

    return info


def build_scene_prompt(frame: "ProcessedFrame") -> str:
    """把当前帧的识别信息转成 prompt 文本。"""
    d = _frame_to_dict(frame)

    lines = ["请基于下列实时视觉感知数据，对当前场景做一段连贯、有层次的解读。"]
    lines.append("")
    lines.append("【感知数据】")

    if d["action"]:
        lines.append(f"- 主体动作：{d['action']}")
    if d["hands"]:
        for h in d["hands"]:
            lines.append(f"- {h['side']}：{h['gesture']}")
    else:
        lines.append("- 双手：未在画面中检测到手部")

    if d["dynamic_gesture"]:
        lines.append(f"- 刚刚做出动态手势：{d['dynamic_gesture']}")

    lines.append(f"- 距离镜头：{_format_distance(d['person_distance_m'])}")

    if d["too_close"]:
        lines.append("- 状态：距离过近（< 0.6 米）")
    elif d["too_far"]:
        lines.append("- 状态：距离过远（> 3 米）")

    if "angles_deg" in d and d["angles_deg"]:
        ang = d["angles_deg"]
        ang_str = ", ".join(f"{k}={v}°" for k, v in ang.items())
        lines.append(f"- 关节角度：{ang_str}")

    lines.append("")
    lines.append(
        "请按照系统人设中规定的层次（行为 → 状态 → 意图推测 → 陪护建议），"
        "写成一段约 80-160 字的连贯文字。注意：不要罗列原始数据，要"
        "把数据翻译成场景；语气自然温和。"
    )
    return "\n".join(lines)


# ============================================================
# 高层 API
# ============================================================

def describe_scene(client: "LLMClient", frame: "ProcessedFrame",
                   max_tokens: int = 120) -> str:
    """
    根据当前帧让 LLM 生成场景描述（一句话）。
    """
    prompt = build_scene_prompt(frame)
    return client.chat(
        prompt=prompt,
        system=SYSTEM_PROMPT_BASE,
        max_tokens=max_tokens,
        temperature=0.5,
    )


def summarize_history(client: "LLMClient",
                      events: List[str],
                      duration_sec: float,
                      max_tokens: int = 300) -> str:
    """
    根据一段时间累积的事件列表生成活动总结。
    events 是已经预处理好的中文事件描述列表。
    """
    if not events:
        return "这段时间内没有显著的活动事件。"

    bullet = "\n".join(f"- {e}" for e in events[-50:])  # 最多取最近 50 条
    minutes = duration_sec / 60
    prompt = (
        f"用户在过去 {minutes:.1f} 分钟内的活动事件如下：\n"
        f"{bullet}\n\n"
        "请生成一段 3-5 句话的活动总结，提到：\n"
        "1) 用户大致在做什么、状态如何；\n"
        "2) 是否有异常或值得关注的行为；\n"
        "3) 对用户的简短关怀建议。"
    )
    return client.chat(
        prompt=prompt,
        system=SYSTEM_PROMPT_BASE,
        max_tokens=max_tokens,
        temperature=0.5,
    )


def build_fall_alert_prompt(context: dict) -> str:
    """
    构造跌倒告警 prompt。
    context 期望键：
        timestamp_str, torso_angle_deg, head_drop, duration_sec,
        person_distance_m, last_action
    """
    lines = ["系统刚检测到一次疑似跌倒事件，请生成一段简洁的告警通知。"]
    lines.append("")
    lines.append("【事件信息】")
    lines.append(f"- 触发时间：{context.get('timestamp_str', '未知')}")
    lines.append(f"- 躯干倾角：{context.get('torso_angle_deg', '?')}°"
                 "（站立 ≈ 0°，平躺 ≈ 90°）")
    lines.append(f"- 头部下落幅度：{context.get('head_drop', '?')}（归一化）")
    lines.append(
        f"- 已倒地确认时长：{context.get('duration_sec', '?')} 秒"
    )
    if context.get("person_distance_m") is not None:
        lines.append(f"- 距离镜头：{context['person_distance_m']:.2f} 米")
    if context.get("last_action"):
        lines.append(f"- 跌倒前最后动作：{context['last_action']}")
    lines.append("")
    lines.append(
        "请生成一段约 80-120 字的告警通知，要求：\n"
        "1) 第一句明确说明发生跌倒事件；\n"
        "2) 给出关键现场信息（时间 / 距离 / 严重程度）；\n"
        "3) 给出建议的应对措施（联系家属 / 立即查看 / 拨打急救电话）；\n"
        "4) 语气克制专业，不要使用感叹号过多，不要 Markdown 标题。"
    )
    return "\n".join(lines)


def explain_alert(client: "LLMClient",
                  alert_type: str,
                  context: dict,
                  max_tokens: int = 200) -> str:
    """
    生成异常告警的自然语言描述（用于跌倒等场景）。
    alert_type: 'fall' | 'too_close' | 'idle_long' | 'leave' ...
    context:    任意上下文字典（距离、动作、时间等）
    """
    ctx_lines = "\n".join(f"- {k}: {v}" for k, v in context.items())
    prompt = (
        f"系统检测到异常事件：{alert_type}\n"
        f"现场信息：\n{ctx_lines}\n\n"
        "请生成一条简洁明确的告警通知（约 50-80 字），"
        "包含事件类型、发生时间/地点暗示、严重程度建议、可采取的操作。"
        "语气克制专业，避免夸张。"
    )
    return client.chat(
        prompt=prompt,
        system=SYSTEM_PROMPT_BASE,
        max_tokens=max_tokens,
        temperature=0.4,
    )
