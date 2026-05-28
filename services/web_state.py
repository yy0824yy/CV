"""
Web 服务的共享状态。

设计：
    - 主程序线程持续向其推送 JPEG 帧 + 事件
    - Flask 端从这里读取最新帧 + 订阅事件流
    - 完全线程安全；不直接依赖 PyQt 或 OpenCV，便于后续替换

事件类型（type 字段）：
    'fall'         : 跌倒告警
    'fall_text'    : 跌倒告警的 LLM 文案（独立事件，便于流式更新）
    'report'       : AI 活动日报生成完成
    'describe'     : 单次场景解读
    'status'       : 周期性状态心跳（可选）

每个事件都自带 timestamp / time_str。
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Iterator, List, Optional


class WebState:
    """单例线程安全的视频流 + 事件状态。"""

    def __init__(self, max_event_history: int = 200,
                 max_subscribers: int = 32):
        # 最新帧（JPEG bytes）
        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._frame_cond = threading.Condition(self._frame_lock)
        self._frame_seq: int = 0
        # 事件历史（用于新订阅者回放最近事件）
        self._events: List[dict] = []
        self._events_lock = threading.Lock()
        self._max_history = int(max_event_history)
        # 订阅者队列（每个 Flask 客户端 SSE 连接一个 queue）
        self._subscribers: List[queue.Queue] = []
        self._subs_lock = threading.Lock()
        self._max_subscribers = int(max_subscribers)
        # 运行时元信息
        self._meta: dict = {}
        self._meta_lock = threading.Lock()

    # --------------- 帧 ---------------
    def push_frame(self, jpeg_bytes: bytes) -> None:
        """主程序线程调用：推一帧 JPEG。"""
        with self._frame_cond:
            self._latest_jpeg = jpeg_bytes
            self._frame_seq += 1
            self._frame_cond.notify_all()

    def get_latest_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def wait_new_frame(self, last_seq: int,
                       timeout: float = 1.0) -> tuple[Optional[bytes], int]:
        """阻塞直到有新帧或超时。返回 (jpeg_bytes, current_seq)。"""
        with self._frame_cond:
            if self._frame_seq <= last_seq:
                self._frame_cond.wait(timeout=timeout)
            return self._latest_jpeg, self._frame_seq

    # --------------- 事件 ---------------
    def push_event(self, event: dict) -> None:
        """主程序调用：广播一个事件给所有订阅者 + 加入历史。"""
        ev = dict(event)
        ev.setdefault("timestamp", time.time())
        ev.setdefault("time_str", time.strftime(
            "%H:%M:%S", time.localtime(ev["timestamp"])))
        with self._events_lock:
            self._events.append(ev)
            if len(self._events) > self._max_history:
                self._events = self._events[-self._max_history:]
        with self._subs_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def recent_events(self, since_seconds: float = 600.0) -> List[dict]:
        cutoff = time.time() - since_seconds
        with self._events_lock:
            return [e for e in self._events if e["timestamp"] >= cutoff]

    # --------------- 订阅 SSE ---------------
    def subscribe(self) -> Iterator[dict]:
        """生成器：阻塞地产出后续事件，连接断开时自动清理。"""
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._subs_lock:
            if len(self._subscribers) >= self._max_subscribers:
                # 拒绝过多订阅者
                return
            self._subscribers.append(q)
        try:
            while True:
                try:
                    ev = q.get(timeout=15.0)
                    yield ev
                except queue.Empty:
                    # 心跳：让客户端连接保持活
                    yield {"type": "ping",
                           "timestamp": time.time(),
                           "time_str": time.strftime("%H:%M:%S")}
        finally:
            with self._subs_lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    # --------------- 元信息 ---------------
    def set_meta(self, **kwargs) -> None:
        with self._meta_lock:
            self._meta.update(kwargs)

    def get_meta(self) -> dict:
        with self._meta_lock:
            return dict(self._meta)

    @property
    def subscriber_count(self) -> int:
        with self._subs_lock:
            return len(self._subscribers)


# ---------------- 单例 ----------------
_GLOBAL_STATE: Optional[WebState] = None
_GLOBAL_LOCK = threading.Lock()


def get_state() -> WebState:
    """全局单例 WebState 访问入口。"""
    global _GLOBAL_STATE
    with _GLOBAL_LOCK:
        if _GLOBAL_STATE is None:
            _GLOBAL_STATE = WebState()
        return _GLOBAL_STATE
