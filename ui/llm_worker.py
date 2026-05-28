"""
LLM 异步调用 Worker。

设计：
    - QThread 在后台执行 LLM API 调用，不阻塞 UI 主线程
    - 通过信号回传：chunk(流式片段) / done(完整结果) / failed(错误)
    - 一次性使用：每次调用新建一个 Worker 实例
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from core.llm_client import LLMClient


class LLMWorker(QThread):
    """后台执行单次 LLM 调用。"""

    chunk = pyqtSignal(str)       # 流式片段（如使用流式）
    done = pyqtSignal(str)        # 完整结果
    failed = pyqtSignal(str)      # 错误信息

    def __init__(
        self,
        client: LLMClient,
        prompt_builder: Callable[[], str],
        system: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.5,
        stream: bool = True,
        parent=None,
    ):
        """
        参数：
            client          : LLMClient 实例
            prompt_builder  : 在 worker 启动后调用，返回最新的 prompt 字符串
                              （用 lambda 避免捕获过期数据）
            stream          : 是否流式输出（流式更早出字，体验更好）
        """
        super().__init__(parent)
        self._client = client
        self._prompt_builder = prompt_builder
        self._system = system
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._stream = bool(stream)

    def run(self):
        try:
            prompt = self._prompt_builder()
        except Exception as e:
            self.failed.emit(f"prompt 构建失败: {e}")
            return

        if not self._client.available:
            self.failed.emit(self._client.last_error or "LLM 不可用")
            return

        try:
            if self._stream:
                buffer: list[str] = []
                for piece in self._client.chat_stream(
                    prompt=prompt,
                    system=self._system,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                ):
                    buffer.append(piece)
                    self.chunk.emit(piece)
                self.done.emit("".join(buffer).strip())
            else:
                text = self._client.chat(
                    prompt=prompt,
                    system=self._system,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                self.done.emit(text)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
