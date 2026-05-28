"""
统一 LLM 客户端封装。

支持四家 OpenAI-兼容厂商，按环境变量优先级自动选择：
    1. 硅基流动 (SILICONFLOW_API_KEY)  -> deepseek-ai/DeepSeek-V3   [推荐]
    2. 智谱     (ZHIPU_API_KEY)        -> glm-4-flash
    3. DeepSeek (DEEPSEEK_API_KEY)     -> deepseek-chat
    4. 通义千问 (DASHSCOPE_API_KEY)    -> qwen-turbo

设计要点：
    - 完全兼容 OpenAI Python SDK
    - 自动重试：503 / 网络抖动 时退避重试
    - 线程安全：可在 QThread 中直接调用
    - 简洁接口：chat(prompt) / chat_messages(messages)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class LLMProvider:
    name: str
    env_key: str
    base_url: str
    model: str


_PROVIDERS: List[LLMProvider] = [
    LLMProvider("硅基流动",  "SILICONFLOW_API_KEY",
                "https://api.siliconflow.cn/v1",
                # 可选模型：
                #   deepseek-ai/DeepSeek-V4-Pro    最强，推理模型（限时 2.5 折）
                #   deepseek-ai/DeepSeek-V4-Flash  更快更便宜
                #   deepseek-ai/DeepSeek-V3        通用，便宜
                #   Qwen/Qwen2.5-7B-Instruct      免费
                "deepseek-ai/DeepSeek-V4-Flash"),
    LLMProvider("智谱",      "ZHIPU_API_KEY",
                "https://open.bigmodel.cn/api/paas/v4",
                "glm-4-flash"),
    LLMProvider("DeepSeek",  "DEEPSEEK_API_KEY",
                "https://api.deepseek.com",
                "deepseek-chat"),
    LLMProvider("通义千问",  "DASHSCOPE_API_KEY",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen-turbo"),
]


def detect_provider() -> Optional[LLMProvider]:
    """按优先级返回第一个有 Key 的厂商。"""
    for p in _PROVIDERS:
        if os.getenv(p.env_key):
            return p
    return None


class LLMClient:
    """
    统一封装：自动选择厂商 + 重试 + 流式输出。
    """

    # 哪些异常类型触发重试（暂时型）
    _RETRY_STATUS = (502, 503, 504, 524, 429)

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ):
        self.provider = provider or detect_provider()
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self._client = None
        self._last_error: Optional[str] = None

        if self.provider is None:
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv(self.provider.env_key),
                base_url=self.provider.base_url,
                timeout=self.timeout,
            )
        except Exception as e:
            self._last_error = f"初始化失败: {e}"
            self._client = None

    # ---------------- 状态 ----------------
    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def info(self) -> str:
        if self.provider is None:
            return "未检测到任何 LLM API Key"
        return f"{self.provider.name}  ({self.provider.model})"

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    # ---------------- 同步聊天 ----------------
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.6,
    ) -> str:
        """单轮：传一段 prompt，返回字符串。"""
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_messages(messages, max_tokens=max_tokens,
                                  temperature=temperature)

    def chat_messages(
        self,
        messages: List[dict],
        max_tokens: int = 400,
        temperature: float = 0.6,
    ) -> str:
        """多轮：完整 messages 列表。"""
        if not self.available:
            raise RuntimeError(
                self._last_error or "LLM 客户端不可用（未设置 API Key）"
            )

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.provider.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                content = resp.choices[0].message.content or ""
                self._last_error = None
                return content.strip()
            except Exception as e:
                last_exc = e
                if not self._is_retryable(e) or attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff ** attempt)

        msg = f"{type(last_exc).__name__}: {last_exc}"
        self._last_error = msg
        raise RuntimeError(msg) from last_exc

    # ---------------- 流式聊天（生成器） ----------------
    def chat_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.6,
    ) -> Iterable[str]:
        """流式：逐 chunk 产出文本片段。"""
        if not self.available:
            raise RuntimeError(
                self._last_error or "LLM 客户端不可用"
            )

        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self._client.chat.completions.create(
            model=self.provider.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    # ---------------- 内部 ----------------
    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        # openai SDK 错误都带 status_code
        code = getattr(exc, "status_code", None)
        if code in LLMClient._RETRY_STATUS:
            return True
        # 网络层错误
        name = type(exc).__name__
        if name in ("APITimeoutError", "APIConnectionError",
                    "RateLimitError"):
            return True
        return False
