"""
LLM API 连通性测试脚本（支持多厂商）。

支持三家厂商，按优先级自动选择已配置 Key 的那家：
    1. 智谱  GLM-4 Flash    : 环境变量 ZHIPU_API_KEY        (推荐 - 免费)
    2. DeepSeek             : 环境变量 DEEPSEEK_API_KEY
    3. 通义千问 Qwen        : 环境变量 DASHSCOPE_API_KEY

用法：
    1. 设置任意一个环境变量：
       $env:ZHIPU_API_KEY = "xxxxxxxx.xxxxxxxx"
       或
       $env:DEEPSEEK_API_KEY = "sk-..."
    2. 运行：
       python tools\test_llm.py

    指定厂商（可选）：
       python tools\test_llm.py --provider zhipu
       python tools\test_llm.py --provider deepseek
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass


@dataclass
class Provider:
    name: str
    env_key: str
    base_url: str
    model: str


PROVIDERS = {
    "siliconflow": Provider(
        name="\u7845\u57fa\u6d41\u52a8 (DeepSeek-V3)",
        env_key="SILICONFLOW_API_KEY",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V3",
    ),
    "zhipu": Provider(
        name="\u667a\u8c31 GLM-4 Flash",
        env_key="ZHIPU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
    ),
    "deepseek": Provider(
        name="DeepSeek \u5b98\u65b9",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    ),
    "qwen": Provider(
        name="\u901a\u4e49\u5343\u95ee Qwen",
        env_key="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
    ),
}


def auto_select() -> "Provider | None":
    """\u6309\u4f18\u5148\u7ea7\u9009\u62e9\u5df2\u914d\u7f6e Key \u7684\u5382\u5546\u3002"""
    for key in ("siliconflow", "zhipu", "deepseek", "qwen"):
        p = PROVIDERS[key]
        if os.getenv(p.env_key):
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()),
                        help="\u6307\u5b9a\u5382\u5546\uff1bnot set \u65f6\u81ea\u52a8\u9009\u62e9")
    args = parser.parse_args()

    if args.provider:
        provider = PROVIDERS[args.provider]
        key = os.getenv(provider.env_key)
        if not key:
            print(f"[ERROR] \u73af\u5883\u53d8\u91cf {provider.env_key} \u672a\u8bbe\u7f6e")
            return 1
    else:
        provider = auto_select()
        if provider is None:
            print("[ERROR] \u6ca1\u8bfb\u5230\u4efb\u4f55\u53ef\u7528\u7684 API Key\u3002\u8bf7\u8bbe\u7f6e\u4ee5\u4e0b\u4e4b\u4e00\uff1a")
            for p in PROVIDERS.values():
                print(f"   $env:{p.env_key} = ...   ({p.name})")
            return 1
        key = os.getenv(provider.env_key)

    print(f"[INFO] \u5382\u5546   : {provider.name}")
    print(f"[INFO] base_url: {provider.base_url}")
    print(f"[INFO] model   : {provider.model}")
    print(f"[INFO] key     : {key[:8]}...{key[-4:]}  (len={len(key)})")

    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai \u5305\u672a\u5b89\u88c5\u3002\u8bf7\u5148\u8fd0\u884c pip install openai")
        return 1

    client = OpenAI(api_key=key, base_url=provider.base_url, timeout=30.0)

    print("\n[STEP 1] \u53d1\u8d77 ping \u8bf7\u6c42 (max_tokens=10)...")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=provider.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10,
            stream=False,
        )
    except Exception as e:
        print(f"[ERROR] \u8bf7\u6c42\u5931\u8d25 ({time.time()-t0:.1f}s): {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    print(f"[OK] \u54cd\u5e94\u8017\u65f6 {dt:.2f} s")
    print(f"     content: {resp.choices[0].message.content!r}")
    print(f"     usage  : {resp.usage}")

    print("\n[STEP 2] \u6d41\u5f0f\u8f93\u51fa\u6d4b\u8bd5...")
    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=provider.model,
            messages=[{"role": "user", "content": "\u7528\u4e00\u53e5\u8bdd\u4ecb\u7ecd\u4f60\u81ea\u5df1"}],
            max_tokens=80,
            stream=True,
        )
        first_t = None
        for chunk in stream:
            if first_t is None:
                first_t = time.time() - t0
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
        print()
        print(f"\n[OK] \u9996 token: {first_t:.2f}s  \u603b\u8017\u65f6: {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"[ERROR] \u6d41\u5f0f\u5931\u8d25: {type(e).__name__}: {e}")
        return 1

    print(f"\n[SUCCESS] {provider.name} API \u8fde\u901a\u6b63\u5e38\uff0c\u53ef\u4ee5\u63a5\u5165\u9879\u76ee\u3002")
    return 0


if __name__ == "__main__":
    sys.exit(main())
