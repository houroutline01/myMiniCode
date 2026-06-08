"""
agent.py — Agent Loop（对应课程 s01 + s02 + s03 的核心逻辑）

职责：
  - 维护 messages 列表（对话历史）
  - 调用 LLM API，解析响应
  - 识别工具调用，经权限检查后执行，将结果喂回模型
  - 循环直到模型不再调用工具

与课程的关键差异（API 格式）：
  课程使用 Anthropic SDK，本项目使用 OpenAI 兼容格式（DeepSeek）。
  两者消息格式对比：

  Anthropic:
    response.stop_reason == "tool_use"
    response.content 是 content block 列表
    block.type == "tool_use", block.id, block.name, block.input
    工具结果格式: {"type": "tool_result", "tool_use_id": ..., "content": ...}
    放进 user role 的 content 列表

  OpenAI（本项目）:
    response.choices[0].finish_reason == "tool_calls"
    response.choices[0].message.tool_calls 是工具调用列表
    tc.id, tc.function.name, tc.function.arguments（JSON 字符串）
    工具结果格式: {"role": "tool", "tool_call_id": ..., "content": ...}
    每个结果是独立的 message，直接 append 到 messages
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOL_HANDLERS, TOOL_DEFINITIONS
from permissions import check_permission

# ── 初始化 ────────────────────────────────────────────────────
load_dotenv(override=True)

# DeepSeek 兼容 OpenAI 格式，只需替换 base_url
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"

# System prompt：告诉模型它是编码助手，要用工具解决问题
SYSTEM = (
    f"你是一个编码助手，工作目录为 {os.getcwd()}。"
    "优先使用工具完成任务，不要只说不做。"
    "所有破坏性操作都会经过权限系统，无需你自行判断是否安全。"
)


# ═══════════════════════════════════════════════════════════════
#  核心：Agent Loop
# ═══════════════════════════════════════════════════════════════

def agent_loop(messages: list):
    """
    驱动一轮用户请求的完整处理流程。

    参数 messages：调用方传入完整对话历史（含本轮 user message）。
    函数直接修改 messages（append），调用方读取最后几条即可获取结果。

    循环逻辑（s01 的核心模式）：
      1. 调用 LLM，拿到 response
      2. 把 assistant 消息追加到 messages
      3. 如果没有工具调用（finish_reason != "tool_calls"），退出循环
      4. 遍历所有工具调用，权限检查后执行，每个结果单独 append 到 messages
      5. 回到步骤 1，把工具结果喂给模型继续处理
    """
    while True:
        # ── 调用 LLM ──────────────────────────────────────────
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",   # 让模型自己决定是否调用工具
            max_tokens=8000,
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # 将 assistant 回复加入历史
        # OpenAI SDK 的 message 对象可以直接 append，后续请求时 SDK 会自动序列化
        messages.append(assistant_message)

        # ── 判断是否需要调用工具 ──────────────────────────────
        # finish_reason == "tool_calls"：模型想调用工具，继续循环
        # finish_reason == "stop"：模型认为任务完成，退出循环
        if choice.finish_reason != "tool_calls":
            return

        # ── 遍历工具调用（模型可能一次调用多个工具）────────────
        for tc in assistant_message.tool_calls:
            tool_name = tc.function.name
            # arguments 是 JSON 字符串，需要解析为 dict
            # 临时加在 agent.py 第 101 行之前
            print(f"[DEBUG] raw arguments: {tc.function.arguments[:300]}")
            args = json.loads(tc.function.arguments)

            print(f"\n\033[36m▶ {tool_name}\033[0m", end="  ")
            # 打印参数摘要（截断避免刷屏）
            args_preview = json.dumps(args, ensure_ascii=False)[:100]
            print(f"\033[90m{args_preview}\033[0m")

            # ── 权限检查（s03）────────────────────────────────
            # check_permission 内部走三关流水线，返回 bool
            if not check_permission(tool_name, args):
                result = "Permission denied."
            else:
                # ── 工具执行（s02）────────────────────────────
                handler = TOOL_HANDLERS.get(tool_name)
                if handler is None:
                    result = f"Error: 未知工具 {tool_name}"
                else:
                    try:
                        result = handler(**args)
                    except Exception as e:
                        result = f"Error: {e}"

            # 打印结果摘要（前 200 字符）
            print(f"\033[90m{str(result)[:200]}\033[0m")

            # ── 将工具结果追加到 messages ─────────────────────
            # OpenAI 格式：每个工具结果是独立的 {"role": "tool", ...} 消息
            # tool_call_id 必须与请求中的 tc.id 对应，模型靠此匹配结果
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

        # 所有工具执行完毕，带着结果进入下一轮循环，继续调用 LLM
