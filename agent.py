"""
agent.py — Agent Loop（s01-s04 + s10 的核心逻辑）

阶段二相对阶段一的变化：
  s04: check_permission() 直接调用 → trigger_hooks("PreToolUse", ...) 事件触发
       PostToolUse 和 Stop 事件点新增
  s10: 硬编码 SYSTEM 字符串 → get_system_prompt(context) 动态组装
       agent_loop 新增 context 参数，每轮工具执行后更新 context

与课程的关键差异（OpenAI vs Anthropic 格式）：见阶段一注释，本文件不重复。
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOL_HANDLERS, TOOL_DEFINITIONS
from hooks import trigger_hooks
from prompt import get_system_prompt, update_context

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"


# ═══════════════════════════════════════════════════════════════
#  核心：Agent Loop
# ═══════════════════════════════════════════════════════════════

def agent_loop(messages: list, context: dict):
    """
    驱动一轮用户请求的完整处理流程。

    参数：
      messages — 完整对话历史（含本轮 user message），函数直接修改此列表
      context  — 当前运行状态（工作目录、记忆等），用于组装 system prompt

    循环逻辑：
      1. 从 context 组装 system prompt（带缓存，context 不变时直接复用）
      2. 调用 LLM
      3. 把 assistant 消息追加到 messages
      4. finish_reason != "tool_calls" → 触发 Stop hook → 退出或继续
      5. 遍历工具调用：PreToolUse hook → 执行 → PostToolUse hook → 结果入 messages
      6. 更新 context，回到步骤 1
    """
    while True:
        # ── s10: 动态组装 system prompt ───────────────────────
        # 每轮循环重新获取，context 变化时会重新组装，否则命中缓存
        system = get_system_prompt(context)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=8000,
        )

        choice = response.choices[0]
        assistant_message = choice.message
        messages.append(assistant_message)

        # ── 判断是否继续循环 ───────────────────────────────────
        if choice.finish_reason != "tool_calls":
            # s04: Stop hook
            # 返回 None → 正常退出
            # 返回字符串 → 注入为 user 消息，强制继续循环（s05 Nag 机制会用到）
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        # ── 遍历工具调用 ────────────────────────────────────────
        for tc in assistant_message.tool_calls:
            tool_name = tc.function.name

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                # DeepSeek 偶发：长内容时 arguments JSON 被截断
                print(f"\n\033[31m✗ {tool_name} 参数解析失败（JSON 截断）：{e}\033[0m")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: 工具参数解析失败，请缩短内容后重试。({e})",
                })
                continue

            print(f"\n\033[36m▶ {tool_name}\033[0m  "
                  f"\033[90m{json.dumps(args, ensure_ascii=False)[:100]}\033[0m")

            # ── s04: PreToolUse hook（含权限检查）─────────────
            # trigger_hooks 返回非 None = 被拦截，字符串即拦截原因
            blocked = trigger_hooks("PreToolUse", tool_name, args)
            if blocked:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(blocked),
                })
                continue

            # ── 工具执行 ────────────────────────────────────────
            handler = TOOL_HANDLERS.get(tool_name)
            if handler is None:
                result = f"Error: 未知工具 {tool_name}"
            else:
                try:
                    result = handler(**args)
                except Exception as e:
                    result = f"Error: {e}"

            print(f"\033[90m{str(result)[:200]}\033[0m")

            # ── s04: PostToolUse hook ──────────────────────────
            # 返回值被忽略，纯副作用（日志、统计、警告等）
            trigger_hooks("PostToolUse", tool_name, args, str(result))

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

        # ── s10: 每轮工具执行后更新 context ────────────────────
        # 此时文件系统可能已变化（write_file 等），重新派生状态
        context = update_context(messages)
