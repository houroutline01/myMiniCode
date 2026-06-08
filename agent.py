"""
agent.py — Agent Loop（s01-s04 + s05 + s06 + s10）

阶段三相对阶段二的变化：
  s05: 新增 todo_write 工具处理函数（来自 todo.py）
       PreLLMCall hook 在每轮 LLM 调用前触发（Nag 机制）
  s06: 新增 task 工具处理函数（来自 subagent.py::spawn_subagent）

阶段二相对阶段一的变化（保留注释）：
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

# tools.py 的基础工具（bash/read/write/edit/glob）
from tools import TOOL_HANDLERS as _BASE_HANDLERS, TOOL_DEFINITIONS
# s05: todo_write / todo_config 工具实现
from todo import run_todo_write, run_todo_config
# s06: task 工具实现（启动子 agent）
from subagent import spawn_subagent
from hooks import trigger_hooks
from prompt import get_system_prompt, update_context

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"

# ── 主 Agent 完整工具分发表 ────────────────────────────────────
# 在 tools.py 的基础工具之上，追加 todo_write 和 task
# 子 agent（subagent.py）有自己的 SUB_TOOL_HANDLERS（不含 task，防递归）
TOOL_HANDLERS = {
    **_BASE_HANDLERS,                   # bash, read_file, write_file, edit_file, glob
    "todo_write":  run_todo_write,      # s05: 更新待办事项列表
    "todo_config": run_todo_config,     # s05: 开启/关闭 todo 规划模式
    "task":        spawn_subagent,      # s06: 委托子 agent 执行子任务
}


# ═══════════════════════════════════════════════════════════════
#  核心：Agent Loop
# ═══════════════════════════════════════════════════════════════

def agent_loop(messages: list, context: dict):
    """
    驱动一轮用户请求的完整处理流程。

    参数：
      messages — 完整对话历史（含本轮 user message），函数直接修改此列表
      context  — 当前运行状态（工作目录、记忆、todo 等），用于组装 system prompt

    循环逻辑（阶段三更新）：
      0. PreLLMCall hook（s05 新增）：Nag 检查，必要时注入 <reminder>
      1. 从 context 组装 system prompt（带缓存，context 不变时直接复用）
      2. 调用 LLM
      3. 把 assistant 消息追加到 messages
      4. finish_reason != "tool_calls" → 触发 Stop hook → 退出或继续
      5. 遍历工具调用：PreToolUse hook → 执行 → PostToolUse hook → 结果入 messages
      6. 更新 context，回到步骤 0
    """
    while True:
        # ── s05: PreLLMCall hook（Nag 机制）──────────────────────
        # 在每次调用 LLM 之前触发，nag_hook 检测工具调用轮次
        # 若计数 >= 阈值，nag_hook 会直接向 messages 注入 <reminder>
        # 返回值被忽略（PreLLMCall 不拦截流程，只做副作用）
        trigger_hooks("PreLLMCall", messages)

        # ── s10: 动态组装 system prompt ───────────────────────────
        # 每轮循环重新获取，context 变化时会重新组装，否则命中缓存
        # context 现在包含 todos 字段，所以 todo 更新后 system prompt 也会更新
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

        # ── 判断是否继续循环 ────────────────────────────────────
        if choice.finish_reason != "tool_calls":
            # s04: Stop hook
            # 返回 None     → 正常退出
            # 返回字符串    → 注入为 user 消息，强制继续循环
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        # ── 遍历工具调用 ─────────────────────────────────────────
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

            # ── s04: PreToolUse hook（含权限检查）──────────────
            # 主 agent 调用时不传 interactive，默认 True（会弹用户确认框）
            # 子 agent 在 subagent.py 中传 False（自动拒绝危险操作）
            blocked = trigger_hooks("PreToolUse", tool_name, args)
            if blocked:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(blocked),
                })
                continue

            # ── 工具执行 ───────────────────────────────────────
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
            # 返回值被忽略，纯副作用（日志、统计、大输出警告等）
            trigger_hooks("PostToolUse", tool_name, args, str(result))

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

        # ── s10: 每轮工具执行后更新 context ─────────────────────
        # 此时 todo 列表可能已变化（todo_write），文件系统也可能变化
        # update_context 读取最新状态，缓存失效，下一轮 system prompt 重新组装
        context = update_context(messages)
