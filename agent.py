"""
agent.py — Agent Loop（s01-s04 + s05 + s06 + s08 + s10 + s11）

阶段四新增（相对阶段三）：
  s08: run_compaction_pipeline — 每轮 LLM 调用前运行四层压缩管道
       compact 工具 — agent 可主动触发 L4 摘要压缩
  s11: RecoveryState + with_retry — 三条错误恢复路径：
         Path 1: finish_reason=="length" → 升级 max_tokens 8K→64K → continuation
         Path 2: prompt_too_long        → reactive_compact（紧急压缩）
         Path 3: 429/529               → 指数退避（with_retry 内部处理）

阶段三相对阶段二的变化（保留注释）：
  s05: todo_write / todo_config，PreLLMCall hook（Nag 机制）
  s06: task 工具，spawn_subagent

阶段二相对阶段一的变化（保留注释）：
  s04: Hook 事件系统（PreToolUse / PostToolUse / Stop）
  s10: 动态 system prompt（get_system_prompt + update_context）

与课程的关键差异（OpenAI vs Anthropic 格式）：
  - finish_reason: "tool_calls" / "length" vs stop_reason: "tool_use" / "max_tokens"
  - 工具结果格式：role=="tool" dict vs tool_result content block
  - SDK 错误类型：openai.RateLimitError vs anthropic.RateLimitError
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOL_HANDLERS as _BASE_HANDLERS, TOOL_DEFINITIONS
from todo import run_todo_write, run_todo_config
from subagent import spawn_subagent
from hooks import trigger_hooks
from prompt import get_system_prompt, update_context
from compact import run_compaction_pipeline, compact_history, reactive_compact
from recovery import (
    RecoveryState,
    with_retry,
    is_context_too_long,
    CONTINUATION_PROMPT,
    MAX_RECOVERY_RETRIES,
    ESCALATED_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
)

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"

# ── 主 Agent 完整工具分发表 ────────────────────────────────────
# compact 工具在 agent_loop 里特殊处理，不经过此表
TOOL_HANDLERS = {
    **_BASE_HANDLERS,
    "todo_write":  run_todo_write,
    "todo_config": run_todo_config,
    "task":        spawn_subagent,
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

    循环逻辑（阶段四更新）：
      0. s08 压缩管道：L3→L1→L2→（可能 L4）
      1. s05 PreLLMCall hook（Nag 检查）
      2. s10 动态 system prompt（带缓存）
      3. s11 LLM 调用（with_retry 处理 429/529，外层 try/except 处理 Path 1/2）
      4. finish_reason=="length" → Path 1 max_tokens 升级/continuation
      5. finish_reason!="tool_calls" → Stop hook → 退出或继续
      6. 遍历工具调用：compact 特殊处理，其余走权限+执行+hook
      7. 更新 context
    """
    # RecoveryState 每次 agent_loop 调用重建（每个用户轮次独立）
    state = RecoveryState(model=MODEL)

    while True:
        # ── s08: 压缩管道（每轮 LLM 调用前执行）─────────────────
        # L3→L1→L2 零 API 调用，超阈值才触发 L4（1 次 API 调用）
        messages[:] = run_compaction_pipeline(messages)

        # ── s05: PreLLMCall hook（Nag 机制）──────────────────────
        trigger_hooks("PreLLMCall", messages)

        # ── s10: 动态组装 system prompt ───────────────────────────
        system = get_system_prompt(context)

        # ── s11: LLM 调用（with_retry 内部处理 429/529）─────────
        try:
            response = with_retry(
                lambda: client.chat.completions.create(
                    model=state.current_model,
                    messages=[{"role": "system", "content": system}] + messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=state.max_tokens,
                ),
                state,
            )
        except Exception as e:
            # Path 2: context 太长 → 紧急压缩，仅重试一次
            if is_context_too_long(e) and not state.has_attempted_reactive_compact:
                print(f"\033[31m[RECOVERY] Context 太长，触发紧急压缩\033[0m")
                messages[:] = reactive_compact(messages)
                state.has_attempted_reactive_compact = True
                continue
            # 不可恢复的错误
            print(f"\033[31m[RECOVERY] 不可恢复：{type(e).__name__}: {str(e)[:100]}\033[0m")
            return

        choice = response.choices[0]
        assistant_message = choice.message

        # ── s11 Path 1: max_tokens 截断恢复 ──────────────────────
        if choice.finish_reason == "length":
            if not state.has_escalated:
                # 第一次截断：升级 max_tokens，不保存截断输出，直接重试
                state.max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f"\033[33m[RECOVERY] 输出被截断，升级 max_tokens: "
                      f"{DEFAULT_MAX_TOKENS} → {ESCALATED_MAX_TOKENS}\033[0m")
                continue
            # 64K 仍被截断：保存截断输出 + 注入 continuation prompt
            messages.append(assistant_message)
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(f"\033[33m[RECOVERY] 仍被截断，注入 continuation prompt "
                      f"（{state.recovery_count}/{MAX_RECOVERY_RETRIES}）\033[0m")
                continue
            print("\033[31m[RECOVERY] 已达 continuation 上限，停止\033[0m")
            return

        # ── 正常完成：追加 assistant 消息 ────────────────────────
        messages.append(assistant_message)

        # ── 判断是否继续循环 ────────────────────────────────────
        if choice.finish_reason != "tool_calls":
            # s04: Stop hook（返回字符串则注入继续，返回 None 则退出）
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
                print(f"\n\033[31m✗ {tool_name} 参数解析失败（JSON 截断）：{e}\033[0m")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: 工具参数解析失败，请缩短内容后重试。({e})",
                })
                continue

            print(f"\n\033[36m▶ {tool_name}\033[0m  "
                  f"\033[90m{json.dumps(args, ensure_ascii=False)[:100]}\033[0m")

            # ── s08: compact 工具特殊处理 ────────────────────────
            # compact 直接修改 messages，不走 TOOL_HANDLERS，不需要权限检查
            # 压缩后 break，让 while 重新用压缩后的 context 调 LLM
            if tool_name == "compact":
                messages[:] = compact_history(messages)
                break

            # ── s04: PreToolUse hook（含权限检查）──────────────
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
            trigger_hooks("PostToolUse", tool_name, args, str(result))

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

        # ── s10: 每轮工具执行后更新 context（compact 后也更新）──
        context = update_context(messages)
