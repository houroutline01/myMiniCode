"""
subagent.py — 子 Agent（对应课程 s06）

核心思想：
  主 agent 把独立子任务"外包"给子 agent。子 agent 是一个完全独立的 agent loop：
    - 独立消息历史，不继承父 context（子任务不需要知道父对话细节）
    - 独立工具集：没有 task 工具，防止子 agent 再创建子子 agent（无限递归）
    - 非交互式权限模式：危险操作自动拒绝，不打断用户
    - 30 轮上限：安全措施，防止失控

权限设计（完整分析见 permissions.py 注释）：
  ┌──────────────────────┬──────────────────┬──────────────────┐
  │ 操作类型              │ 主 Agent          │ 子 Agent          │
  ├──────────────────────┼──────────────────┼──────────────────┤
  │ 读文件/glob           │ 直接放行          │ 直接放行          │
  │ 写文件/编辑文件        │ 直接放行          │ 直接放行          │
  │ 黑名单 bash（rm -rf/) │ 硬拒绝            │ 硬拒绝            │
  │ 危险 bash（rm/wget等）│ 询问用户          │ 自动拒绝          │
  │ session 白名单内      │ 直接放行          │ 直接放行 ← 继承   │
  └──────────────────────┴──────────────────┴──────────────────┘

  session 白名单自动继承的原因：
    _SESSION_ALLOWLIST 是 permissions.py 的模块级变量，
    Python 进程内所有代码共享同一份，子 agent 天然继承，无需额外处理。

设计决策记录：
  Q: 为什么子 agent 没有 task 工具？
  A: 防止递归。主 agent 可以派生子 agent，子 agent 只能干活，不能再派生。

  Q: 为什么子 agent 使用固定 system prompt 而不是动态组装？
  A: 子 agent 的任务是聚焦执行，不需要 memory/todos 等状态感知。
     动态 system prompt 留给主 agent 的复杂感知需求。

  Q: 为什么返回文本后丢弃整个消息历史？
  A: 子任务结果通过返回值传回主 agent，工具调用历史对主 agent 没价值，
     丢弃可以避免 context 膨胀（s08 context 管理时更重要）。
"""

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

# 工具实现函数（与主 agent 共用同一份实现）
from tools import (
    run_bash, run_read, run_write, run_edit, run_glob,
    WORKDIR, TOOL_DEFINITIONS as _MAIN_TOOL_DEFINITIONS,
)
# todo_write 工具（子 agent 也可以更新 todo，记录子任务进度）
from todo import run_todo_write
# hook 系统（权限检查、日志等复用主 agent 的 hook）
from hooks import trigger_hooks

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-chat"

# 子 agent 最大轮次：超过后强制终止，避免失控
MAX_TURNS = 30


# ═══════════════════════════════════════════════════════════════
#  子 Agent 系统提示（固定字符串，不动态组装）
# ═══════════════════════════════════════════════════════════════

SUB_SYSTEM = (
    "你是一个编码子任务执行器，专注完成指派的单一编码任务。\n\n"
    f"工作目录：{WORKDIR}\n\n"
    "【运行环境：Windows + cmd.exe】\n"
    "- 使用 python，不使用 python3\n"
    "- 使用 where，不使用 which\n"
    "- echo \"5\" 会输出带引号的 \"5\"，测试交互程序时直接编写 test_xxx.py\n"
    "- 不支持 <<< heredoc\n"
    "- 路径分隔符使用 \\\\ 或 /\n\n"
    "规则：\n"
    "- 优先使用工具完成任务，不要只说不做\n"
    "- 危险的 bash 操作（如 rm、wget、pip install）会被系统自动拒绝，"
    "遇到拒绝时改用更安全的替代方式\n"
    "- 完成后输出简洁的结果摘要（一段话，不超过 200 字）"
)


# ═══════════════════════════════════════════════════════════════
#  子 Agent 工具集（没有 "task" 工具，防止递归）
# ═══════════════════════════════════════════════════════════════

# 工具分发表：从主工具表复用所有实现，唯独排除 task
# todo_write 包含在内：子 agent 可以更新 todo 记录进度
SUB_TOOL_HANDLERS: dict = {
    "bash":       run_bash,
    "read_file":  run_read,
    "write_file": run_write,
    "edit_file":  run_edit,
    "glob":       run_glob,
    "todo_write": run_todo_write,
    # "task" 故意缺失 → 子 agent 无法再派生子 agent
}

# Schema 列表：从主 agent 的 TOOL_DEFINITIONS 过滤掉 task
# 这样保证 schema 和实现始终同步，只需在一个地方维护 schema
SUB_TOOL_DEFINITIONS: list[dict] = [
    tool for tool in _MAIN_TOOL_DEFINITIONS
    if tool["function"]["name"] != "task"
]


# ═══════════════════════════════════════════════════════════════
#  子 Agent 主函数（供主 agent 通过 task 工具调用）
# ═══════════════════════════════════════════════════════════════

def spawn_subagent(description: str) -> str:
    """
    创建并运行一个子 agent，完成独立的编码子任务。

    参数：
      description — 子任务的完整描述（主 agent 传过来，通常已足够详细）

    返回：
      子 agent 完成后的最终文本回复（整个消息历史随后丢弃）
      如果超出 MAX_TURNS，返回带错误前缀的说明字符串

    关键设计：
      1. messages 从头开始 — 不继承父 agent 的对话历史
      2. trigger_hooks("PreToolUse", ..., False) — interactive=False，危险操作自动拒绝
      3. 30 轮上限 — 超出时强制返回，避免 API 费用失控
      4. 不调用 PreLLMCall hook — 子 agent 自己的循环不触发 nag 机制
         （nag 机制只适用于主 agent，追踪主 agent 的进度）
    """
    print(f"\n\033[35m╔══ 子 Agent 启动 ════════════════════════╗\033[0m")
    print(f"\033[35m║ 任务：{description[:58]}\033[0m")
    print(f"\033[35m╚═════════════════════════════════════════╝\033[0m")

    # 全新消息历史，仅包含任务描述作为第一条用户消息
    messages: list = [{"role": "user", "content": description}]

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUB_SYSTEM}] + messages,
            tools=SUB_TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=8000,
        )

        choice = response.choices[0]
        assistant_message = choice.message
        messages.append(assistant_message)

        # ── 子 agent 决定停止 ──────────────────────────────────
        # finish_reason 不是 tool_calls，说明子任务已完成（或无法继续）
        if choice.finish_reason != "tool_calls":
            final_text = assistant_message.content or "(子 agent 未返回文本)"
            print(f"\n\033[35m╔══ 子 Agent 完成（第 {turn + 1} 轮）══════════════╗\033[0m")
            # 预览最多 100 字
            preview = final_text[:100].replace("\n", " ")
            print(f"\033[35m║ {preview}\033[0m")
            print(f"\033[35m╚═════════════════════════════════════════╝\033[0m")
            return final_text

        # ── 遍历工具调用 ───────────────────────────────────────
        for tc in assistant_message.tool_calls:
            tool_name = tc.function.name

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                # DeepSeek 偶发：长内容时 arguments JSON 被截断
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: 工具参数解析失败（{e}），请缩短内容后重试。",
                })
                continue

            print(f"\n\033[35m[子 Agent·轮{turn+1}]\033[0m "
                  f"\033[36m▶ {tool_name}\033[0m  "
                  f"\033[90m{json.dumps(args, ensure_ascii=False)[:80]}\033[0m")

            # ── PreToolUse hook（interactive=False）─────────────
            # 关键：第三个参数 False 表示非交互式模式
            # permission_hook 会把 False 传给 check_permission，
            # check_permission 在 Gate3 时不询问用户，直接拒绝
            blocked = trigger_hooks("PreToolUse", tool_name, args, False)
            if blocked:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(blocked),
                })
                continue

            # ── 执行工具 ──────────────────────────────────────
            handler = SUB_TOOL_HANDLERS.get(tool_name)
            if handler is None:
                result = f"Error: 未知工具 {tool_name}（子 agent 不支持此工具）"
            else:
                try:
                    result = handler(**args)
                except Exception as e:
                    result = f"Error: {e}"

            print(f"\033[90m{str(result)[:200]}\033[0m")

            # ── PostToolUse hook（纯副作用，不影响流程）─────────
            trigger_hooks("PostToolUse", tool_name, args, str(result))

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    # ── 超出轮次上限，强制终止 ─────────────────────────────────
    print(f"\n\033[31m[子 Agent] ⚠ 已达最大轮次 {MAX_TURNS}，强制终止\033[0m")
    return (
        f"Error: 子 agent 在 {MAX_TURNS} 轮内未完成任务，已强制终止。"
        "请尝试进一步拆解子任务，或直接描述更具体的操作步骤。"
    )
