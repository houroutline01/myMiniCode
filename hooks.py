"""
hooks.py — Hook 注册与触发系统（对应课程 s04 + s05）

核心思想：
  把"loop 里的扩展逻辑"从 agent_loop 中剥离，挂到事件点上。
  agent_loop 自身只做最核心的事：调 LLM → 执行工具 → 循环。
  其他一切（权限、日志、统计、催促）通过 hook 注入，互不耦合。

五个事件点（阶段三新增 PreLLMCall）：
  PreLLMCall       — 每次 LLM 调用之前触发（阶段三新增，Nag 机制挂在此处）
                     参数：messages（完整消息列表，可直接修改）
                     不可拦截：返回值被忽略；但 hook 可直接 append 到 messages
  UserPromptSubmit — 用户输入刚提交，LLM 还没收到
  PreToolUse       — 工具即将执行（可拦截：返回非 None = 阻止执行）
                     参数：tool_name, args[, interactive=True]
  PostToolUse      — 工具执行完毕（不可拦截：返回值被忽略）
  Stop             — LLM 决定不再调用工具，loop 即将退出
                     （可注入：返回字符串 = 作为 user 消息强制继续循环）

Nag 机制工作原理（对应课程 s05）：
  agent 在连续执行多轮工具调用时，容易专注干活而忘记更新 todo 列表。
  nag_hook 在每次 LLM 调用前扫描 messages：
    1. 若 messages[-1] 是工具结果（role=="tool"），说明刚完成了一轮工具调用
       → nag 计数器 +1
    2. 若计数器 >= NAG_THRESHOLD（3），向 messages 注入 <reminder> 用户消息
       → LLM 下一轮会调用 todo_write，run_todo_write 会自动重置计数器

  为什么 nag_hook 在 PreLLMCall 而不是 Stop？
    Stop 只在 LLM "决定停止" 时触发，但 agent 可能连续多轮调用工具而不停止。
    PreLLMCall 在每轮工具调用后都会触发，是检测计数的正确时机。

  为什么不在 agent_loop 里直接做计数？
    违反 hook 哲学：agent_loop 应该只做核心事务，扩展逻辑通过 hook 注入。
    hook 可以独立测试、替换、禁用，不需要修改 agent_loop。

PreToolUse 的 interactive 参数：
  主 agent：trigger_hooks("PreToolUse", tool_name, args)         → interactive 默认 True
  子 agent：trigger_hooks("PreToolUse", tool_name, args, False)  → interactive=False
  permission_hook 把 interactive 传给 check_permission，
  check_permission 在 Gate3 时用它决定"询问用户"还是"自动拒绝"。

注意：按课程原设计，hook 内部异常不捕获，直接向上传播。
这样在开发阶段能立刻暴露 bug，不会被静默吞掉。
"""

import json
from permissions import check_permission

# ── Hook 注册表 ────────────────────────────────────────────────
# 事件名 -> 回调函数列表（按注册顺序执行）
HOOKS: dict[str, list] = {
    "PreLLMCall":       [],  # s05 新增：每次 LLM 调用前触发（Nag 机制）
    "UserPromptSubmit": [],
    "PreToolUse":       [],
    "PostToolUse":      [],
    "Stop":             [],
}


def register_hook(event: str, callback) -> None:
    """将回调函数注册到指定事件。同一事件可注册多个回调，按顺序执行。"""
    if event not in HOOKS:
        raise ValueError(f"未知事件：{event}，可用事件：{list(HOOKS.keys())}")
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """
    触发指定事件的所有回调。

    遍历回调列表，将 *args 透传给每个回调：
      - 如果某回调返回非 None，立即停止遍历，将该返回值返回给调用方
      - 全部回调返回 None，则 trigger_hooks 也返回 None

    调用方根据返回值是否为 None 决定是否拦截/注入/继续。

    注意：PreLLMCall 的 hook 通过直接修改 messages 列表来注入消息，
          而不是通过返回值，所以 PreLLMCall 的返回值通常被 agent_loop 忽略。
    """
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


# ═══════════════════════════════════════════════════════════════
#  内置 Hook 实现
# ═══════════════════════════════════════════════════════════════

# ── PreLLMCall（s05 新增）────────────────────────────────────

# Nag 阈值：连续多少轮有工具调用后，触发 <reminder> 催促
NAG_THRESHOLD = 3


def nag_hook(messages: list):
    """
    PreLLMCall hook：检测工具调用轮次，必要时向 messages 注入 <reminder>。

    参数：
      messages — 完整消息列表（传引用，可直接 append 修改）

    工作逻辑：
      1. 检测本次 PreLLMCall 是否紧跟一轮工具调用：
           如果 messages[-1] 是 dict 且 role=="tool"，则刚完成一轮工具调用
           （注意：messages[-1] 也可能是 SDK 对象，用 isinstance 安全判断）
      2. 若刚完成工具调用：nag 计数器 +1
      3. 若计数器 >= NAG_THRESHOLD：注入 <reminder>，重置计数器

    为什么检查 messages[-1] 而不是单独维护状态？
      PreLLMCall 在每次 LLM 调用前触发，包括：
        - 第一次调用（messages[-1] 是用户消息）→ 不算工具轮
        - 工具执行后继续循环（messages[-1] 是工具结果）→ 算工具轮
        - Stop hook 注入继续后（messages[-1] 是注入的用户消息）→ 不算工具轮
      messages 已经包含了所有必要信息，不需要额外状态。

    返回 None（PreLLMCall 不使用返回值拦截，直接修改 messages）。
    """
    # 延迟导入，避免循环依赖（hooks.py 被 todo.py 间接依赖）
    from todo import get_nag_counter, increment_nag_counter, reset_nag_counter, is_todos_enabled

    # ── 开关检查：todo 模式关闭时完全静默 ─────────────────────
    # agent 调用 todo_config(false) 表示"此任务不需要 todo 规划"，
    # nag_hook 尊重这个决定，不做任何计数或注入。
    if not is_todos_enabled():
        print(f"\033[90m[HOOK] PreLLMCall（todo 模式关闭）\033[0m")
        return None

    # ── 检测是否刚完成一轮工具调用 ────────────────────────────
    # messages[-1] 是 dict 且 role=="tool" → 工具结果，说明刚完成工具调用
    # messages 可能为空（理论上不会，但防御性检查）
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "tool":
        increment_nag_counter()
        counter = get_nag_counter()
        print(f"\033[90m[HOOK] PreLLMCall: nag 计数 = {counter}/{NAG_THRESHOLD}\033[0m")

        # ── 计数器达到阈值，注入 reminder ──────────────────────
        if counter >= NAG_THRESHOLD:
            print(f"\033[33m[HOOK] ⏰ Nag 触发：已执行 {counter} 轮工具调用，催促更新 todo\033[0m")
            messages.append({
                "role": "user",
                "content": (
                    "<reminder>请更新你的待办事项列表（调用 todo_write 工具），"
                    "标记已完成的项目，并将当前正在进行的任务标记为 in_progress。</reminder>"
                ),
            })
            reset_nag_counter()
    else:
        # 不是工具调用结束（是第一次调用，或 Stop 注入后的继续），不计数
        print(f"\033[90m[HOOK] PreLLMCall\033[0m")

    return None  # PreLLMCall 不拦截，永远返回 None


# ── UserPromptSubmit ──────────────────────────────────────────

def context_inject_hook(query: str):
    """
    用户提交输入后触发。
    目前仅打印日志，未来可在此注入上下文（如 todo 状态、记忆摘要等）。
    返回 None = 不拦截，正常继续。
    """
    print(f"\033[90m[HOOK] UserPromptSubmit\033[0m")
    return None


# ── PreToolUse ────────────────────────────────────────────────

def permission_hook(tool_name: str, args: dict, interactive: bool = True):
    """
    工具执行前触发，调用 permissions.py 的三关权限系统。

    与 s03 的关系：
      s03 中 check_permission 是在 agent_loop 里直接调用的。
      s04 中把它包装成 hook，agent_loop 只知道"有个 PreToolUse 事件"，
      不再直接依赖 permissions 模块。

    interactive 参数（阶段三新增）：
      主 agent 调用：trigger_hooks("PreToolUse", tool_name, args)
                     → interactive 默认 True → 危险操作弹出用户确认框
      子 agent 调用：trigger_hooks("PreToolUse", tool_name, args, False)
                     → interactive=False    → 危险操作自动拒绝，不打断用户

    返回 None      = 允许执行
    返回字符串     = 拒绝执行，字符串作为工具结果告知模型
    """
    allowed = check_permission(tool_name, args, interactive)
    if not allowed:
        return "Permission denied."
    return None


def log_hook(tool_name: str, args: dict, interactive: bool = True):
    """
    工具执行前触发，打印工具调用日志（在权限通过后）。
    注意：注册在 permission_hook 之后，权限拒绝时不会执行到这里。
    interactive 参数：接受但不使用（签名保持与 permission_hook 一致）。
    返回 None = 不拦截。
    """
    preview = json.dumps(args, ensure_ascii=False)[:80]
    print(f"\033[90m[HOOK] PreToolUse: {tool_name}({preview})\033[0m")
    return None


# ── PostToolUse ───────────────────────────────────────────────

def large_output_hook(tool_name: str, args: dict, output: str):
    """
    工具执行后触发，对超大输出发出警告。
    PostToolUse 的返回值被调用方忽略（不可拦截），纯副作用。
    阈值 10000 字符：超过此长度可能撑大 context，未来接 s08 压缩模块。
    """
    if len(str(output)) > 10000:
        print(f"\033[33m[HOOK] ⚠ 大输出：{tool_name} 返回了 {len(str(output))} 字符，"
              f"注意 context 占用\033[0m")
    return None


# ── Stop ──────────────────────────────────────────────────────

def summary_hook(messages: list):
    """
    LLM 决定停止时触发，统计本轮 tool 调用次数并打印 todo 状态摘要。

    OpenAI 格式中，工具结果是 role=="tool" 的独立 message（dict）。
    Anthropic 格式中是 type=="tool_result" 的 content block，计数方式不同。

    返回 None = 正常退出循环（不注入消息）。
    Nag 机制在 PreLLMCall 处理，而非 Stop，所以这里不需要注入 reminder。
    """
    tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )

    # 顺便打印 todo 状态，方便用户看到当前进度
    from todo import CURRENT_TODOS
    if CURRENT_TODOS:
        done = sum(1 for t in CURRENT_TODOS if t["status"] == "done")
        total = len(CURRENT_TODOS)
        print(f"\033[90m[HOOK] Stop: 执行 {tool_count} 次工具调用，"
              f"todo 进度 {done}/{total}\033[0m")
    else:
        print(f"\033[90m[HOOK] Stop: 执行 {tool_count} 次工具调用\033[0m")

    return None


# ═══════════════════════════════════════════════════════════════
#  注册内置 Hook（顺序即执行顺序）
# ═══════════════════════════════════════════════════════════════

register_hook("PreLLMCall",       nag_hook)           # s05 新增：Nag 催促机制
register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse",       permission_hook)    # 先检查权限
register_hook("PreToolUse",       log_hook)           # 权限通过后再打印日志
register_hook("PostToolUse",      large_output_hook)
register_hook("Stop",             summary_hook)
