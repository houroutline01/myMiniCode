"""
hooks.py — Hook 注册与触发系统（对应课程 s04）

核心思想：
  把"loop 里的扩展逻辑"从 agent_loop 中剥离，挂到事件点上。
  agent_loop 自身只做最核心的事：调 LLM → 执行工具 → 循环。
  其他一切（权限、日志、统计）通过 hook 注入，互不耦合。

四个事件点：
  UserPromptSubmit — 用户输入刚提交，LLM 还没收到
  PreToolUse       — 工具即将执行（可拦截：返回非 None = 阻止执行）
  PostToolUse      — 工具执行完毕（不可拦截：返回值被忽略）
  Stop             — LLM 决定不再调用工具，loop 即将退出
                     （可注入：返回字符串 = 作为 user 消息强制继续循环）

拦截机制（PreToolUse / UserPromptSubmit / Stop）：
  trigger_hooks 遍历回调列表，第一个返回非 None 的回调即视为"拦截"，
  后续回调不再执行，拦截原因字符串作为工具结果返回给模型。

注意：按课程原设计，hook 内部异常不捕获，直接向上传播。
这样在开发阶段能立刻暴露 bug，不会被静默吞掉。
"""

import json
from permissions import check_permission

# ── Hook 注册表 ────────────────────────────────────────────────
# 事件名 -> 回调函数列表（按注册顺序执行）
HOOKS: dict[str, list] = {
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

    遍历回调列表，将 *args 传入每个回调：
      - 如果某回调返回非 None，立即停止遍历，将该返回值返回给调用方
      - 全部回调返回 None，则 trigger_hooks 也返回 None

    调用方根据返回值是否为 None 决定是否拦截/注入。
    """
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


# ═══════════════════════════════════════════════════════════════
#  内置 Hook 实现
# ═══════════════════════════════════════════════════════════════

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

def permission_hook(tool_name: str, args: dict):
    """
    工具执行前触发，调用 permissions.py 的三关权限系统。

    与 s03 的关系：
      s03 中 check_permission 是在 agent_loop 里直接调用的。
      s04 中把它包装成 hook，agent_loop 只知道"有个 PreToolUse 事件"，
      不再直接依赖 permissions 模块。

    返回 None      = 允许执行
    返回字符串     = 拒绝执行，字符串作为工具结果告知模型
    """
    allowed = check_permission(tool_name, args)
    if not allowed:
        return "Permission denied."
    return None


def log_hook(tool_name: str, args: dict):
    """
    工具执行前触发，打印工具调用日志（在权限通过后）。
    注意：注册在 permission_hook 之后，权限拒绝时不会执行到这里。
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
    LLM 决定停止时触发，统计本轮 tool 调用次数。

    OpenAI 格式中，工具结果是 role=="tool" 的独立 message（dict）。
    Anthropic 格式中是 type=="tool_result" 的 content block，计数方式不同。

    返回 None = 正常退出循环（不注入消息）。
    未来 s05 的 Nag 机制会在这里返回 <reminder> 字符串，强制模型继续。
    """
    tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )
    print(f"\033[90m[HOOK] Stop: 本轮共执行 {tool_count} 次工具调用\033[0m")
    return None


# ═══════════════════════════════════════════════════════════════
#  注册内置 Hook（顺序即执行顺序，permission 必须在 log 之前）
# ═══════════════════════════════════════════════════════════════

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse",       permission_hook)   # 先检查权限
register_hook("PreToolUse",       log_hook)          # 通过后再打印日志
register_hook("PostToolUse",      large_output_hook)
register_hook("Stop",             summary_hook)
