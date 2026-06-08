"""
todo.py — 待办事项状态管理 + Nag 计数器（对应课程 s05）

职责：
  1. 维护全局 CURRENT_TODOS 列表（agent 的工作计划，内存级，进程重启清空）
  2. 提供 run_todo_write：todo_write 工具的处理函数，供 agent 调用
  3. 维护 nag 计数器，供 hooks.py 的 nag_hook 使用

Nag 机制原理（nag = "催促"）：
  agent 在每轮工具调用后自然倾向于继续干活，可能忘记维护 todo 列表。
  nag 计数器记录"自上次 todo_write 以来经过了多少轮有工具调用的循环"。
  超过阈值时，nag_hook 向 messages 注入 <reminder>，强制提醒模型更新 todo。

设计原则：
  - run_todo_write 是唯一的写入入口，保证 CURRENT_TODOS 状态一致
  - nag 计数器与 todo 模块放在一起，因为它们语义紧耦合
  - 状态纯内存，不持久化（s12 阶段会加文件持久化）
"""

import json

# ── 待办事项全局状态 ────────────────────────────────────────────
# 每项结构：{"content": "任务描述", "status": "pending|in_progress|done"}
# 所有模块都可以读取，只有 run_todo_write 负责写入
CURRENT_TODOS: list[dict] = []

# ── Todo 模式开关（默认关闭）─────────────────────────────────────
# agent 通过 todo_config 工具显式开启/关闭。
# 关闭时：nag_hook 静默，不催促；system prompt 不展示 todo 列表。
# 开启时：nag 机制激活，system prompt 实时展示 todo 进度。
# 默认 False：简单任务无需 todos 仪式，由 agent 自主决定是否开启。
_todos_enabled: bool = False

# ── Nag 计数器（私有，通过下方函数访问）──────────────────────────
# 记录：自上次 todo_write 以来，经过了多少轮"包含工具调用"的 LLM 循环
# 递增由 nag_hook 负责，重置由 run_todo_write / run_todo_config 负责
_rounds_since_todo: int = 0


# ═══════════════════════════════════════════════════════════════
#  输入校验（内部函数）
# ═══════════════════════════════════════════════════════════════

def _normalize_todos(todos) -> list[dict]:
    """
    校验并标准化 todos 输入，返回干净的 list[dict]。

    接受两种格式（模型调用工具时可能传两种格式之一）：
      1. list[dict]  — 最常见，直接使用
      2. str         — JSON 字符串，先解析

    校验规则（违反则抛出 ValueError，由 run_todo_write 捕获并返回错误信息给模型）：
      - 必须能解析为列表
      - 每项必须是 dict
      - 每项必须有 "content" 字段（str 类型）
      - 每项必须有 "status" 字段，且值是 pending / in_progress / done 之一
    """
    # 接受 JSON 字符串格式（模型偶尔会把列表序列化成字符串传来）
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError as e:
            raise ValueError(f"todos 参数不是合法 JSON：{e}")

    if not isinstance(todos, list):
        raise ValueError(f"todos 必须是列表，收到 {type(todos).__name__}")

    VALID_STATUSES = {"pending", "in_progress", "done"}
    normalized = []

    for i, item in enumerate(todos):
        if not isinstance(item, dict):
            raise ValueError(f"todos[{i}] 必须是字典，收到 {type(item).__name__}")
        if "content" not in item or not isinstance(item["content"], str):
            raise ValueError(f"todos[{i}] 缺少 'content' 字段或类型不是字符串")
        if "status" not in item:
            raise ValueError(f"todos[{i}] 缺少 'status' 字段")
        if item["status"] not in VALID_STATUSES:
            raise ValueError(
                f"todos[{i}].status={item['status']!r} 不合法，"
                f"合法值：{sorted(VALID_STATUSES)}"
            )
        # 只保留已知字段，丢弃模型可能附加的多余字段
        normalized.append({"content": item["content"], "status": item["status"]})

    return normalized


# ═══════════════════════════════════════════════════════════════
#  工具处理函数（agent 调用 todo_write 工具时触发）
# ═══════════════════════════════════════════════════════════════

def run_todo_write(todos: list) -> str:
    """
    更新全局待办事项列表，打印可视化摘要，重置 nag 计数器。

    这是 todo_write 工具的处理函数，供 agent 通过工具调用触发。
    主 agent 和子 agent 均可调用（子 agent 的进度也应该更新 todo）。

    副作用（按顺序）：
      1. 校验并标准化输入
      2. 更新 CURRENT_TODOS
      3. 打印带颜色图标的列表（用户可实时看到进度）
      4. 重置 nag 计数器（agent 主动更新了 todo，停止催促）

    返回：
      成功 → 简短统计字符串（反馈给模型，让它知道操作已成功）
      失败 → Error: 开头的错误描述
    """
    global CURRENT_TODOS

    try:
        normalized = _normalize_todos(todos)
    except ValueError as e:
        return f"Error: todo_write 参数校验失败 — {e}"

    CURRENT_TODOS = normalized

    # ── 打印可视化列表 ─────────────────────────────────────────
    # 用颜色 + 图标直观展示进度，帮助用户和开发者快速理解当前状态
    STATUS_ICONS = {
        "pending":     "○",   # 未开始（灰色）
        "in_progress": "◉",   # 进行中（蓝色）
        "done":        "✓",   # 已完成（绿色）
    }
    STATUS_COLORS = {
        "pending":     "\033[90m",   # 灰色
        "in_progress": "\033[34m",   # 蓝色
        "done":        "\033[32m",   # 绿色
    }
    RESET = "\033[0m"

    print(f"\n\033[1m📋 待办事项（共 {len(CURRENT_TODOS)} 项）：\033[0m")
    if not CURRENT_TODOS:
        print("  （空）")
    for item in CURRENT_TODOS:
        status = item["status"]
        icon  = STATUS_ICONS.get(status, "?")
        color = STATUS_COLORS.get(status, "")
        print(f"  {color}{icon} {item['content']}{RESET}")

    # ── 重置 nag 计数器 ────────────────────────────────────────
    # agent 主动更新了 todo，不需要再催促
    reset_nag_counter()

    done_count = sum(1 for t in CURRENT_TODOS if t["status"] == "done")
    return (
        f"待办事项已更新，共 {len(CURRENT_TODOS)} 项，"
        f"其中 {done_count} 项已完成。"
    )


# ═══════════════════════════════════════════════════════════════
#  Todo 模式开关 API
# ═══════════════════════════════════════════════════════════════

def is_todos_enabled() -> bool:
    """返回 todo 模式是否已开启。供 nag_hook 和 prompt.py 查询。"""
    return _todos_enabled


def run_todo_config(enabled: bool) -> str:
    """
    开启或关闭 todo 规划模式。这是 todo_config 工具的处理函数。

    enabled=True  → 开启：激活 nag 机制，system prompt 展示 todo 列表
    enabled=False → 关闭：nag 静默，清空 todo 列表，system prompt 不展示

    典型用法：
      任务开始时：todo_config(true)  → 然后 todo_write([...]) 制定计划
      任务完成时：todo_config(false) → 收尾，清理状态

    关闭时会同时清空 CURRENT_TODOS，避免旧列表残留影响下一个任务。
    """
    global _todos_enabled, CURRENT_TODOS

    _todos_enabled = enabled

    if enabled:
        # 开启：重置 nag 计数器，准备接受新的 todo 计划
        reset_nag_counter()
        print(f"\n\033[32m✓ Todo 规划模式已开启\033[0m  "
              f"\033[90m（nag 机制激活，请用 todo_write 制定计划）\033[0m")
        return "Todo 规划模式已开启。请调用 todo_write 制定任务计划。"
    else:
        # 关闭：清空列表 + 重置计数器，状态彻底归零
        CURRENT_TODOS = []
        reset_nag_counter()
        print(f"\n\033[90m○ Todo 规划模式已关闭\033[0m")
        return "Todo 规划模式已关闭，待办事项已清空。"


# ═══════════════════════════════════════════════════════════════
#  Nag 计数器 API（仅供 hooks.py 的 nag_hook 调用）
# ═══════════════════════════════════════════════════════════════

def get_nag_counter() -> int:
    """
    返回当前 nag 计数器的值。
    语义：自上次 todo_write 以来，经过了多少轮包含工具调用的 LLM 循环。
    """
    return _rounds_since_todo


def increment_nag_counter() -> None:
    """
    将 nag 计数器加 1。
    调用方：nag_hook（检测到一轮工具调用完成时递增）。
    """
    global _rounds_since_todo
    _rounds_since_todo += 1


def reset_nag_counter() -> None:
    """
    将 nag 计数器归零。
    调用方：run_todo_write（agent 主动更新了 todo，停止催促）。
    """
    global _rounds_since_todo
    _rounds_since_todo = 0
