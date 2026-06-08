"""
prompt.py — 动态 System Prompt 组装与缓存（对应课程 s10 + s05）

核心思想：
  System prompt 不再是一个硬编码字符串，而是由多个"碎片（section）"
  按当前运行状态动态拼接而成。这样：
  1. 每个功能模块只管理自己的 section，互不干扰
  2. 没激活的功能不占用 token（如 memory 在 s09 接入前不出现）
  3. 未来新增功能只需加一个条件分支

缓存机制：
  每次调 LLM 前都需要 system prompt，但 context 变化时才需要重新组装。
  用 json.dumps(context) 作为 key，context 不变就直接返回上次的字符串。
  避免每轮都做字符串拼接（虽然开销小，但养成好习惯）。

context 更新时机（对应课程 s10 的 update_context）：
  每轮工具调用完成后重新计算 context，system prompt 随之更新。
  阶段三新增：
    - todos 字段：从 todo.CURRENT_TODOS 读取，todo_write 调用后自动更新
  s09（跨会话记忆）接入后，memories 字段会真正动态化。

context 字段说明：
  workspace — 工作目录路径（几乎不变）
  memories  — MEMORY.md 内容（s09 接入前为空，s09 后会动态变化）
  todos     — 当前待办事项列表摘要（s05 接入，todo_write 后立即更新）
"""

import json
from pathlib import Path

WORKDIR = Path.cwd()
MEMORY_DIR  = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"


# ═══════════════════════════════════════════════════════════════
#  Prompt 碎片字典
#  key = section 名称，value = 文本内容（或空字符串占位）
# ═══════════════════════════════════════════════════════════════

PROMPT_SECTIONS: dict[str, str] = {

    # 始终加载：身份定义
    "identity": (
        "你是一个编码助手。优先使用工具完成任务，不要只说不做。\n"
        "所有破坏性操作都会经过权限系统，无需你自行判断是否安全。\n\n"
        "【Todo 规划模式（默认关闭，由你决定是否开启）】\n"
        "- 简单单步任务：直接执行，无需 todo。\n"
        "- 多步骤复杂任务：任务开始时调用 todo_config(true) 开启规划模式，"
        "再用 todo_write 制定计划，执行中及时更新状态，完成后调用 todo_config(false) 收尾。"
    ),

    # 始终加载：工作目录
    "workspace": f"工作目录：{WORKDIR}",

    # 始终加载：Windows 运行环境提示（优化点：模型训练数据以 Linux 为主，需明确纠正）
    "platform": (
        "【运行环境：Windows + cmd.exe】\n"
        "- 使用 python，不使用 python3\n"
        "- 使用 where，不使用 which\n"
        "- echo \"5\" 会输出带引号的 \"5\"，"
          "测试交互程序时直接编写 test_xxx.py，用 subprocess 模拟输入\n"
        "- 不支持 <<< heredoc\n"
        "- 路径分隔符使用 \\\\ 或 /"
    ),

    # 条件加载：当前待办事项（s05 接入，todo_write 后更新）
    # 占位符，实际内容在 assemble_system_prompt 里从 context["todos"] 读取
    "todos": "",

    # 条件加载：跨会话记忆（s09 阶段接入，目前为空占位）
    # 当 .memory/MEMORY.md 存在时，内容会被注入此处
    "memory": "",
}


# ═══════════════════════════════════════════════════════════════
#  组装函数
# ═══════════════════════════════════════════════════════════════

def assemble_system_prompt(context: dict) -> str:
    """
    根据当前 context 选择并拼接 section，返回完整 system prompt 字符串。

    哪些 section 始终加载，哪些条件加载，都在这里决定。
    新增功能只需在此加一个条件分支，不需要改其他地方。
    """
    sections = []

    # 始终加载的三个基础 section
    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["workspace"])
    sections.append(PROMPT_SECTIONS["platform"])

    # 条件加载：有待办事项时注入（s05 接入，todo_write 调用后更新）
    # agent 需要在 system prompt 里看到自己的 todo 列表，才能追踪进度
    todos = context.get("todos", "")
    if todos:
        sections.append(f"当前待办事项：\n{todos}")

    # 条件加载：有记忆内容时注入（s09 阶段激活）
    memories = context.get("memories", "")
    if memories:
        sections.append(f"以下是与当前任务相关的历史记忆：\n{memories}")

    return "\n\n".join(sections)


# ── 进程内缓存（模块级变量，进程结束清空） ────────────────────
_cached_key: str | None = None
_cached_prompt: str | None = None


def get_system_prompt(context: dict) -> str:
    """
    带缓存的 system prompt 获取入口。

    使用 json.dumps(context, sort_keys=True) 作为缓存 key，
    而不是 hash()，原因：
      - Python 的 hash() 对 dict/list 不可用
      - hash() 有进程级随机化（Python 3.3+），重启后结果不同
      - json.dumps 保证相同内容每次产生相同字符串

    context 不变 → 直接返回缓存，不重新拼接字符串
    context 变化 → 重新组装，更新缓存
    """
    global _cached_key, _cached_prompt

    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)

    if key == _cached_key and _cached_prompt is not None:
        return _cached_prompt

    # context 变化，重新组装
    _cached_key = key
    _cached_prompt = assemble_system_prompt(context)

    # 打印已加载的 section 列表，方便调试
    loaded = ["identity", "workspace", "platform"]
    if context.get("todos"):
        loaded.append("todos")
    if context.get("memories"):
        loaded.append("memory")
    print(f"\033[90m[PROMPT] 重新组装：{', '.join(loaded)}\033[0m")

    return _cached_prompt


# ═══════════════════════════════════════════════════════════════
#  Context 派生函数
# ═══════════════════════════════════════════════════════════════

def _format_todos(todos: list) -> str:
    """
    将 CURRENT_TODOS 列表格式化为适合放进 system prompt 的纯文本。

    格式示例：
      ○ [pending]     分析需求文档
      ◉ [in_progress] 实现 parse_config 函数
      ✓ [done]        编写单元测试
    """
    if not todos:
        return ""
    STATUS_ICONS = {
        "pending":     "○",
        "in_progress": "◉",
        "done":        "✓",
    }
    lines = []
    for t in todos:
        icon = STATUS_ICONS.get(t["status"], "?")
        # 对齐：status 固定宽度，方便阅读
        lines.append(f"  {icon} [{t['status']:<11}] {t['content']}")
    return "\n".join(lines)


def update_context(messages: list) -> dict:
    """
    从当前真实状态派生 context dict。

    原则：context 只反映"现在是什么状态"，不存历史。
    调用时机：每轮工具调用完成后，由 agent_loop 调用。

    当前字段：
      workspace — 工作目录路径（几乎不变）
      todos     — 当前 todo 列表格式化文本（s05 接入，todo_write 后变化）
      memories  — MEMORY.md 内容（s09 接入后会动态变化）

    messages 参数保留但未使用：
      预留给 s09 的记忆提取逻辑——届时会扫描消息历史，
      用 LLM 提取关键信息写入 MEMORY.md，然后 memories 字段才会动态变化。

    未来扩展字段：
      tasks     — 当前任务状态（s12 持久化任务）
    """
    # ── 读取 todo 列表（s05）──────────────────────────────────
    # 延迟导入避免循环依赖（todo.py 间接依赖 hooks.py）
    from todo import CURRENT_TODOS
    todos = _format_todos(CURRENT_TODOS)

    # ── 读取跨会话记忆（s09 接入后才有内容）────────────────────
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            memories = content

    return {
        "workspace": str(WORKDIR),
        "todos":     todos,
        "memories":  memories,
    }
