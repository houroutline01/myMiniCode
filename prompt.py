"""
prompt.py — 动态 System Prompt 组装与缓存（对应课程 s10）

核心思想：
  System prompt 不再是一个硬编码字符串，而是由多个"碎片（section）"
  按当前运行状态动态拼接而成。这样：
  1. 每个功能模块只管理自己的 section，互不干扰
  2. 没激活的功能不占用 token（如 memory 在 s09 接入前不出现）
  3. 未来新增功能只需加一个 section，不用改其他地方

缓存机制：
  每次调 LLM 前都需要 system prompt，但 context 变化时才需要重新组装。
  用 json.dumps(context) 作为 key，context 不变就直接返回上次的字符串。
  避免每轮都做字符串拼接（虽然开销小，但养成好习惯）。

context 更新时机（对应课程 s10 的 update_context）：
  每轮工具调用完成后重新计算 context，system prompt 随之更新。
  目前 context 只含工作目录和记忆，几乎不变；
  s09（跨会话记忆）接入后，memories 字段会真正动态化。
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
        "你是一个编码助手。"
        "优先使用工具完成任务，不要只说不做。"
        "所有破坏性操作都会经过权限系统，无需你自行判断是否安全。"
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
    未来新增功能（todo、技能等）只需在此加一个条件分支。
    """
    sections = []

    # 始终加载的三个基础 section
    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["workspace"])
    sections.append(PROMPT_SECTIONS["platform"])

    # 条件加载：有记忆内容时才注入（s09 阶段激活）
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
    if context.get("memories"):
        loaded.append("memory")
    print(f"\033[90m[PROMPT] 重新组装：{', '.join(loaded)}\033[0m")

    return _cached_prompt


# ═══════════════════════════════════════════════════════════════
#  Context 派生函数
# ═══════════════════════════════════════════════════════════════

def update_context(messages: list) -> dict:
    """
    从当前真实状态派生 context dict。

    原则：context 只反映"现在是什么状态"，不存历史。
    调用时机：每轮工具调用完成后，由 agent_loop 调用。

    当前字段：
      workspace — 工作目录路径（几乎不变）
      memories  — MEMORY.md 内容（s09 接入后会动态变化）

    未来扩展字段（接入对应课程后添加）：
      todos     — 当前 todo 列表摘要（s05）
      tasks     — 当前任务状态（s12）
    """
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            memories = content

    return {
        "workspace": str(WORKDIR),
        "memories": memories,
    }
