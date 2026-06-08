"""
tools.py — 工具实现层（对应课程 s02 + s05 + s06）

职责：
  1. 实现基础工具的具体逻辑（bash / read_file / write_file / edit_file / glob）
  2. 定义所有工具的 OpenAI 格式 schema（供 API 调用时使用）
  3. 为每个工具标注风险级别，供权限系统使用

阶段三新增工具 schema（实现在其他模块，此处只登记 schema 和风险级别）：
  todo_write — 更新待办事项列表（实现：todo.py::run_todo_write）
  task       — 启动子 agent 完成独立子任务（实现：subagent.py::spawn_subagent）

风险级别（risk）：
  "safe"   — 永远放行，不经过任何权限检查
  "prompt" — 需要经过 Gate1（黑名单）→ Gate2（规则匹配）→ Gate3（用户确认）

阶段三权限调整（重要设计决策）：
  write_file / edit_file：从 "prompt" 改为 "safe"
  原因分析：
    safe_path() 在 OS 层面强制保证所有写操作都在 WORKDIR 内，
    这是真正的边界防护（不可绕过）。
    之前的 Gate2 规则（lambda: True）只是"用户每次手动确认"，
    对编码助手来说这是噪音而非保护——用户要求写文件时当然应该写。
    真正危险的操作（rm、wget 等）留给 bash 的 Gate2+Gate3 处理。
"""

import os
import subprocess
import glob as _glob
from pathlib import Path

# ── 工作目录 ──────────────────────────────────────────────────
# 以脚本启动时的当前目录为工作目录，所有文件操作都限制在此目录内
WORKDIR = Path.cwd()


# ═══════════════════════════════════════════════════════════════
#  路径安全校验
# ═══════════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """
    将相对路径解析为绝对路径，并验证其在 WORKDIR 内。
    如果路径试图逃出工作目录（如 ../../etc/passwd），抛出 ValueError。

    这是文件操作的第一道防线，在工具函数内部调用。
    """
    resolved = (WORKDIR / p).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"路径越界，禁止访问工作目录之外：{p!r}")
    return resolved


# ═══════════════════════════════════════════════════════════════
#  工具实现
# ═══════════════════════════════════════════════════════════════

def run_bash(command: str) -> str:
    """
    在工作目录下执行 shell 命令，返回 stdout + stderr 合并结果。

    - timeout=120s：防止命令挂死
    - 输出截断至 50000 字符：防止超大输出撑爆 context
    - 注意：危险命令拦截由权限系统（Gate1）负责，这里不重复处理
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到无法解码的字节用 ? 替代，不崩溃
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: 命令超时（120s）"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    """
    读取文件内容，返回字符串。

    - limit：可选，最多返回前 N 行，超出部分显示省略提示
    - safe_path 保证路径在工作目录内
    """
    try:
        lines = safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... (省略 {len(lines) - limit} 行)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    将内容写入文件（覆盖）。父目录不存在时自动创建。
    """
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 字符到 {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    替换文件中第一处匹配的字符串。
    如果找不到 old_text，返回错误而不是静默失败。
    """
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if old_text not in text:
            return f"Error: 在 {path} 中未找到目标字符串"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"已编辑 {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """
    在工作目录内用 glob 模式匹配文件，返回相对路径列表。
    匹配结果会二次校验是否在 WORKDIR 内，防止符号链接逃逸。
    """
    try:
        matches = []
        for match in _glob.glob(pattern, root_dir=WORKDIR, recursive=True):
            abs_match = (WORKDIR / match).resolve()
            if abs_match.is_relative_to(WORKDIR):
                matches.append(match)
        return "\n".join(matches) if matches else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════
#  工具分发表
# ═══════════════════════════════════════════════════════════════

# TOOL_HANDLERS：工具名 -> 函数对象
# agent_loop 通过 TOOL_HANDLERS[name](**args) 调用工具，避免 if-else 硬编码
TOOL_HANDLERS = {
    "bash":       run_bash,
    "read_file":  run_read,
    "write_file": run_write,
    "edit_file":  run_edit,
    "glob":       run_glob,
}

# ═══════════════════════════════════════════════════════════════
#  工具风险分级（权限系统直接查表，不需要 if-else 分支）
#
#  "safe"   → 直接放行，不经过权限检查
#  "prompt" → 需要经过完整权限流水线（Gate1 黑名单 → Gate2 规则 → Gate3 确认）
#
#  阶段三变更：
#    write_file: "prompt" → "safe"（safe_path 是真正的边界，过度确认是噪音）
#    edit_file:  "prompt" → "safe"（同上）
#    todo_write: "safe"   （状态管理操作，无安全风险）
#    task:       "safe"   （子 agent 内部有自己的权限检查）
# ═══════════════════════════════════════════════════════════════
TOOL_RISK = {
    "bash":       "prompt",  # 命令执行，危险性最高，走完整权限流水线
    "read_file":  "safe",    # 只读操作
    "write_file": "safe",    # safe_path 已保证边界安全（阶段三从 prompt 改为 safe）
    "edit_file":  "safe",    # 同上（阶段三从 prompt 改为 safe）
    "glob":       "safe",    # 只读操作
    "todo_write": "safe",    # 内存状态操作，无文件系统风险
    "todo_config": "safe",   # 开关操作，无安全风险
    "task":       "safe",    # 子 agent 内部有独立权限检查
}

# ═══════════════════════════════════════════════════════════════
#  工具 Schema（OpenAI 格式，直接传给 API）
#
#  与 Anthropic 格式的差异：
#    Anthropic: {"name": ..., "input_schema": {...}}
#    OpenAI:    {"type": "function", "function": {"name": ..., "parameters": {...}}}
# ═══════════════════════════════════════════════════════════════
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "在工作目录下执行 shell 命令，返回输出。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 shell 命令"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作目录内的文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":  {"type": "string",  "description": "相对于工作目录的文件路径"},
                    "limit": {"type": "integer", "description": "最多读取的行数（可选）"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入文件（覆盖），父目录不存在时自动创建。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "写入内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "替换文件中第一处匹配的字符串。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":     {"type": "string", "description": "文件路径"},
                    "old_text": {"type": "string", "description": "要替换的原始字符串（必须精确匹配）"},
                    "new_text": {"type": "string", "description": "替换后的新字符串"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "用 glob 模式匹配工作目录内的文件，返回匹配路径列表。支持 ** 递归。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 模式，如 **/*.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    # ── 阶段三新增：todo_write ─────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "更新当前会话的待办事项列表。"
                "每次开始新任务时创建列表，执行过程中及时更新状态（pending/in_progress/done）。"
                "完整替换整个列表，不支持追加单项。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "待办事项完整列表（全量替换）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "任务描述（简洁明确，动词开头）",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done"],
                                    "description": "任务状态：pending=未开始，in_progress=进行中，done=已完成",
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    # ── 阶段三新增：todo_config ───────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "todo_config",
            "description": (
                "开启或关闭 todo 规划模式。\n"
                "enabled=true：激活规划模式，nag 机制开始计数，"
                "system prompt 会展示 todo 列表。开启后应立即调用 todo_write 制定计划。\n"
                "enabled=false：关闭规划模式，nag 静默，清空 todo 列表。"
                "任务完成后调用以收尾。\n"
                "默认关闭。简单单步任务无需开启；多步骤复杂任务应在开始时开启。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "true=开启规划模式，false=关闭规划模式",
                    },
                },
                "required": ["enabled"],
            },
        },
    },
    # ── 阶段三新增：task ───────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "将独立的编码子任务委托给子 agent 执行。"
                "子 agent 有独立的工具集和消息历史，完成后返回结果摘要。\n"
                "适用场景：可独立完成的、与当前对话上下文无关的具体编码任务。\n"
                "重要：子 agent 可以自行使用 glob/read_file 等工具探索文件系统，"
                "不需要你提前收集数据。'自包含'是指任务的目标和要求要写清楚，"
                "而不是把文件内容预先读出来塞进描述里。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "子任务的完整描述。必须自包含：包含文件路径、期望结果、"
                            "技术要求等所有必要信息，子 agent 没有任何其他上下文。"
                        ),
                    },
                },
                "required": ["description"],
            },
        },
    },
]
