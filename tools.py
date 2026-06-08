"""
tools.py — 工具实现层（对应课程 s02）

职责：
  1. 实现每个工具的具体逻辑（read_file / write_file / edit_file / glob / bash）
  2. 定义工具的 OpenAI 格式 schema（供 API 调用时使用）
  3. 为每个工具标注风险级别，供权限系统使用

风险级别（risk）：
  "safe"   — 只读操作，永远放行，不经过权限检查
  "prompt" — 写操作或 bash，需要经过 Gate2+Gate3 权限检查
  "block"  — （扩展用）直接拒绝，目前未使用
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
#  工具风险分级（优化点：在定义时标注，权限系统直接查表）
#
#  "safe"   → 只读，权限系统直接放行
#  "prompt" → 写操作/命令执行，需经过权限 Gate2+Gate3
# ═══════════════════════════════════════════════════════════════
TOOL_RISK = {
    "bash":       "prompt",
    "read_file":  "safe",
    "write_file": "prompt",
    "edit_file":  "prompt",
    "glob":       "safe",
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
]
