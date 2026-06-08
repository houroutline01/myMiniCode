"""
permissions.py — 三关权限系统（对应课程 s03 + 优化）

三关流水线：
  Gate 1 — 黑名单硬拒绝：bash 命令包含危险模式时直接 block，不询问用户
  Gate 2 — 规则匹配：命中规则的操作需要用户确认（而不是全部询问）
  Gate 3 — 用户确认：提供三个选项
             y = 仅此次允许
             a = 本次会话始终允许（session 白名单，不再询问）【优化点】
             N = 拒绝（默认）

优化说明（相对课程 s03）：
  - 课程 s03：Gate3 每次命中规则都询问，重复操作很烦
  - 本项目：用户可以选 "a" 将该工具加入 session 白名单，后续同类操作直接放行

工具风险分级（配合 tools.py 中的 TOOL_RISK）：
  - "safe"   → 跳过全部权限检查，直接执行
  - "prompt" → 经过 Gate1（仅 bash）→ Gate2 → Gate3
"""

import re
from tools import TOOL_RISK

# ── 辅助函数 ───────────────────────────────────────────────────

def _bash_writes_file(command: str) -> bool:
    """
    判断 bash 命令是否通过非 write_file 途径写入文件。
    目的：防止模型绕过权限系统，用 bash 悄悄写文件。

    检测两类模式：
      1. 输出重定向到文件：> filename 或 >> filename
         排除无害的：>nul、>/dev/null、数字开头的错误重定向（2>）
      2. Python 内联写文件：open(..., 'w') 或 open(..., "w")
    """
    # 匹配 > 或 >> 但排除 >nul、>/dev/null、2>、&> 等无害形式
    if re.search(r'(?<![0-9&])\s*>{1,2}\s*(?!nul\b|/dev/null)', command):
        return True
    # 匹配 Python open() 写模式
    if re.search(r'open\s*\(.*["\']w["\']', command):
        return True
    return False


# ═══════════════════════════════════════════════════════════════
#  Gate 1：bash 黑名单（硬拒绝，不询问用户）
# ═══════════════════════════════════════════════════════════════

# 包含这些子串的 bash 命令直接拒绝
# 原则：只列真正灾难性的操作，不要过度拦截
BASH_DENY_PATTERNS = [
    "rm -rf /",      # 删除根目录
    "sudo",          # 提权
    "shutdown",      # 关机
    "reboot",        # 重启
    "mkfs",          # 格式化磁盘
    "dd if=",        # 磁盘底层写入
    "> /dev/sda",    # 写入块设备
    ":(){:|:&};:",   # fork 炸弹
]

def _check_deny_list(command: str) -> str | None:
    """
    检查 bash 命令是否命中黑名单。
    返回命中的模式字符串（用于打印原因），未命中返回 None。
    """
    for pattern in BASH_DENY_PATTERNS:
        if pattern in command:
            return pattern
    return None


# ═══════════════════════════════════════════════════════════════
#  Gate 2：规则匹配（命中才进入 Gate3，不命中直接放行）
# ═══════════════════════════════════════════════════════════════

# 每条规则是一个 dict：
#   tools   — 适用的工具名列表
#   check   — lambda(args) -> bool，返回 True 表示"需要确认"
#   message — 显示给用户的风险说明
PERMISSION_RULES = [
    {
        # bash 命令包含常见破坏性关键词时，提示用户确认
        "tools": ["bash"],
        "check": lambda args: any(
            kw in args.get("command", "")
            for kw in ["rm ", "rmdir", "> /etc/", "chmod 777", "chown", "curl", "wget", "pip install"]
        ),
        "message": "bash 命令包含潜在危险操作",
    },
    {
        # bash 通过重定向写文件，效果等同于 write_file，同样需要确认
        # 检测 "> 非空设备" 的模式，排除 >nul / >/dev/null / 2> 等无害重定向
        # 典型绕过方式：echo content > file.py、python -c "open(...).write(...)"
        "tools": ["bash"],
        "check": lambda args: _bash_writes_file(args.get("command", "")),
        "message": "bash 命令包含文件写入操作（重定向或 open() 写入）",
    },
    {
        # write_file / edit_file 始终需要确认（写操作影响文件系统）
        "tools": ["write_file", "edit_file"],
        "check": lambda args: True,
        "message": "文件写入/编辑操作",
    },
]

def _check_rules(tool_name: str, args: dict) -> str | None:
    """
    遍历规则列表，返回第一条命中规则的 message，未命中返回 None。
    """
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"]:
            try:
                if rule["check"](args):
                    return rule["message"]
            except Exception:
                pass  # 规则 check 出错时跳过，不影响主流程
    return None


# ═══════════════════════════════════════════════════════════════
#  Gate 3：用户确认（带 session 白名单）
# ═══════════════════════════════════════════════════════════════

# session 白名单：用户选择 "a" 后，工具名加入这个集合
# 进程结束后自动清空（不持久化），这是 session 级别的记忆
_SESSION_ALLOWLIST: set[str] = set()

def _ask_user(tool_name: str, args: dict, reason: str) -> str:
    """
    向用户展示风险信息，等待输入。
    返回 "allow" / "allow_always" / "deny"。
    """
    # 构造简洁的操作摘要，方便用户判断
    if tool_name == "bash":
        summary = f"$ {args.get('command', '')}"
    elif tool_name in ("write_file", "edit_file"):
        summary = f"{tool_name}({args.get('path', '')})"
    else:
        summary = f"{tool_name}({args})"

    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   {summary}")
    print("   允许？[\033[32my\033[0m=仅此次  \033[32ma\033[0m=本次会话始终允许  \033[31mN\033[0m=拒绝（默认）] ", end="")

    choice = input().strip().lower()
    if choice == "a":
        return "allow_always"
    elif choice == "y":
        return "allow"
    else:
        return "deny"


# ═══════════════════════════════════════════════════════════════
#  主入口：check_permission
# ═══════════════════════════════════════════════════════════════

def check_permission(tool_name: str, args: dict) -> bool:
    """
    三关权限检查总入口，返回 True 表示允许执行，False 表示拒绝。

    调用方（agent_loop）只需关心返回值，不需要了解内部逻辑。

    流程：
      1. 查 TOOL_RISK：safe 级别直接返回 True，跳过所有检查
      2. 查 session 白名单：在白名单内直接返回 True
      3. Gate1（仅 bash）：黑名单命中直接返回 False
      4. Gate2：规则匹配，未命中直接返回 True
      5. Gate3：命中规则时询问用户，根据答案更新白名单并返回结果
    """
    risk = TOOL_RISK.get(tool_name, "prompt")

    # ── 风险级别：safe，直接放行 ──────────────────────────────
    if risk == "safe":
        return True

    # ── session 白名单：用户已说过"始终允许" ─────────────────
    if tool_name in _SESSION_ALLOWLIST:
        print(f"\033[90m[权限] {tool_name} 在会话白名单中，自动放行\033[0m")
        return True

    # ── Gate 1：bash 黑名单硬拒绝 ─────────────────────────────
    if tool_name == "bash":
        blocked = _check_deny_list(args.get("command", ""))
        if blocked:
            print(f"\n\033[31m⛔ 黑名单拦截：命令包含 {blocked!r}\033[0m")
            return False

    # ── Gate 2：规则匹配 ──────────────────────────────────────
    reason = _check_rules(tool_name, args)
    if reason is None:
        # 没有命中任何规则，直接放行（例如：bash 运行 echo、ls 等安全命令）
        return True

    # ── Gate 3：用户确认 ──────────────────────────────────────
    decision = _ask_user(tool_name, args, reason)

    if decision == "allow_always":
        # 加入 session 白名单，本次会话后续不再询问
        _SESSION_ALLOWLIST.add(tool_name)
        print(f"\033[90m[权限] {tool_name} 已加入会话白名单\033[0m")
        return True
    elif decision == "allow":
        return True
    else:
        print("\033[31m[权限] 用户拒绝\033[0m")
        return False
