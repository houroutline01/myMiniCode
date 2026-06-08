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

import json
from tools import TOOL_RISK

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
        # 注：write_file / edit_file 规则在阶段三已删除，
        #     原因：TOOL_RISK 改为 "safe"，safe_path() 是真正的文件边界保护。
        #     合法文件写入不需要每次用户确认，过度询问只会制造噪音。
        "tools": ["bash"],
        "check": lambda args: any(
            kw in args.get("command", "")
            for kw in ["rm ", "rmdir", "> /etc/", "chmod 777", "chown", "curl", "wget", "pip install"]
        ),
        "message": "bash 命令包含潜在危险操作",
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

def check_permission(tool_name: str, args: dict, interactive: bool = True) -> bool:
    """
    三关权限检查总入口，返回 True 表示允许执行，False 表示拒绝。

    参数：
      tool_name   — 工具名，用于查询风险级别和匹配规则
      args        — 工具参数，用于规则检查（如 bash 命令内容）
      interactive — 是否可以交互式询问用户（默认 True）
                    主 agent 调用：True  → Gate3 弹出确认框
                    子 agent 调用：False → Gate3 直接拒绝（不打断用户）

    流程：
      1. 查 TOOL_RISK：safe 级别直接返回 True，跳过所有检查
      2. 查 session 白名单：在白名单内直接返回 True（主子 agent 共享）
      3. Gate1（仅 bash）：黑名单命中直接返回 False（无论 interactive）
      4. Gate2：规则匹配，未命中直接返回 True
      5. Gate3：
           interactive=True  → 询问用户，根据答案更新白名单
           interactive=False → 打印提示，直接返回 False

    session 白名单继承机制：
      _SESSION_ALLOWLIST 是模块级变量，进程内所有代码共享同一个集合。
      主 agent 通过 Gate3 批准的工具，子 agent 自动继承，无需额外处理。
    """
    risk = TOOL_RISK.get(tool_name, "prompt")

    # ── 风险级别：safe，直接放行（write_file/edit_file 在阶段三变为 safe）──
    if risk == "safe":
        return True

    # ── session 白名单：用户已说过"始终允许"（主子 agent 天然共享）──────
    if tool_name in _SESSION_ALLOWLIST:
        print(f"\033[90m[权限] {tool_name} 在会话白名单中，自动放行\033[0m")
        return True

    # ── Gate 1：bash 黑名单硬拒绝（无论 interactive，黑名单不讲情面）────
    if tool_name == "bash":
        blocked = _check_deny_list(args.get("command", ""))
        if blocked:
            print(f"\n\033[31m⛔ 黑名单拦截：命令包含 {blocked!r}\033[0m")
            return False

    # ── Gate 2：规则匹配 ──────────────────────────────────────────────
    reason = _check_rules(tool_name, args)
    if reason is None:
        # 没有命中任何规则，直接放行（如 bash 运行 echo、ls 等安全命令）
        return True

    # ── Gate 3：用户确认（interactive 参数在此分叉）─────────────────
    if not interactive:
        # 子 agent 模式：不能打断用户，命中规则直接拒绝
        # 打印提示，方便用户事后了解为何子任务失败
        print(
            f"\n\033[33m[子 Agent 权限] 操作被拒绝（非交互模式）：{reason}\033[0m"
            f"\n  工具：{tool_name}，参数：{json.dumps(args, ensure_ascii=False)[:80]}"
            f"\n  提示：若需执行此操作，请在主 agent 中进行（会弹出确认框）\033[0m"
        )
        return False

    # interactive=True：正常询问用户
    decision = _ask_user(tool_name, args, reason)

    if decision == "allow_always":
        # 加入 session 白名单，本次会话后续不再询问（子 agent 也会继承）
        _SESSION_ALLOWLIST.add(tool_name)
        print(f"\033[90m[权限] {tool_name} 已加入会话白名单\033[0m")
        return True
    elif decision == "allow":
        return True
    else:
        print("\033[31m[权限] 用户拒绝\033[0m")
        return False
