"""
compact.py — 四层 Context 压缩管道（对应课程 s08）

核心思想：
  context 不是无限大的；长对话会逐渐撑满 token 窗口。
  策略遵循 "cheap first, expensive last"：
    L3 tool_result_budget — 持久化超大工具输出（无 API 调用）
    L1 snip_compact       — 裁剪中间消息（无 API 调用）
    L2 micro_compact      — 旧工具结果替换占位符（无 API 调用）
    L4 compact_history    — LLM 生成对话摘要（1 次 API 调用，仅超阈值触发）

  执行顺序与课程原版一致（CC 源码顺序）：budget → snip → micro → 判断

紧急情况（reactive_compact）：
  API 返回 context 太长错误时触发，保留最近 5 条消息 + LLM 摘要。

OpenAI 格式适配（与课程 Anthropic 格式的差异）：
  tool result 是独立的 role=="tool" dict 消息，不是嵌套在 user message 里的 block。
  所有函数均用 isinstance(m, dict) 判断，兼容 SDK 对象与 dict 混合的 messages 列表。
"""

import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR           = Path.cwd()
TRANSCRIPT_DIR    = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR  = WORKDIR / ".task_outputs" / "tool-results"

# ── 压缩触发阈值 ──────────────────────────────────────────────
CONTEXT_LIMIT     = 80_000  # 字符数估算，超过才触发 L4 LLM 摘要
KEEP_RECENT       = 3       # L2 保留最近 N 条工具结果，不替换为占位符
PERSIST_THRESHOLD = 30_000  # L3 单条输出超过此字符数才持久化

_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)
_MODEL = "deepseek-chat"


def estimate_size(messages: list) -> int:
    """粗略估算 messages 的字符数（用于判断是否触发 L4）。"""
    return len(json.dumps(messages, default=str, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════
#  L1: snip_compact — 裁剪中间消息
# ═══════════════════════════════════════════════════════════════

def snip_compact(messages: list, max_messages: int = 50) -> list:
    """
    消息数量超过 max_messages 时，裁剪中间部分，保留头尾。

    保留最早 3 条（通常是用户最初的任务描述），
    保留最新 max_messages-3 条（保证 LLM 有最新上下文）。
    被裁剪的部分替换为一条 [snipped N messages] 占位符。
    """
    if len(messages) <= max_messages:
        return messages
    keep_head, keep_tail = 3, max_messages - 3
    snipped = len(messages) - keep_head - keep_tail
    print(f"\033[90m[COMPACT L1] 裁剪中间 {snipped} 条消息\033[0m")
    return (
        messages[:keep_head]
        + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
        + messages[-keep_tail:]
    )


# ═══════════════════════════════════════════════════════════════
#  L2: micro_compact — 旧工具结果替换为占位符
# ═══════════════════════════════════════════════════════════════

def micro_compact(messages: list) -> list:
    """
    保留最近 KEEP_RECENT 条工具结果，更早的大段输出替换为占位符。

    OpenAI 格式的 tool result = {"role": "tool", "tool_call_id": ..., "content": str}
    只对 content 较长（>120 字符）的旧结果做替换，短结果保留。
    直接修改 messages 内的 dict，返回同一个列表（in-place）。
    """
    tool_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    if len(tool_indices) <= KEEP_RECENT:
        return messages
    count = 0
    for i in tool_indices[:-KEEP_RECENT]:
        content = messages[i].get("content", "")
        if isinstance(content, str) and len(content) > 120:
            messages[i] = {
                **messages[i],
                "content": "[Earlier tool result compacted. Re-run if needed.]",
            }
            count += 1
    if count:
        print(f"\033[90m[COMPACT L2] 压缩 {count} 条旧工具输出\033[0m")
    return messages


# ═══════════════════════════════════════════════════════════════
#  L3: tool_result_budget — 持久化超大工具输出
# ═══════════════════════════════════════════════════════════════

def _persist_large_output(tool_call_id: str, output: str) -> str:
    """将大输出写入 .task_outputs/tool-results/<id>.txt，返回引用摘要。"""
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_call_id}.txt"
    if not path.exists():
        path.write_text(output, encoding="utf-8")
    return (
        f"<persisted-output>\n"
        f"完整输出已保存：{path}\n"
        f"预览（前 2000 字符）：\n{output[:2000]}\n"
        f"</persisted-output>"
    )


def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    """
    当所有工具结果总大小超过 max_bytes 时，将最大的几条持久化到磁盘。

    只处理超过 PERSIST_THRESHOLD 的单条结果（小结果不值得持久化）。
    按大小降序处理，优先压缩最大的，直到总大小降到 max_bytes 以下。
    """
    tool_msgs = [
        (i, m) for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    if not tool_msgs:
        return messages

    total = sum(len(str(m.get("content", ""))) for _, m in tool_msgs)
    if total <= max_bytes:
        return messages

    ranked = sorted(
        tool_msgs,
        key=lambda p: len(str(p[1].get("content", ""))),
        reverse=True,
    )
    for idx, msg in ranked:
        if total <= max_bytes:
            break
        content = str(msg.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD:
            continue
        tid = msg.get("tool_call_id", "unknown")
        new_content = _persist_large_output(tid, content)
        messages[idx] = {**msg, "content": new_content}
        total = sum(len(str(m.get("content", ""))) for _, m in tool_msgs)
        print(f"\033[90m[COMPACT L3] 持久化大输出 {tid[:12]}...\033[0m")

    return messages


# ═══════════════════════════════════════════════════════════════
#  L4: compact_history — LLM 生成对话摘要
# ═══════════════════════════════════════════════════════════════

def _write_transcript(messages: list) -> Path:
    """将当前对话历史保存为 .transcripts/transcript_<timestamp>.jsonl。"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
    return path


def _summarize_history(messages: list) -> str:
    """调用 LLM 生成对话摘要，供压缩后作为新的起点。"""
    conversation = json.dumps(messages, default=str, ensure_ascii=False)[:80_000]
    prompt = (
        "请总结以下 AI 编码助手对话，使对话可以从摘要处继续工作。\n"
        "必须保留：1. 当前目标  2. 关键发现与决策  3. 已读/已修改的文件  "
        "4. 剩余工作  5. 用户给出的约束条件\n"
        "保持简洁具体，不超过 500 字。\n\n"
        + conversation
    )
    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return response.choices[0].message.content or "(empty summary)"
    except Exception as e:
        return f"(摘要生成失败：{e})"


def compact_history(messages: list) -> list:
    """
    L4 主动压缩：保存记录 → 生成摘要 → 返回仅含摘要的新 messages 列表。

    调用方用返回值替换原 messages（in-place）：
        messages[:] = compact_history(messages)
    """
    transcript_path = _write_transcript(messages)
    print(f"\033[90m[COMPACT L4] 对话记录已保存：{transcript_path.name}\033[0m")
    summary = _summarize_history(messages)
    print(f"\033[33m[COMPACT L4] 对话已压缩（摘要 {len(summary)} 字）\033[0m")
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ═══════════════════════════════════════════════════════════════
#  Emergency: reactive_compact — API 报 context 太长时触发
# ═══════════════════════════════════════════════════════════════

def reactive_compact(messages: list) -> list:
    """
    紧急压缩：保存记录 → LLM 摘要 → 摘要 + 最近 5 条消息。

    与 L4 的区别：额外保留最近 5 条消息，让 LLM 能从最新状态续写。
    触发时机：API 调用本身返回 prompt_too_long，说明已超出 token 窗口。
    """
    print("\033[31m[COMPACT reactive] 紧急压缩中...\033[0m")
    transcript_path = _write_transcript(messages)
    print(f"\033[90m[COMPACT reactive] 对话记录已保存：{transcript_path.name}\033[0m")
    summary = _summarize_history(messages)
    return [
        {"role": "user", "content": f"[Reactive compact]\n\n{summary}"},
        *messages[-5:],
    ]


# ═══════════════════════════════════════════════════════════════
#  Pipeline 入口（agent_loop 每轮调用）
# ═══════════════════════════════════════════════════════════════

def run_compaction_pipeline(messages: list) -> list:
    """
    在每次 LLM 调用前，按顺序执行 L3→L1→L2，然后判断是否触发 L4。

    "cheap first"：前三层零 API 调用；只在估算大小超过 CONTEXT_LIMIT 时
    才触发 L4（1 次 API 调用）。
    """
    messages = tool_result_budget(messages)  # L3: 先持久化超大输出
    messages = snip_compact(messages)        # L1: 裁剪中间消息
    messages = micro_compact(messages)       # L2: 旧结果替换占位符

    if estimate_size(messages) > CONTEXT_LIMIT:
        print(f"\033[33m[COMPACT] 估算 context 超过 {CONTEXT_LIMIT} 字符，触发 L4\033[0m")
        messages = compact_history(messages)

    return messages
