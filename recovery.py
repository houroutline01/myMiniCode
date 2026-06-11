"""
recovery.py — LLM 调用错误恢复（对应课程 s11）

三条恢复路径（agent_loop 负责调用）：
  Path 1: finish_reason=="length"（输出被截断）
    → 第一次：max_tokens 从 8K 升级到 64K，重试（不保存截断输出）
    → 再次截断：保存截断输出 + 注入 continuation prompt，最多 3 次
  Path 2: prompt_too_long（context 超出窗口，API 直接拒绝）
    → 调用 reactive_compact，仅重试一次
  Path 3: 429（限流）/ 529（过载）
    → with_retry 指数退避 + 随机抖动，最多重试 10 次
    → 连续 3 次 529 时，切换到 DEEPSEEK_FALLBACK_MODEL（若配置）

与课程 Anthropic SDK 的差异（OpenAI SDK）：
  - RateLimitError       → 对应 429 限流
  - APIStatusError(529)  → 对应 529 过载
  - BadRequestError      → 可能包含 context_length_exceeded
  - finish_reason=="length" → 对应 Anthropic 的 stop_reason=="max_tokens"
"""

import os
import random
import time

from openai import RateLimitError, APIStatusError

# ── 常量 ──────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS   = 8_000
ESCALATED_MAX_TOKENS = 64_000
MAX_RECOVERY_RETRIES = 3     # continuation prompt 最大注入次数
MAX_RETRIES          = 10    # with_retry 最大重试次数
BASE_DELAY_MS        = 500   # 退避基础延迟（毫秒）
MAX_CONSECUTIVE_529  = 3     # 触发 fallback 的连续 529 次数

CONTINUATION_PROMPT = (
    "输出已达 token 上限，请直接继续——不要道歉，不要重述，从中断处接着写。"
)

FALLBACK_MODEL = os.getenv("DEEPSEEK_FALLBACK_MODEL")  # 未配置则不切换


# ═══════════════════════════════════════════════════════════════
#  RecoveryState — 跟踪单次 agent_loop 调用内的恢复状态
# ═══════════════════════════════════════════════════════════════

class RecoveryState:
    """
    每次 agent_loop 调用时创建（每个用户轮次独立），跟踪：
      max_tokens     — 当前请求的 token 上限（Path 1 升级后变为 64K）
      current_model  — 当前使用的模型（Path 3 可能切换到 fallback）
      has_escalated  — 是否已升级过 max_tokens
      recovery_count — continuation prompt 注入次数
      consecutive_529 — 连续 529 次数（累计到阈值触发 fallback 切换）
      has_attempted_reactive_compact — Path 2 只尝试一次紧急压缩
    """
    def __init__(self, model: str = "deepseek-chat"):
        self.max_tokens                    = DEFAULT_MAX_TOKENS
        self.current_model                 = model
        self.has_escalated                 = False
        self.recovery_count                = 0
        self.consecutive_529               = 0
        self.has_attempted_reactive_compact = False


# ═══════════════════════════════════════════════════════════════
#  with_retry — Path 3: 429/529 指数退避
# ═══════════════════════════════════════════════════════════════

def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """
    指数退避 + 随机抖动。
    retry_after 由 API 响应头提供时优先使用（更精确）。
    """
    if retry_after:
        return retry_after
    base = min(BASE_DELAY_MS * (2 ** attempt), 32_000) / 1000
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


def with_retry(fn, state: RecoveryState):
    """
    执行 fn()，遇到 429/529 时指数退避重试，其他错误直接 raise。

    fn 通常是封装了 client.chat.completions.create 的 lambda。
    state.current_model 和 state.max_tokens 在 fn() 内部读取，
    所以 with_retry 内部修改 state（切换 fallback 模型）后，
    下一次重试自动使用新模型。
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except RateLimitError:
            delay = _retry_delay(attempt)
            print(f"\033[33m[RECOVERY] 429 限流，{delay:.1f}s 后重试 "
                  f"（{attempt + 1}/{MAX_RETRIES}）\033[0m")
            time.sleep(delay)
        except APIStatusError as e:
            if e.status_code == 529 or "overloaded" in str(e).lower():
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529 and FALLBACK_MODEL:
                    state.current_model = FALLBACK_MODEL
                    state.consecutive_529 = 0
                    print(f"\033[31m[RECOVERY] 连续 {MAX_CONSECUTIVE_529} 次 529，"
                          f"切换到 {FALLBACK_MODEL}\033[0m")
                delay = _retry_delay(attempt)
                print(f"\033[33m[RECOVERY] 529 过载，{delay:.1f}s 后重试 "
                      f"（{attempt + 1}/{MAX_RETRIES}）\033[0m")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"已达最大重试次数（{MAX_RETRIES} 次），放弃。")


# ═══════════════════════════════════════════════════════════════
#  is_context_too_long — Path 2: 判断是否为 context 超长错误
# ═══════════════════════════════════════════════════════════════

def is_context_too_long(e: Exception) -> bool:
    """
    检测 API 错误是否因为 prompt/context 太长。
    DeepSeek 可能使用不同的错误描述，多关键词兜底。
    """
    msg = str(e).lower()
    return (
        ("prompt" in msg and "long" in msg)
        or "prompt_is_too_long" in msg
        or "context_length_exceeded" in msg
        or "max_context_window" in msg
        or "too many tokens" in msg
        or "input is too long" in msg
    )
