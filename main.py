"""
main.py — 程序入口，REPL 交互循环

职责：
  - 维护跨轮次的 messages 列表（多轮对话共享同一历史）
  - 接收用户输入，调用 agent_loop
  - 打印模型最终文本回复

为什么 messages 在 main 里维护，而不是在 agent_loop 里？
  agent_loop 处理"单次请求的工具调用循环"，而 messages 承载的是"整个会话历史"。
  分离职责：main 管会话生命周期，agent_loop 管单轮处理逻辑。
"""

from agent import agent_loop, SYSTEM

# ── readline：改善终端输入体验（支持方向键历史、中文输入） ──
# Windows 没有 readline，ImportError 时静默跳过
try:
    import readline
except ImportError:
    pass


def print_response(messages: list):
    """
    从 messages 末尾找到最后一条 assistant 消息，打印其文本内容。

    OpenAI SDK 返回的 assistant message 是 ChatCompletionMessage 对象，
    其 .content 属性即为文本（如果模型只返回文本，没有工具调用时）。
    """
    last = messages[-1]

    # last 可能是 ChatCompletionMessage 对象（assistant），也可能是 dict（tool result）
    # 只打印 assistant 的文本回复
    if hasattr(last, "content") and last.content:
        print(f"\n\033[32m助手：\033[0m{last.content}")
    elif isinstance(last, dict) and last.get("role") == "assistant":
        print(f"\n\033[32m助手：\033[0m{last.get('content', '')}")


def main():
    print("=" * 50)
    print("  编码助手（阶段一：s01-s03）")
    print("  输入问题后回车发送，输入 q 退出")
    print("=" * 50)
    print(f"\033[90mSystem: {SYSTEM}\033[0m\n")

    # messages 是整个会话的历史记录，多轮对话共享
    # 格式遵循 OpenAI Chat Completions API：
    #   [{"role": "user"|"assistant"|"tool", "content": ...}, ...]
    messages = []

    while True:
        try:
            user_input = input("\033[36m你：\033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        # 将用户输入追加到历史，然后交给 agent_loop 处理
        messages.append({"role": "user", "content": user_input})
        agent_loop(messages)

        # agent_loop 返回后，messages 已包含本轮所有 assistant 和 tool 消息
        print_response(messages)
        print()


if __name__ == "__main__":
    main()
