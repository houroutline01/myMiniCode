"""
main.py — 程序入口，REPL 交互循环

阶段三相对阶段二的变化：
  - 标题更新到阶段三（s01-s06 + s10）

阶段二相对阶段一的变化（保留注释）：
  - 初始化 context，传给 agent_loop
  - 每轮对话后更新 context（接 s10 的 update_context）
  - 用户输入提交后触发 UserPromptSubmit hook（接 s04）
"""

from agent import agent_loop, MODEL
from hooks import trigger_hooks
from prompt import update_context, get_system_prompt

try:
    import readline
except ImportError:
    pass


def print_response(messages: list):
    """打印最后一条 assistant 消息的文本内容。"""
    last = messages[-1]
    if hasattr(last, "content") and last.content:
        print(f"\n\033[32m助手：\033[0m{last.content}")
    elif isinstance(last, dict) and last.get("role") == "assistant":
        print(f"\n\033[32m助手：\033[0m{last.get('content', '')}")


def main():
    print("=" * 50)
    print("  编码助手（阶段三：s01-s06 + s10）")
    print(f"  模型：{MODEL}")
    print("  输入问题后回车发送，输入 q 退出")
    print("=" * 50)

    # s10: 初始化 context，派生自当前真实状态
    context = update_context([])

    # 打印初始 system prompt，方便确认 section 已正确加载
    print(f"\033[90m{get_system_prompt(context)}\033[0m\n")

    # messages 承载整个会话历史（跨轮次共享）
    # 注意：system prompt 不放在这里，而是在 agent_loop 里每轮动态传入
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

        # s04: UserPromptSubmit hook（在消息入 messages 之前触发）
        trigger_hooks("UserPromptSubmit", user_input)

        messages.append({"role": "user", "content": user_input})

        # s10: 传入 context，agent_loop 内部会在每轮工具执行后更新它
        agent_loop(messages, context)

        # s10: agent_loop 返回后，同步更新 main 里的 context
        # （agent_loop 内部用局部变量更新，这里同步到外层）
        context = update_context(messages)

        print_response(messages)
        print()


if __name__ == "__main__":
    main()
