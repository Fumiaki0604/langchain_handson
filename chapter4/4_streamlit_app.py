# 4_streamlit_app.py

import uuid
import json
import streamlit as st
import asyncio # 追加
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from x_agent_core import agent


# =========================
# 初期化
# =========================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "waiting_for_approval" not in st.session_state:
        st.session_state.waiting_for_approval = False
    if "final_result" not in st.session_state:
        st.session_state.final_result = None
    if "tool_info" not in st.session_state:
        st.session_state.tool_info = None
    if "thread_id" not in st.session_state or not st.session_state.thread_id:
        # セッション開始時に一度だけ生成→以後固定
        st.session_state.thread_id = str(uuid.uuid4())


def reset_session():
    st.session_state.messages = []
    st.session_state.waiting_for_approval = False
    st.session_state.final_result = None
    st.session_state.tool_info = None
    # 新規会話用に thread_id を再採番（明示的に「新規チャット」ボタンでのみ）
    st.session_state.thread_id = str(uuid.uuid4())


# =========================
# エージェント呼び出し
# =========================
def run_agent(input_data):
    """
    input_data: [HumanMessage(...)] もしくは Command(resume="APPROVE"/"DENY")
    """
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    payload = input_data if isinstance(input_data, Command) else {"messages": input_data}

    with st.spinner("処理中...", show_time=True):
        for chunk in agent.stream(payload, stream_mode="updates", config=config):
            for task_name, result in chunk.items():
                # ---- interrupt（承認待ち）----
                if task_name == "__interrupt__":
                    data = result
                    # list/tupleなら先頭要素
                    if isinstance(data, (list, tuple)) and data:
                        data = data[0]
                    # .value を持つ場合
                    if hasattr(data, "value"):
                        data = data.value
                    # さらに list/tuple でネストされていたら dict を優先
                    if isinstance(data, (list, tuple)):
                        picked = next((x for x in data if isinstance(x, dict)), None)
                        data = picked or {"args": str(result)}
                    elif not isinstance(data, dict):
                        data = {"args": str(result)}
                    st.session_state.tool_info = data
                    st.session_state.waiting_for_approval = True

                # ---- agent（最終結果）----
                elif task_name == "agent":
                    # result が {"messages":[AIMessage(...)]} or AIMessage の両方に対応
                    ai_msg = None
                    if isinstance(result, dict) and "messages" in result:
                        msgs = result["messages"]
                        if isinstance(msgs, list) and msgs:
                            ai_msg = msgs[-1]
                    else:
                        ai_msg = result

                    content = getattr(ai_msg, "content", ai_msg)
                    if isinstance(content, list):
                        texts = [
                            c.get("text")
                            for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        ]
                        st.session_state.final_result = "\n".join(t for t in texts if t)
                    elif isinstance(content, str):
                        st.session_state.final_result = content
                    else:
                        st.session_state.final_result = str(content)

                # ---- invoke_llm（途中経過）----
                elif task_name == "invoke_llm":
                    content = getattr(result, "content", result)
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": c["text"]}
                                )
                    elif isinstance(content, str):
                        st.session_state.messages.append(
                            {"role": "assistant", "content": content}
                        )

                # ---- use_tool（ツール結果の可視化）----
                elif task_name == "use_tool":
                    name = getattr(result, "name", None)
                    raw = getattr(result, "content", "")

                    parsed = None
                    if isinstance(raw, str):
                        try:
                            parsed = json.loads(raw)
                        except Exception:
                            parsed = None

                    # write_file の成功JSONなら保存パスを明示
                    if (
                        name == "write_file"
                        and isinstance(parsed, dict)
                        and parsed.get("status") == "ok"
                    ):
                        fp = parsed.get("file_path")
                        ap = parsed.get("abs_path")
                        msg = "✅ レポートを保存しました。"
                        if fp:
                            msg += f"\n- 相対パス: `{fp}`"
                        if ap:
                            msg += f"\n- 絶対パス: `{ap}`"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg}
                        )
                    elif isinstance(parsed, dict):
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"ツール({name})を実行しました: {parsed.get('status')}",
                            }
                        )
                    else:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"ツール({name})を実行しました。"}
                        )
    return True


# =========================
# 承認ボタン
# =========================
def feedback_buttons():
    col1, col2 = st.columns(2)
    feedback_result = None
    with col1:
        if st.button("APPROVE", use_container_width=True):
            st.session_state.waiting_for_approval = False
            feedback_result = "APPROVE"
    with col2:
        if st.button("DENY", use_container_width=True):
            st.session_state.waiting_for_approval = False
            feedback_result = "DENY"
    return feedback_result


# =========================
# アプリ本体
# =========================
def app():
    st.set_page_config(page_title="Webリサーチエージェント", page_icon="🔎", layout="centered")
    st.title("Webリサーチエージェント")

    init_session_state()

    # 新規チャット
    if st.button("＋ 新規チャットを開始", type="secondary"):
        reset_session()
        st.rerun()

    # これまでのメッセージ
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 承認待ちUI
    if st.session_state.waiting_for_approval and st.session_state.tool_info:
        ti = st.session_state.tool_info if isinstance(st.session_state.tool_info, dict) else {}

        if "args" in ti and ti["args"]:
            st.info(ti["args"])

        if ti.get("name") == "write_file" and isinstance(ti.get("html"), str):
            with st.container(height=420, border=True):
                html_content = f"""
                <style>
                body {{
                    background-color: #fdfdfd;
                    color: #111;
                    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Hiragino Sans","Noto Sans JP","Helvetica Neue",Arial;
                    line-height: 1.6;
                }}
                pre,code {{
                    background-color: #eee;
                    color: #000;
                    padding: 4px 6px;
                    border-radius: 4px;
                }}
                </style>
                {ti["html"]}
                """
                st.components.v1.html(html_content, height=420, scrolling=True)

        fb = feedback_buttons()
        if fb:
            st.chat_message("user").write(fb)
            st.session_state.messages.append({"role": "user", "content": fb})
            run_agent(Command(resume=fb))
            st.rerun()
        return

    # 最終結果
    if st.session_state.final_result and not st.session_state.waiting_for_approval:
        st.subheader("最終結果")
        st.success(st.session_state.final_result)

    # 入力欄
    if not st.session_state.waiting_for_approval:
        user_input = st.chat_input("メッセージを入力してください")
        if user_input:
            st.chat_message("user").write(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            messages = [HumanMessage(content=user_input)]
            run_agent(messages)
            st.rerun()
    else:
        st.info("ツールの承認待ちです。上のボタンで応答してください。")


if __name__ == "__main__":
    app()
