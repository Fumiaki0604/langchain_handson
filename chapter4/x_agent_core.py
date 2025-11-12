# x_agent_core.py

from __future__ import annotations
from typing import TypedDict, List
from typing import Annotated
import os

from dotenv import load_dotenv
from botocore.config import Config

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)
from langchain_tavily import TavilySearch
from langchain_community.agent_toolkits import FileManagementToolkit

from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages


# ========= 環境変数 =========
load_dotenv(override=True)

# ========= ツール定義 =========
web_search = TavilySearch(max_results=2, topic="general")

working_directory = "report"
os.makedirs(working_directory, exist_ok=True)
file_toolkit = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["write_file"],
)
write_file = file_toolkit.get_tools()[0]

tools = [web_search, write_file]
tools_by_name = {t.name: t for t in tools}

# ========= LLM 初期化 =========
cfg = Config(read_timeout=300)
llm_with_tools = init_chat_model(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_provider="bedrock_converse",
    config=cfg,
).bind_tools(tools)

# ========= システムプロンプト =========
system_prompt = """
あなたの責務はユーザーの依頼を調査し、結果をHTMLレポートとして保存することです。
- 調査にWeb検索が必要ならWeb検索ツールを使う。
- 必要な情報が集まったと判断したら検索は終了する。
- 厳守: web_search は最大2回まで。3回目以降は検索せず、これまでの結果を要約して結論を書き出す。
- ファイル出力はHTML(.html)で保存する。
  * Web検索が拒否されたら検索せずレポート作成。
  * レポート保存が拒否されたらレポート作成をやめ、内容を直接伝える。
  # --- system_prompt を強化（抜粋差分） ---
- HTML保存が完了したと宣言してよいのは、直近の ToolMessage(name="write_file") の content が
  {"status":"ok","file_path": "..."} というJSONであることを確認した場合に限る。
  それ以外は「まだ保存していません」と明確に述べる。

""".strip()


# ========= 状態（メッセージ履歴を add_messages で集約） =========
class MsgState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ========= タスク =========
@task(name="invoke_llm")
def invoke_llm(state: MsgState) -> AIMessage:
    msgs_in = state["messages"] if isinstance(state, dict) else []
    msgs = [SystemMessage(content=system_prompt)] + msgs_in
    return llm_with_tools.invoke(msgs)


# --- use_tool を堅牢化 ---
import json
import traceback

@task(name="use_tool")
def use_tool(tool_call: ToolCall) -> ToolMessage:
    tool = tools_by_name[tool_call["name"]]
    try:
        observation = tool.invoke(tool_call["args"])
        # FileManagementToolkit の write_file は通常 None / メッセージを返すので
        # 自分で成功JSONを作る
        if tool_call["name"] == write_file.name:
            file_path = tool_call["args"].get("file_path")
            abs_path = os.path.abspath(os.path.join(working_directory, file_path)) \
                       if file_path else None
            payload = {"status": "ok", "tool": "write_file",
                       "file_path": file_path, "abs_path": abs_path}
        else:
            payload = {"status": "ok", "tool": tool_call["name"], "data": observation}
    except Exception as e:
        payload = {"status": "error", "tool": tool_call["name"],
                   "error": str(e), "trace": traceback.format_exc()}

    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )


# ========= interrupt =========
def ask_human(tool_call: ToolCall):
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_data = {"name": tool_name}

    if tool_name == web_search.name:
        args_txt = "* ツール名\n"
        args_txt += f"  * {tool_name}\n"
        args_txt += "* 引数\n"
        for k, v in tool_args.items():
            args_txt += f"  * {k}\n    * {v}\n"
        tool_data["args"] = args_txt

    elif tool_name == write_file.name:
        args_txt = "* ツール名\n"
        args_txt += f"  * {tool_name}\n"
        args_txt += "* 保存ファイル名\n"
        args_txt += f"  * {tool_args.get('file_path','(不明)')}"
        tool_data["args"] = args_txt
        tool_data["html"] = tool_args.get("text", "")

    feedback = interrupt(tool_data)
    if feedback == "APPROVE":
        return tool_call

    return ToolMessage(
        content="ツール利用が拒否されたため、処理を終了してください。",
        name=tool_name,
        tool_call_id=tool_call["id"],
    )


# ========= チェックポインタ =========
checkpointer = MemorySaver()


# ========= エージェント本体 =========
@entrypoint(checkpointer=checkpointer)
def agent(state: MsgState) -> MsgState:
    llm_response = invoke_llm(state).result()

    loop_count = 0
    MAX_LOOP = 8
    search_count = 0

    while True:
        loop_count += 1
        if loop_count > MAX_LOOP:
            return {
                "messages": [
                    ToolMessage(
                        content="内部ガードにより処理を終了しました（ループ上限）。これまでの情報でレポートをまとめてください。",
                        name="system_guard",
                        tool_call_id=getattr(llm_response, "id", "guard"),
                    )
                ]
            }

        if not llm_response.tool_calls:
            break

        approve_calls, tool_results = [], []

        for tc in llm_response.tool_calls:
            if tc["name"] == web_search.name and search_count >= 2:
                tool_results.append(
                    ToolMessage(
                        content="検索上限(2回)に達しました。これまでの結果を要約し、結論をまとめてください。",
                        name=web_search.name,
                        tool_call_id=tc["id"],
                    )
                )
                continue

            decision = ask_human(tc)
            if isinstance(decision, ToolMessage):
                tool_results.append(decision)
            else:
                approve_calls.append(decision)

        executed = [use_tool(tc).result() for tc in approve_calls]
        search_count += sum(1 for tc in approve_calls if tc["name"] == web_search.name)

        state = {"messages": [llm_response, *executed, *tool_results]}
        llm_response = invoke_llm(state).result()

    return {"messages": [llm_response]}
