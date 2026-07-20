import streamlit as st
import json
import os
from config import Settings

st.set_page_config(page_title="历史记录", page_icon="📋", layout="wide")

settings = Settings.from_env()

st.title("📋 历史旅行方案")

history_dir = settings.history_dir
if not os.path.exists(history_dir):
    st.info("暂无历史记录，去首页生成你的第一份旅行方案吧！")
    st.stop()

files = sorted(
    [f for f in os.listdir(history_dir) if f.endswith(".json")],
    reverse=True,
)

if not files:
    st.info("暂无历史记录")
    st.stop()

search_term = st.text_input("搜索目的地...")

for filename in files:
    filepath = os.path.join(history_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            session = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            st.warning(f"⚠️ 跳过损坏文件: {filename}")
            continue

    # 兼容旧格式 (input 字段) 和新格式 (meta 字段)
    meta = session.get("meta", session.get("input", {}))
    destination = meta.get("destination", "未知")
    if search_term and search_term not in destination:
        continue

    days = meta.get("days", "?")
    budget = meta.get("budget", [0, 0])
    messages = session.get("messages", session.get("conversation", []))
    conv_count = len(messages)
    status = "✅已确认" if session.get("status") == "confirmed" else "⚠️未确认"

    with st.expander(
        f"{session.get('created_at','')[:16]} | {destination} {days}天 | {conv_count}轮对话 | {status}"
    ):
        tabs = st.tabs(["最终方案", "对话历史", "初始方案"])

        with tabs[0]:
            if session.get("final_report"):
                st.markdown(session["final_report"])
                st.download_button(
                    "📥 下载 Markdown",
                    data=session["final_report"],
                    file_name=f"{destination}_{days}天_旅行方案.md",
                    key=f"dl_{filename}",
                )

        with tabs[1]:
            for msg in messages:
                role = "🙋" if msg["role"] == "user" else "🤖"
                st.caption(f"{role} {msg.get('timestamp', '')}")
                st.write(msg["content"][:500])
                # RAG 引用 popover
                if msg.get("sources"):
                    with st.popover("📖 引用来源"):
                        for src in msg["sources"]:
                            st.caption(f"**{src['label']}**")
                            st.markdown(src["text"][:300])
                st.divider()
            if not messages:
                st.caption("(无对话记录)")

        with tabs[2]:
            if session.get("initial_report"):
                st.markdown(session["initial_report"][:1000])
            else:
                st.caption("(无初始方案)")

        if st.button("🗑️ 删除此记录", key=f"del_{filename}"):
            os.remove(filepath)
            st.success("已删除")
            st.rerun()
