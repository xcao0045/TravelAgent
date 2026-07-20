# Home.py
import streamlit as st
import json
import os
from datetime import datetime
from config import Settings
from agents.graph import run_travel_plan

st.set_page_config(
    page_title="Multi-Agent 智能旅行规划",
    page_icon="🧭",
    layout="wide",
)

# --- 初始化 ---
settings = Settings.from_env()

# 线程模型：每个旅行方案是一个 thread，含 meta / messages / final_report / rag_refs
if "history" not in st.session_state:
    st.session_state.history = {}         # {thread_id: {...}}
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


def _get_thread():
    """返回当前活跃线程的引用，若无则返回 None。"""
    tid = st.session_state.active_thread_id
    if tid and tid in st.session_state.history:
        return st.session_state.history[tid]
    return None


def _new_thread(meta: dict) -> str:
    """创建新线程，返回 thread_id。"""
    import uuid
    tid = uuid.uuid4().hex[:12]
    st.session_state.history[tid] = {
        "thread_id": tid,
        "meta": meta,
        "messages": [],
        "final_report": "",
        "rag_refs": {},          # {source_id: text} 用于 popover 渲染
        "status": "draft",
        "created_at": datetime.now().isoformat(),
    }
    st.session_state.active_thread_id = tid
    return tid

st.title("🧭 Multi-Agent 智能旅行规划")
st.caption("输入需求，AI自动生成覆盖天气、景点、酒店、餐饮、交通、预算的完整旅行方案")

# === 第1步：输入需求 ===
st.header("第1步：输入你的旅行需求")

col1, col2, col3 = st.columns(3)
with col1:
    destination = st.text_input("目的地", placeholder="如：成都")
with col2:
    travel_date = st.date_input("出发日期")
with col3:
    days = st.number_input("天数", min_value=1, max_value=30, value=3)

col4, col5 = st.columns(2)
with col4:
    budget_min, budget_max = st.slider("预算区间(元)", 0, 50000, (3000, 8000), step=500)

preferences = st.text_area("偏好", placeholder="如：亲子、安静、美食、适合老人")

plan_clicked = st.button(
    "🚀 开始规划", type="primary", use_container_width=True,
    disabled=st.session_state.is_processing,
)

# === 第2步：AI执行过程 ===
if plan_clicked:
    st.header("第2步：AI 执行过程")

    if not settings.bailian_api_key or not settings.amap_api_key:
        st.error("""
        ⚠️ 请先在项目根目录创建 `.env` 文件，配置 API Key：
        ```
        BAILIAN_API_KEY=your_bailian_api_key_here
        AMAP_API_KEY=your_amap_api_key_here
        ```
        """)
    else:
        st.session_state.is_processing = True
        with st.spinner("🔄 Agent 协作中，请稍候..."):
            try:
                user_input = {
                    "destination": destination,
                    "travel_date": str(travel_date),
                    "days": days,
                    "preferences": preferences,
                    "budget_total": float(budget_max),
                }

                result = run_travel_plan(user_input, settings)

                # 创建新线程
                tid = _new_thread(meta={
                    "destination": destination,
                    "travel_date": str(travel_date),
                    "days": days,
                    "preferences": preferences,
                    "budget": [budget_min, budget_max],
                })
                thread = _get_thread()
                thread["final_report"] = result.get("final_report", "")
                thread["initial_report"] = result.get("final_report", "")
                # 收集 RAG 引用源文本
                thread["rag_refs"] = result.get("rag_refs", {})

                st.success("✅ 方案生成完成！")
            except Exception as e:
                st.error(f"❌ 生成失败: {str(e)}")
            finally:
                st.session_state.is_processing = False

# === 第3步：展示方案 ===
thread = _get_thread()
if thread and thread.get("final_report"):
    report = thread["final_report"]
    st.header("第3步：旅行方案")
    st.markdown(report)

    # RAG 引用：初始报告下方展示所有检索到的源文本（规避缺陷 A — 初始报告不走 messages 渲染）
    rag = thread.get("rag_refs", {})
    if rag:
        with st.expander(f"📖 RAG 引用来源（{len(rag)} 条）"):
            for rid, text in rag.items():
                st.caption(f"**{rid}**")
                st.markdown(text[:300])
                st.divider()
    else:
        st.caption("💡 当前知识库未匹配到相关内容")

    col_actions = st.columns(4)
    with col_actions[0]:
        st.download_button(
            "📥 下载 Markdown",
            data=report,
            file_name=f"{destination}_{days}天_旅行方案.md",
            mime="text/markdown",
        )

    # 源码查看器放到按钮行下方全宽（规避 Issue 2 — 列内狭窄 + disabled 禁止光标）
    with st.expander("📋 查看 Markdown 源码"):
        st.text_area(
            "Markdown 源码", report,
            height=400, label_visibility="collapsed",
        )

    # === 第4步：微调对话 ===
    st.header("第4步：微调优化")
    st.caption("对方案有修改意见？在这里和AI对话调整")

    # 显示对话历史
    for msg in thread.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.popover("📖 引用来源"):
                    for src in msg["sources"]:
                        st.caption(f"**{src['label']}**")
                        st.markdown(src["text"][:300])

    # 用户输入
    user_feedback = st.chat_input(
        "输入修改意见...",
        disabled=st.session_state.is_processing,
    )
    if user_feedback:
        thread["messages"].append({"role": "user", "content": user_feedback})
        st.session_state.is_processing = True

        with st.spinner("🔄 AI 正在优化方案..."):
            from agents.graph import _get_llm
            llm = _get_llm()

            history_str = "\n".join(
                [f"{m['role']}: {m['content']}" for m in thread["messages"][-10:]]
            )

            refine_prompt = f"""以下是当前的旅行方案:
{report}

用户提出了新的修改意见。请根据修改意见更新方案，只修改用户要求的部分，其他保持不变。

对话历史:
{history_str}

请输出更新后的完整Markdown方案。"""
            response = llm.invoke(refine_prompt)
            thread["final_report"] = response.content

            # 对话消息的 RAG 引用：全量携带 rag_refs，不依赖 LLM 输出格式（规避缺陷 B）
            thread["messages"].append({
                "role": "assistant",
                "content": response.content,
                "sources": [{"label": rid, "text": text}
                            for rid, text in thread.get("rag_refs", {}).items()],
            })

        st.session_state.is_processing = False
        st.rerun()

    # 确认按钮
    if st.button("✅ 确认最终方案，保存到历史", type="primary"):
        thread["status"] = "confirmed"
        os.makedirs(settings.history_dir, exist_ok=True)
        filename = f"{thread['thread_id']}_{destination}_{days}天.json"
        filepath = os.path.join(settings.history_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(thread, f, ensure_ascii=False, indent=2)
        st.success("✅ 方案已保存到历史记录！")
        st.balloons()
