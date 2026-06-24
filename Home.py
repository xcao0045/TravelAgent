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

if "plan_state" not in st.session_state:
    st.session_state.plan_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "final_report" not in st.session_state:
    st.session_state.final_report = ""

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

plan_clicked = st.button("🚀 开始规划", type="primary", use_container_width=True)

# === 第2步：AI执行过程 ===
if plan_clicked:
    st.header("第2步：AI 执行过程")
    progress_container = st.container()

    with progress_container:
        status_placeholder = st.empty()

        if not settings.bailian_api_key or not settings.amap_api_key:
            st.error("""
            ⚠️ 请先在项目根目录创建 `.env` 文件，配置 API Key：
            ```
            BAILIAN_API_KEY=your_bailian_api_key_here
            AMAP_API_KEY=your_amap_api_key_here
            ```
            """)
        else:
            status_placeholder.info("🔄 主控Agent 解析需求...")
            try:
                user_input = {
                    "destination": destination,
                    "travel_date": str(travel_date),
                    "days": days,
                    "preferences": preferences,
                    "budget_total": float(budget_max),
                }

                result = run_travel_plan(user_input, settings)
                st.session_state.plan_state = result
                st.session_state.final_report = result.get("final_report", "")
                st.session_state.chat_history = []

                status_placeholder.success("✅ 方案生成完成！")
            except Exception as e:
                status_placeholder.error(f"❌ 生成失败: {str(e)}")
                st.stop()

# === 第3步：展示方案 ===
if st.session_state.final_report:
    st.header("第3步：旅行方案")
    st.markdown(st.session_state.final_report)

    col_actions = st.columns(4)
    with col_actions[0]:
        st.download_button(
            "📥 下载 Markdown",
            data=st.session_state.final_report,
            file_name=f"{destination}_{days}天_旅行方案.md",
            mime="text/markdown",
        )
    with col_actions[1]:
        st.components.v1.html(
            f"""
            <button onclick="navigator.clipboard.writeText({json.dumps(st.session_state.final_report)}).then(()=>{{this.innerText='✅ 已复制'}})"
                    style="padding:6px 16px;border:1px solid #ddd;border-radius:8px;background:white;cursor:pointer;font-size:14px;line-height:1.5;">
                📋 复制
            </button>
            """,
            height=40,
        )

    # === 第4步：微调对话 ===
    st.header("第4步：微调优化")
    st.caption("对方案有修改意见？在这里和AI对话调整")

    # 显示对话历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 用户输入
    user_feedback = st.chat_input("输入修改意见...")
    if user_feedback:
        st.session_state.chat_history.append({"role": "user", "content": user_feedback})

        # 再次调用LLM处理修改
        from agents.graph import _get_llm
        llm = _get_llm()

        current_report = st.session_state.final_report
        history_str = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-10:]]
        )

        refine_prompt = f"""以下是当前的旅行方案:
{current_report}

用户提出了新的修改意见。请根据修改意见更新方案，只修改用户要求的部分，其他保持不变。

对话历史:
{history_str}

请输出更新后的完整Markdown方案。"""
        response = llm.invoke(refine_prompt)
        st.session_state.final_report = response.content
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"方案已更新。\n\n{response.content[:200]}..."}
        )
        st.rerun()

    # 确认按钮
    if st.button("✅ 确认最终方案，保存到历史", type="primary"):
        session_data = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(),
            "status": "confirmed",
            "input": {
                "destination": destination,
                "travel_date": str(travel_date),
                "days": days,
                "preferences": preferences,
                "budget": [budget_min, budget_max],
            },
            "initial_report": st.session_state.plan_state.get("final_report", ""),
            "conversation": st.session_state.chat_history,
            "final_report": st.session_state.final_report,
        }
        os.makedirs(settings.history_dir, exist_ok=True)
        filename = f"{session_data['session_id']}_{destination}_{days}天.json"
        filepath = os.path.join(settings.history_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        st.success("✅ 方案已保存到历史记录！")
        st.balloons()
