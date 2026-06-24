import streamlit as st
import os
import json
from datetime import datetime
from config import Settings
from rag.embedding import create_embeddings
from rag.vector_store import VectorStoreManager
from rag.data_loader import load_file_to_docs
from rag.dedup import md5_hash, dedup_pipeline
from langchain.schema import Document

st.set_page_config(page_title="知识库管理", page_icon="📚")

settings = Settings.from_env()

# 初始化
@st.cache_resource
def get_vector_store():
    embeddings = create_embeddings(settings.bailian_api_key, settings.embedding_model)
    return VectorStoreManager(settings.chroma_persist_dir, embeddings)

vs = get_vector_store()

st.title("📚 知识库管理")

# 状态概览
try:
    prefs_count = vs.get_preferences_collection()._collection.count()
except Exception:
    prefs_count = 0
try:
    cases_count = vs.get_cases_collection()._collection.count()
except Exception:
    cases_count = 0

col1, col2 = st.columns(2)
with col1:
    st.metric("偏好库", f"{prefs_count} 条")
with col2:
    st.metric("案例库", f"{cases_count} 条")

st.divider()

# 录入目标库选择
collection_type = st.radio("选择目标库", ["preferences", "cases"],
                           format_func=lambda x: "偏好库 (酒店/景点/餐厅评价)" if x == "preferences" else "案例库 (优质旅行方案)")

tab1, tab2 = st.tabs(["📝 手动录入", "📤 文件上传"])

# --- 手动录入 ---
with tab1:
    if collection_type == "preferences":
        with st.form("pref_form"):
            category = st.selectbox("品类", ["hotel", "restaurant", "attraction"],
                                    format_func=lambda x: {"hotel": "酒店", "restaurant": "餐厅", "attraction": "景点"}[x])
            name = st.text_input("名称")
            tags = st.text_input("标签（用竖线|分隔）", placeholder="亲子|隔音好|卫生好")
            rating = st.slider("评分", 0.0, 5.0, 4.0, 0.5)
            text = st.text_area("评价文本")
            submitted = st.form_submit_button("提交入库")
            if submitted and name and text:
                doc = Document(
                    page_content=text,
                    metadata={
                        "category": category, "name": name,
                        "tags": [t.strip() for t in tags.split("|") if t.strip()],
                        "rating": rating, "source": "manual",
                        "created_at": datetime.now().isoformat(),
                    }
                )
                vs.add_to_preferences([doc])
                st.success(f"✅ '{name}' 已入库")
                st.rerun()
    else:
        with st.form("case_form"):
            destination = st.text_input("目的地")
            days = st.number_input("天数", 1, 30, 3)
            season = st.selectbox("季节", ["春季", "夏季", "秋季", "冬季"])
            budget_range = st.text_input("预算区间", "3000-5000")
            tags = st.text_input("标签（用竖线|分隔）", placeholder="美食|休闲")
            rating = st.slider("案例质量分", 0.0, 5.0, 4.0, 0.5)
            content = st.text_area("完整旅行方案 (Markdown)", height=300)
            submitted = st.form_submit_button("提交入库")
            if submitted and destination and content:
                doc = Document(
                    page_content=content,
                    metadata={
                        "destination": destination, "days": days,
                        "season": season, "budget_range": budget_range,
                        "tags": [t.strip() for t in tags.split("|") if t.strip()],
                        "rating": rating, "source": "manual",
                        "created_at": datetime.now().isoformat(),
                    }
                )
                vs.add_to_cases([doc])
                st.success(f"✅ '{destination} {days}天' 案例已入库")
                st.rerun()

# --- 文件上传 ---
with tab2:
    st.caption(f"上传文件到{collection_type}库")
    supported = "CSV, JSON" if collection_type == "preferences" else "JSON, Markdown, TXT, CSV"
    st.info(f"支持格式: {supported}")

    uploaded_file = st.file_uploader("选择文件", type=["csv", "json", "md", "txt"])
    if uploaded_file:
        # Save to temp
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            docs = load_file_to_docs(tmp_path, collection_type)
            if collection_type == "preferences":
                vs.add_to_preferences(docs)
            else:
                vs.add_to_cases(docs)
            st.success(f"✅ 成功导入 {len(docs)} 条数据")
        except Exception as e:
            st.error(f"导入失败: {e}")
        finally:
            os.unlink(tmp_path)

st.divider()

# 已有数据概览
st.subheader("📋 已有数据")
search = st.text_input("搜索", placeholder="输入关键词...")

if search:
    collection = vs.get_preferences_collection() if collection_type == "preferences" else vs.get_cases_collection()
    results = collection.similarity_search(search, k=10)
    for doc in results:
        meta = doc.metadata
        with st.expander(f"{meta.get('name', meta.get('destination', '未命名'))} - {meta.get('rating', 'N/A')}★"):
            st.write(doc.page_content)
            st.caption(f"标签: {meta.get('tags', [])}")
