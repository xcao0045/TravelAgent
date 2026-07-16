import streamlit as st
import os
import tempfile
from datetime import datetime
from config import Settings
from rag.embedding import create_embeddings
from rag.vector_store import VectorStoreManager
from rag.data_loader import load_file_to_docs
from rag.dedup import md5_hash, dedup_pipeline
from langchain_core.documents import Document

PAGE_SIZE = 5

st.set_page_config(page_title="知识库管理", page_icon="📚")

settings = Settings.from_env()

# 初始化
@st.cache_resource
def get_vector_store():
    embeddings = create_embeddings(settings.bailian_api_key, settings.embedding_model)
    return VectorStoreManager(settings.chroma_persist_dir, embeddings,
                              chunk_size=settings.chunk_size,
                              chunk_overlap=settings.chunk_overlap)

vs = get_vector_store()

st.title("📚 知识库管理")

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
                # 去重检查
                coll = vs.get_preferences_collection()
                existing = coll.get()
                result = dedup_pipeline(
                    doc=doc, collection=coll,
                    collection_type="preferences",
                    existing_texts=existing.get("documents", []) or [],
                    existing_metas=existing.get("metadatas", []) or [],
                    options={"md5": True, "field": False, "semantic": False},
                )
                if result["status"] == "duplicate":
                    st.warning(f"⚠️ '{name}' 内容重复，跳过入库")
                else:
                    if result["status"] == "suspected":
                        st.warning(f"⚠️ '{name}' 疑似重复 ({result['reason']})，仍将入库")
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
                # 去重检查
                coll = vs.get_cases_collection()
                existing = coll.get()
                result = dedup_pipeline(
                    doc=doc, collection=coll,
                    collection_type="cases",
                    existing_texts=existing.get("documents", []) or [],
                    existing_metas=existing.get("metadatas", []) or [],
                    options={"md5": True, "field": False, "semantic": False},
                )
                if result["status"] == "duplicate":
                    st.warning(f"⚠️ '{destination} {days}天' 案例内容重复，跳过入库")
                else:
                    if result["status"] == "suspected":
                        st.warning(f"⚠️ '{destination} {days}天' 疑似重复 ({result['reason']})，仍将入库")
                    vs.add_to_cases([doc])
                    st.success(f"✅ '{destination} {days}天' 案例已入库")
                    st.rerun()

# --- 文件上传 ---
with tab2:
    if collection_type == "preferences":
        allowed_types = ["csv", "json"]
        st.info("支持格式: CSV, JSON")
    else:
        allowed_types = ["json", "md", "txt", "csv"]
        st.info("支持格式: JSON, Markdown, TXT, CSV")

    uploaded_file = st.file_uploader("选择文件", type=allowed_types)
    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_kb_last_upload") == file_key:
            st.stop()

        # Save original to local archive
        archive_dir = os.path.join("data", "knowledge_base", collection_type)
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, uploaded_file.name)
        with open(archive_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Save to temp for processing
        tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            docs = load_file_to_docs(tmp_path, collection_type)
            # 文件批量去重（MD5检查）
            coll = vs.get_preferences_collection() if collection_type == "preferences" else vs.get_cases_collection()
            existing = coll.get()
            existing_texts = existing.get("documents", []) or []
            existing_metas = existing.get("metadatas", []) or []

            new_docs = []
            dup_count = 0
            for doc in docs:
                result = dedup_pipeline(
                    doc=doc, collection=coll,
                    collection_type=collection_type,
                    existing_texts=existing_texts,
                    existing_metas=existing_metas,
                    options={"md5": True, "field": False, "semantic": False},
                )
                if result["status"] == "duplicate":
                    dup_count += 1
                    continue
                if result["status"] == "suspected":
                    st.warning(f"⚠️ 疑似重复: {doc.page_content[:50]}... ({result['reason']})")
                new_docs.append(doc)
                # 更新已检查列表，防止本次上传内部重复
                existing_texts.append(doc.page_content)
                existing_metas.append(doc.metadata)

            if new_docs:
                for d in new_docs:
                    d.metadata["source_file"] = archive_path
                if collection_type == "preferences":
                    vs.add_to_preferences(new_docs)
                else:
                    vs.add_to_cases(new_docs)

            if dup_count > 0:
                st.warning(f"⚠️ {dup_count}条跳过(重复), {len(new_docs)}条新增")
            else:
                st.success(f"✅ 成功导入 {len(new_docs)} 条数据")
            st.session_state["_kb_last_upload"] = file_key
        except Exception as e:
            st.error(f"导入失败: {e}")
        finally:
            os.unlink(tmp_path)

st.divider()

# 文档列表
st.subheader("📋 已有文档")
search = st.text_input("搜索（可选）", placeholder="输入关键词筛选，留空查看全部...")

docs = vs.list_documents(collection_type)

# 语义搜索筛选
if search.strip():
    coll = vs.get_preferences_collection() if collection_type == "preferences" else vs.get_cases_collection()
    raw_results = coll.similarity_search_with_relevance_scores(search, k=len(docs) * 3 or 20)
    matched_md5s = {doc.metadata.get("source_md5", "") for doc, score in raw_results if score >= 0.3}
    docs = [d for d in docs if d["source_md5"] in matched_md5s]

if not docs:
    st.info("暂无文档，去录入或上传一些数据吧！")
else:
    total_pages = max(1, (len(docs) + PAGE_SIZE - 1) // PAGE_SIZE)
    if "kb_page" not in st.session_state:
        st.session_state.kb_page = 0
    page = st.session_state.kb_page
    if page >= total_pages:
        page = 0
        st.session_state.kb_page = 0

    start = page * PAGE_SIZE
    page_docs = docs[start:start + PAGE_SIZE]

    st.caption(f"共 {len(docs)} 个文档，第 {page+1}/{total_pages} 页")

    for d in page_docs:
        sm5 = d["source_md5"]
        label = d["title"]
        if d.get("category"):
            label = f"[{d['category']}] {label}"
        meta_line = f"⭐ {d['rating']} · {d['chunk_count']} chunks"
        if d.get("tags"):
            meta_line += f" · 标签: {', '.join(d['tags'])}"

        with st.container(border=True):
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f"**{label}**")
                st.caption(meta_line)
                with st.expander("🔍 预览"):
                    coll = vs.get_preferences_collection() if collection_type == "preferences" else vs.get_cases_collection()
                    chunks = coll.get(
                        where={"source_md5": sm5}, limit=2
                    )
                    if chunks["documents"]:
                        preview = "\n\n".join(chunks["documents"])
                        st.write(preview[:1000])
            with cols[1]:
                if st.button("🗑️ 删除", key=f"del_{sm5}", use_container_width=True):
                    deleted = vs.delete_by_source(sm5, collection_type)
                    st.success(f"已删除 {deleted} 个片段" + (" 及归档文件" if d.get("source_file") else ""))
                    st.session_state.kb_page = 0
                    st.rerun()

    # 分页控件
    if total_pages > 1:
        cols = st.columns(5)
        with cols[0]:
            if st.button("⏮️ 首页", disabled=(page == 0)):
                st.session_state.kb_page = 0
                st.rerun()
        with cols[1]:
            if st.button("◀ 上一页", disabled=(page == 0)):
                st.session_state.kb_page = page - 1
                st.rerun()
        with cols[2]:
            st.caption(f"{page+1} / {total_pages}")
        with cols[3]:
            if st.button("下一页 ▶", disabled=(page >= total_pages - 1)):
                st.session_state.kb_page = page + 1
                st.rerun()
        with cols[4]:
            if st.button("末页 ⏭️", disabled=(page >= total_pages - 1)):
                st.session_state.kb_page = total_pages - 1
                st.rerun()
