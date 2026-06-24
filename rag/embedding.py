from langchain_community.embeddings import DashScopeEmbeddings


def create_embeddings(api_key: str, model: str = "text-embedding-v3") -> DashScopeEmbeddings:
    """创建阿里百炼 Embedding 实例"""
    return DashScopeEmbeddings(
        dashscope_api_key=api_key,
        model=model,
    )
