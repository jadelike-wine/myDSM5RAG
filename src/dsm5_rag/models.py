"""Pydantic 数据模型定义"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field, StringConstraints

# 单条提问的长度上限：防止一个请求就把 LLM 配额或网关打爆
QuestionText = Annotated[
    str, StringConstraints(min_length=1, max_length=4000)
]


class QueryRequest(BaseModel):
    """查询请求模型"""

    question: QuestionText = Field(..., description="要查询的问题（1~4000 字符）")
    mode: str = Field(
        default="HYBRID",
        description="检索模式: HYBRID, VECTOR_ONLY, KEYWORD_ONLY, FUSION",
    )
    similarity_top_k: int = Field(
        default=5, ge=1, le=20, description="检索数量"
    )
    vector_weight: float = Field(
        default=0.7, ge=0, le=1, description="向量权重（FUSION模式有效）"
    )
    stream: bool = Field(default=False, description="是否流式输出")
    verbose: bool = Field(
        default=False,
        description="是否在 sources 中返回完整原文 full_text（默认只返回预览）",
    )


# 关键词搜索的输入约束：空关键词在 BM25 里没有意义（分词后无 token）
KeywordText = Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class KeywordSearchRequest(BaseModel):
    """关键词搜索请求模型"""

    keyword: KeywordText = Field(..., description="要搜索的关键词（1~1000 字符）")
    top_k: int = Field(default=5, ge=1, le=20, description="检索数量")
    verbose: bool = Field(
        default=False,
        description="是否在结果中返回完整原文 full_text（默认只返回预览）",
    )


class BatchQueryRequest(BaseModel):
    """批量查询请求模型"""

    questions: List[QuestionText] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="问题列表（单次最多 20 条，每条 1~4000 字符）",
    )
    mode: str = Field(default="HYBRID", description="检索模式")


class SystemStatus(BaseModel):
    """系统状态模型"""

    status: str = Field(..., description="系统状态")
    documents_path: Optional[str] = Field(None, description="文档路径")
    db_dir: Optional[str] = Field(None, description="数据库目录")
    embedding_model: Optional[str] = Field(None, description="Embedding模型")
    index_built: bool = Field(..., description="索引是否已构建")
    index_stats: Optional[Dict[str, Any]] = Field(None, description="索引统计信息")
    uptime: Optional[float] = Field(None, description="运行时间（秒）")
    document_count: Optional[int] = Field(None, description="文档数量")
    supported_formats: Optional[List[str]] = Field(
        None, description="支持的文件格式"
    )


class IndexStatus(BaseModel):
    """索引状态模型"""

    status: str = Field(..., description="索引状态")
    message: str = Field(..., description="状态信息")
    progress: Optional[float] = Field(None, description="构建进度")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
