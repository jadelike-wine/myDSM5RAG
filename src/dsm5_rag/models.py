"""Pydantic 数据模型定义"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求模型"""

    question: str = Field(..., description="要查询的问题")
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


class KeywordSearchRequest(BaseModel):
    """关键词搜索请求模型"""

    keyword: str = Field(..., description="要搜索的关键词")
    top_k: int = Field(default=5, ge=1, le=20, description="检索数量")


class BatchQueryRequest(BaseModel):
    """批量查询请求模型"""

    questions: List[str] = Field(..., description="问题列表")
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