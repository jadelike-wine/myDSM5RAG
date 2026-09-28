"""给 API/端点测试用的假 RAG 系统。

它的职责不是复现检索质量（那由 tests/test_system_integration.py 用真实小语料验证），
而是把 `DeepSeekRAGSystem` 的**输出契约**如实钉住：字段名、流式事件的顺序与行格式、
错误分支。因此这里的返回值结构与 system.py 的真实实现保持一致，
新增字段时两边要一起改。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from dsm5_rag.config import RagConfig
from dsm5_rag.models import IndexStatus, SystemStatus

VALID_MODES = ("HYBRID", "VECTOR_ONLY", "KEYWORD_ONLY", "FUSION")


def _source(rank: int, retrieval_type: str = "both") -> Dict[str, Any]:
    return {
        "rank": rank,
        "file_name": "dsm5_sample.txt",
        "file_type": "txt",
        "page": 12,
        "retrieval_type": retrieval_type,
        "score": round(1.0 - 0.2 * rank, 4),
        "text_length": 512,
        "text_preview": "Major Depressive Disorder (MDD) ...",
        "full_text": None,
    }


class FakeRAGSystem:
    """实现 api.py 用到的全部方法，可控制 index 状态与抛错。"""

    def __init__(
        self,
        index_built: bool = True,
        index_building: bool = False,
        initialized: bool = True,
        query_error: str | None = None,
        stream_chunks: List[str] | None = None,
    ) -> None:
        self.config = RagConfig()
        self.index_built = index_built
        self.index_building = index_building
        self.initialized = initialized
        self.query_error = query_error
        self.stream_chunks = stream_chunks or ["第一段", "第二段", "第三段"]
        self.default_mode = "HYBRID"
        self.default_top_k = 5
        self.default_vector_weight = 0.7
        self.total_documents = 1
        self.documents_path = "/tmp/documents"
        self.persist_dir = "/tmp/persist"
        self.embedding_model = self.config.embedding_model
        self.app_start_time = datetime.now()
        self.queries: List[Any] = []
        self.rebuild_calls: List[bool] = []

    # ------------------------------------------------------------------
    def get_status(self) -> SystemStatus:
        return SystemStatus(
            status="running" if self.initialized else "initializing",
            documents_path=self.documents_path,
            db_dir=self.persist_dir,
            embedding_model=self.embedding_model,
            index_built=self.index_built,
            index_stats={
                "vector_index_type": "VectorStoreIndex",
                "keyword_index_type": "BM25KeywordIndex",
                "keyword_index_nodes": 7,
                "query_engine_ready": self.index_built,
                "current_mode": self.default_mode,
                "document_count": self.total_documents,
            },
            uptime=1.5,
            document_count=self.total_documents,
            supported_formats=[".pdf", ".docx", ".txt"],
        )

    def get_index_status(self) -> IndexStatus:
        if self.index_building:
            return IndexStatus(
                status="building", message="索引正在构建中", progress=0.5, details={}
            )
        if self.index_built:
            return IndexStatus(
                status="ready",
                message="DSM-5知识库索引已构建完成",
                progress=1.0,
                details={"document_count": self.total_documents, "mode": self.default_mode},
            )
        return IndexStatus(
            status="not_built",
            message="索引未构建",
            progress=0.0,
            details={"document_count": 0},
        )

    async def list_documents(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "documents": [
                {
                    "file_name": "dsm5_sample.txt",
                    "file_path": "/tmp/documents/dsm5_sample.txt",
                    "file_size_mb": 0.01,
                    "file_type": ".txt",
                    "last_modified": datetime.now().isoformat(),
                }
            ],
            "supported_formats": [".pdf", ".docx", ".txt"],
        }

    # ------------------------------------------------------------------
    def _answer(self, question: str, mode: str, top_k: int) -> Dict[str, Any]:
        return {
            "question": question,
            "answer": f"基于 DSM-5 的回答：{question}",
            "mode": mode,
            "similarity_top_k": top_k,
            "vector_weight": self.default_vector_weight,
            "timestamp": datetime.now().isoformat(),
            "response_time": 0.123,
            "retrieval_ms": 40.5,
            "llm_ms": 82.0,
            "total_ms": 123.0,
            "sources": [_source(1, "both"), _source(2, "keyword")],
            "retrieval_stats": {"vector": 0, "keyword": 1, "both": 1},
            "sources_limited": "显示前2个来源（共2个）",
        }

    async def query(self, request, verbose: bool | None = None) -> Dict[str, Any]:
        self.queries.append(request)
        if self.query_error:
            raise RuntimeError(self.query_error)
        if request.stream:
            return self._stream(request, verbose)
        result = self._answer(request.question, request.mode, request.similarity_top_k)
        if verbose if verbose is not None else request.verbose:
            for source in result["sources"]:
                source["full_text"] = "完整原文"
        return result

    def _stream(self, request, verbose: bool | None) -> Dict[str, Any]:
        def ndjson(payload: Dict[str, Any]) -> str:
            return json.dumps(payload, ensure_ascii=False) + "\n"

        async def generate():
            yield ndjson(
                {
                    "event": "query_started",
                    "question": request.question,
                    "mode": request.mode,
                    "similarity_top_k": request.similarity_top_k,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            yield ndjson(
                {"event": "retrieval_completed", "count": 2, "retrieval_ms": 40.5}
            )
            for chunk in self.stream_chunks:
                yield ndjson(
                    {"event": "answer_chunk", "chunk": chunk, "is_complete": False}
                )
            yield ndjson(
                {
                    "event": "answer_chunk",
                    "chunk": "",
                    "is_complete": True,
                    "answer_chars": sum(len(c) for c in self.stream_chunks),
                    "response_time": 0.123,
                    "retrieval_ms": 40.5,
                    "llm_ms": 82.0,
                    "ttft_ms": 1.2,
                    "total_ms": 123.0,
                }
            )
            yield ndjson(
                {
                    "event": "sources",
                    "sources": [
                        {
                            "rank": 1,
                            "file_name": "dsm5_sample.txt",
                            "retrieval_type": "both",
                            "score": 0.8,
                        }
                    ],
                    "total_sources": 1,
                    "retrieval_stats": {"vector": 0, "keyword": 0, "both": 1},
                }
            )
            yield ndjson(
                {
                    "event": "query_completed",
                    "timestamp": datetime.now().isoformat(),
                    "total_ms": 123.0,
                }
            )

        return {
            "streaming_response": generate(),
            "media_type": "application/x-ndjson",
        }

    async def keyword_search(
        self, request, verbose: bool | None = None
    ) -> Dict[str, Any]:
        include_full = request.verbose if verbose is None else bool(verbose)
        item = {
            "rank": 1,
            "file_name": "dsm5_sample.txt",
            "file_type": "txt",
            "page": 3,
            "retrieval_type": "keyword",
            "score": 1.0,
            "text_length": 200,
            "text_preview": "Major Depressive Disorder (MDD) ...",
            "full_text": "Major Depressive Disorder 全文" if include_full else None,
        }
        return {
            "keyword": request.keyword,
            "results": [item],
            "count": 1,
            "total_matched": 2,
            "filtered_by_cutoff": 1,
            "similarity_cutoff": self.config.similarity_cutoff,
            "max_sources": self.config.max_sources,
            "retrieval_ms": 3.2,
            "timestamp": datetime.now().isoformat(),
        }

    async def batch_query(self, request) -> Dict[str, Any]:
        return {
            "mode": request.mode,
            "results": [
                {"question": q, "answer": f"回答：{q}", "success": True}
                for q in request.questions
            ],
            "total": len(request.questions),
            "successful": len(request.questions),
            "concurrency": self.config.batch_concurrency,
            "response_time": 0.5,
            "timestamp": datetime.now().isoformat(),
        }

    def set_default_search_params(
        self,
        mode: str | None = None,
        similarity_top_k: int | None = None,
        vector_weight: float | None = None,
    ) -> tuple:
        if mode is not None:
            if mode not in VALID_MODES:
                raise ValueError(f"无效的检索模式: {mode}")
            self.default_mode = mode
        if similarity_top_k is not None:
            self.default_top_k = int(similarity_top_k)
        if vector_weight is not None:
            self.default_vector_weight = float(vector_weight)
        return (
            self.default_mode,
            self.default_top_k,
            self.default_vector_weight,
        )

    async def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        self.rebuild_calls.append(force_rebuild)
        return {
            "status": "built" if force_rebuild else "loaded",
            "message": "索引构建成功",
            "documents": 5,
            "nodes": 12,
            "file_count": 5,
            "document_path": self.documents_path,
        }

    async def index_diff(self) -> Dict[str, Any]:
        return {
            "manifest_present": True,
            "up_to_date": True,
            "indexed_file_count": 5,
            "current_file_count": 5,
            "changed_since_index": {},
            "manifest_updated_at": datetime.now().isoformat(),
        }
