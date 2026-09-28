"""真实系统（假 embedding + 临时 Chroma + MockLLM）的检索链路集成测试。

这组测试刻意不使用 mock 节点：它跑的是 `tests/fixtures/corpus` 里的真实 DSM-5 片段，
覆盖 system.py 的构建/加载/查询/关键词搜索/参数隔离，能永久挡住历史上的 P0 缺陷
（关键词通道返回全量节点且 score=None → HYBRID 里关键词贡献 0 条）。
"""

import asyncio
from pathlib import Path

import pytest
from llama_index.core import QueryBundle

from dsm5_rag.config import RagConfig
from dsm5_rag.keyword_index import BM25KeywordIndex
from dsm5_rag.models import BatchQueryRequest, KeywordSearchRequest, QueryRequest
from dsm5_rag.system import DeepSeekRAGSystem

MDD_QUERY = "What are the diagnostic criteria for Major Depressive Disorder?"


def _make_config(corpus_dir: Path, root: Path) -> RagConfig:
    config = RagConfig()
    config.documents_path = str(corpus_dir)
    config.persist_dir = str(root / "persist")
    config.models_dir = str(root / "models")
    config.max_sources = 3
    config.similarity_cutoff = 0.2
    config.default_top_k = 5
    config.batch_concurrency = 2
    return config


@pytest.fixture(scope="module")
def system(corpus_dir, tmp_path_factory):
    """构建一次真实索引（用测试向量替身，不下载模型权重）。"""
    root = tmp_path_factory.mktemp("system")
    instance = DeepSeekRAGSystem(_make_config(corpus_dir, root), auto_build_index=False)
    instance._init_storage()
    asyncio.run(instance.build_index(force_rebuild=True))
    assert instance.index_built is True
    return instance


def file_names(nodes) -> list:
    return [n.node.metadata.get("file_name") for n in nodes]


class TestIndexBuild:
    def test_keyword_index_is_bm25(self, system):
        assert isinstance(system.keyword_index, BM25KeywordIndex)
        assert system.keyword_index.node_count > 0

    def test_status_reports_bm25_keyword_index(self, system):
        stats = system.get_status().index_stats
        assert stats["keyword_index_type"] == "BM25KeywordIndex"
        assert stats["keyword_index_nodes"] == system.keyword_index.node_count

    def test_manifest_written_after_build(self, system):
        assert (system.persist_dir / "index_manifest.json").is_file()

    def test_unchanged_rebuild_loads_instead_of_rebuilding(self, system):
        result = asyncio.run(system.build_index())
        assert result["status"] == "loaded"


class TestKeywordChannel:
    def test_keyword_only_respects_top_k(self, system):
        retriever = system.keyword_index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(QueryBundle(MDD_QUERY))
        assert 0 < len(nodes) <= 2

    def test_keyword_scores_are_normalized(self, system):
        nodes = system.keyword_index.as_retriever(similarity_top_k=3).retrieve(
            QueryBundle(MDD_QUERY)
        )
        assert nodes
        assert all(n.score is not None and 0.0 < n.score <= 1.0 for n in nodes)

    def test_icd_code_query_hits_mdd(self, system):
        nodes = system.keyword_index.as_retriever(similarity_top_k=2).retrieve(
            QueryBundle("296.3x")
        )
        assert "mdd_major_dep.txt" in file_names(nodes)

    async def test_keyword_search_endpoint_shape(self, system):
        result = await system.keyword_search(
            KeywordSearchRequest(keyword="296.3x", top_k=20)
        )
        assert result["count"] <= system.config.max_sources
        assert result["total_matched"] >= result["count"]
        assert result["similarity_cutoff"] == system.config.similarity_cutoff
        assert all(r["full_text"] is None for r in result["results"])
        assert all(r["score"] is not None for r in result["results"])

    async def test_keyword_search_verbose_only_when_asked(self, system):
        request = KeywordSearchRequest(keyword="anhedonia", top_k=3)
        plain = await system.keyword_search(request)
        verbose = await system.keyword_search(request, verbose=True)
        assert all(r["full_text"] is None for r in plain["results"])
        assert any(r["full_text"] for r in verbose["results"])


class TestHybridFusion:
    async def test_hybrid_keyword_channel_contributes(self, system):
        """关键词通道必须真的进上下文 —— 这就是当年 P0 缺陷的反面断言。"""
        result = await system.query(
            QueryRequest(question=MDD_QUERY, mode="HYBRID", similarity_top_k=5)
        )
        kinds = [s["retrieval_type"] for s in result["sources"]]
        assert any(kind in ("keyword", "both") for kind in kinds), kinds
        assert all(s["score"] is not None for s in result["sources"])

    async def test_vector_weight_and_top_k_are_honored_per_request(self, system):
        small = await system.query(
            QueryRequest(question=MDD_QUERY, mode="HYBRID", similarity_top_k=1)
        )
        large = await system.query(
            QueryRequest(question=MDD_QUERY, mode="HYBRID", similarity_top_k=5)
        )
        assert small["similarity_top_k"] == 1
        assert large["similarity_top_k"] == 5
        assert len(small["sources"]) <= 1
        assert len(large["sources"]) > len(small["sources"])

    async def test_concurrent_requests_do_not_pollute_each_other(self, system):
        requests = [
            QueryRequest(question=MDD_QUERY, mode="KEYWORD_ONLY", similarity_top_k=1),
            QueryRequest(question=MDD_QUERY, mode="VECTOR_ONLY", similarity_top_k=5),
            QueryRequest(question=MDD_QUERY, mode="FUSION", similarity_top_k=2, vector_weight=0.2),
            QueryRequest(question="296.3x", mode="KEYWORD_ONLY", similarity_top_k=3),
        ]
        results = await asyncio.gather(*(system.query(r) for r in requests))

        for request, result in zip(requests, results):
            assert result["mode"] == request.mode
            assert result["similarity_top_k"] == request.similarity_top_k
        # 服务级默认值不受请求影响
        assert system.default_mode in ("HYBRID", "VECTOR_ONLY", "KEYWORD_ONLY", "FUSION")
        assert system.default_top_k == 5

    async def test_timings_are_real(self, system):
        result = await system.query(QueryRequest(question=MDD_QUERY, mode="HYBRID"))
        assert result["retrieval_ms"] > 0
        assert result["llm_ms"] >= 0
        assert result["response_time"] < 60  # 旧实现返回的是 uptime，必然 >= 查询耗时
        assert result["total_ms"] >= result["retrieval_ms"]


class TestBatchQuery:
    async def test_batch_results_keep_order_and_limit(self, system):
        request = BatchQueryRequest(
            questions=["296.3x", "300.02", "anhedonia"], mode="KEYWORD_ONLY"
        )
        result = await system.batch_query(request)
        assert [r["question"] for r in result["results"]] == request.questions
        assert result["successful"] == 3
        assert result["concurrency"] == system.config.batch_concurrency

    def test_batch_rejects_too_many_questions(self, system):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchQueryRequest(questions=[f"q{i}" for i in range(21)])


class TestDefaultSearchParams:
    def test_mode_change_then_request_without_mode_uses_default(self, system, corpus_dir):
        system.set_default_search_params(mode="KEYWORD_ONLY", similarity_top_k=2)
        try:
            # 请求没写 mode/top_k → 用服务默认值
            request = QueryRequest(question="296.3x")
            mode, top_k, _weight = system.params_from_request(request)
            assert (mode, top_k) == ("KEYWORD_ONLY", 2)
            # 请求显式写了 → 用自己的值，但不改动服务默认值
            explicit = QueryRequest(question="296.3x", mode="VECTOR_ONLY", similarity_top_k=4)
            assert system.params_from_request(explicit)[:2] == ("VECTOR_ONLY", 4)
            assert system.default_mode == "KEYWORD_ONLY"
        finally:
            system.set_default_search_params(mode="HYBRID", similarity_top_k=5)

    def test_invalid_mode_rejected(self, system):
        with pytest.raises(ValueError):
            system.set_default_search_params(mode="NOPE")


class TestReloadFromDisk:
    def test_second_instance_loads_bm25(self, system, corpus_dir, tmp_path_factory):
        """模拟服务重启：新实例从同一 persist_dir 加载，关键词通道依然可用。"""
        fresh = DeepSeekRAGSystem(
            _make_config(corpus_dir, tmp_path_factory.mktemp("reload")),
            auto_build_index=False,
        )
        fresh.config.persist_dir = str(system.persist_dir)
        fresh.persist_dir = system.persist_dir
        fresh._init_storage()
        asyncio.run(fresh._load_existing_index())

        assert fresh.index_built is True
        assert isinstance(fresh.keyword_index, BM25KeywordIndex)
        nodes = fresh.keyword_index.as_retriever(similarity_top_k=2).retrieve(
            QueryBundle("296.3x")
        )
        assert nodes and all(n.score is not None for n in nodes)
        assert "mdd_major_dep.txt" in file_names(nodes)
