"""HybridRetriever 单元测试（使用 mock 通道检索器）。

这些测试锁的是"融合语义"本身，与底层向量/关键词实现无关：
  1. 融合只看名次或归一化后的分数 —— 把某一路的分数整体放大 100 倍，
     结果顺序必须不变（历史版本在 HYBRID 里硬编码 `score * 1.1`，属于跨尺度误判）；
  2. top_k 必须生效（历史版本关键词通道返回全量节点且 score=None，
     被 SimilarityPostprocessor 全部丢弃，"混合检索"实际等于纯向量检索）；
  3. vector_weight 必须真正参与 FUSION 的加权；
  4. retrieval_type / 分数范围契约（0~1、非 None）—— MCP 展示层和
     retrieval_stats 都依赖它。

真实语料的 BM25 召回测试见 tests/test_keyword_index.py。
"""

from unittest.mock import MagicMock

import pytest
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode

from dsm5_rag.retriever import HybridRetriever, rescale_scores_to_unit


def test_bm25_dependency_is_importable():
    """关键词通道依赖 llama-index-retrievers-bm25，装不上应当立刻暴露。"""
    import llama_index.retrievers.bm25 as bm25_module

    assert hasattr(bm25_module, "BM25Retriever")


def make_node(node_id: str, text: str = "", score: float = 0.5) -> NodeWithScore:
    """创建 NodeWithScore（每个节点是独立对象，模拟两路各自反序列化出的副本）。"""
    node = TextNode(text=text or f"text of {node_id}", id_=node_id)
    return NodeWithScore(node=node, score=score)


def make_retriever(nodes: list) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = nodes
    retriever.similarity_top_k = len(nodes)
    return retriever


def scaled(nodes: list, factor: float) -> list:
    """把一路的分数整体放大 factor 倍，用来验证融合不被分数量纲影响。"""
    return [make_node(n.node.node_id, n.node.text, (n.score or 0.0) * factor) for n in nodes]


class TestHybridRetrieverSingleChannel:
    """VECTOR_ONLY / KEYWORD_ONLY 两个单通道模式。"""

    def test_vector_only_mode(self):
        vector_nodes = [make_node("v1", "vector text", 0.9)]
        keyword_nodes = [make_node("k1", "keyword text", 0.8)]

        retriever = HybridRetriever(
            vector_retriever=make_retriever(vector_nodes),
            keyword_retriever=make_retriever(keyword_nodes),
            mode="VECTOR_ONLY",
        )

        result = retriever.retrieve(QueryBundle("test query"))
        assert len(result) == 1
        assert result[0].node.node_id == "v1"
        assert result[0].node.metadata["retrieval_type"] == "vector"

    def test_keyword_only_mode(self):
        vector_nodes = [make_node("v1", "vector text", 0.9)]
        keyword_nodes = [make_node("k1", "keyword text", 0.8)]

        retriever = HybridRetriever(
            vector_retriever=make_retriever(vector_nodes),
            keyword_retriever=make_retriever(keyword_nodes),
            mode="KEYWORD_ONLY",
        )

        result = retriever.retrieve(QueryBundle("test query"))
        assert len(result) == 1
        assert result[0].node.node_id == "k1"
        assert result[0].node.metadata["retrieval_type"] == "keyword"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            HybridRetriever(
                vector_retriever=make_retriever([]),
                keyword_retriever=make_retriever([]),
                mode="NOT_A_MODE",
            )


class TestHybridModeRankFusion:
    """HYBRID = Reciprocal Rank Fusion（core 自带实现）。"""

    def _nodes(self):
        vector = [make_node("c1", "common", 0.9), make_node("v1", "vector only", 0.6)]
        keyword = [make_node("k1", "keyword only", 0.8), make_node("c1", "common", 0.5)]
        return vector, keyword

    def test_hybrid_orders_by_rank_not_raw_score(self):
        vector, keyword = self._nodes()
        retriever = HybridRetriever(
            vector_retriever=make_retriever(vector),
            keyword_retriever=make_retriever(keyword),
            mode="HYBRID",
            similarity_top_k=3,
        )
        result = retriever.retrieve(QueryBundle("q"))

        # c1 在两路都命中（名次 0 和 1）→ RRF 分数最高；k1 单路第 0 名次之；v1 单路第 1 名最后
        assert [n.node.node_id for n in result] == ["c1", "k1", "v1"]
        assert [n.node.metadata["retrieval_type"] for n in result] == [
            "both",
            "keyword",
            "vector",
        ]

    def test_hybrid_is_scale_invariant(self):
        """把关键词那一路的分数整体放大 100 倍，融合结果必须一模一样。

        这是对历史缺陷 `node.score = node.score * 1.1` 的回归锁定：
        任何跨尺度的分数放大都会让这个断言失败。
        """
        vector_a, keyword_a = self._nodes()
        vector_b, keyword_b = self._nodes()
        plain = HybridRetriever(
            vector_retriever=make_retriever(vector_a),
            keyword_retriever=make_retriever(keyword_a),
            mode="HYBRID",
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))
        inflated = HybridRetriever(
            vector_retriever=make_retriever(vector_b),
            keyword_retriever=make_retriever(scaled(keyword_b, 100.0)),
            mode="HYBRID",
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))

        assert [n.node.node_id for n in plain] == [n.node.node_id for n in inflated]
        for a, b in zip(plain, inflated):
            assert abs((a.score or 0) - (b.score or 0)) < 1e-9

    def test_hybrid_scores_are_not_none_and_within_unit(self):
        vector, keyword = self._nodes()
        result = HybridRetriever(
            vector_retriever=make_retriever(vector),
            keyword_retriever=make_retriever(keyword),
            mode="HYBRID",
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))

        assert result
        for node in result:
            assert node.score is not None
            assert 0.0 <= node.score <= 1.0
        assert result[0].score == 1.0  # 缩放后最高分为 1.0

    def test_hybrid_respects_top_k(self):
        vector = [make_node(f"v{i}", score=1.0 - i * 0.1) for i in range(5)]
        keyword = [make_node(f"k{i}", score=1.0 - i * 0.1) for i in range(5)]
        result = HybridRetriever(
            vector_retriever=make_retriever(vector),
            keyword_retriever=make_retriever(keyword),
            mode="HYBRID",
            similarity_top_k=2,
        ).retrieve(QueryBundle("q"))
        assert len(result) == 2

    def test_hybrid_keyword_channel_survives_similarity_cutoff(self):
        """历史缺陷：关键词节点 score=None → 被 SimilarityPostprocessor 全删。"""
        from llama_index.core.postprocessor import SimilarityPostprocessor

        vector = [make_node("v1", score=0.3)]
        keyword = [make_node("k1", score=0.95), make_node("k2", score=0.9)]
        result = HybridRetriever(
            vector_retriever=make_retriever(vector),
            keyword_retriever=make_retriever(keyword),
            mode="HYBRID",
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))
        kept = SimilarityPostprocessor(similarity_cutoff=0.2).postprocess_nodes(
            result, QueryBundle("q")
        )
        kinds = {n.node.metadata.get("retrieval_type") for n in kept}
        assert "keyword" in kinds
        assert len(kept) > 0


class TestFusionModeWeighted:
    """FUSION = 尺度归一化后的加权融合，vector_weight 必须生效。"""

    def _retrievers(self):
        """两路各返回自己的节点；c1 在两路都命中（文本与 node_id 完全一致）。"""
        vector = [make_node("c1", "common", 1.0), make_node("v1", "vector only", 0.5)]
        keyword = [make_node("k1", "keyword only", 1.0), make_node("c1", "common", 0.4)]
        return vector, keyword

    def test_vector_weight_changes_winner(self):
        vector_a, keyword_a = self._retrievers()
        vector_b, keyword_b = self._retrievers()

        heavy_vector = HybridRetriever(
            vector_retriever=make_retriever(vector_a),
            keyword_retriever=make_retriever(keyword_a),
            mode="FUSION",
            vector_weight=0.9,
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))
        heavy_keyword = HybridRetriever(
            vector_retriever=make_retriever(vector_b),
            keyword_retriever=make_retriever(keyword_b),
            mode="FUSION",
            vector_weight=0.1,
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))

        assert heavy_vector[0].node.node_id == "c1"
        assert heavy_keyword[0].node.node_id == "k1"

    def test_fusion_scores_within_unit_and_not_none(self):
        vector, keyword = self._retrievers()
        result = HybridRetriever(
            vector_retriever=make_retriever(vector),
            keyword_retriever=make_retriever(keyword),
            mode="FUSION",
            vector_weight=0.7,
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))
        assert result
        for node in result:
            assert node.score is not None
            assert 0.0 <= node.score <= 1.0

    def test_fusion_records_channel_scores(self):
        vector, keyword = self._retrievers()
        result = HybridRetriever(
            vector_retriever=make_retriever(vector),
            keyword_retriever=make_retriever(keyword),
            mode="FUSION",
            vector_weight=0.7,
            similarity_top_k=3,
        ).retrieve(QueryBundle("q"))
        by_id = {n.node.node_id: n for n in result}
        assert by_id["c1"].node.metadata["retrieval_type"] == "both"
        assert "vector_score" in by_id["c1"].node.metadata
        assert "keyword_score" in by_id["c1"].node.metadata
        assert by_id["k1"].node.metadata["retrieval_type"] == "keyword"


class TestScoreRescaling:
    def test_rescale_empty(self):
        assert rescale_scores_to_unit([]) == []

    def test_rescale_all_zero(self):
        nodes = [make_node("a", score=0.0), make_node("b", score=0.0)]
        assert [n.score for n in rescale_scores_to_unit(nodes)] == [0.0, 0.0]

    def test_rescale_maps_max_to_one(self):
        nodes = [make_node("a", score=0.25), make_node("b", score=0.05)]
        result = rescale_scores_to_unit(nodes)
        assert result[0].score == 1.0
        assert abs(result[1].score - 0.2) < 1e-9
