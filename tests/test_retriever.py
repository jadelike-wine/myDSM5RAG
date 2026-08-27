"""HybridRetriever 单元测试（使用 mock）"""

from unittest.mock import MagicMock, patch

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode

from dsm5_rag.retriever import HybridRetriever


def make_mock_node(node_id: str, text: str, score: float = 0.5):
    """创建 mock NodeWithScore"""
    node = TextNode(text=text, id_=node_id)
    return NodeWithScore(node=node, score=score)


def make_mock_retriever(nodes: list):
    """创建 mock retriever"""
    retriever = MagicMock()
    retriever.retrieve.return_value = nodes
    return retriever


class TestHybridRetriever:
    """测试 HybridRetriever"""

    def test_vector_only_mode(self):
        """测试 VECTOR_ONLY 模式"""
        vector_nodes = [make_mock_node("v1", "vector text", 0.9)]
        keyword_nodes = [make_mock_node("k1", "keyword text", 0.8)]

        retriever = HybridRetriever(
            vector_retriever=make_mock_retriever(vector_nodes),
            keyword_retriever=make_mock_retriever(keyword_nodes),
            mode="VECTOR_ONLY",
        )

        result = retriever.retrieve(QueryBundle("test query"))
        assert len(result) == 1
        assert result[0].node.node_id == "v1"

    def test_keyword_only_mode(self):
        """测试 KEYWORD_ONLY 模式"""
        vector_nodes = [make_mock_node("v1", "vector text", 0.9)]
        keyword_nodes = [make_mock_node("k1", "keyword text", 0.8)]

        retriever = HybridRetriever(
            vector_retriever=make_mock_retriever(vector_nodes),
            keyword_retriever=make_mock_retriever(keyword_nodes),
            mode="KEYWORD_ONLY",
        )

        result = retriever.retrieve(QueryBundle("test query"))
        assert len(result) == 1
        assert result[0].node.node_id == "k1"

    def test_hybrid_mode_merge(self):
        """测试 HYBRID 模式合并去重"""
        # 两个检索器有重叠的节点
        common_node = make_mock_node("c1", "common text", 0.7)
        vector_only = make_mock_node("v1", "vector only", 0.6)
        keyword_only = make_mock_node("k1", "keyword only", 0.8)

        retriever = HybridRetriever(
            vector_retriever=make_mock_retriever([common_node, vector_only]),
            keyword_retriever=make_mock_retriever([common_node, keyword_only]),
            mode="HYBRID",
        )

        result = retriever.retrieve(QueryBundle("test query"))
        # 应该去重后只有3个节点
        assert len(result) == 3
        # 按分数排序，k1(0.8*1.1=0.88) > c1(0.7) > v1(0.6)
        assert result[0].node.node_id == "k1"
        assert result[1].node.node_id == "c1"
        assert result[2].node.node_id == "v1"

    def test_fusion_mode(self):
        """测试 FUSION 模式加权融合"""
        vector_nodes = [make_mock_node("c1", "common", 0.8)]
        keyword_nodes = [make_mock_node("c1", "common", 0.6)]

        retriever = HybridRetriever(
            vector_retriever=make_mock_retriever(vector_nodes),
            keyword_retriever=make_mock_retriever(keyword_nodes),
            mode="FUSION",
            vector_weight=0.7,
        )

        result = retriever.retrieve(QueryBundle("test query"))
        assert len(result) == 1
        # 融合分数 = 0.8 * 0.7 + 0.6 * 0.3 = 0.56 + 0.18 = 0.74
        assert abs(result[0].score - 0.74) < 0.01
        assert result[0].node.metadata["retrieval_type"] == "both"