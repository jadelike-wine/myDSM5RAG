"""混合检索器模块"""

from typing import List

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


class HybridRetriever(BaseRetriever):
    """结合向量检索和关键词检索的混合检索器"""

    def __init__(
        self,
        vector_retriever,
        keyword_retriever,
        mode: str = "HYBRID",
        vector_weight: float = 0.7,
    ):
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._mode = mode
        self._vector_weight = vector_weight
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """执行混合检索"""
        if self._mode == "VECTOR_ONLY":
            return self._vector_retriever.retrieve(query_bundle)
        elif self._mode == "KEYWORD_ONLY":
            return self._keyword_retriever.retrieve(query_bundle)
        elif self._mode == "FUSION":
            return self._fusion_retrieve(query_bundle)
        else:  # HYBRID 模式（默认）
            return self._hybrid_retrieve(query_bundle)

    def _hybrid_retrieve(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """HYBRID模式：分别检索，合并去重，加权排序"""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        # 为不同类型节点添加元数据标记
        for node in vector_nodes:
            node.node.metadata["retrieval_type"] = "vector"
        for node in keyword_nodes:
            node.node.metadata["retrieval_type"] = "keyword"
            if node.score is not None:
                node.score = node.score * 1.1

        # 合并节点并去重
        all_nodes_dict = {}
        for node in vector_nodes:
            node_id = node.node.node_id
            if node_id not in all_nodes_dict:
                all_nodes_dict[node_id] = node

        for node in keyword_nodes:
            node_id = node.node.node_id
            if node_id in all_nodes_dict:
                existing_node = all_nodes_dict[node_id]
                existing_score = (
                    existing_node.score if existing_node.score is not None else 0
                )
                new_score = node.score if node.score is not None else 0
                if new_score > existing_score:
                    all_nodes_dict[node_id] = node
            else:
                all_nodes_dict[node_id] = node

        all_nodes = list(all_nodes_dict.values())
        all_nodes.sort(
            key=lambda x: x.score if x.score is not None else 0, reverse=True
        )
        return all_nodes

    def _fusion_retrieve(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """FUSION模式：更复杂的融合检索策略"""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_dict = {node.node.node_id: node for node in vector_nodes}
        keyword_dict = {node.node.node_id: node for node in keyword_nodes}

        all_node_ids = set(vector_dict.keys()) | set(keyword_dict.keys())

        fused_nodes = []
        for node_id in all_node_ids:
            vector_node = vector_dict.get(node_id)
            keyword_node = keyword_dict.get(node_id)

            vector_score = (
                vector_node.score
                if vector_node and vector_node.score is not None
                else 0
            )
            keyword_score = (
                keyword_node.score
                if keyword_node and keyword_node.score is not None
                else 0
            )
            fusion_score = (
                vector_score * self._vector_weight
                + keyword_score * (1 - self._vector_weight)
            )

            if vector_node:
                final_node = vector_node
                retrieval_type = "both" if keyword_node else "vector"
            else:
                final_node = keyword_node
                retrieval_type = "keyword"

            final_node.score = fusion_score
            final_node.node.metadata["retrieval_type"] = retrieval_type
            final_node.node.metadata["vector_score"] = vector_score
            final_node.node.metadata["keyword_score"] = keyword_score
            final_node.node.metadata["fusion_score"] = fusion_score
            fused_nodes.append(final_node)

        fused_nodes.sort(
            key=lambda x: x.score if x.score is not None else 0, reverse=True
        )
        return fused_nodes