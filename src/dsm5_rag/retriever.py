"""混合检索器模块。

四个通道/融合语义：
  - VECTOR_ONLY   仅向量（余弦分数，0~1）
  - KEYWORD_ONLY  仅关键词（BM25，经 `normalize_keyword_nodes` 归一到 0~1）
  - HYBRID        两路各自检索后按名次做 Reciprocal Rank Fusion（core 自带实现）
  - FUSION        两路各自检索后做尺度归一化（min-max）的加权融合，
                  `vector_weight` 在这里真正生效

历史包袱：早期版本在 HYBRID 里对关键词分数硬编码 `score * 1.1`，并在 FUSION 里
把 BM25 分数和余弦分数直接线性相加 —— 两种分数尺度不同，属于跨尺度误判；而且
当时关键词通道返回的是 `score=None` 的全量节点，融合结果里关键词实际贡献 0 条。
现在统一为：各通道分数先归一到 0~1，再用 core 的 `QueryFusionRetriever`
（`num_queries=1`，不请求 LLM 生成查询变体）融合，最后把融合分数线性缩放到 0~1，
使 `similarity_cutoff`、MCP `_format_nodes`、`retrieval_stats` 的语义保持一致。
"""

from typing import Dict, List, Optional, Set

from llama_index.core import QueryBundle
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.retrievers.fusion_retriever import (
    FUSION_MODES,
    QueryFusionRetriever,
)
from llama_index.core.schema import NodeWithScore

from dsm5_rag.background import run_in_executor

VALID_MODES = ("VECTOR_ONLY", "KEYWORD_ONLY", "HYBRID", "FUSION")


def resolve_top_k(
    explicit: Optional[int], *retrievers: object
) -> int:
    """确定融合后返回的条数：显式参数 > 子检索器 > 全局默认。"""
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    for retriever in retrievers:
        value = getattr(retriever, "similarity_top_k", None)
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_SIMILARITY_TOP_K


def rescale_scores_to_unit(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
    """把分数线性缩放到 0~1（最高分为 1.0），不改变名次。

    RRF 的原始分数是 `1/(rank+60)` 量级（约 0.016~0.033），若原样透出，
    `SimilarityPostprocessor(similarity_cutoff=0.2)` 会把结果全部删掉。
    """
    if not nodes:
        return nodes
    max_score = max(node.score or 0.0 for node in nodes)
    if max_score <= 0:
        for node in nodes:
            node.score = 0.0
        return nodes
    for node in nodes:
        node.score = float(node.score or 0.0) / max_score
    return nodes


class _ChannelRetriever(BaseRetriever):
    """包装一路检索器，记录本次命中的 node_id，供融合后回填 retrieval_type。

    注意：这里**不**往节点 metadata 里写 `retrieval_type`。节点的 `hash` 包含
    metadata，两路若各自打上不同标记，`QueryFusionRetriever` 按 hash 去重就会
    失效（同一节点被当成两个不同节点）。标记统一由 `HybridRetriever` 在融合
    之后写。
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        retrieval_type: str,
        offload_sync: bool = False,
    ):
        super().__init__()
        self._retriever = retriever
        self._retrieval_type = retrieval_type
        # 向量通道（Chroma 查询是同步调用）在异步路径里要挪到线程池，
        # 否则会把服务的事件循环堵住。关键词通道自己已经做了 offload。
        self._offload_sync = offload_sync
        self.seen_scores: Dict[str, float] = {}

    @property
    def retrieval_type(self) -> str:
        return self._retrieval_type

    def reset(self) -> None:
        self.seen_scores = {}

    @property
    def seen_ids(self) -> Set[str]:
        return set(self.seen_scores.keys())

    def _record(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        for node in nodes:
            self.seen_scores[node.node.node_id] = float(node.score or 0.0)
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._record(self._retriever.retrieve(query_bundle))

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._offload_sync:
            return self._record(
                await run_in_executor(self._retriever.retrieve, query_bundle)
            )
        return self._record(await self._retriever.aretrieve(query_bundle))


class HybridRetriever(BaseRetriever):
    """结合向量检索和关键词检索的混合检索器。

    实例不是并发安全的（`_ChannelRetriever` 会记录本次命中的 node_id），
    因此请按请求构造、用完即弃 —— `DeepSeekRAGSystem` 就是这么做的。
    """

    def __init__(
        self,
        vector_retriever,
        keyword_retriever,
        mode: str = "HYBRID",
        vector_weight: float = 0.7,
        similarity_top_k: Optional[int] = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(
                f"无效的检索模式: {mode}，可选: {', '.join(VALID_MODES)}"
            )

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._mode = mode
        self._vector_weight = max(0.0, min(1.0, float(vector_weight)))
        self._similarity_top_k = resolve_top_k(
            similarity_top_k, vector_retriever, keyword_retriever
        )
        super().__init__()

        self._vector_channel = _ChannelRetriever(
            vector_retriever, "vector", offload_sync=True
        )
        # 关键词通道（KeywordRetriever）自己已经把 BM25 打分放在线程池里，
        # 这里不能再 offload，否则会在线程里再次取事件循环而报错。
        self._keyword_channel = _ChannelRetriever(
            keyword_retriever, "keyword", offload_sync=False
        )
        self._fusion_retriever = self._build_fusion_retriever()

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def similarity_top_k(self) -> int:
        return self._similarity_top_k

    @property
    def vector_weight(self) -> float:
        return self._vector_weight

    def _build_fusion_retriever(self) -> Optional[QueryFusionRetriever]:
        """HYBRID/FUSION 复用 core 的 QueryFusionRetriever，不做查询改写。"""
        if self._mode not in ("HYBRID", "FUSION"):
            return None

        if self._mode == "HYBRID":
            mode = FUSION_MODES.RECIPROCAL_RANK
            weights: Optional[List[float]] = None
        else:
            mode = FUSION_MODES.RELATIVE_SCORE
            weights = [self._vector_weight, 1.0 - self._vector_weight]

        return QueryFusionRetriever(
            [self._vector_channel, self._keyword_channel],
            # num_queries=1：不让 LLM 生成查询变体；传 MockLLM 只是为了
            # 避免 core 去读全局 Settings.llm —— 纯检索的 MCP 服务不应该
            # 依赖任何 LLM 配置（Settings.llm 未设置时会直接报缺 OPENAI_API_KEY）。
            llm=MockLLM(),
            num_queries=1,
            mode=mode,
            similarity_top_k=self._similarity_top_k,
            retriever_weights=weights,
            use_async=False,
        )

    # ------------------------------------------------------------------
    def _tag_single_channel(
        self, nodes: List[NodeWithScore], retrieval_type: str
    ) -> List[NodeWithScore]:
        for node in nodes:
            if node.score is None:
                node.score = 0.0
            node.node.metadata["retrieval_type"] = retrieval_type
        return nodes

    def _finalize(
        self,
        nodes: List[NodeWithScore],
        vector_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
    ) -> List[NodeWithScore]:
        """回填 retrieval_type（both/vector/keyword）、各通道原始分数，
        并把融合分数缩放到 0~1。"""
        for node in nodes:
            node_id = node.node.node_id
            in_vector = node_id in vector_scores
            in_keyword = node_id in keyword_scores
            if in_vector and in_keyword:
                node.node.metadata["retrieval_type"] = "both"
            elif in_vector:
                node.node.metadata["retrieval_type"] = "vector"
            elif in_keyword:
                node.node.metadata["retrieval_type"] = "keyword"
            if in_vector:
                node.node.metadata["vector_score"] = vector_scores[node_id]
            if in_keyword:
                node.node.metadata["keyword_score"] = keyword_scores[node_id]
            if node.score is None:
                node.score = 0.0

        nodes = rescale_scores_to_unit(nodes)
        for node in nodes:
            # 融合分数统一为缩放后的 0~1 值，供 MCP 展示层与排查使用
            node.node.metadata["fusion_score"] = float(node.score or 0.0)
        return nodes

    @property
    def _active_channel(self) -> Optional[_ChannelRetriever]:
        if self._mode == "VECTOR_ONLY":
            return self._vector_channel
        if self._mode == "KEYWORD_ONLY":
            return self._keyword_channel
        return None

    # ------------------------------------------------------------------
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """执行混合检索。"""
        channel = self._active_channel
        if channel is not None:
            return self._tag_single_channel(
                channel.retrieve(query_bundle), channel.retrieval_type
            )

        self._vector_channel.reset()
        self._keyword_channel.reset()
        assert self._fusion_retriever is not None
        nodes = self._fusion_retriever.retrieve(query_bundle)
        return self._finalize(
            nodes,
            self._vector_channel.seen_scores,
            self._keyword_channel.seen_scores,
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        channel = self._active_channel
        if channel is not None:
            return self._tag_single_channel(
                await channel.aretrieve(query_bundle), channel.retrieval_type
            )

        self._vector_channel.reset()
        self._keyword_channel.reset()
        assert self._fusion_retriever is not None
        nodes = await self._fusion_retriever.aretrieve(query_bundle)
        return self._finalize(
            nodes,
            self._vector_channel.seen_scores,
            self._keyword_channel.seen_scores,
        )
