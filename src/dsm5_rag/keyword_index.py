"""BM25 关键词索引模块。

替代原先被误当作"关键词索引"使用的 `SummaryIndex`：
`SummaryIndex.as_retriever()` 返回的 `SummaryIndexRetriever` 会忽略查询、返回
docstore 里的全部节点且 `score=None`，导致关键词通道在 HYBRID/FUSION 融合、
MCP 展示层和 `/search/keyword` 中全部失效。

本模块提供：
  - `BM25KeywordIndex`：基于 `llama-index-retrievers-bm25`（bm25s 实现）的关键词索引，
    支持构建 / 持久化 / 加载 / 旧格式迁移；
  - `KeywordRetriever`：LlamaIndex 检索器封装，按请求指定 top_k，
    并把 BM25 原始分（无上界）归一化为 0~1 的相关性分数，
    使 `SimilarityPostprocessor`、MCP `_format_nodes`、`retrieval_stats`
    三处的分数语义与向量通道一致。

持久化目录沿用现有 `persist_dir` 结构：`<persist_dir>/keyword_index/`
（与 `<persist_dir>/vector_index/` 同级）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeRelationship, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from dsm5_rag.background import run_in_executor

logger = logging.getLogger(__name__)

# BM25Retriever 落盘时写入的参数文件名，用来判定"新格式"索引是否存在
RETRIEVER_PARAMS_FILENAME = "retriever.json"
LEGACY_DOCSTORE_FILENAME = "docstore.json"
# 本模块附加写的元数据（language / skip_stemming / node_count 等）
META_FILENAME = "keyword_index_meta.json"
# 落盘时写进 retriever.json 的默认 top_k（加载后由 KeywordRetriever 按请求覆盖）
PERSIST_DEFAULT_TOP_K = 10


def normalize_keyword_nodes(nodes: Sequence[NodeWithScore]) -> List[NodeWithScore]:
    """丢弃无词重叠（BM25 分数 <= 0）的节点，并把剩余分数归一化到 0~1。

    BM25 的原始分数没有上界（语料越大分数越高），直接和向量通道的余弦分数
    比较是跨尺度误判，因此这里统一为"相对本次检索最好结果"的 0~1 分数，
    最高分为 1.0。
    """
    kept = [n for n in nodes if n.score is not None and float(n.score) > 0.0]
    if not kept:
        return []
    max_score = max(float(n.score) for n in kept)
    if max_score <= 0.0:
        return []
    for node in kept:
        node.score = float(node.score) / max_score
    return kept


def _child_ids(node: BaseNode) -> List[str]:
    """取节点的子节点 ID（HierarchicalNodeParser 用 relationships 表达层级，
    0.14.x 的 TextNode 上没有 child_ids 属性）。"""
    relationships = getattr(node, "relationships", None) or {}
    children = relationships.get(NodeRelationship.CHILD) or []
    return [getattr(child, "node_id", "") for child in children]


def leaf_nodes(nodes: Iterable[BaseNode]) -> List[BaseNode]:
    """只保留叶子节点。

    `HierarchicalNodeParser` 会同时产出父节点（chunk_size）和子节点
    （chunk_size//2），父节点的文本是子节点的拼接，语料较小时两者文本甚至完全
    一样。若两级都进关键词索引，同一段落会被重复命中，白白挤掉 top_k 名额，
    因此默认只索引叶子层；没有父子关系时（普通切分/旧索引）原样返回。
    """
    all_nodes = list(nodes)
    leaves = [n for n in all_nodes if not _child_ids(n)]
    return leaves or all_nodes


class KeywordRetriever(BaseRetriever):
    """关键词通道检索器：共享已构建好的 BM25 倒排，按请求返回 top_k。"""

    def __init__(
        self,
        index: "BM25KeywordIndex",
        similarity_top_k: int = 5,
        verbose: bool = False,
    ) -> None:
        self._index = index
        self._similarity_top_k = max(1, int(similarity_top_k))
        super().__init__()
        self._verbose = verbose

    @property
    def similarity_top_k(self) -> int:
        return self._similarity_top_k

    def _base_retriever(self) -> BM25Retriever:
        return self._index.base_retriever(similarity_top_k=self._similarity_top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._base_retriever().retrieve(query_bundle)
        return normalize_keyword_nodes(nodes)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # BM25 分词+打分是同步 CPU 操作，放到线程池里，避免阻塞事件循环
        retriever = self._base_retriever()
        nodes = await run_in_executor(retriever.retrieve, query_bundle)
        return normalize_keyword_nodes(nodes)


class BM25KeywordIndex:
    """BM25 关键词索引：构建、持久化、加载。

    用法与 LlamaIndex 的 Index 类似：`as_retriever(similarity_top_k=k)` 返回一个
    真正按查询打分的 `KeywordRetriever`。倒排表只构建一次，多个 top_k 共用。
    """

    def __init__(
        self,
        bm25: Any,
        corpus: List[Any],
        node_count: int,
        language: str = "en",
        skip_stemming: bool = False,
    ) -> None:
        self._bm25 = bm25
        self._corpus = corpus
        self._node_count = node_count
        self.language = language
        self.skip_stemming = skip_stemming

    # ------------------------------------------------------------------
    # 构建 / 加载
    # ------------------------------------------------------------------
    @classmethod
    def from_nodes(
        cls,
        nodes: Iterable[BaseNode],
        language: str = "en",
        skip_stemming: bool = False,
        verbose: bool = False,
        only_leaf_nodes: bool = True,
    ) -> "BM25KeywordIndex":
        source_nodes = list(nodes)
        node_list = leaf_nodes(source_nodes) if only_leaf_nodes else source_nodes
        if not node_list:
            raise ValueError("没有可用于构建 BM25 关键词索引的节点")
        logger.info(
            "构建 BM25 关键词索引：%d/%d 个节点入索引（语言=%s，词干化=%s）",
            len(node_list),
            len(source_nodes),
            language,
            "关闭" if skip_stemming else "开启",
        )

        # 倒排表只构建一次；返回条数由 KeywordRetriever 按请求决定，
        # 这里的 similarity_top_k 只是构造参数占位。
        retriever = BM25Retriever.from_defaults(
            nodes=node_list,
            language=language,
            skip_stemming=skip_stemming,
            similarity_top_k=min(len(node_list), PERSIST_DEFAULT_TOP_K),
            verbose=verbose,
        )
        return cls(
            bm25=retriever.bm25,
            corpus=list(retriever.corpus),
            node_count=len(node_list),
            language=language,
            skip_stemming=skip_stemming,
        )

    @staticmethod
    def is_persisted(path: Path | str) -> bool:
        """目录里是否已有 BM25 关键词索引（新格式）。"""
        return (Path(path) / RETRIEVER_PARAMS_FILENAME).is_file()

    @staticmethod
    def has_legacy_index(path: Path | str) -> bool:
        """目录里是否是旧的 SummaryIndex 持久化结构（需要迁移）。"""
        p = Path(path)
        return (p / LEGACY_DOCSTORE_FILENAME).is_file() and not (
            p / RETRIEVER_PARAMS_FILENAME
        ).is_file()

    @classmethod
    def from_persist_dir(cls, path: Path | str) -> "BM25KeywordIndex":
        p = Path(path)
        retriever = BM25Retriever.from_persist_dir(str(p))
        corpus = list(retriever.corpus or [])
        meta = cls._read_meta(p)
        node_count = int(
            meta.get("node_count") or retriever.bm25.scores.get("num_docs") or len(corpus)
        )
        logger.info("已加载 BM25 关键词索引: %s（%d 个节点）", p, node_count)
        return cls(
            bm25=retriever.bm25,
            corpus=corpus,
            node_count=node_count,
            language=str(meta.get("language", "en")) or "en",
            skip_stemming=bool(meta.get("skip_stemming", False)),
        )

    @classmethod
    def load_or_migrate(
        cls, path: Path | str, verbose: bool = False
    ) -> Optional["BM25KeywordIndex"]:
        """加载新格式索引；遇到旧的 SummaryIndex 目录则就地迁移。

        迁移只需要旧 docstore 里的节点文本，不会重新计算 embedding。
        返回 None 表示目录不可用，调用方应重建。
        """
        p = Path(path)
        if cls.is_persisted(p):
            try:
                return cls.from_persist_dir(p)
            except Exception as e:  # 损坏的索引：交给上层重建
                logger.warning("BM25 关键词索引加载失败（%s），将重新构建", e)
                return None

        if cls.has_legacy_index(p):
            nodes = cls._nodes_from_legacy_dir(p)
            if not nodes:
                logger.warning("旧关键词索引中没有可用节点，将重新构建索引")
                return None
            logger.info(
                "检测到旧版 SummaryIndex 关键词索引，迁移为 BM25（%d 个节点）",
                len(nodes),
            )
            index = cls.from_nodes(nodes, verbose=verbose)
            index.persist(p)
            return index

        return None

    @staticmethod
    def _nodes_from_legacy_dir(path: Path) -> List[BaseNode]:
        """从旧的 SummaryIndex 持久化目录里取出节点（用于就地迁移）。"""
        from llama_index.core import StorageContext, load_index_from_storage

        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(path))
            legacy_index = load_index_from_storage(storage_context)
            docstore = legacy_index.docstore
            return [doc for doc in docstore.docs.values() if doc is not None]  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("读取旧关键词索引失败: %s", e)
            return []

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def persist(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        retriever = self.base_retriever(
            similarity_top_k=min(self._node_count, PERSIST_DEFAULT_TOP_K)
        )
        retriever.persist(str(p))
        self._write_meta(p)
        logger.info("BM25 关键词索引已持久化: %s（%d 个节点）", p, self._node_count)

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path / META_FILENAME

    @staticmethod
    def _read_meta(path: Path) -> Dict[str, Any]:
        meta_path = BM25KeywordIndex._meta_path(path)
        if not meta_path.is_file():
            return {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            logger.warning("读取关键词索引元数据失败 %s: %s", meta_path, e)
            return {}

    def _write_meta(self, path: Path) -> None:
        payload = {
            "index_type": "bm25",
            "language": self.language,
            "skip_stemming": self.skip_stemming,
            "node_count": self._node_count,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            with open(self._meta_path(path), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:  # 元数据缺失不影响检索，只降级为日志
            logger.warning("写入关键词索引元数据失败 %s: %s", path, e)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def base_retriever(self, similarity_top_k: int = 5) -> BM25Retriever:
        """复用一个已构建好的倒排表，按给定 top_k 生成 BM25Retriever。"""
        retriever = BM25Retriever(
            existing_bm25=self._bm25,
            similarity_top_k=max(1, int(similarity_top_k)),
        )
        retriever.corpus = self._corpus
        return retriever

    def as_retriever(
        self,
        similarity_top_k: int = 5,
        **kwargs: Any,
    ) -> KeywordRetriever:
        """与 LlamaIndex Index.as_retriever 保持一致的调用方式。"""
        return KeywordRetriever(
            index=self,
            similarity_top_k=similarity_top_k,
            verbose=bool(kwargs.get("verbose", False)),
        )

    @property
    def node_count(self) -> int:
        return self._node_count

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"BM25KeywordIndex(node_count={self._node_count}, language={self.language!r})"
