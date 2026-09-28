"""BM25 关键词索引测试（真实小语料）。

这组测试用 tests/fixtures/corpus 里的 DSM-5 片段做真实分词与打分，
专门用来永久锁住历史上的 P0 缺陷：

    关键词通道曾经用 SummaryIndex 充当，它的检索器会**忽略查询**、
    返回 docstore 里的全部节点且 score=None，于是
      - KEYWORD_ONLY 返回"整个知识库"而不是 top_k；
      - 分数为 None 的节点被 SimilarityPostprocessor 全部丢弃，
        HYBRID/FUSION 里关键词通道贡献 0 条；
      - MCP 的 search(mode="KEYWORD_ONLY") 恒返回"所有结果均低于相似度阈值"。
"""

from pathlib import Path

import pytest
from llama_index.core import Document, QueryBundle

from dsm5_rag.keyword_index import (
    BM25KeywordIndex,
    KeywordRetriever,
    leaf_nodes,
    normalize_keyword_nodes,
)

CORPUS_DIR = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def corpus_nodes():
    """把 DSM-5 片段读成节点（每个文件一个节点）。"""
    files = sorted(CORPUS_DIR.glob("*.txt"))
    assert len(files) >= 3, "语料目录缺少足够的测试文件"
    return [
        Document(
            text=path.read_text(encoding="utf-8"),
            metadata={"file_name": path.name, "file_type": "txt"},
        )
        for path in files
    ]


@pytest.fixture
def keyword_index(corpus_nodes):
    return BM25KeywordIndex.from_nodes(corpus_nodes, language="en")


def file_names(nodes):
    return [n.node.metadata.get("file_name") for n in nodes]


class TestKeywordIndexBuild:
    def test_index_reports_node_count(self, keyword_index, corpus_nodes):
        assert keyword_index.node_count == len(corpus_nodes)

    def test_as_retriever_returns_keyword_retriever(self, keyword_index):
        retriever = keyword_index.as_retriever(similarity_top_k=2)
        assert isinstance(retriever, KeywordRetriever)
        assert retriever.similarity_top_k == 2

    def test_empty_nodes_raise(self):
        with pytest.raises(ValueError):
            BM25KeywordIndex.from_nodes([])


class TestKeywordRetrievalIsQueryAware:
    """核心回归：检索必须与查询相关，且受 top_k 限制。"""

    def test_returns_at_most_top_k(self, keyword_index):
        for top_k in (1, 2, 3):
            nodes = keyword_index.as_retriever(similarity_top_k=top_k).retrieve(
                QueryBundle("Major Depressive Disorder")
            )
            assert len(nodes) <= top_k

    def test_does_not_return_whole_corpus(self, keyword_index):
        nodes = keyword_index.as_retriever(similarity_top_k=1).retrieve(
            QueryBundle("296.3x")
        )
        assert 0 < len(nodes) < keyword_index.node_count

    def test_irrelevant_query_returns_nothing(self, keyword_index):
        """无词重叠的查询应返回空，而不是整个知识库。"""
        nodes = keyword_index.as_retriever(similarity_top_k=5).retrieve(
            QueryBundle("pneumoultramicroscopicsilicovolcanoconiosis zzzzqqqq")
        )
        assert nodes == []

    def test_different_queries_give_different_results(self, keyword_index):
        retriever = keyword_index.as_retriever(similarity_top_k=2)
        first = file_names(retriever.retrieve(QueryBundle("296.3x")))
        second = file_names(retriever.retrieve(QueryBundle("300.02 generalized anxiety")))
        assert first != second


class TestKeywordRecall:
    """DSM-5 场景最依赖术语/编码精确匹配。"""

    @pytest.mark.parametrize(
        "query,expected_file",
        [
            ("296.3x", "mdd_major_dep.txt"),
            ("anhedonia", "mdd_major_dep.txt"),
            ("300.02", "generalized_anxiety.txt"),
            ("296.4x manic episode", "bipolar_i.txt"),
            ("300.4 dysthymia", "persistent_depressive.txt"),
            ("314 hyperactivity", "adhd.txt"),
        ],
    )
    def test_top_hit_is_the_right_document(self, keyword_index, query, expected_file):
        nodes = keyword_index.as_retriever(similarity_top_k=3).retrieve(
            QueryBundle(query)
        )
        assert nodes, f"查询 {query!r} 没有命中任何节点"
        assert nodes[0].node.metadata["file_name"] == expected_file

    def test_chinese_term_matches(self, keyword_index):
        """中文术语（与英文混排在同一节点里）也应命中。"""
        nodes = keyword_index.as_retriever(similarity_top_k=2).retrieve(
            QueryBundle("快感缺失")
        )
        assert "mdd_major_dep.txt" in file_names(nodes)


class TestKeywordScores:
    def test_scores_are_not_none_and_within_unit(self, keyword_index):
        nodes = keyword_index.as_retriever(similarity_top_k=3).retrieve(
            QueryBundle("diagnostic criteria for Major Depressive Disorder")
        )
        assert nodes
        for node in nodes:
            assert node.score is not None
            assert 0.0 < node.score <= 1.0

    def test_top_score_is_normalized_to_one(self, keyword_index):
        nodes = keyword_index.as_retriever(similarity_top_k=3).retrieve(
            QueryBundle("major depressive episode")
        )
        assert max(n.score for n in nodes) == pytest.approx(1.0)

    def test_scores_are_ordered_descending(self, keyword_index):
        nodes = keyword_index.as_retriever(similarity_top_k=3).retrieve(
            QueryBundle("anxiety worry muscle tension")
        )
        scores = [n.score for n in nodes]
        assert scores == sorted(scores, reverse=True)

    def test_normalize_drops_zero_score_nodes(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        nodes = [
            NodeWithScore(node=TextNode(text="a", id_="a"), score=2.0),
            NodeWithScore(node=TextNode(text="b", id_="b"), score=1.0),
            NodeWithScore(node=TextNode(text="c", id_="c"), score=0.0),
            NodeWithScore(node=TextNode(text="d", id_="d"), score=None),
        ]
        kept = normalize_keyword_nodes(nodes)
        assert [n.node.node_id for n in kept] == ["a", "b"]
        assert [n.score for n in kept] == [1.0, 0.5]


class TestLeafNodeFiltering:
    def test_parents_are_excluded(self):
        from llama_index.core.schema import (
            NodeRelationship,
            RelatedNodeInfo,
            TextNode,
        )

        parent = TextNode(text="parent text", id_="p")
        child = TextNode(text="child text", id_="c")
        parent.relationships[NodeRelationship.CHILD] = [
            RelatedNodeInfo(node_id=child.node_id)
        ]
        assert [n.node_id for n in leaf_nodes([parent, child])] == ["c"]

    def test_fallback_when_no_hierarchy(self, corpus_nodes):
        assert len(leaf_nodes(corpus_nodes)) == len(corpus_nodes)


class TestPersistence:
    def test_persist_and_reload(self, keyword_index, corpus_nodes, tmp_path):
        persist_dir = tmp_path / "keyword_index"
        keyword_index.persist(persist_dir)

        assert BM25KeywordIndex.is_persisted(persist_dir)
        reloaded = BM25KeywordIndex.from_persist_dir(persist_dir)
        assert reloaded.node_count == keyword_index.node_count

        query = QueryBundle("296.3x")
        original = keyword_index.as_retriever(similarity_top_k=2).retrieve(query)
        restored = reloaded.as_retriever(similarity_top_k=2).retrieve(query)
        assert file_names(original) == file_names(restored)
        assert [round(n.score, 6) for n in original] == [
            round(n.score, 6) for n in restored
        ]

    def test_metadata_and_text_survive_reload(self, keyword_index, tmp_path):
        persist_dir = tmp_path / "keyword_index"
        keyword_index.persist(persist_dir)
        reloaded = BM25KeywordIndex.from_persist_dir(persist_dir)
        nodes = reloaded.as_retriever(similarity_top_k=1).retrieve(
            QueryBundle("296.3x")
        )
        assert nodes[0].node.metadata["file_name"] == "mdd_major_dep.txt"
        assert "Major Depressive Disorder" in nodes[0].node.text

    def test_missing_dir_is_not_persisted(self, tmp_path):
        assert not BM25KeywordIndex.is_persisted(tmp_path / "nothing")
        assert BM25KeywordIndex.load_or_migrate(tmp_path / "nothing") is None

    def test_legacy_summary_index_is_migrated(self, corpus_nodes, tmp_path):
        """旧版 SummaryIndex 目录（docstore.json）可就地迁移，不用重新 embedding。"""
        from llama_index.core import StorageContext, SummaryIndex

        persist_dir = tmp_path / "keyword_index"
        storage_context = StorageContext.from_defaults()
        SummaryIndex(nodes=corpus_nodes, storage_context=storage_context)
        storage_context.persist(persist_dir=str(persist_dir))

        assert BM25KeywordIndex.has_legacy_index(persist_dir)
        assert not BM25KeywordIndex.is_persisted(persist_dir)

        migrated = BM25KeywordIndex.load_or_migrate(persist_dir)
        assert migrated is not None
        assert BM25KeywordIndex.is_persisted(persist_dir)
        nodes = migrated.as_retriever(similarity_top_k=2).retrieve(
            QueryBundle("296.3x")
        )
        assert nodes and nodes[0].node.metadata["file_name"] == "mdd_major_dep.txt"

    def test_corrupted_dir_returns_none(self, tmp_path):
        persist_dir = tmp_path / "keyword_index"
        persist_dir.mkdir()
        (persist_dir / "retriever.json").write_text("{}", encoding="utf-8")
        assert BM25KeywordIndex.load_or_migrate(persist_dir) is None
