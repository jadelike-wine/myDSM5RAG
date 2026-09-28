"""测试 DSM-5 RAG MCP Server 的工具注册与工具行为。

注册类测试是纯静态的；`TestToolBehaviour` 会把模块级单例 `_rag_system` 换成一个
用真实小语料建好索引的系统，通过 fastmcp 客户端真正调用工具，
锁住历史上"MCP 的 KEYWORD_ONLY 永远返回空结果"的缺陷。
"""

import asyncio

import pytest
from fastmcp import Client

from dsm5_rag import mcp_server
from dsm5_rag.config import RagConfig
from dsm5_rag.mcp_server import mcp


@pytest.mark.asyncio
async def test_tools_registered():
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "search",
        "keyword_search",
        "build_index",
        "get_index_status",
        "list_documents",
    }, f"缺少工具: {tool_names}"


@pytest.mark.asyncio
async def test_resources_registered():
    resources = await mcp.list_resources()
    uris = {str(r.uri) for r in resources}
    assert uris == {
        "dsm5://status",
        "dsm5://index/status",
        "dsm5://config",
        "dsm5://documents",
    }, f"缺少资源: {uris}"


@pytest.mark.asyncio
async def test_prompts_registered():
    prompts = await mcp.list_prompts()
    prompt_names = {p.name for p in prompts}
    assert "dsm5_search_assistant" in prompt_names
    assert "dsm5_differential_diagnosis" in prompt_names


@pytest.mark.asyncio
async def test_search_tool_has_params():
    tools = await mcp.list_tools()
    search_tool = next(t for t in tools if t.name == "search")
    props = search_tool.parameters.get("properties", {})
    assert "question" in props
    assert "mode" in props
    assert "top_k" in props
    assert "vector_weight" in props


@pytest.mark.asyncio
async def test_keyword_search_tool_has_params():
    tools = await mcp.list_tools()
    kw_tool = next(t for t in tools if t.name == "keyword_search")
    props = kw_tool.parameters.get("properties", {})
    assert "keyword" in props
    assert "top_k" in props


@pytest.mark.asyncio
async def test_build_index_tool_has_params():
    tools = await mcp.list_tools()
    tool = next(t for t in tools if t.name == "build_index")
    props = tool.parameters.get("properties", {})
    assert "force_rebuild" in props


# ---------------------------------------------------------------------------
# 工具行为（真实小语料）
# ---------------------------------------------------------------------------


def _tool_text(result) -> str:
    blocks = getattr(result, "content", None) or []
    return "\n".join(
        getattr(block, "text", str(block)) for block in blocks
    ) or str(getattr(result, "data", result))


@pytest.fixture(scope="module")
def built_system(corpus_dir, tmp_path_factory):
    config = RagConfig()
    config.documents_path = str(corpus_dir)
    config.persist_dir = str(tmp_path_factory.mktemp("mcp-persist"))
    config.models_dir = str(tmp_path_factory.mktemp("mcp-models"))
    config.max_sources = 3
    config.similarity_cutoff = 0.2
    system = mcp_server.DeepSeekRAGSystem(config, auto_build_index=False)
    system._init_storage()
    asyncio.run(system.build_index(force_rebuild=True))
    assert system.index_built is True
    return system


@pytest.fixture
def client(built_system, monkeypatch):
    monkeypatch.setattr(mcp_server, "_rag_system", built_system)

    async def call(name: str, arguments: dict) -> str:
        async with Client(mcp) as c:
            result = await c.call_tool(name, arguments)
        return _tool_text(result)

    return call


class TestToolBehaviour:
    async def test_keyword_only_search_returns_visible_results(self, client):
        text = await client("search", {"question": "296.3x", "mode": "KEYWORD_ONLY", "top_k": 3})
        assert "低于相似度阈值" not in text
        assert "### 结果 #1" in text
        assert "mdd_major_dep.txt" in text

    async def test_keyword_search_tool_is_truncated(self, client):
        text = await client("keyword_search", {"keyword": "depressive", "top_k": 20})
        assert "### #" in text
        # 只展示 max_sources 条，且不带整段原文
        assert text.count("### #") <= 3

    async def test_hybrid_search_tags_retrieval_types(self, client):
        text = await client(
            "search",
            {"question": "diagnostic criteria for Major Depressive Disorder", "mode": "HYBRID", "top_k": 5},
        )
        assert "[keyword]" in text or "[both]" in text

    async def test_status_and_index_tools_work(self, client):
        status = await client("get_index_status", {})
        assert '"status": "ready"' in status
        documents = await client("list_documents", {})
        assert "mdd_major_dep.txt" in documents

    async def test_build_index_tool_reports_state(self, client):
        text = await client("build_index", {"force_rebuild": False})
        assert "状态:" in text
