"""测试 DSM-5 RAG MCP Server 的工具注册"""

import asyncio

import pytest

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