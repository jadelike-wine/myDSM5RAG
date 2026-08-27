"""
DSM-5 RAG HTTP API 客户端示例

通过 httpx 调用 REST API，展示如何与运行中的 API 服务交互。

用法:
    # 先启动 API 服务（另一个终端）:
    uv run uvicorn dsm5_rag.api:app --host 127.0.0.1 --port 8031

    # 再运行本示例:
    uv run python examples/api_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 从 .env 加载端口配置（仅示例用，实际可硬编码或传参）
_API_PORT = 8031
_BASE_URL = f"http://127.0.0.1:{_API_PORT}"

# ============================================================


async def check_health(client: httpx.AsyncClient) -> dict:
    """健康检查"""
    resp = await client.get("/health")
    resp.raise_for_status()
    data = resp.json()
    print(f"  状态: {data['status']}, 时间戳: {data['timestamp']}")
    return data


async def get_status(client: httpx.AsyncClient) -> dict:
    """获取系统状态"""
    resp = await client.get("/status")
    resp.raise_for_status()
    data = resp.json()
    print(f"  系统状态: {data['status']}")
    print(f"  文档数: {data['document_count']}")
    print(f"  索引已就绪: {data['index_built']}")
    return data


async def query(client: httpx.AsyncClient, question: str, **kwargs) -> dict:
    """执行查询（非流式）"""
    payload = {
        "question": question,
        "mode": kwargs.get("mode", "HYBRID"),
        "similarity_top_k": kwargs.get("top_k", 5),
        "stream": False,
    }
    resp = await client.post("/query", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(f"  问题: {question}")
    answer = data.get("answer", "")
    print(f"  回答: {answer[:150]}..." if len(answer) > 150 else f"  回答: {answer}")
    sources = data.get("sources", [])
    print(f"  来源 ({len(sources)} 个):")
    for s in sources[:3]:
        print(f"    [{s['rank']}] {s['file_name']} (score: {s['score']:.3f})")
    return data


async def query_stream(client: httpx.AsyncClient, question: str) -> None:
    """执行流式查询"""
    payload = {
        "question": question,
        "mode": "HYBRID",
        "similarity_top_k": 3,
        "stream": True,
    }
    async with client.stream("POST", "/query", json=payload) as resp:
        resp.raise_for_status()
        print(f"  [流式] 问题: {question}")
        async for line in resp.aiter_lines():
            if line.strip():
                try:
                    evt = json.loads(line)
                    if evt.get("event") == "answer_chunk":
                        print(evt.get("chunk", ""), end="", flush=True)
                except json.JSONDecodeError:
                    pass
        print()


async def keyword_search(client: httpx.AsyncClient, keyword: str, top_k: int = 3) -> dict:
    """关键词搜索"""
    payload = {"keyword": keyword, "top_k": top_k}
    resp = await client.post("/search/keyword", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(f"  关键词: '{keyword}' → {data['count']} 个结果")
    for r in data.get("results", [])[:3]:
        print(f"    [{r['rank']}] {r['file_name']} (score: {r['score']:.3f})")
    return data


async def batch_query(client: httpx.AsyncClient, questions: list[str]) -> dict:
    """批量查询"""
    payload = {"questions": questions, "mode": "HYBRID"}
    resp = await client.post("/query/batch", json=payload)
    resp.raise_for_status()
    data = resp.json()
    for i, r in enumerate(data.get("results", []), 1):
        icon = "✓" if r["success"] else "✗"
        answer = r.get("answer", "")
        print(f"  [{icon}] Q{i}: {r['question'][:40]} -> {answer[:80]}...")
    return data


async def change_mode(client: httpx.AsyncClient, mode: str) -> dict:
    """切换检索模式"""
    resp = await client.post(f"/mode/change?mode={mode}&similarity_top_k=5")
    resp.raise_for_status()
    data = resp.json()
    print(f"  模式: {data['mode']}, 状态: {data['status']}")
    return data

# ============================================================


async def main():
    print("=" * 60)
    print("DSM-5 RAG HTTP API 客户端示例")
    print(f"服务地址: {_BASE_URL}")
    print("=" * 60)

    # 确保服务已启动
    try:
        async with httpx.AsyncClient(base_url=_BASE_URL, timeout=5) as c:
            await c.get("/health")
    except httpx.ConnectError:
        print("\n❌ 无法连接到 API 服务，请先启动服务:")
        print("   uv run uvicorn dsm5_rag.api:app --host 127.0.0.1 --port 8031")
        sys.exit(1)

    async with httpx.AsyncClient(base_url=_BASE_URL, timeout=30) as client:
        # 1. 健康检查
        print("\n[1/7] 健康检查")
        await check_health(client)

        # 2. 系统状态
        print("\n[2/7] 系统状态")
        status = await get_status(client)

        if not status.get("index_built"):
            print("\n⚠️  索引未就绪，跳过后续查询步骤")
            return

        # 3. 基本查询（非流式）
        print("\n[3/7] 基本查询")
        await query(client, "抑郁症的诊断标准是什么？")

        # 4. 流式查询
        print("\n[4/7] 流式查询")
        await query_stream(client, "What is Bipolar I Disorder?")

        # 5. 关键词搜索
        print("\n[5/7] 关键词搜索")
        await keyword_search(client, "anxiety")

        # 6. 批量查询
        print("\n[6/7] 批量查询")
        await batch_query(client, [
            "What is Major Depressive Disorder?",
            "广泛性焦虑障碍的症状有哪些？",
        ])

        # 7. 切换检索模式
        print("\n[7/7] 切换检索模式")
        await change_mode(client, "VECTOR_ONLY")
        await change_mode(client, "HYBRID")

    print("\n" + "=" * 60)
    print("示例完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())