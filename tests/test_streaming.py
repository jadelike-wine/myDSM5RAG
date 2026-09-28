"""真流式测试：分块下发、首 token 提前、且不阻塞事件循环。

历史实现是"假流式"：引擎以 streaming=False 创建，`_stream_query` 内部同步 query
拿到完整答案后一次性 yield，首 token 延迟等于整段生成时间。
"""

import asyncio
import json
import time
from pathlib import Path

import pytest
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms import MockLLM

from dsm5_rag.config import RagConfig
from dsm5_rag.models import QueryRequest
from dsm5_rag.system import DeepSeekRAGSystem

TOKENS = ["DSM-5", " 重性抑郁", " 障碍（MDD）", " 诊断标准", " A：", " 五条以上症状"]


class SlowStreamingLLM(MockLLM):
    """每 20ms 产出一个 token 的假 LLM，用来把"是否真流式"变成可观测的时间差。"""

    async def astream_complete(self, prompt, formatted=False, **kwargs):
        async def gen():
            accumulated = ""
            for token in TOKENS:
                await asyncio.sleep(0.02)
                accumulated += token
                yield CompletionResponse(text=accumulated, delta=token)

        return gen()

    async def astream_chat(self, messages, **kwargs):  # pragma: no cover
        from llama_index.core.llms import ChatResponse

        async def gen():
            accumulated = ""
            for token in TOKENS:
                await asyncio.sleep(0.02)
                accumulated += token
                yield ChatResponse(message=None, text=accumulated, delta=token)

        return gen()


def _make_config(corpus_dir: Path, root: Path) -> RagConfig:
    config = RagConfig()
    config.documents_path = str(corpus_dir)
    config.persist_dir = str(root / "persist")
    config.models_dir = str(root / "models")
    config.max_sources = 2
    return config


@pytest.fixture(scope="module")
def streaming_system(corpus_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("stream")
    instance = DeepSeekRAGSystem(
        _make_config(corpus_dir, root), auto_build_index=False
    )
    instance._init_storage()
    asyncio.run(instance.build_index(force_rebuild=True))
    return instance


async def _collect(streaming_response) -> tuple[list[dict], float, float]:
    started = time.perf_counter()
    first_at = None
    events = []
    async for line in streaming_response:
        payload = json.loads(line)
        if first_at is None and payload.get("event") == "answer_chunk" and payload.get("chunk"):
            first_at = time.perf_counter() - started
        events.append(payload)
    return events, (first_at or 0.0), (time.perf_counter() - started)


class TestStreamProtocol:
    async def test_answer_arrives_in_multiple_chunks(self, streaming_system):
        streaming_system.llm = SlowStreamingLLM()
        result = await streaming_system.query(
            QueryRequest(
                question="Major Depressive Disorder 的诊断标准", stream=True
            )
        )
        events, first_at, total = await _collect(result["streaming_response"])

        names = [e["event"] for e in events]
        assert names[0] == "query_started"
        assert names[1] == "retrieval_completed"
        assert names[-1] == "query_completed"

        content_chunks = [
            e for e in events if e["event"] == "answer_chunk" and e["chunk"]
        ]
        assert len(content_chunks) == len(TOKENS), (
            "流式应逐 token 下发，而不是整段一次给完"
        )
        assert "".join(e["chunk"] for e in content_chunks) == "".join(TOKENS)
        assert sum(1 for e in events if e.get("is_complete")) == 1

    async def test_first_token_is_early(self, streaming_system):
        streaming_system.llm = SlowStreamingLLM()
        result = await streaming_system.query(
            QueryRequest(question="296.3x", mode="KEYWORD_ONLY", stream=True)
        )
        events, first_at, total = await _collect(result["streaming_response"])

        final = next(e for e in events if e.get("is_complete"))
        # 6 个 token * 20ms：首 token 应该远早于全部生成完
        assert first_at < total * 0.5, (first_at, total)
        assert final["ttft_ms"] < final["total_ms"]
        assert final["retrieval_ms"] >= 0

    async def test_event_loop_stays_responsive(self, streaming_system):
        """消费流期间事件循环必须能跑别的协程（旧实现是同步 LLM 调用，会堵住循环）。"""
        streaming_system.llm = SlowStreamingLLM()
        result = await streaming_system.query(
            QueryRequest(question="anxiety criteria", stream=True)
        )

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            await _collect(result["streaming_response"])
        finally:
            task.cancel()

        assert ticks >= 5, f"流式期间事件循环只被调度了 {ticks} 次"

    async def test_error_is_reported_as_event(self, streaming_system):
        class BrokenLLM(MockLLM):
            async def astream_complete(self, prompt, formatted=False, **kwargs):
                raise RuntimeError("LLM 挂了")

        original = streaming_system.llm
        streaming_system.llm = BrokenLLM()
        try:
            result = await streaming_system.query(
                QueryRequest(question="296.3x", stream=True)
            )
            events, _first, _total = await _collect(result["streaming_response"])
        finally:
            streaming_system.llm = original

        # 流式失败时不能让请求静默：要么给出内容，要么给出 error 事件
        names = [e["event"] for e in events]
        assert "error" in names or "answer_chunk" in names
        if "error" in names:
            assert names[-1] == "error"


class TestContextPacking:
    def test_context_is_capped(self, streaming_system, corpus_dir):
        from llama_index.core import QueryBundle

        streaming_system.config.stream_context_char_limit = 800
        nodes = streaming_system.keyword_index.as_retriever(
            similarity_top_k=10
        ).retrieve(QueryBundle("diagnostic criteria"))
        assert nodes
        context = streaming_system._pack_context(nodes)
        assert len(context) <= 800 + 2 * len(nodes)

        streaming_system.config.stream_context_char_limit = 24000

    async def test_empty_nodes_yield_empty_response(self, streaming_system):
        chunks = [
            chunk async for chunk in streaming_system._astream_answer("q", [])
        ]
        assert chunks == ["Empty Response"]
