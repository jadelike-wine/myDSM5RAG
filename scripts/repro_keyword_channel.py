"""关键词通道诊断脚本（阶段 A 的验收基线）。

不依赖 torch / HuggingFace 权重：用 llama-index-core 自带的 MockEmbedding 替换
`llama_index.embeddings.huggingface.HuggingFaceEmbedding`，用 MockLLM（不设
DEEPSEEK_API_KEY 时系统自动使用）替换 DeepSeek，从而只考察检索链路本身。

用法：
    PYTHONPATH=src python scripts/repro_keyword_channel.py [corpus_dir]

同一脚本可在改动前后各跑一次，用于对比关键数字。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(
    sys.argv[1] if len(sys.argv) > 1 else ROOT / "tests" / "fixtures" / "corpus"
)
TMP = Path(tempfile.mkdtemp(prefix="dsm5-repro-"))

# --- 环境隔离：必须在 import dsm5_rag 之前设置 -------------------------------
os.environ["DOCUMENTS_PATH"] = str(CORPUS)
os.environ["PERSIST_DIR"] = str(TMP / "chroma")
os.environ["MODELS_DIR"] = str(TMP / "models")
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["DEFAULT_MODE"] = "HYBRID"
os.environ["DEFAULT_TOP_K"] = "5"
os.environ["SIMILARITY_CUTOFF"] = "0.2"
os.environ["MAX_SOURCES"] = "3"
os.environ["ENABLE_CONTEXT_REORDER"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"

# --- 用测试替身顶掉 HuggingFace embedding（避免下载几百 MB 权重 + torch）-----
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.lexical_embedding import install_fake_huggingface_embedding  # noqa: E402

install_fake_huggingface_embedding(embed_dim=256)

from llama_index.core import QueryBundle  # noqa: E402
from llama_index.core.postprocessor import SimilarityPostprocessor  # noqa: E402

from dsm5_rag.config import RagConfig  # noqa: E402
from dsm5_rag.mcp_server import _format_nodes  # noqa: E402
from dsm5_rag.models import (  # noqa: E402
    BatchQueryRequest,
    KeywordSearchRequest,
    QueryRequest,
)
from dsm5_rag.retriever import HybridRetriever  # noqa: E402
from dsm5_rag.system import DeepSeekRAGSystem  # noqa: E402

QUERY = "What are the diagnostic criteria for Major Depressive Disorder?"
CODE_QUERY = "296.3x"
EXACT_QUERY = "anhedonia 快感缺失"
TOP_K = 2
HYBRID_TOP_K = 5

RESULTS: dict[str, object] = {}


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


async def main() -> int:
    hr(f"构建索引：语料={CORPUS} 临时目录={TMP}")
    config = RagConfig()
    system = DeepSeekRAGSystem(config, auto_build_index=False)
    system._init_storage()
    built = await system.build_index(force_rebuild=True)
    print("build_index ->", json.dumps(built, ensure_ascii=False))
    total_nodes = built.get("nodes") or 0
    print(f"索引节点总数 = {total_nodes}")

    # ---- 1. KEYWORD_ONLY：请求 top_k=2，实际返回多少条？分数是否为 None？ ----
    hr("[1] KEYWORD_ONLY 通道（请求 top_k=2）")
    keyword_retriever = system.keyword_index.as_retriever(similarity_top_k=TOP_K)
    print(f"retriever 类型 = {type(keyword_retriever).__name__}")
    kw_nodes_out = keyword_retriever.retrieve(QueryBundle(QUERY))
    scores = [n.score for n in kw_nodes_out]
    print(f"返回节点数 = {len(kw_nodes_out)}")
    print(f"scores = {scores}")
    check("KEYWORD_ONLY 未返回全部节点", len(kw_nodes_out) < total_nodes,
          f"{len(kw_nodes_out)} < 全部 {total_nodes}")
    check("KEYWORD_ONLY 条数 <= top_k", len(kw_nodes_out) <= TOP_K, f"{len(kw_nodes_out)} <= {TOP_K}")
    check("KEYWORD_ONLY score 不为 None", all(s is not None for s in scores))
    check("KEYWORD_ONLY score 落在 0~1", all(
        s is not None and 0.0 <= s <= 1.0 for s in scores
    ))
    hit_code = any(
        "Major Depressive Disorder" in n.node.text
        for n in keyword_retriever.retrieve(QueryBundle(CODE_QUERY))
    )
    check("KEYWORD_ONLY 用 ICD 编码 '296.3x' 命中 MDD 段落", hit_code)
    hit_exact = any(
        "Major Depressive Disorder" in n.node.text
        for n in keyword_retriever.retrieve(QueryBundle(EXACT_QUERY))
    )
    check("KEYWORD_ONLY 用术语 'anhedonia 快感缺失' 命中 MDD 段落", hit_exact)

    # ---- 2. HYBRID：关键词通道对最终上下文有没有贡献？ --------------------
    hr("[2] HYBRID 融合 + SimilarityPostprocessor(cutoff=0.2)")
    hybrid_kw_retriever = system.keyword_index.as_retriever(
        similarity_top_k=HYBRID_TOP_K
    )
    vector_retriever = system.vector_index.as_retriever(similarity_top_k=HYBRID_TOP_K)
    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=hybrid_kw_retriever,
        mode="HYBRID",
        vector_weight=config.default_vector_weight,
        similarity_top_k=HYBRID_TOP_K,
    )
    merged = hybrid.retrieve(QueryBundle(QUERY))
    print(f"合并后节点数 = {len(merged)}  (top_k 本应是 {HYBRID_TOP_K})")
    print(f"scores = {[None if n.score is None else round(n.score, 4) for n in merged]}")
    kept = SimilarityPostprocessor(similarity_cutoff=config.similarity_cutoff).postprocess_nodes(
        merged, QueryBundle(QUERY)
    )
    types_kept = [n.node.metadata.get("retrieval_type", "未知") for n in kept]
    print(f"cutoff 过滤后 = {len(kept)} 条, retrieval_type = {types_kept}")
    kw_any = sum(1 for t in types_kept if t in ("keyword", "both"))
    kw_only = sum(1 for t in types_kept if t == "keyword")
    check("HYBRID 结果条数 <= top_k", len(kept) <= HYBRID_TOP_K, f"{len(kept)}")
    check(
        "HYBRID 关键词通道贡献 > 0 条（keyword 或 both）",
        kw_any > 0,
        f"keyword_or_both={kw_any}, 其中纯 keyword={kw_only}",
    )

    hr("[3] FUSION 融合（vector_weight 应当参与，且不得跨尺度硬编码）")
    fusion = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=hybrid_kw_retriever,
        mode="FUSION",
        vector_weight=0.9,
        similarity_top_k=HYBRID_TOP_K,
    )
    fused = fusion.retrieve(QueryBundle(QUERY))
    print(f"FUSION 返回 = {len(fused)} 条")
    print(f"scores = {[None if n.score is None else round(n.score, 4) for n in fused]}")
    check("FUSION score 不为 None 且 <= 1", bool(fused) and all(
        n.score is not None and 0.0 <= n.score <= 1.0 for n in fused
    ))
    check("FUSION 条数 <= top_k", len(fused) <= HYBRID_TOP_K, f"{len(fused)} <= {HYBRID_TOP_K}")

    # ---- 4. MCP 展示层：KEYWORD_ONLY 的结果能不能显示出来？ ----------------
    hr("[4] MCP _format_nodes 对 KEYWORD_ONLY 的渲染")
    rendered = _format_nodes(kw_nodes_out, system)
    print(rendered[:600])
    shown = rendered.count("### 结果 #")
    print(f"渲染出的结果条数 = {shown}")
    check("MCP _format_nodes 能展示 KEYWORD_ONLY 结果", shown > 0, f"shown={shown}")

    hr("[5] MCP search(mode='KEYWORD_ONLY') 走完整工具路径")
    from dsm5_rag import mcp_server  # noqa: E402

    mcp_server._rag_system = system  # 复用已构建好的实例，避免重复初始化
    try:
        from fastmcp import Client  # noqa: E402

        async with Client(mcp_server.mcp) as client:
            call_res = await client.call_tool("search", {
                "question": CODE_QUERY, "mode": "KEYWORD_ONLY", "top_k": 3
            })
        text = "\n".join(
            getattr(block, "text", str(block))
            for block in getattr(call_res, "content", [call_res])
        )
    except Exception as e:
        text = f"(无法通过 fastmcp Client 调用工具: {type(e).__name__}: {e})"
    print(text[:700] + ("..." if len(text) > 700 else ""))
    check("MCP KEYWORD_ONLY 不返回『低于阈值』空结果", "低于相似度阈值" not in text and not text.startswith("("))

    # ---- 6. /search/keyword：有没有截断？倒出多少全文？ --------------------
    hr("[6] system.keyword_search（HTTP /search/keyword 的实现）")
    res = await system.keyword_search(KeywordSearchRequest(keyword=CODE_QUERY, top_k=TOP_K))
    full_chars = sum(
        len(r["full_text"]) for r in res["results"] if r.get("full_text")
    )
    print(f"keyword='{CODE_QUERY}' -> count={res['count']}  full_text 总字符={full_chars}")
    print(f"scores = {[r['score'] for r in res['results']]}")
    check("keyword_search 条数 <= top_k", res["count"] <= TOP_K, f"{res['count']}")
    check("keyword_search 默认不带 full_text", full_chars == 0, f"full_chars={full_chars}")

    try:
        res_v = await system.keyword_search(
            KeywordSearchRequest(keyword=CODE_QUERY, top_k=TOP_K), verbose=True
        )
        chars_v = sum(len(r["full_text"]) for r in res_v["results"] if r.get("full_text"))
        print(f"verbose=True -> full_text 总字符={chars_v}")
    except TypeError as e:
        print(f"verbose 参数尚不存在（改动前）：{e}")

    # ---- 7. response_time 是不是真实耗时？ --------------------------------
    hr("[7] query() 的 response_time")
    t0 = time.perf_counter()
    q = await system.query(QueryRequest(question=QUERY, mode="HYBRID", similarity_top_k=TOP_K))
    elapsed = time.perf_counter() - t0
    print(f"真实墙钟耗时 = {elapsed:.3f}s")
    print(f"返回的 response_time = {q.get('response_time')}")
    print(f"retrieval_ms={q.get('retrieval_ms')} llm_ms={q.get('llm_ms')}")
    print(f"retrieval_stats = {q.get('retrieval_stats')}")
    rt = q.get("response_time")
    check(
        "response_time 接近真实墙钟耗时（而非 uptime）",
        rt is not None and abs(rt - elapsed) <= max(0.25, 0.5 * elapsed),
        f"response_time={rt} wall={elapsed:.3f} uptime={system.get_status().uptime:.3f}",
    )
    check(
        "retrieval_ms / llm_ms 已分段上报",
        q.get("retrieval_ms") is not None and q.get("llm_ms") is not None,
        f"retrieval_ms={q.get('retrieval_ms')} llm_ms={q.get('llm_ms')}",
    )

    hr("[8] 从磁盘重新加载索引（第二个系统实例，模拟服务重启）")
    kw_dir = TMP / "chroma" / "keyword_index"
    print(f"{kw_dir} 目录内容: {sorted(os.listdir(kw_dir)) if kw_dir.is_dir() else '(不存在)'}")
    system2 = DeepSeekRAGSystem(RagConfig(), auto_build_index=False)
    system2._init_storage()
    await system2._load_existing_index()
    print(f"index_built = {system2.index_built}")
    print(f"关键词索引类型 = {type(system2.keyword_index).__name__}")
    kw2 = system2.keyword_index.as_retriever(similarity_top_k=TOP_K).retrieve(
        QueryBundle(CODE_QUERY)
    )
    print(f"重启后 KEYWORD_ONLY('296.3x') -> {len(kw2)} 条, scores={[round(n.score,4) for n in kw2 if n.score is not None]}")
    check("重启后关键词索引类型是 BM25", type(system2.keyword_index).__name__ == "BM25KeywordIndex",
          type(system2.keyword_index).__name__)
    check("重启后 KEYWORD_ONLY 仍有结果", len(kw2) > 0 and all(n.score is not None for n in kw2))
    check(
        "重启后 KEYWORD_ONLY 仍命中 MDD",
        any("Major Depressive Disorder" in n.node.text for n in kw2),
    )

    hr("[9] 请求参数隔离（不再共享引擎/全局模式）")
    small = QueryRequest(question=QUERY, mode="KEYWORD_ONLY", similarity_top_k=1)
    large = QueryRequest(question=QUERY, mode="KEYWORD_ONLY", similarity_top_k=5)
    r_small, r_large = await asyncio.gather(system.query(small), system.query(large))
    n_small = len(r_small.get("sources", []))
    n_large = len(r_large.get("sources", []))
    print(f"并发 top_k=1 -> {n_small} 条; top_k=5 -> {n_large} 条")
    check("并发请求各自遵守自己的 top_k", n_small <= 1 and n_large > n_small,
          f"{n_small} vs {n_large}")
    check("回显生效的 similarity_top_k", r_small.get("similarity_top_k") == 1 and r_large.get("similarity_top_k") == 5)
    before_mode = system.default_mode
    await system.query(QueryRequest(question=QUERY, mode="VECTOR_ONLY", similarity_top_k=2))
    check("请求不会改动服务默认模式", system.default_mode == before_mode,
          f"default_mode={system.default_mode}")

    hr("[10] keyword_search 阈值与条数上限")
    res_bounded = await system.keyword_search(KeywordSearchRequest(keyword="depression", top_k=20))
    print(f"count={res_bounded['count']} total_matched={res_bounded['total_matched']} "
          f"filtered={res_bounded['filtered_by_cutoff']} max_sources={res_bounded['max_sources']}")
    check("keyword_search 受 max_sources 截断", res_bounded["count"] <= config.max_sources,
          f"{res_bounded['count']} <= {config.max_sources}")

    hr("[11] 真流式：answer_chunk 是否分多次下发、首 token 是否早于结束")
    stream_res = await system.query(
        QueryRequest(question=QUERY, mode="HYBRID", similarity_top_k=3, stream=True)
    )
    events = []
    complete_payload: dict = {}
    first_chunk_at = None
    stream_started = time.perf_counter()
    async for line in stream_res["streaming_response"]:
        data = json.loads(line)
        events.append(data.get("event"))
        if data.get("event") == "answer_chunk" and data.get("chunk") and first_chunk_at is None:
            first_chunk_at = time.perf_counter() - stream_started
        if data.get("event") == "answer_chunk" and data.get("is_complete"):
            complete_payload = data
    stream_elapsed = time.perf_counter() - stream_started
    n_chunks = sum(1 for e in events if e == "answer_chunk")
    print(f"事件序列 = {events[:6]} ... 共 {len(events)} 个, answer_chunk={n_chunks}")
    print(f"首块耗时 = {first_chunk_at:.4f}s / 全部耗时 = {stream_elapsed:.4f}s")
    print(f"完成事件里的指标 = {json.dumps(complete_payload, ensure_ascii=False)[:220]}")
    check("流式返回多个 answer_chunk（不是一次性倒完）", n_chunks > 2, f"answer_chunk={n_chunks}")
    check(
        "首 token 明显早于流结束（真流式）",
        first_chunk_at is not None and first_chunk_at < stream_elapsed * 0.9,
        f"ttft={first_chunk_at:.4f}s total={stream_elapsed:.4f}s",
    )
    check("事件顺序含 query_started/answer_chunk/query_completed",
          events[0] == "query_started" and events[-1] == "query_completed", str(events[:2]) + "..." + str(events[-1]))

    hr("[12] 批量查询并发上限")
    batch = await system.batch_query(
        BatchQueryRequest(
            questions=["296.3x", "300.02", "anhedonia", "manic episode"],
            mode="KEYWORD_ONLY",
        )
    )
    print(f"total={batch['total']} successful={batch['successful']} concurrency={batch['concurrency']} 耗时={batch['response_time']}s")
    check("批量查询结果条数与输入一致", batch["total"] == 4)
    check("批量查询全部成功", batch["successful"] == 4, str([r.get("error") for r in batch["results"] if not r.get("success")]))
    check("批量查询走 Semaphore 并发上限", batch["concurrency"] == config.batch_concurrency,
          f"{batch['concurrency']} == {config.batch_concurrency}")

    hr("汇总")
    for k, v in RESULTS.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    failed = [k for k, v in RESULTS.items() if not v]
    print(f"\n共 {len(RESULTS)} 项断言，失败 {len(failed)} 项")
    print(f"临时目录保留在 {TMP}（可 rm -rf 清理）")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
