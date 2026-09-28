"""
DSM-5 RAG 批量查询示例

用法:
    uv run python examples/batch_query.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsm5_rag import (
    BatchQueryRequest,
    DeepSeekRAGSystem,
    KeywordSearchRequest,
    RagConfig,
)


async def main():
    print("=" * 60)
    print("DSM-5 RAG 批量查询 & 关键词搜索示例")
    print("=" * 60)

    config = RagConfig()
    rag = DeepSeekRAGSystem(config)

    print("\n等待索引就绪...")
    for _ in range(30):
        if rag.get_status().index_built:
            print("  索引已就绪 ✓")
            break
        await asyncio.sleep(1)
    else:
        print("  索引未就绪")
        return

    # ===== 批量查询 =====
    print("\n" + "-" * 40)
    print("批量查询")
    print("-" * 40)

    batch_req = BatchQueryRequest(
        questions=[
            "What is Bipolar I Disorder?",
            "广泛性焦虑障碍的症状有哪些？",
            "精神分裂症的主要特征",
        ],
        mode="HYBRID",
    )
    batch_result = await rag.batch_query(batch_req)

    for i, r in enumerate(batch_result["results"], 1):
        status = "✓" if r["success"] else "✗"
        print(f"\n[{status}] Q{i}: {r['question'][:40]}...")
        answer = r.get("answer", "")
        print(f"   回答: {answer[:100]}..." if len(answer) > 100 else f"   回答: {answer}")

    # ===== 关键词搜索 =====
    print("\n" + "-" * 40)
    print("关键词搜索")
    print("-" * 40)

    keywords = ["depression", "anxiety", "schizophrenia"]
    for kw in keywords:
        kw_req = KeywordSearchRequest(keyword=kw, top_k=3)
        kw_result = await rag.keyword_search(kw_req)
        print(f"\n关键词: '{kw}' → 找到 {kw_result['count']} 个结果")
        for res in kw_result["results"][:2]:
            print(f"  [{res['rank']}] {res['file_name']} (score: {res['score']:.3f})")

    print("\n" + "=" * 60)
    print("示例完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
