"""
DSM-5 RAG 基本查询示例

用法:
    uv run python examples/basic_query.py

要求:
    - 已配置 .env 文件中的 DEEPSEEK_API_KEY
    - 文档目录（默认 ./dsm5_documents）中存在 DSM-5 文档
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保包可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsm5_rag import DeepSeekRAGSystem, QueryRequest, RagConfig


async def main():
    print("=" * 60)
    print("DSM-5 RAG 基本查询示例")
    print("=" * 60)

    # 1. 初始化 RAG 系统（使用默认配置，自动从 .env 加载）
    print("\n[1/4] 初始化 RAG 系统...")
    config = RagConfig()
    rag = DeepSeekRAGSystem(config)

    # 2. 等待索引准备就绪
    print("\n[2/4] 等待索引加载...")
    for _ in range(30):
        status = rag.get_status()
        if status.index_built:
            print("   索引已就绪 ✓")
            break
        await asyncio.sleep(1)
    else:
        print("   索引未就绪，请检查文档配置")
        return

    # 3. 执行查询
    print("\n[3/4] 执行查询...")
    questions = [
        ("EN", "What are the diagnostic criteria for Major Depressive Disorder?"),
        ("ZH", "抑郁症的诊断标准是什么？"),
    ]

    for lang, question in questions:
        print(f"\n--- [{lang}] {question} ---")
        request = QueryRequest(question=question, mode="HYBRID")
        result = await rag.query(request)

        answer = result.get("answer", "无回答")
        sources = result.get("sources", [])

        print(f"\n回答: {answer[:200]}..." if len(answer) > 200 else f"\n回答: {answer}")
        print(f"\n来源 ({len(sources)} 个):")
        for src in sources:
            print(f"  [{src['rank']}] {src['file_name']} (score: {src['score']:.3f})")

    # 4. 打印系统状态
    print("\n[4/4] 系统状态:")
    status = rag.get_status()
    print(f"   状态: {status.status}")
    print(f"   文档数: {status.document_count}")
    print(f"   运行时间: {status.uptime:.1f}s")

    print("\n" + "=" * 60)
    print("示例完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
