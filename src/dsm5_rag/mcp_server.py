"""
DSM-5 RAG MCP Server — 纯检索服务

复用 dsm5_rag 底层混合检索逻辑（向量检索、关键词检索、混合/FUSION），
完全剥离外部 LLM 调用（无 DeepSeek API、无 MockLLM），
通过 MCP 协议暴露为 Tool / Resource / Prompt 三种原语。

设计原则:
  - 每个工具仅做一件事
  - 所有返回结果均为纯检索数据，不含 LLM 生成的回答
  - 资源的 URI 语义化且稳定
  - 提示模板可复用，指导 LLM 使用检索工具
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from llama_index.core import QueryBundle

from dsm5_rag.config import RagConfig
from dsm5_rag.models import KeywordSearchRequest
from dsm5_rag.retriever import HybridRetriever
from dsm5_rag.system import DeepSeekRAGSystem

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dsm5-rag-mcp")

# ---------------------------------------------------------------------------
# 全局单例 — 复用 DeepSeekRAGSystem 的检索能力，但本模块绝不调用 LLM
# ---------------------------------------------------------------------------
_rag_system: Optional[DeepSeekRAGSystem] = None


def _get_system() -> DeepSeekRAGSystem:
    """延迟初始化 RAG 系统（仅在首次调用时初始化一次）。"""
    global _rag_system
    if _rag_system is None:
        logger.info("正在初始化 DSM-5 检索系统（纯检索模式，无 LLM 调用）...")
        config = RagConfig()
        # DeepSeekRAGSystem 内部会初始化 MockLLM（当无 API Key 时），
        # 但本模块完全不访问 query_engine 或 llm，仅使用 index + retriever。
        _rag_system = DeepSeekRAGSystem(config, auto_build_index=True)
        logger.info("DSM-5 检索系统初始化完成，索引将在后台自动管理")
    return _rag_system


def _ensure_index(system: DeepSeekRAGSystem) -> None:
    """确保索引已就绪，否则抛出 ToolError。"""
    if not system.index_built:
        status = system.get_index_status()
        if status.status == "building":
            pct = (status.progress or 0) * 100
            logger.warning("索引正在构建中（进度 %.0f%%），请求被拒绝", pct)
            raise ToolError(f"索引正在构建中（进度 {pct:.0f}%），请稍后重试")
        logger.warning("索引未就绪: %s", status.message)
        raise ToolError(f"索引未就绪: {status.message}")


def _format_nodes(nodes, system: DeepSeekRAGSystem) -> str:
    """将检索节点格式化为可读的 Markdown 文本。"""
    cutoff = system.config.similarity_cutoff
    max_sources = system.config.max_sources

    lines = [f"检索到 {len(nodes)} 个结果:"]
    count = 0
    for i, node in enumerate(nodes):
        score = node.score if node.score is not None else 0
        if score < cutoff:
            continue
        count += 1
        if count > max_sources:
            break

        meta = node.node.metadata
        file_name = meta.get("file_name", "未知文件")
        ret_type = meta.get("retrieval_type", "")
        type_tag = f" [{ret_type}]" if ret_type else ""

        lines.append(f"\n### 结果 #{count}{type_tag}")
        lines.append(f"- **文件**: {file_name}")
        lines.append(f"- **分数**: {score:.4f}")
        lines.append(f"- **长度**: {len(node.node.text)} 字符")
        if "page" in meta:
            lines.append(f"- **页码**: {meta['page']}")
        if "fusion_score" in meta:
            lines.append(f"- **融合分数**: {meta['fusion_score']:.4f}")

        lines.append(f"- **文本预览**:")
        preview = node.node.text[:300].replace("\n", " ")
        lines.append(f"  {preview}...")

    if count == 0:
        lines.append("\n（所有结果均低于相似度阈值，未展示）")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FastMCP 实例
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "dsm5-rag-mcp",
    instructions="DSM-5 RAG 纯检索服务 — 复用底层混合检索逻辑，无外部 LLM 依赖",
    version="1.0.0",
)

# ═══════════════════════════════════════════════════════════════════════════
# Resources（只读数据，LLM 按 URI 获取）
# ═══════════════════════════════════════════════════════════════════════════


@mcp.resource("dsm5://status")
async def resource_system_status() -> str:
    """DSM-5 RAG 系统整体状态（运行状态、嵌入模型、文档数量等）。"""
    system = _get_system()
    return system.get_status().model_dump_json(indent=2, ensure_ascii=False)


@mcp.resource("dsm5://index/status")
async def resource_index_status() -> str:
    """索引构建状态（ready / building / not_built）。"""
    system = _get_system()
    return system.get_index_status().model_dump_json(indent=2, ensure_ascii=False)


@mcp.resource("dsm5://config")
async def resource_config() -> str:
    """当前配置概览（不包含密钥等敏感信息）。"""
    config = RagConfig()
    info = {
        "embedding_model": config.embedding_model,
        "embedding_device": config.embedding_device,
        "default_mode": config.default_mode,
        "default_top_k": config.default_top_k,
        "default_vector_weight": config.default_vector_weight,
        "similarity_cutoff": config.similarity_cutoff,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "max_sources": config.max_sources,
        "supported_extensions": list(config.supported_extensions),
        "documents_path": str(config.documents_path),
        "persist_dir": str(config.persist_dir),
    }
    return json.dumps(info, indent=2, ensure_ascii=False)


@mcp.resource("dsm5://documents")
async def resource_documents() -> str:
    """列出知识库中所有已索引的文档文件信息。"""
    system = _get_system()
    docs = await system.list_documents()
    return json.dumps(docs, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Tools（LLM 主动调用，带参数、有副作用）
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool
async def search(
    question: str,
    mode: str = "HYBRID",
    top_k: int = 5,
    vector_weight: float = 0.7,
) -> str:
    """对 DSM-5 知识库执行混合检索，返回匹配的文本块及分数（纯检索，无 LLM 生成）。

    不会调用任何外部 LLM，仅返回检索到的原始文本块。

    Args:
        question: 要检索的问题或关键词。
        mode: 检索模式 — HYBRID（混合去重）、VECTOR_ONLY（仅向量）、
               KEYWORD_ONLY（仅关键词）、FUSION（加权融合）。
        top_k: 每个通道返回的最多结果数（1–20）。
        vector_weight: FUSION 模式下向量分数的权重（0–1），其余模式忽略。
    """
    system = _get_system()
    _ensure_index(system)

    logger.info("search: question=%s mode=%s top_k=%d", question, mode, top_k)

    # 直接构建 HybridRetriever，完全绕过 LLM
    vector_retriever = system.vector_index.as_retriever(similarity_top_k=top_k)
    keyword_retriever = system.keyword_index.as_retriever(similarity_top_k=top_k)
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        mode=mode,
        vector_weight=vector_weight,
    )

    query_bundle = QueryBundle(query_str=question)
    nodes = hybrid_retriever.retrieve(query_bundle)

    return _format_nodes(nodes, system)


@mcp.tool
async def keyword_search(keyword: str, top_k: int = 5) -> str:
    """对 DSM-5 知识库执行纯关键词搜索，返回匹配的文本块。

    Args:
        keyword: 要搜索的关键词（如症状名称、诊断代码等）。
        top_k: 返回的最多结果数（1–20）。
    """
    system = _get_system()
    _ensure_index(system)

    logger.info("keyword_search: keyword=%s top_k=%d", keyword, top_k)

    request = KeywordSearchRequest(keyword=keyword, top_k=top_k)
    result = await system.keyword_search(request)

    lines = [f"关键词「{keyword}」检索到 {result['count']} 个结果:"]
    for item in result["results"]:
        lines.append(f"\n### #{item['rank']}")
        lines.append(f"- **文件**: {item['file_name']}")
        lines.append(f"- **分数**: {item['score']:.4f}")
        if item.get("page"):
            lines.append(f"- **页码**: {item['page']}")
        lines.append(f"- **文本**: {item['text_preview']}")

    if not result["results"]:
        lines.append("\n（未找到匹配结果）")

    return "\n".join(lines)


@mcp.tool
async def build_index(force_rebuild: bool = False) -> str:
    """构建或重建 DSM-5 知识库的检索索引。

    解析文档文件 → 切分为文本节点 → 构建向量索引 + 关键词索引。
    构建完成后即可使用 search / keyword_search 进行检索。

    Args:
        force_rebuild: 是否强制重建。为 False 时若已有索引则直接加载，
                        为 True 时无论是否存在都重新构建。
    """
    system = _get_system()
    logger.info("build_index: force_rebuild=%s", force_rebuild)
    result = await system.build_index(force_rebuild=force_rebuild)
    logger.info("build_index 完成: status=%s nodes=%s", result.get("status"), result.get("nodes"))
    return (
        f"索引构建结果:\n"
        f"- 状态: {result.get('status')}\n"
        f"- 消息: {result.get('message')}\n"
        f"- 文档数: {result.get('documents', 'N/A')}\n"
        f"- 节点数: {result.get('nodes', 'N/A')}\n"
        f"- 文件数: {result.get('file_count', 'N/A')}"
    )


@mcp.tool
async def get_index_status() -> str:
    """查询索引构建状态（ready / building / not_built）。"""
    system = _get_system()
    status = system.get_index_status()
    logger.info("get_index_status: status=%s", status.status)
    return status.model_dump_json(indent=2, ensure_ascii=False)


@mcp.tool
async def list_documents() -> str:
    """列出知识库中已索引的所有文档文件信息。"""
    system = _get_system()
    docs = await system.list_documents()
    logger.info("list_documents: total=%d", docs.get("total_documents", 0))
    lines = [f"文档列表（共 {docs['total_documents']} 个）:"]
    lines.append(f"支持格式: {', '.join(docs['supported_formats'])}")
    for doc in docs["documents"]:
        lines.append(f"\n- **{doc['file_name']}**")
        lines.append(f"  - 类型: {doc['file_type']}")
        lines.append(f"  - 大小: {doc['file_size_mb']:.1f} MB")
        lines.append(f"  - 路径: {doc['file_path']}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Prompts（可复用提示模板，供 LLM 作为指令使用）
# ═══════════════════════════════════════════════════════════════════════════


@mcp.prompt
async def dsm5_search_assistant() -> str:
    """DSM-5 检索助手系统提示 —— 通过检索工具查找诊断标准。"""
    return """你是一个专注于 **DSM-5 诊断标准** 的检索助手。

## 可用工具
| 工具 | 说明 |
|---|---|
| `search` | 混合/向量/关键词检索 |
| `keyword_search` | 纯关键词精确检索 |
| `build_index` | 构建或重建检索索引 |
| `get_index_status` | 查询索引构建状态 |
| `list_documents` | 列出已索引的文档 |

## 检索模式选择
| 模式 | 适用场景 |
|---|---|
| `HYBRID`（默认） | 兼顾精确匹配和语义相似度 |
| `KEYWORD_ONLY` | 精确匹配诊断标准名称或症状术语 |
| `VECTOR_ONLY` | 语义相近但表达不同的查询 |
| `FUSION` | 需要精细控制向量/关键词权重时 |

## 工作流程
1. 如需确保索引就绪，先调用 `get_index_status` 或 `build_index`
2. 理解用户的临床问题或诊断需求
3. 选择合适的检索模式和关键词，调用 `search` 或 `keyword_search`
4. 基于检索结果给出专业、准确的回答"""


@mcp.prompt
async def dsm5_differential_diagnosis(symptoms: str) -> str:
    """基于症状生成鉴别诊断的检索提示模板。

    Args:
        symptoms: 患者的症状描述。
    """
    return f"""请对以下症状进行 DSM-5 鉴别诊断检索：

**症状描述**: {symptoms}

请按以下步骤操作：
1. 使用 `search` 工具（模式: HYBRID）检索相关诊断标准
2. 使用 `keyword_search` 工具补充检索关键症状术语
3. 汇总检索结果，列出可能的诊断方向及对应的 DSM-5 诊断标准"""


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════


def _startup():
    """服务启动时主动初始化 RAG 系统并显示索引状态。"""
    logger.info("正在初始化 RAG 系统（首次加载可能较慢，请耐心等待）...")
    system = _get_system()
    status = system.get_index_status()
    logger.info("索引状态: %s — %s", status.status, status.message)
    if status.status == "ready":
        logger.info("索引已就绪，共 %d 个文档", status.details.get("document_count", 0))
    elif status.status == "building":
        logger.info("索引正在后台构建中，进度 %.0f%%", (status.progress or 0) * 100)
    else:
        logger.info("索引未构建，请通过 build_index 工具构建")


def main():
    """运行 dsm5-rag-mcp 服务器（默认 stdio 传输，适配本地客户端）。"""
    logger.info("启动 dsm5-rag-mcp 服务器（stdio）...")
    _startup()
    logger.info("服务就绪，等待客户端连接")
    mcp.run()


def main_http():
    """运行 dsm5-rag-mcp 服务器（Streamable HTTP 传输，适用于远程/生产）。"""
    logger.info("启动 dsm5-rag-mcp 服务器（Streamable HTTP）...")
    _startup()
    logger.info("服务就绪，等待客户端连接")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()