"""DSM-5 RAG FastAPI 微服务模块"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse

from dsm5_rag.config import RagConfig
from dsm5_rag.models import (
    BatchQueryRequest,
    IndexStatus,
    KeywordSearchRequest,
    QueryRequest,
    SystemStatus,
)
from dsm5_rag.system import DeepSeekRAGSystem

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rag_system: DeepSeekRAGSystem | None = None


# ==================== 流式响应生成器 ====================


async def simple_stream_generator(
    stream_response: AsyncGenerator,
) -> AsyncGenerator:
    """简化流式响应生成器，更适合Swagger UI显示"""
    async for chunk in stream_response:
        try:
            data = json.loads(chunk)
            event_type = data.get("event", "")
            if event_type == "query_started":
                yield f"查询开始: {data.get('question', '')}\n\n"
            elif event_type == "answer_chunk":
                yield f"回答: {data.get('chunk', '')}\n\n"
            elif event_type == "sources":
                sources = data.get("sources", [])
                yield f"来源 ({len(sources)}个):\n"
                for source in sources:
                    yield (
                        f"  {source.get('rank', '')}. "
                        f"{source.get('file_name', '未知文件')} "
                        f"({source.get('retrieval_type', '未知')})\n"
                    )
                yield "\n"
            elif event_type == "query_completed":
                yield f"查询完成于: {data.get('timestamp', '')}\n"
            elif event_type == "error":
                yield f"错误: {data.get('error', '未知错误')}\n"
        except Exception as e:
            yield f"解析错误: {str(e)}\n原始数据: {chunk}\n\n"


async def json_stream_generator(
    stream_response: AsyncGenerator,
) -> AsyncGenerator:
    """JSON流式响应生成器（保持原始格式）"""
    async for chunk in stream_response:
        yield chunk


# ==================== 应用生命周期 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_system
    try:
        logger.info("正在初始化DSM-5 RAG系统...")
        rag_config = RagConfig()
        logger.info("DSM-5 RAG系统配置摘要:")
        logger.info(f"  Embedding模型: {rag_config.embedding_model}")
        logger.info(f"  数据目录: {rag_config.persist_dir}")
        logger.info(f"  文档路径: {rag_config.documents_path}")
        logger.info(f"  检索模式: {rag_config.default_mode}")
        logger.info(
            f"  DeepSeek API密钥: {'已设置' if rag_config.deepseek_api_key else '未设置（使用模拟模式）'}"
        )
        rag_system = DeepSeekRAGSystem(rag_config)
        logger.info("DSM-5 RAG系统初始化完成，索引将在后台自动管理！")
    except Exception as e:
        logger.error(f"DSM-5 RAG系统初始化失败: {e}")
        raise
    yield
    logger.info("正在关闭DSM-5 RAG系统...")


# ==================== FastAPI应用 ====================

cors_origins = (
    os.getenv("CORS_ORIGINS", "*").split(",")
    if os.getenv("CORS_ORIGINS", "*") != "*"
    else ["*"]
)

app = FastAPI(
    title="DSM-5 RAG API",
    description="基于混合检索的DSM-5诊断标准知识问答系统 - 多语言版本",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="DSM-5 RAG API",
        version="1.0.0",
        description="基于混合检索的DSM-5诊断标准知识问答系统 - 多语言版本",
        routes=app.routes,
    )
    api_port = int(os.getenv("API_PORT", "8000"))
    openapi_schema["servers"] = [{"url": f"http://localhost:{api_port}", "description": "本地服务器"}]
    for path, path_item in openapi_schema["paths"].items():
        for method, operation in path_item.items():
            if "query" in path or "test/stream" in path:
                operation["description"] = (
                    operation.get("description", "")
                    + "\n\n**流式响应说明:**\n"
                    + "- 当`stream=true`时，返回流式响应\n"
                    + "- 格式: `application/x-ndjson` (每行一个JSON对象)\n"
                    + "- Swagger UI可能无法正确显示流式响应，建议使用curl或专门的API测试工具\n"
                    + "- 使用`/test/simple`端点可获得更适合Swagger UI的流式响应"
                )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API端点 ====================


@app.get("/", tags=["健康检查"])
async def root():
    """根端点，返回服务信息"""
    global rag_system
    status = "uninitialized"
    if rag_system:
        system_status = rag_system.get_status()
        status = system_status.status
        index_status = "ready" if system_status.index_built else "not_built"
        document_count = system_status.document_count
    else:
        index_status = "unknown"
        document_count = 0

    return {
        "service": "DSM-5 RAG API",
        "version": "1.0.0",
        "status": status,
        "index_status": index_status,
        "document_count": document_count,
        "description": "基于混合检索的DSM-5诊断标准知识问答系统 - 多语言版本",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "supported_formats": [".pdf", ".docx", ".txt"],
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "index_status": "/index/status",
            "documents": "/documents",
            "query": "/query",
            "keyword_search": "/search/keyword",
            "batch_query": "/query/batch",
            "change_mode": "/mode/change",
            "test": "/test",
            "test_stream": "/test/stream",
            "test_simple": "/test/simple",
        },
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    global rag_system
    if rag_system and rag_system.initialized:
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    return {"status": "initializing", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=SystemStatus, tags=["系统状态"])
async def get_system_status():
    """获取系统状态"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    return rag_system.get_status()


@app.get("/index/status", response_model=IndexStatus, tags=["系统状态"])
async def get_index_status():
    """获取索引状态"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    return rag_system.get_index_status()


@app.get("/documents", tags=["文档管理"])
async def list_documents():
    """列出所有文档文件信息"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    try:
        documents_info = await rag_system.list_documents()
        return JSONResponse(content=documents_info)
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@app.post("/query", tags=["查询"])
async def query_endpoint(
    request: QueryRequest,
    format: str = Query(
        "json", description="响应格式: json (默认), text (纯文本流)"
    ),
):
    """执行查询"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")

    if not rag_system.index_built:
        if rag_system.index_building:
            raise HTTPException(
                status_code=425, detail="索引正在构建中，请稍后再试"
            )
        index_status = rag_system.get_index_status()
        raise HTTPException(
            status_code=503, detail=f"索引未就绪: {index_status.message}"
        )

    try:
        result = await rag_system.query(request)

        if request.stream:
            if isinstance(result, dict) and "streaming_response" in result:
                stream_response = result["streaming_response"]
                if format == "text":
                    return StreamingResponse(
                        simple_stream_generator(stream_response),
                        media_type="text/plain; charset=utf-8",
                    )
                return StreamingResponse(
                    json_stream_generator(stream_response),
                    media_type=result.get("media_type", "application/x-ndjson"),
                )
            else:

                async def default_stream():
                    yield (
                        json.dumps(
                            {
                                "event": "query_started",
                                "question": request.question,
                                "mode": request.mode,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        + "\n"
                    )
                    if isinstance(result, dict) and "answer" in result:
                        yield (
                            json.dumps(
                                {
                                    "event": "answer_chunk",
                                    "chunk": result["answer"],
                                    "is_complete": True,
                                }
                            )
                            + "\n"
                        )
                    yield (
                        json.dumps(
                            {
                                "event": "query_completed",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        + "\n"
                    )

                return StreamingResponse(
                    default_stream(), media_type="application/x-ndjson"
                )

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/search/keyword", tags=["搜索"])
async def keyword_search_endpoint(request: KeywordSearchRequest):
    """执行纯关键词搜索"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    if not rag_system.index_built:
        raise HTTPException(status_code=503, detail="索引未构建")

    try:
        result = await rag_system.keyword_search(request)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"关键词搜索失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"关键词搜索失败: {str(e)}"
        )


@app.post("/query/batch", tags=["查询"])
async def batch_query_endpoint(request: BatchQueryRequest):
    """批量查询"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    if not rag_system.index_built:
        raise HTTPException(status_code=503, detail="索引未构建")

    try:
        result = await rag_system.batch_query(request)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量查询失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"批量查询失败: {str(e)}"
        )


@app.post("/mode/change", tags=["配置"])
async def change_mode(
    mode: str = Query(
        ...,
        description="检索模式: HYBRID, VECTOR_ONLY, KEYWORD_ONLY, FUSION",
    ),
    similarity_top_k: int = Query(
        default=5, ge=1, le=20, description="检索数量"
    ),
    vector_weight: float = Query(
        default=0.7,
        ge=0,
        le=1,
        description="向量权重（仅FUSION模式有效）",
    ),
):
    """更改检索模式"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")

    try:
        if mode not in ["HYBRID", "VECTOR_ONLY", "KEYWORD_ONLY", "FUSION"]:
            raise HTTPException(status_code=400, detail="无效的检索模式")

        await rag_system.create_query_engine(
            hybrid_mode=mode,
            similarity_top_k=similarity_top_k,
            vector_weight=vector_weight,
        )

        return JSONResponse(
            content={
                "status": "success",
                "message": f"检索模式已更改为 {mode}",
                "mode": mode,
                "similarity_top_k": similarity_top_k,
                "vector_weight": vector_weight if mode == "FUSION" else None,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"模式更改失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"模式更改失败: {str(e)}"
        )


@app.get("/test", tags=["测试"])
async def test_endpoint():
    """测试端点，验证系统基本功能（非流式）"""
    global rag_system
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "RAG系统未初始化"},
        )

    try:
        status = rag_system.get_status()
        test_result = None
        if rag_system.index_built:
            try:
                request = QueryRequest(
                    question="What are the diagnostic criteria for Major Depressive Disorder?",
                    mode="HYBRID",
                    similarity_top_k=2,
                    stream=False,
                )
                test_result = await rag_system.query(request)
            except Exception as e:
                test_result = {"error": str(e)}

        return JSONResponse(
            content={
                "status": "success",
                "system_status": status.model_dump(),
                "test_query": test_result,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.get("/test/stream", tags=["测试"])
async def test_stream_endpoint(
    format: str = Query(
        "json", description="响应格式: json (默认), text (纯文本)"
    ),
):
    """测试流式查询端点"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    if not rag_system.index_built:
        raise HTTPException(status_code=503, detail="索引未构建")

    try:
        request = QueryRequest(
            question="What are the diagnostic criteria for Major Depressive Disorder?",
            mode="HYBRID",
            similarity_top_k=2,
            stream=True,
        )
        result = await rag_system.query(request)

        if isinstance(result, dict) and "streaming_response" in result:
            stream_response = result["streaming_response"]
            if format == "text":
                return StreamingResponse(
                    simple_stream_generator(stream_response),
                    media_type="text/plain; charset=utf-8",
                )
            return StreamingResponse(
                json_stream_generator(stream_response),
                media_type=result.get("media_type", "application/x-ndjson"),
            )
        else:
            async def default_stream():
                yield (
                    json.dumps(
                        {
                            "event": "query_started",
                            "question": request.question,
                            "mode": request.mode,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    + "\n"
                )
                yield (
                    json.dumps(
                        {
                            "event": "answer_chunk",
                            "chunk": "DSM-5流式查询测试响应",
                            "is_complete": True,
                        }
                    )
                    + "\n"
                )
                yield (
                    json.dumps(
                        {
                            "event": "query_completed",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    + "\n"
                )

            return StreamingResponse(
                default_stream(), media_type="application/x-ndjson"
            )
    except Exception as e:
        logger.error(f"流式查询测试失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"流式查询测试失败: {str(e)}"
        )


@app.get("/test/simple", tags=["测试"])
async def test_simple_stream_endpoint():
    """简单流式测试端点（适合Swagger UI）"""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    if not rag_system.index_built:
        raise HTTPException(status_code=503, detail="索引未构建")

    async def simple_test_stream():
        yield "DSM-5知识库测试流式查询开始\n\n"
        yield "这是一个DSM-5诊断标准测试响应，展示流式查询功能\n\n"
        yield "示例来源:\n"
        yield "  1. Major Depressive Disorder (MDD)诊断标准\n"
        yield "  2. Bipolar I Disorder诊断标准\n"
        yield "  3. Generalized Anxiety Disorder (GAD)诊断标准\n\n"
        yield f"查询完成于: {datetime.now().isoformat()}\n"
        await asyncio.sleep(0.5)
        if rag_system.index_built:
            yield "\n系统信息:\n"
            status = rag_system.get_status()
            yield f"  索引状态: {'已构建' if status.index_built else '未构建'}\n"
            yield f"  文档数量: {status.document_count}\n"
            yield f"  嵌入模型: {status.embedding_model}\n"
            yield f"  运行时间: {status.uptime:.1f} 秒\n"

    return StreamingResponse(
        simple_test_stream(), media_type="text/plain; charset=utf-8"
    )


# ==================== 错误处理 ====================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


# ==================== 主程序入口 ====================


def main():
    """主程序入口"""
    import uvicorn

    config = RagConfig()

    logger.info("正在启动DSM-5 RAG API服务...")
    logger.info(f"服务地址: http://{config.api_host}:{config.api_port}")
    logger.info(f"文档地址: http://{config.api_host}:{config.api_port}/docs")

    uvicorn.run(
        "dsm5_rag.api:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()