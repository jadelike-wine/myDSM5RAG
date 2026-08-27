"""RAG 系统核心模块"""

import asyncio
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# 优先加载环境变量，确保 HF_ENDPOINT 等变量在 huggingface_hub 导入前生效
_here = Path(__file__).resolve().parent.parent.parent
load_dotenv(_here / ".env")

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
    QueryBundle,
)
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.postprocessor import (
    LongContextReorder,
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.chroma import ChromaVectorStore

from dsm5_rag.config import RagConfig, current_dir
from dsm5_rag.models import (
    BatchQueryRequest,
    IndexStatus,
    KeywordSearchRequest,
    QueryRequest,
    SystemStatus,
)
from dsm5_rag.parser import DocumentParser
from dsm5_rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# 线程池用于执行同步操作
executor = ThreadPoolExecutor(max_workers=4)


class DeepSeekRAGSystem:
    """DeepSeek混合检索RAG系统 - DSM-5知识库版本"""

    def __init__(self, config: RagConfig | None = None, auto_build_index: bool = True):
        """初始化RAG系统 - 自动管理索引"""
        self.app_start_time = datetime.now()
        self.config = config or RagConfig()

        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))
        os.environ["HF_ENDPOINT"] = self.config.hf_endpoint

        self.documents_path = self._resolve_path(self.config.documents_path)
        self.persist_dir = self._resolve_path(self.config.persist_dir)
        self.models_dir = self._resolve_path(self.config.models_dir)

        self.embedding_model = None
        self.llm = None
        self.vector_index = None
        self.keyword_index = None
        self.query_engine = None
        self.initialized = False
        self.index_built = False
        self.current_mode = self.config.default_mode
        self.index_building = False
        self.index_build_progress = 0.0
        self._async_lock = asyncio.Lock()
        self.document_files: list[Path] = []
        self.total_documents = 0

        self._create_directories()
        logger.info("开始初始化DSM-5 RAG系统组件...")
        logger.info("配置信息:")
        logger.info(f"  DeepSeek API密钥: {'已设置' if self.config.deepseek_api_key else '未设置'}")
        logger.info(f"  文档路径: {self.documents_path}")
        logger.info(f"  持久化目录: {self.persist_dir}")
        logger.info(f"  检索模式: {self.config.default_mode}")

        self._init_embedding_model()
        self._init_llm()
        self.initialized = True
        logger.info("RAG系统组件初始化成功")

        self._scan_document_files()
        if auto_build_index and self.document_files:
            self._init_storage()
            threading.Thread(target=self._auto_manage_index_thread, daemon=True).start()
        elif not self.document_files:
            logger.warning("未找到支持的文档文件，请检查DOCUMENTS_PATH配置")

        logger.info("DeepSeekRAGSystem初始化完成 - DSM-5知识库版本")

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            return current_dir / path
        return path

    def _create_directories(self):
        dirs = [self.models_dir, self.persist_dir]
        if self.documents_path.is_dir():
            dirs.append(self.documents_path)
        else:
            dirs.append(self.documents_path.parent)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _scan_document_files(self):
        self.document_files = []
        if self.documents_path.is_file():
            if self.documents_path.suffix.lower() in self.config.supported_extensions:
                self.document_files.append(self.documents_path)
        elif self.documents_path.is_dir():
            for ext in self.config.supported_extensions:
                for file_path in self.documents_path.rglob(f"*{ext}"):
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if file_size_mb <= self.config.max_file_size_mb:
                        self.document_files.append(file_path)
                        logger.info(f"找到文档文件: {file_path.name} ({file_size_mb:.1f}MB)")
                    else:
                        logger.warning(f"跳过大文件: {file_path.name} ({file_size_mb:.1f}MB)")
        self.total_documents = len(self.document_files)
        logger.info(f"共找到 {self.total_documents} 个文档文件")

    def _init_embedding_model(self):
        model_name = self.config.embedding_model
        logger.info(f"正在加载多语言Embedding模型: {model_name}")
        try:
            self.embedding_model = HuggingFaceEmbedding(
                model_name=model_name,
                cache_folder=str(self.models_dir),
                device=self.config.embedding_device,
            )
            Settings.embed_model = self.embedding_model
            logger.info("多语言Embedding模型加载成功")
        except Exception as e:
            logger.error(f"多语言Embedding模型加载失败: {e}")
            raise

    def _init_llm(self):
        api_key = self.config.deepseek_api_key
        logger.info("正在初始化DeepSeek LLM...")
        if not api_key:
            logger.warning("未提供DeepSeek API密钥，将使用模拟模式")
            from llama_index.core.llms import MockLLM

            self.llm = MockLLM()
        else:
            self.llm = DeepSeek(
                api_key=api_key,
                base_url=self.config.deepseek_base_url,
                model=self.config.deepseek_model,
                temperature=0.1,
                timeout=60,
            )
        Settings.llm = self.llm
        logger.info("DeepSeek LLM初始化成功")

    def _init_storage(self):
        logger.info(f"初始化存储系统: {self.persist_dir}")
        self.persist_dir.mkdir(exist_ok=True)
        chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        vector_collection = chroma_client.get_or_create_collection(
            name="dsm5_vector_index",
            metadata={"description": "DSM-5诊断标准向量索引 - 多语言版本"},
        )
        self.vector_store = ChromaVectorStore(chroma_collection=vector_collection)
        logger.info("存储系统初始化成功")

    def _auto_manage_index_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._auto_manage_index_async())
        except Exception as e:
            logger.error(f"索引自动管理失败: {e}")
        finally:
            if loop:
                loop.close()

    async def _auto_manage_index_async(self):
        try:
            vector_index_path = self.persist_dir / "vector_index"
            keyword_index_path = self.persist_dir / "keyword_index"
            if vector_index_path.exists() and keyword_index_path.exists():
                logger.info("检测到已有索引，尝试加载...")
                await self._load_existing_index()
            else:
                logger.info("未检测到现有索引，将自动构建...")
                await self.build_index()
        except Exception as e:
            logger.error(f"索引自动管理失败: {e}")

    async def _load_existing_index(self):
        try:
            vector_index_path = self.persist_dir / "vector_index"
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store, persist_dir=str(vector_index_path)
            )
            self.vector_index = load_index_from_storage(storage_context)

            keyword_index_path = self.persist_dir / "keyword_index"
            keyword_storage_context = StorageContext.from_defaults(
                persist_dir=str(keyword_index_path)
            )
            self.keyword_index = load_index_from_storage(keyword_storage_context)

            self.index_built = True
            logger.info("索引加载成功")
            await self.create_query_engine()
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            self.index_built = False

    async def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """构建混合索引"""
        async with self._async_lock:
            if self.index_building:
                return {"status": "building", "message": "索引正在构建中，请稍候"}

            self.index_building = True
            self.index_build_progress = 0.0

            try:
                if not self.document_files:
                    raise FileNotFoundError(
                        f"未找到支持的文档文件: {self.documents_path}"
                    )

                logger.info(
                    f"开始构建索引，共 {len(self.document_files)} 个文档文件"
                )

                vector_index_path = self.persist_dir / "vector_index"
                keyword_index_path = self.persist_dir / "keyword_index"

                if not force_rebuild and vector_index_path.exists() and keyword_index_path.exists():
                    logger.info("检测到已有索引，正在加载...")
                    try:
                        await self._load_existing_index()
                        self.index_building = False
                        return {"status": "loaded", "message": "索引加载成功"}
                    except Exception as e:
                        logger.error(f"索引加载失败，将重新构建: {e}")
                        force_rebuild = True

                logger.info("重新构建索引...")
                self.index_build_progress = 0.1
                documents = await self._parse_documents()
                if not documents:
                    self.index_building = False
                    raise ValueError("未解析到任何文档内容")

                self.index_build_progress = 0.3
                node_parser = HierarchicalNodeParser.from_defaults(
                    chunk_sizes=[self.config.chunk_size, self.config.chunk_size // 2],
                    chunk_overlap=self.config.chunk_overlap,
                    include_metadata=True,
                )
                nodes = node_parser.get_nodes_from_documents(documents)
                logger.info(f"生成 {len(nodes)} 个文本节点")

                self.index_build_progress = 0.6
                logger.info("构建向量索引...")
                vector_storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
                self.vector_index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=vector_storage_context,
                    embed_model=self.embedding_model,
                )
                vector_storage_context.persist(persist_dir=str(vector_index_path))

                self.index_build_progress = 0.8
                logger.info("构建关键词索引...")
                keyword_storage_context = StorageContext.from_defaults()
                self.keyword_index = SummaryIndex(
                    nodes=nodes, storage_context=keyword_storage_context
                )
                keyword_storage_context.persist(persist_dir=str(keyword_index_path))

                self.index_built = True
                self.index_build_progress = 1.0
                self.index_building = False
                logger.info("混合索引构建完成")
                await self.create_query_engine()

                return {
                    "status": "built",
                    "message": "索引构建成功",
                    "documents": len(documents),
                    "nodes": len(nodes),
                    "file_count": len(self.document_files),
                    "document_path": str(self.documents_path),
                }

            except Exception as e:
                self.index_building = False
                logger.error(f"索引构建失败: {e}")
                raise

    async def _parse_documents(self) -> List[Document]:
        """解析所有支持格式的文档"""
        logger.info(f"正在解析文档文件: {len(self.document_files)} 个文件")
        all_documents: List[Document] = []
        loop = asyncio.get_event_loop()
        batch_size = 5
        total_files = len(self.document_files)

        for i in range(0, total_files, batch_size):
            batch_files = self.document_files[i : i + batch_size]
            logger.info(
                f"处理文件批次 {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size}"
            )

            batch_tasks = [
                loop.run_in_executor(executor, self._sync_parse_document, fp)
                for fp in batch_files
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                file_path = batch_files[j]
                if isinstance(result, Exception):
                    logger.error(f"文档解析失败 {file_path}: {result}")
                elif result:
                    all_documents.extend(result)

            self.index_build_progress = 0.1 + 0.2 * (
                min(i + batch_size, total_files) / total_files
            )

        logger.info(f"解析完成，共 {len(all_documents)} 个文档块")
        return all_documents

    def _sync_parse_document(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return DocumentParser.parse_pdf(file_path)
        elif ext == ".docx":
            return DocumentParser.parse_docx(file_path)
        elif ext == ".txt":
            return DocumentParser.parse_txt(file_path)
        else:
            return []

    async def create_query_engine(
        self,
        hybrid_mode: str | None = None,
        similarity_top_k: int | None = None,
        vector_weight: float | None = None,
    ):
        """创建简化的混合查询引擎"""
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        mode = hybrid_mode or self.current_mode
        top_k = similarity_top_k or self.config.default_top_k
        weight = vector_weight or self.config.default_vector_weight

        logger.info(f"创建混合查询引擎（模式: {mode}, top_k: {top_k}）")

        vector_retriever = self.vector_index.as_retriever(similarity_top_k=top_k)
        keyword_retriever = self.keyword_index.as_retriever(similarity_top_k=top_k)
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            mode=mode,
            vector_weight=weight,
        )

        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=self.config.similarity_cutoff)
        ]
        if self.config.enable_context_reorder:
            node_postprocessors.append(LongContextReorder())
        if self.config.enable_metadata_replacement:
            node_postprocessors.append(
                MetadataReplacementPostProcessor(target_metadata_key="window")
            )

        qa_prompt_template = PromptTemplate(
            """你是一个专业的DSM-5诊断标准助手，请基于以下上下文信息，用简洁、专业的语言回答用户的问题。

            上下文信息:
            {context_str}

            用户问题: {query_str}

            请根据DSM-5诊断标准提供准确、专业的回答。如果上下文信息不足，请说明并给出基于你知识的一般性建议。

            回答:"""
        )

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            llm=self.llm,
            response_mode="compact",
            node_postprocessors=node_postprocessors,
            text_qa_template=qa_prompt_template,
            verbose=False,
            streaming=False,
        )

        self.current_mode = mode
        logger.info("混合查询引擎创建成功")
        return self.query_engine

    async def query(
        self, request: QueryRequest, verbose: bool = False
    ) -> Dict[str, Any]:
        """执行查询"""
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        logger.info(
            f"查询问题: {request.question[:50]}... (模式: {request.mode})"
        )

        if not self.query_engine or self.current_mode != request.mode:
            await self.create_query_engine(
                hybrid_mode=request.mode,
                similarity_top_k=request.similarity_top_k,
                vector_weight=request.vector_weight,
            )

        if request.stream:
            return await self._stream_query(request)

        try:
            response = self.query_engine.query(request.question)
            result = {
                "question": request.question,
                "answer": str(response),
                "mode": request.mode,
                "timestamp": datetime.now().isoformat(),
                "response_time": None,
            }

            if hasattr(response, "source_nodes") and response.source_nodes:
                sources = []
                retrieval_stats: Dict[str, int] = {"vector": 0, "keyword": 0, "both": 0}
                max_sources = min(self.config.max_sources, len(response.source_nodes))

                for i, node in enumerate(response.source_nodes[:max_sources]):
                    file_name = node.node.metadata.get("file_name", "未知文件")
                    file_type = node.node.metadata.get("file_type", "未知类型")
                    page = node.node.metadata.get("page", None)
                    retrieval_type = node.node.metadata.get("retrieval_type", "未知")
                    score = node.score if node.score is not None else 0

                    if retrieval_type in retrieval_stats:
                        retrieval_stats[retrieval_type] += 1

                    text_preview = (
                        node.node.text[:150] + "..."
                        if len(node.node.text) > 150
                        else node.node.text
                    )

                    sources.append(
                        {
                            "rank": i + 1,
                            "file_name": file_name,
                            "file_type": file_type,
                            "page": page,
                            "retrieval_type": retrieval_type,
                            "score": round(score, 4),
                            "text_length": len(node.node.text),
                            "text_preview": text_preview,
                            "full_text": node.node.text if verbose else None,
                        }
                    )

                result["sources"] = sources
                result["retrieval_stats"] = retrieval_stats
                result["sources_limited"] = (
                    f"显示前{max_sources}个来源（共{len(response.source_nodes)}个）"
                )

            result["response_time"] = (
                datetime.now() - self.app_start_time
            ).total_seconds()
            logger.info(
                f"查询完成，返回 {len(sources) if 'sources' in result else 0} 个来源"
            )
            return result

        except Exception as e:
            logger.error(f"查询过程中发生错误: {e}")
            return {
                "question": request.question,
                "answer": f"查询过程中发生错误: {str(e)}",
                "error": str(e),
                "mode": request.mode,
                "timestamp": datetime.now().isoformat(),
            }

    async def _stream_query(self, request: QueryRequest) -> Dict[str, Any]:
        """流式查询"""
        async def generate():
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
            try:
                response = self.query_engine.query(request.question)
                yield (
                    json.dumps(
                        {
                            "event": "answer_chunk",
                            "chunk": str(response),
                            "is_complete": True,
                        }
                    )
                    + "\n"
                )
                if hasattr(response, "source_nodes") and response.source_nodes:
                    sources = []
                    max_sources = min(3, len(response.source_nodes))
                    for i, node in enumerate(response.source_nodes[:max_sources]):
                        file_name = node.node.metadata.get("file_name", "未知文件")
                        retrieval_type = node.node.metadata.get(
                            "retrieval_type", "未知"
                        )
                        sources.append(
                            {
                                "rank": i + 1,
                                "file_name": file_name,
                                "retrieval_type": retrieval_type,
                            }
                        )
                    yield (
                        json.dumps(
                            {
                                "event": "sources",
                                "sources": sources,
                                "total_sources": len(response.source_nodes),
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
            except Exception as e:
                yield (
                    json.dumps({"event": "error", "error": str(e)}) + "\n"
                )

        return {"streaming_response": generate(), "media_type": "application/x-ndjson"}

    async def keyword_search(
        self, request: KeywordSearchRequest
    ) -> Dict[str, Any]:
        """执行纯关键词搜索"""
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        query_bundle = QueryBundle(query_str=request.keyword)
        keyword_retriever = self.keyword_index.as_retriever(
            similarity_top_k=request.top_k
        )
        nodes = keyword_retriever.retrieve(query_bundle)

        results = []
        for i, node in enumerate(nodes):
            file_name = node.node.metadata.get("file_name", "未知文件")
            file_type = node.node.metadata.get("file_type", "未知类型")
            page = node.node.metadata.get("page", None)
            text_preview = (
                node.node.text[:150] + "..."
                if len(node.node.text) > 150
                else node.node.text
            )
            results.append(
                {
                    "rank": i + 1,
                    "file_name": file_name,
                    "file_type": file_type,
                    "page": page,
                    "score": node.score if node.score is not None else 0,
                    "text_preview": text_preview,
                    "full_text": node.node.text,
                }
            )

        return {
            "keyword": request.keyword,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat(),
        }

    async def batch_query(
        self, request: BatchQueryRequest
    ) -> Dict[str, Any]:
        """批量查询"""
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        await self.create_query_engine(hybrid_mode=request.mode)

        results = []
        for i, question in enumerate(request.questions):
            try:
                response = self.query_engine.query(question)
                results.append(
                    {"question": question, "answer": str(response), "success": True}
                )
            except Exception as e:
                results.append(
                    {
                        "question": question,
                        "answer": f"查询失败: {str(e)}",
                        "success": False,
                        "error": str(e),
                    }
                )

        return {
            "mode": request.mode,
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "timestamp": datetime.now().isoformat(),
        }

    def get_status(self) -> SystemStatus:
        """获取系统状态"""
        uptime = (datetime.now() - self.app_start_time).total_seconds()
        index_stats = None
        if self.index_built and self.vector_index:
            try:
                index_stats = {
                    "vector_index_type": type(self.vector_index).__name__,
                    "keyword_index_type": (
                        type(self.keyword_index).__name__
                        if self.keyword_index
                        else None
                    ),
                    "query_engine_ready": self.query_engine is not None,
                    "current_mode": self.current_mode,
                    "document_count": self.total_documents,
                }
            except Exception:
                pass

        return SystemStatus(
            status="running" if self.initialized else "initializing",
            documents_path=str(self.documents_path) if self.documents_path else None,
            db_dir=str(self.persist_dir) if self.persist_dir else None,
            embedding_model=self.config.embedding_model,
            index_built=self.index_built,
            index_stats=index_stats,
            uptime=uptime,
            document_count=self.total_documents,
            supported_formats=list(self.config.supported_extensions),
        )

    def get_index_status(self) -> IndexStatus:
        """获取索引构建状态"""
        if self.index_building:
            return IndexStatus(
                status="building",
                message="索引正在构建中",
                progress=self.index_build_progress,
                details={
                    "progress_percentage": round(self.index_build_progress * 100, 1),
                    "document_count": self.total_documents,
                    "files_processed": int(
                        self.index_build_progress * self.total_documents
                    ),
                },
            )
        elif self.index_built:
            return IndexStatus(
                status="ready",
                message="DSM-5知识库索引已构建完成",
                progress=1.0,
                details={
                    "document_count": self.total_documents,
                    "query_engine_ready": self.query_engine is not None,
                    "mode": self.current_mode,
                },
            )
        else:
            return IndexStatus(
                status="not_built",
                message="索引未构建",
                progress=0.0,
                details={
                    "document_count": self.total_documents,
                    "documents_path": str(self.documents_path),
                },
            )

    async def list_documents(self) -> Dict[str, Any]:
        """列出所有文档文件信息"""
        documents_info = []
        for file_path in self.document_files:
            try:
                file_size = file_path.stat().st_size
                documents_info.append(
                    {
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "file_type": file_path.suffix.lower(),
                        "last_modified": datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"获取文件信息失败 {file_path}: {e}")

        return {
            "total_documents": len(documents_info),
            "documents": documents_info,
            "supported_formats": list(self.config.supported_extensions),
        }