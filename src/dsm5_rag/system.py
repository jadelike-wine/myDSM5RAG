"""RAG 系统核心模块"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from dotenv import load_dotenv

# 优先加载环境变量，确保 HF_ENDPOINT 等变量在 huggingface_hub 导入前生效
_here = Path(__file__).resolve().parent.parent.parent
load_dotenv(_here / ".env")

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import (
    Document,
    QueryBundle,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.postprocessor import (
    LongContextReorder,
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.chroma import ChromaVectorStore

from dsm5_rag.background import executor, run_in_executor
from dsm5_rag.config import RagConfig, current_dir
from dsm5_rag.index_manifest import IndexManifest, describe_files, status_of
from dsm5_rag.keyword_index import BM25KeywordIndex
from dsm5_rag.models import (
    BatchQueryRequest,
    IndexStatus,
    KeywordSearchRequest,
    QueryRequest,
    SystemStatus,
)
from dsm5_rag.parser import DocumentParser
from dsm5_rag.retriever import VALID_MODES, HybridRetriever

logger = logging.getLogger(__name__)


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
        self.initialized = False
        self.index_built = False
        # "服务级默认检索参数"：只有 /mode/change 会显式改它们，
        # 单个请求不会留下任何副作用（历史版本用 self.query_engine /
        # self.current_mode 存引擎，导致并发请求互相污染）。
        self.default_mode = (self.config.default_mode or "HYBRID").upper()
        self.default_top_k = self.config.default_top_k
        self.default_vector_weight = self.config.default_vector_weight
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

    def _build_embedding_model(self):
        """构造 embedding 模型。

        延迟导入 HuggingFaceEmbedding（它依赖 sentence-transformers/torch），
        这样只用到关键词通道或做检索测试时不需要装几百 MB 的推理框架，
        也让测试可以注入替身模型。
        """
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(
            model_name=self.config.embedding_model,
            cache_folder=str(self.models_dir),
            device=self.config.embedding_device,
        )

    def _init_embedding_model(self):
        model_name = self.config.embedding_model
        logger.info(f"正在加载多语言Embedding模型: {model_name}")
        try:
            self.embedding_model = self._build_embedding_model()
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
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._auto_manage_index_async())
        except Exception as e:
            logger.error(f"索引自动管理失败: {e}")
        finally:
            if loop is not None:
                loop.close()

    async def _auto_manage_index_async(self):
        """启动时的索引自动管理：交给 build_index 的清单比对逻辑处理。"""
        try:
            result = await self.build_index()
            logger.info("索引自动管理完成: %s", result.get("status"))
        except Exception as e:
            logger.error(f"索引自动管理失败: {e}")

    def _reset_index_dirs(self, *dirs: Path) -> None:
        """重建前清空旧的索引目录，避免已删除文档的节点残留在库里。"""
        import shutil

        for directory in dirs:
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)
                logger.info("重建前已清空旧索引目录: %s", directory)

    async def index_diff(self) -> Dict[str, Any]:
        """当前磁盘文档与已索引清单的差异（供 /index/diff 与运维排查用）。"""
        # 先重新扫描目录，否则看不到"放进来但还没建索引"的新文件
        self._scan_document_files()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor, status_of, self.persist_dir, self.document_files, self.config
        )

    async def _load_existing_index(self):
        try:
            vector_index_path = self.persist_dir / "vector_index"
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store, persist_dir=str(vector_index_path)
            )
            self.vector_index = load_index_from_storage(storage_context)

            keyword_index_path = self.persist_dir / "keyword_index"
            keyword_index = BM25KeywordIndex.load_or_migrate(keyword_index_path)
            if keyword_index is None:
                raise ValueError(
                    f"关键词索引缺失或格式过旧: {keyword_index_path}"
                )
            self.keyword_index = keyword_index

            self.index_built = True
            logger.info(
                "索引加载成功（向量索引 + BM25 关键词索引，%d 个关键词节点）",
                keyword_index.node_count,
            )
            await self.create_query_engine()
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            self.index_built = False
            raise

    async def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """构建/重建混合索引。

        会先重新扫描文档目录，再和 `persist_dir/index_manifest.json` 里记录的
        每个文件的 sha256/mtime/size 以及切分相关配置比对：
        没有任何变化时直接加载现有索引（不重新 embedding），有变化才重建。
        """
        async with self._async_lock:
            if self.index_building:
                return {"status": "building", "message": "索引正在构建中，请稍候"}

            self.index_building = True
            self.index_build_progress = 0.0

            try:
                # 关键：重新扫描，否则放进来的新文件永远发现不了
                self._scan_document_files()
                if not self.document_files:
                    raise FileNotFoundError(
                        f"未找到支持的文档文件: {self.documents_path}"
                    )

                current_files = await run_in_executor(
                    describe_files, self.document_files
                )
                signature = IndexManifest.build_config_signature(self.config)

                vector_index_path = self.persist_dir / "vector_index"
                keyword_index_path = self.persist_dir / "keyword_index"

                if (
                    vector_index_path.exists()
                    and keyword_index_path.exists()
                    and not force_rebuild
                ):
                    manifest = IndexManifest.load(self.persist_dir)
                    if manifest is None:
                        logger.info("索引已存在但缺少清单，需要重建以建立基线")
                        needs_rebuild, reasons = True, {"reason": "缺少索引清单"}
                    else:
                        needs_rebuild, reasons = manifest.needs_rebuild(
                            current_files, signature
                        )
                    if not needs_rebuild:
                        logger.info("文档与配置均未变化，直接加载现有索引")
                        await self._load_existing_index()
                        return {
                            "status": "loaded",
                            "message": "文档未变化，索引直接加载",
                            "documents": len(current_files),
                            "file_count": len(current_files),
                            "document_path": str(self.documents_path),
                            "manifest_updated_at": manifest.updated_at
                            if manifest
                            else None,
                        }
                    logger.info("检测到需要重建索引的原因: %s", reasons)
                    rebuild_reasons = reasons
                else:
                    rebuild_reasons = {"force_rebuild": True} if force_rebuild else {}

                # 旧索引目录里可能残留已删除文件的节点；重建前先清掉索引目录，
                # 避免把陈旧向量留在库里。
                self._reset_index_dirs(vector_index_path, keyword_index_path)

                logger.info(
                    f"开始构建索引，共 {len(self.document_files)} 个文档文件"
                )

                logger.info("重新构建索引...")
                self.index_build_progress = 0.1
                documents = await self._parse_documents()
                if not documents:
                    raise ValueError("未解析到任何文档内容")

                self.index_build_progress = 0.3
                nodes = await run_in_executor(self._chunk_documents, documents)

                self.index_build_progress = 0.6
                logger.info("构建向量索引...")
                self.vector_index = await run_in_executor(
                    self._build_vector_index, nodes, vector_index_path
                )

                self.index_build_progress = 0.8
                logger.info("构建 BM25 关键词索引...")
                keyword_index = await run_in_executor(
                    self._build_keyword_index, nodes
                )
                self.keyword_index = keyword_index
                await run_in_executor(keyword_index.persist, keyword_index_path)

                self.index_build_progress = 0.95
                IndexManifest(
                    persist_dir=self.persist_dir,
                    files=current_files,
                    build_config=signature,
                    stats={
                        "documents": len(documents),
                        "nodes": len(nodes),
                        "keyword_nodes": keyword_index.node_count,
                        "file_count": len(current_files),
                    },
                ).save()

                self.index_built = True
                self.index_build_progress = 1.0
                logger.info("混合索引构建完成")
                await self.create_query_engine()

                return {
                    "status": "built",
                    "message": "索引构建成功",
                    "documents": len(documents),
                    "nodes": len(nodes),
                    "keyword_nodes": keyword_index.node_count,
                    "file_count": len(self.document_files),
                    "document_path": str(self.documents_path),
                    "rebuild_reason": rebuild_reasons,
                }

            except Exception as e:
                logger.error(f"索引构建失败: {e}")
                raise
            finally:
                self.index_building = False

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

    def _chunk_documents(self, documents: List[Document]) -> List[Any]:
        """切分文档（CPU 密集，在子线程里跑）。"""
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[self.config.chunk_size, self.config.chunk_size // 2],
            chunk_overlap=self.config.chunk_overlap,
            include_metadata=True,
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"生成 {len(nodes)} 个文本节点")
        return nodes

    def _build_vector_index(
        self, nodes: List[Any], persist_path: Path
    ) -> VectorStoreIndex:
        """构建并持久化向量索引（embedding 是同步重活，必须在子线程里跑）。"""
        vector_storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=vector_storage_context,
            embed_model=self.embedding_model,
        )
        vector_storage_context.persist(persist_dir=str(persist_path))
        return index

    def _build_keyword_index(self, nodes) -> BM25KeywordIndex:
        """在子线程里构建 BM25 倒排（分词是 CPU 密集操作）。"""
        return BM25KeywordIndex.from_nodes(
            nodes,
            language=self.config.bm25_language,
            skip_stemming=self.config.bm25_skip_stemming,
        )

    def _sync_parse_document(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            documents = DocumentParser.parse_pdf(file_path)
        elif ext == ".docx":
            documents = DocumentParser.parse_docx(file_path)
        elif ext == ".txt":
            documents = DocumentParser.parse_txt(file_path)
        else:
            return []

        # 文档 ID 必须稳定：节点通过 ref_doc_id 关联回文档，清单比对、
        # 按文档删除旧节点都依赖它。用绝对路径派生，而不是每次解析都生成 uuid。
        try:
            stem = str(file_path.resolve())
        except OSError:
            stem = str(file_path)
        multi = len(documents) > 1
        for i, document in enumerate(documents):
            document.id_ = f"{stem}::{i}" if multi else stem
        return documents

    QA_PROMPT = PromptTemplate(
        """你是一个专业的DSM-5诊断标准助手，请基于以下上下文信息，用简洁、专业的语言回答用户的问题。

            上下文信息:
            {context_str}

            用户问题: {query_str}

            请根据DSM-5诊断标准提供准确、专业的回答。如果上下文信息不足，请说明并给出基于你知识的一般性建议。

            回答:"""
    )

    def _normalize_search_params(
        self,
        mode: str | None = None,
        similarity_top_k: int | None = None,
        vector_weight: float | None = None,
    ) -> tuple[str, int, float]:
        """把入参与服务默认值归一为 (mode, top_k, vector_weight)。"""
        resolved_mode = (mode or self.default_mode or "HYBRID").upper()
        resolved_top_k = int(similarity_top_k or self.default_top_k)
        resolved_weight = (
            self.default_vector_weight
            if vector_weight is None
            else max(0.0, min(1.0, float(vector_weight)))
        )
        return resolved_mode, resolved_top_k, resolved_weight

    def params_from_request(
        self, request: QueryRequest | KeywordSearchRequest
    ) -> tuple[str, int, float]:
        """从请求里取参数；客户端没写的字段回落到服务默认值。

        用 `model_fields_set` 区分"没传"和"传了默认值"，这样 `/mode/change`
        改过的默认参数对后续请求真正生效，而请求本身不会改动任何全局状态。
        """
        explicit = getattr(request, "model_fields_set", set()) or set()
        return self._normalize_search_params(
            mode=request.mode if "mode" in explicit else None,
            similarity_top_k=(
                request.similarity_top_k
                if "similarity_top_k" in explicit
                else None
            ),
            vector_weight=(
                request.vector_weight if "vector_weight" in explicit else None
            ),
        )

    def set_default_search_params(
        self,
        mode: str | None = None,
        similarity_top_k: int | None = None,
        vector_weight: float | None = None,
    ) -> tuple[str, int, float]:
        """显式修改服务级默认检索参数（`/mode/change` 的实现）。"""
        if mode is not None:
            mode = mode.upper()
            if mode not in VALID_MODES:
                raise ValueError(
                    f"无效的检索模式: {mode}，可选: {', '.join(VALID_MODES)}"
                )
            self.default_mode = mode
        if similarity_top_k is not None:
            top_k = int(similarity_top_k)
            if not 1 <= top_k <= 20:
                raise ValueError("similarity_top_k 必须在 1~20 之间")
            self.default_top_k = top_k
        if vector_weight is not None:
            self.default_vector_weight = max(0.0, min(1.0, float(vector_weight)))
        logger.info(
            "服务默认检索参数已更新: mode=%s top_k=%d vector_weight=%.2f",
            self.default_mode,
            self.default_top_k,
            self.default_vector_weight,
        )
        return self.default_mode, self.default_top_k, self.default_vector_weight

    def _build_hybrid_retriever(
        self,
        mode: str,
        similarity_top_k: int,
        vector_weight: float,
    ) -> HybridRetriever:
        """按 (mode, top_k, vector_weight) 现场组装检索器。

        只有轻量包装对象是每次新建的（向量/BM25 索引本身常驻内存），
        这样每个请求都用自己那份检索器，并发请求之间不会互相污染。
        """
        vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        keyword_retriever = self.keyword_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        return HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            mode=mode,
            vector_weight=vector_weight,
            similarity_top_k=similarity_top_k,
        )

    def _build_node_postprocessors(self) -> List[Any]:
        postprocessors: List[Any] = [
            SimilarityPostprocessor(similarity_cutoff=self.config.similarity_cutoff)
        ]
        if self.config.enable_context_reorder:
            postprocessors.append(LongContextReorder())
        if self.config.enable_metadata_replacement:
            postprocessors.append(
                MetadataReplacementPostProcessor(target_metadata_key="window")
            )
        return postprocessors

    def build_query_engine(
        self,
        mode: str | None = None,
        similarity_top_k: int | None = None,
        vector_weight: float | None = None,
        streaming: bool = False,
    ) -> RetrieverQueryEngine:
        """为单次请求组装查询引擎（无副作用，可并发调用）。

        索引对象（向量索引 + BM25 倒排）常驻内存，这里只新建轻量包装
        （检索器 / 后处理器 / 响应合成器），所以每请求新建的成本可以忽略，
        换来的是并发请求的 top_k、mode、权重互不影响。
        """
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        resolved_mode, top_k, weight = self._normalize_search_params(
            mode=mode, similarity_top_k=similarity_top_k, vector_weight=vector_weight
        )
        return RetrieverQueryEngine.from_args(
            retriever=self._build_hybrid_retriever(resolved_mode, top_k, weight),
            llm=self.llm,
            response_mode="compact",
            node_postprocessors=self._build_node_postprocessors(),
            text_qa_template=self.QA_PROMPT,
            verbose=False,
            streaming=streaming,
        )

    async def create_query_engine(
        self,
        hybrid_mode: str | None = None,
        similarity_top_k: int | None = None,
        vector_weight: float | None = None,
    ) -> RetrieverQueryEngine:
        """兼容旧入口：按给定参数（或服务默认值）创建一个查询引擎并返回。

        注意：这里**不会**把引擎存成实例状态，调用方用完即弃。
        """
        mode, top_k, weight = self._normalize_search_params(
            mode=hybrid_mode,
            similarity_top_k=similarity_top_k,
            vector_weight=vector_weight,
        )
        logger.info(f"创建混合查询引擎（模式: {mode}, top_k: {top_k}）")
        return self.build_query_engine(
            mode=mode, similarity_top_k=top_k, vector_weight=weight
        )

    @staticmethod
    def _preview(text: str, limit: int = 150) -> str:
        return text[:limit] + "..." if len(text) > limit else text

    def _build_sources(
        self, nodes: List[Any], verbose: bool = False
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """把检索节点整理成对外返回的 sources 结构，并统计各通道贡献。"""
        sources: List[Dict[str, Any]] = []
        retrieval_stats: Dict[str, int] = {"vector": 0, "keyword": 0, "both": 0}

        for i, node in enumerate(nodes[: self.config.max_sources]):
            metadata = node.node.metadata
            retrieval_type = metadata.get("retrieval_type", "未知")
            if retrieval_type in retrieval_stats:
                retrieval_stats[retrieval_type] += 1

            score = node.score if node.score is not None else 0
            item = {
                "rank": i + 1,
                "file_name": metadata.get("file_name", "未知文件"),
                "file_type": metadata.get("file_type", "未知类型"),
                "page": metadata.get("page", None),
                "retrieval_type": retrieval_type,
                "score": round(score, 4),
                "text_length": len(node.node.text),
                "text_preview": self._preview(node.node.text),
                "full_text": node.node.text if verbose else None,
            }
            for key in ("vector_score", "keyword_score", "fusion_score"):
                if key in metadata:
                    item[key] = round(float(metadata[key]), 4)
            sources.append(item)

        return sources, retrieval_stats

    def _filter_by_cutoff(self, nodes: List[Any]) -> tuple[List[Any], int]:
        """按配置的相似度阈值过滤，返回(保留的节点, 被过滤掉的数量)。"""
        cutoff = self.config.similarity_cutoff
        kept = [
            node for node in nodes
            if node.score is not None and node.score >= cutoff
        ]
        return kept, len(nodes) - len(kept)

    async def query(
        self, request: QueryRequest, verbose: bool | None = None
    ) -> Dict[str, Any]:
        """执行查询。

        引擎按 (mode, top_k, vector_weight) 现场组装，请求之间不共享任何可变状态。
        """
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        mode, top_k, weight = self.params_from_request(request)
        include_full = request.verbose if verbose is None else bool(verbose)

        logger.info(
            f"查询问题: {request.question[:50]}... (模式: {mode}, top_k: {top_k})"
        )

        if request.stream:
            return await self._stream_query(request, verbose=include_full)

        query_started = time.perf_counter()
        try:
            engine = self.build_query_engine(
                mode=mode, similarity_top_k=top_k, vector_weight=weight
            )
            query_bundle = QueryBundle(request.question)

            # 分两段计时：检索（含节点后处理）与 LLM 生成
            retrieval_started = time.perf_counter()
            nodes = await engine.aretrieve(query_bundle)
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

            llm_started = time.perf_counter()
            response = await engine.asynthesize(query_bundle, nodes)
            llm_ms = (time.perf_counter() - llm_started) * 1000

            total_ms = (time.perf_counter() - query_started) * 1000
            source_nodes = getattr(response, "source_nodes", None) or nodes
            sources, retrieval_stats = self._build_sources(source_nodes, include_full)

            result = {
                "question": request.question,
                "answer": str(response),
                "mode": mode,
                "similarity_top_k": top_k,
                "vector_weight": weight,
                "timestamp": datetime.now().isoformat(),
                # 真实响应耗时（秒），历史版本这里填的是服务已运行时长
                "response_time": round(total_ms / 1000, 4),
                "retrieval_ms": round(retrieval_ms, 2),
                "llm_ms": round(llm_ms, 2),
                "total_ms": round(total_ms, 2),
            }

            if sources:
                result["sources"] = sources
                result["retrieval_stats"] = retrieval_stats
                result["sources_limited"] = (
                    f"显示前{len(sources)}个来源（共{len(source_nodes)}个）"
                )

            logger.info(
                "查询完成，返回 %d 个来源，检索 %.1fms / 生成 %.1fms",
                len(sources),
                retrieval_ms,
                llm_ms,
            )
            return result

        except Exception as e:
            total_ms = (time.perf_counter() - query_started) * 1000
            logger.error(f"查询过程中发生错误: {e}")
            return {
                "question": request.question,
                "answer": f"查询过程中发生错误: {str(e)}",
                "error": str(e),
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
                "response_time": round(total_ms / 1000, 4),
                "total_ms": round(total_ms, 2),
            }

    @staticmethod
    def _llm_context(node: Any) -> str:
        from llama_index.core.schema import MetadataMode

        return node.node.get_content(metadata_mode=MetadataMode.LLM)

    def _pack_context(self, nodes: List[Any]) -> str:
        """把检索到的节点拼成上下文，并按字符上限截尾，避免提示词超出模型窗口。"""
        limit = max(1000, int(self.config.stream_context_char_limit))
        parts: List[str] = []
        used = 0
        dropped = 0
        for node in nodes:
            text = self._llm_context(node)
            if used + len(text) > limit:
                dropped += 1
                continue
            parts.append(text)
            used += len(text)
        if dropped:
            logger.warning(
                "上下文超出 %d 字符，已丢弃末尾 %d 个节点", limit, dropped
            )
        return "\n\n".join(parts)

    async def _astream_answer(
        self, question: str, nodes: List[Any]
    ) -> AsyncGenerator[str, None]:
        """直接走 LLM 的异步流式接口逐 token 产出答案。

        为什么要自己拼提示词：llama-index-core 0.14.23 的 refine/compact 合成器在
        `astream_call()` 里会把所有 token 先累加成完整答案再 yield 一次
        （`answer += token` → 单个 yield），所以通过 `synthesize()` 拿到的"流式"
        响应实际只有一个块，首 token 延迟等于整段生成时间。这里直接
        `await llm.astream(prompt)` 才是真正的 token 级流式。
        """
        if not nodes:
            yield "Empty Response"
            return

        context_str = self._pack_context(nodes)
        prompt = self.QA_PROMPT.format(context_str=context_str, query_str=question)

        # `LLM.astream(str)` 在 0.14.23 里要求传模板对象（会读 prompt.kwargs），
        # 字符串提示词要走 astream_chat / astream_complete 这两条明确的路径。
        if getattr(self.llm.metadata, "is_chat_model", False):
            from llama_index.core.llms import ChatMessage, MessageRole

            stream = await self.llm.astream_chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)]
            )
            async for response in stream:
                delta = getattr(response, "delta", None) or ""
                if delta:
                    yield delta
        else:
            stream = await self.llm.astream_complete(prompt)
            async for response in stream:
                delta = getattr(response, "delta", None) or ""
                if delta:
                    yield delta

    async def _stream_query(
        self, request: QueryRequest, verbose: bool = False
    ) -> Dict[str, Any]:
        """真流式查询：NDJSON，每行一个事件。

        事件顺序：
          query_started -> retrieval_completed -> answer_chunk(多次)
          -> answer_chunk(is_complete=true, 带耗时) -> sources -> query_completed
        出错时输出 error 事件。

        检索走引擎（含节点后处理），生成走 `self.llm.astream()`，
        首 token 一到就下发，不再等整段答案生成完。
        """
        mode, top_k, weight = self.params_from_request(request)

        def ndjson(payload: Dict[str, Any]) -> str:
            return json.dumps(payload, ensure_ascii=False) + "\n"

        async def generate():
            started = time.perf_counter()
            yield ndjson(
                {
                    "event": "query_started",
                    "question": request.question,
                    "mode": mode,
                    "similarity_top_k": top_k,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            try:
                engine = self.build_query_engine(
                    mode=mode,
                    similarity_top_k=top_k,
                    vector_weight=weight,
                )
                query_bundle = QueryBundle(request.question)

                retrieval_started = time.perf_counter()
                nodes = await engine.aretrieve(query_bundle)
                retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
                yield ndjson(
                    {
                        "event": "retrieval_completed",
                        "count": len(nodes),
                        "retrieval_ms": round(retrieval_ms, 2),
                    }
                )

                llm_started = time.perf_counter()
                ttft_ms: float | None = None
                answer_chars = 0

                async def emit(chunk: str) -> str:
                    nonlocal ttft_ms, answer_chars
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - llm_started) * 1000
                    answer_chars += len(chunk)
                    return ndjson(
                        {
                            "event": "answer_chunk",
                            "chunk": chunk,
                            "is_complete": False,
                        }
                    )

                streamed = False
                try:
                    async for token in self._astream_answer(request.question, nodes):
                        streamed = True
                        yield await emit(token)
                except Exception as stream_error:
                    logger.warning(
                        "token 级流式生成失败，回退到一次性生成: %s", stream_error
                    )

                if not streamed:
                    # 兜底：LLM 不支持 astream 时，用合成器整段生成
                    response = await engine.asynthesize(query_bundle, nodes)
                    yield await emit(str(response))

                llm_ms = (time.perf_counter() - llm_started) * 1000
                total_ms = (time.perf_counter() - started) * 1000
                yield ndjson(
                    {
                        "event": "answer_chunk",
                        "chunk": "",
                        "is_complete": True,
                        "answer_chars": answer_chars,
                        "response_time": round(total_ms / 1000, 4),
                        "retrieval_ms": round(retrieval_ms, 2),
                        "llm_ms": round(llm_ms, 2),
                        "ttft_ms": round(ttft_ms or 0.0, 2),
                        "total_ms": round(total_ms, 2),
                    }
                )

                sources, stats = self._build_sources(nodes, verbose)
                if sources:
                    yield ndjson(
                        {
                            "event": "sources",
                            "sources": [
                                {
                                    "rank": s["rank"],
                                    "file_name": s["file_name"],
                                    "retrieval_type": s["retrieval_type"],
                                    "score": s["score"],
                                }
                                for s in sources
                            ],
                            "total_sources": len(sources),
                            "retrieval_stats": stats,
                        }
                    )
                yield ndjson(
                    {
                        "event": "query_completed",
                        "timestamp": datetime.now().isoformat(),
                        "total_ms": round(total_ms, 2),
                    }
                )
            except Exception as e:
                logger.error(f"流式查询失败: {e}")
                yield ndjson({"event": "error", "error": str(e)})

        return {"streaming_response": generate(), "media_type": "application/x-ndjson"}

    async def keyword_search(
        self,
        request: KeywordSearchRequest,
        verbose: bool | None = None,
    ) -> Dict[str, Any]:
        """执行纯关键词搜索（BM25）。

        与 `/query` 走同一套截断规则：先按 `similarity_cutoff` 过滤，再按
        `max_sources` 限制条数；`full_text` 只在显式 verbose 时返回，
        否则一次请求就会把整个知识库倒出来。
        """
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        include_full = request.verbose if verbose is None else bool(verbose)
        query_bundle = QueryBundle(query_str=request.keyword)
        explicit = getattr(request, "model_fields_set", set()) or set()
        top_k = request.top_k if "top_k" in explicit else self.default_top_k

        retrieval_started = time.perf_counter()
        retriever = self._build_hybrid_retriever(
            mode="KEYWORD_ONLY",
            similarity_top_k=top_k,
            vector_weight=self.default_vector_weight,
        )
        nodes = await retriever.aretrieve(query_bundle)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        kept, filtered_out = self._filter_by_cutoff(nodes)
        results, _stats = self._build_sources(kept, verbose=include_full)

        return {
            "keyword": request.keyword,
            "results": results,
            "count": len(results),
            "total_matched": len(nodes),
            "filtered_by_cutoff": filtered_out,
            "similarity_cutoff": self.config.similarity_cutoff,
            "max_sources": self.config.max_sources,
            "retrieval_ms": round(retrieval_ms, 2),
            "timestamp": datetime.now().isoformat(),
        }

    async def batch_query(
        self, request: BatchQueryRequest
    ) -> Dict[str, Any]:
        """批量查询：受 Semaphore 并发上限约束，结果顺序与输入一致。"""
        if not self.index_built:
            raise ValueError("索引未构建，请先构建索引")

        started = time.perf_counter()
        mode, top_k, weight = self._normalize_search_params(mode=request.mode)
        concurrency = max(1, self.config.batch_concurrency)
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(question: str) -> Dict[str, Any]:
            async with semaphore:
                single = QueryRequest(
                    question=question,
                    mode=mode,
                    similarity_top_k=top_k,
                    vector_weight=weight,
                )
                try:
                    result = await self.query(single)
                except Exception as e:
                    return {
                        "question": question,
                        "answer": f"查询失败: {str(e)}",
                        "success": False,
                        "error": str(e),
                    }
                payload = {
                    "question": question,
                    "answer": result.get("answer"),
                    "success": "error" not in result,
                }
                if "error" in result:
                    payload["error"] = result["error"]
                return payload

        results = list(
            await asyncio.gather(*(run_one(q) for q in request.questions))
        )

        return {
            "mode": mode,
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "concurrency": concurrency,
            "response_time": round(time.perf_counter() - started, 4),
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
                    "keyword_index_nodes": (
                        self.keyword_index.node_count if self.keyword_index else 0
                    ),
                    "query_engine_ready": self.index_built,
                    "engine_per_request": True,
                    "current_mode": self.default_mode,
                    "default_top_k": self.default_top_k,
                    "default_vector_weight": self.default_vector_weight,
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
        if self.index_built:
            return IndexStatus(
                status="ready",
                message="DSM-5知识库索引已构建完成",
                progress=1.0,
                details={
                    "document_count": self.total_documents,
                    "query_engine_ready": True,
                    "mode": self.default_mode,
                    "keyword_index_nodes": (
                        self.keyword_index.node_count if self.keyword_index else 0
                    ),
                },
            )
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
        """列出所有文档文件信息（先重新扫描目录，避免只反映启动时的快照）。"""
        self._scan_document_files()
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
