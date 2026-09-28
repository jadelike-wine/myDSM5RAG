"""RAG 系统配置模块"""

import os
from pathlib import Path

current_dir = Path(__file__).parent.parent.parent


class RagConfig:
    """RAG系统配置模型 - DSM-5知识库版本"""

    def __init__(self):
        # =============== API服务器配置 ===============
        self.api_host = os.getenv("API_HOST", "localhost")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_reload = os.getenv("API_RELOAD", "false").lower() == "true"

        # =============== DeepSeek API配置 ===============
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url = os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

        # =============== Embedding配置 ===============
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")

        # =============== 存储配置 ===============
        self.persist_dir = os.getenv("PERSIST_DIR", "./chroma_dsm5_db")

        # =============== 文档文件配置 ===============
        self.documents_path = os.getenv("DOCUMENTS_PATH", "./dsm5_documents")
        self.supported_extensions = {".pdf", ".docx", ".txt"}

        # =============== 检索配置 ===============
        self.default_mode = os.getenv("DEFAULT_MODE", "HYBRID")
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", "5"))
        self.default_vector_weight = float(os.getenv("DEFAULT_VECTOR_WEIGHT", "0.7"))

        # =============== BM25 关键词索引配置 ===============
        # 关键词通道的停用词语言（bm25s 内置 en/de/nl/fr/es/pt/it/ru/sv/no/zh/tr/ko/da）
        self.bm25_language = os.getenv("BM25_LANGUAGE", "en")
        # 关闭词干化可让 "disorders"/"disorder" 严格区分，默认开启词干化
        self.bm25_skip_stemming = (
            os.getenv("BM25_SKIP_STEMMING", "false").lower() == "true"
        )

        # =============== HuggingFace镜像站配置 ===============
        self.hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

        # =============== CORS配置 ===============
        self.cors_origins = os.getenv("CORS_ORIGINS", "*")

        # =============== 日志配置 ===============
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # =============== 模型目录配置 ===============
        self.models_dir = os.getenv("MODELS_DIR", "./models")

        # =============== 优化配置 ===============
        self.similarity_cutoff = float(os.getenv("SIMILARITY_CUTOFF", "0.2"))
        self.enable_context_reorder = (
            os.getenv("ENABLE_CONTEXT_REORDER", "true").lower() == "true"
        )
        self.enable_metadata_replacement = (
            os.getenv("ENABLE_METADATA_REPLACEMENT", "false").lower() == "true"
        )
        self.max_sources = int(os.getenv("MAX_SOURCES", "3"))
        # 流式回答时拼进提示词的上下文长度上限（字符数，超出后从尾部丢弃节点）
        self.stream_context_char_limit = int(
            os.getenv("STREAM_CONTEXT_CHAR_LIMIT", "24000")
        )
        # 批量查询的单请求并发上限（asyncio.Semaphore）
        self.batch_concurrency = int(os.getenv("BATCH_CONCURRENCY", "4"))

        # =============== 文档解析配置 ===============
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
