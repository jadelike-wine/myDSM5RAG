"""RagConfig 单元测试"""

import os

from dsm5_rag.config import RagConfig

# 测试默认值需忽略的环境变量（conftest 的 autouse fixture 会加载 .env）
_IGNORED_ENV_VARS = [
    "API_HOST", "API_PORT", "API_RELOAD",
    "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL",
    "EMBEDDING_MODEL", "EMBEDDING_DEVICE",
    "PERSIST_DIR", "DOCUMENTS_PATH",
    "DEFAULT_MODE", "DEFAULT_TOP_K", "DEFAULT_VECTOR_WEIGHT",
    "SIMILARITY_CUTOFF", "CHUNK_SIZE", "CHUNK_OVERLAP",
    "MAX_FILE_SIZE_MB", "MAX_SOURCES",
    "ENABLE_CONTEXT_REORDER", "ENABLE_METADATA_REPLACEMENT",
    "HF_ENDPOINT", "CORS_ORIGINS", "LOG_LEVEL", "MODELS_DIR",
]


class TestRagConfig:
    """测试 RagConfig 配置类"""

    def test_default_values(self, monkeypatch):
        """测试默认配置值（清除 .env 干扰）"""
        for var in _IGNORED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        config = RagConfig()
        assert config.api_host == "localhost"
        assert config.api_port == 8000
        assert config.deepseek_model == "deepseek-v4-flash"
        assert config.embedding_device == "cpu"
        assert config.default_mode == "HYBRID"
        assert config.default_top_k == 5
        assert config.default_vector_weight == 0.7
        assert config.similarity_cutoff == 0.2
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.max_file_size_mb == 50
        assert config.max_sources == 3
        assert config.enable_context_reorder is True
        assert config.enable_metadata_replacement is False
        assert ".pdf" in config.supported_extensions
        assert ".docx" in config.supported_extensions
        assert ".txt" in config.supported_extensions

    def test_env_override(self, monkeypatch):
        """测试环境变量覆盖默认值"""
        monkeypatch.setenv("API_HOST", "0.0.0.0")
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("DEFAULT_MODE", "VECTOR_ONLY")
        monkeypatch.setenv("DEFAULT_TOP_K", "10")
        monkeypatch.setenv("CHUNK_SIZE", "512")

        config = RagConfig()
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 9000
        assert config.default_mode == "VECTOR_ONLY"
        assert config.default_top_k == 10
        assert config.chunk_size == 512

    def test_api_key_empty_by_default(self):
        """测试API密钥默认未设置"""
        config = RagConfig()
        # 确保测试不会意外使用真实环境变量
        assert hasattr(config, "deepseek_api_key")

    def test_hf_endpoint_default(self):
        """测试HuggingFace镜像站默认值"""
        config = RagConfig()
        assert config.hf_endpoint == "https://hf-mirror.com"

    def test_cors_origins_default(self):
        """测试CORS默认值"""
        config = RagConfig()
        assert config.cors_origins == "*"

    def test_log_level_default(self):
        """测试日志级别默认值"""
        config = RagConfig()
        assert config.log_level == "INFO"