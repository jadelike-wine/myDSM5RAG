"""pytest 共享 fixtures。

关键点：测试必须跑在隔离的临时目录里，并且绝不初始化真实 embedding 模型。
`DeepSeekRAGSystem` 一旦按默认配置构造，就会去 HuggingFace 下载几百 MB 权重并在
真实 `chroma_dsm5_db/` 上建索引 —— 所以下面的 autouse fixture 会把
DOCUMENTS_PATH / PERSIST_DIR / MODELS_DIR 全部指向临时目录，并清空 API Key
（无 Key 时系统自动用 MockLLM）。
"""

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from dsm5_rag.config import RagConfig
from tests.lexical_embedding import install_fake_huggingface_embedding

_ISOLATED_ENV = {
    "DEEPSEEK_API_KEY": "",
    "LOG_LEVEL": "WARNING",
    "ENABLE_METADATA_REPLACEMENT": "false",
    "MAX_SOURCES": "3",
    "SIMILARITY_CUTOFF": "0.2",
    "DEFAULT_MODE": "HYBRID",
    "DEFAULT_TOP_K": "5",
    "BATCH_CONCURRENCY": "2",
}


@pytest.fixture(autouse=True)
def setup_env():
    """自动加载 .env 文件（不覆盖下面隔离出来的测试变量）。"""
    load_dotenv(override=False)
    yield


@pytest.fixture(scope="session", autouse=True)
def isolate_storage_env(tmp_path_factory):
    """把存储路径隔离到临时目录，并装上测试用 embedding 替身。"""
    root = tmp_path_factory.mktemp("dsm5-env")
    documents = root / "documents"
    documents.mkdir()

    previous = {
        key: os.environ.get(key)
        for key in [
            "DOCUMENTS_PATH",
            "PERSIST_DIR",
            "MODELS_DIR",
            *_ISOLATED_ENV,
        ]
    }
    os.environ["DOCUMENTS_PATH"] = str(documents)
    os.environ["PERSIST_DIR"] = str(root / "persist")
    os.environ["MODELS_DIR"] = str(root / "models")
    for key, value in _ISOLATED_ENV.items():
        os.environ[key] = value

    install_fake_huggingface_embedding(embed_dim=128)

    yield {"root": root, "documents": documents}

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """创建测试配置"""
    return RagConfig()


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    """真实 DSM-5 小语料目录（5 个 txt）。"""
    return Path(__file__).parent / "fixtures" / "corpus"
