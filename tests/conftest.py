"""pytest 共享 fixtures"""

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from dsm5_rag.config import RagConfig


@pytest.fixture(autouse=True)
def setup_env():
    """自动加载 .env 文件"""
    load_dotenv()
    yield


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """创建测试配置"""
    return RagConfig()


