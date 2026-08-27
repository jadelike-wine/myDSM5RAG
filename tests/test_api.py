"""API 端点单元测试（使用 TestClient）"""

import pytest
from fastapi.testclient import TestClient

# 在导入 app 之前，设置测试环境变量
import os

os.environ["DEEPSEEK_API_KEY"] = "test-key"
os.environ["API_PORT"] = "8000"

from dsm5_rag.api import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestAPIEndpoints:
    """测试 API 端点"""

    def test_health_endpoint(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_root_endpoint(self, client):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "DSM-5 RAG API"
        assert "endpoints" in data

    def test_status_endpoint(self, client):
        """测试状态端点"""
        response = client.get("/status")
        # 可能返回 503 如果系统未初始化，但不应是 500
        assert response.status_code in (200, 503)

    def test_index_status_endpoint(self, client):
        """测试索引状态端点"""
        response = client.get("/index/status")
        assert response.status_code in (200, 503)

    def test_documents_endpoint(self, client):
        """测试文档列表端点"""
        response = client.get("/documents")
        assert response.status_code in (200, 503)

    def test_swagger_docs_available(self, client):
        """测试 Swagger 文档可访问"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_schema(self, client):
        """测试 OpenAPI 模式"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "DSM-5 RAG API"
        assert "paths" in schema

    def test_redoc_available(self, client):
        """测试 ReDoc 文档可访问"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_query_without_index(self, client):
        """测试索引未就绪时查询"""
        request_data = {
            "question": "测试问题",
            "mode": "HYBRID",
            "similarity_top_k": 5,
            "vector_weight": 0.7,
            "stream": False,
        }
        response = client.post("/query", json=request_data)
        # 索引未就绪时返回 503 或 425
        assert response.status_code in (400, 425, 503)

    def test_keyword_search_without_index(self, client):
        """测试索引未就绪时关键词搜索"""
        response = client.post(
            "/search/keyword",
            json={"keyword": "depression", "top_k": 5},
        )
        assert response.status_code in (400, 503)

    def test_batch_query_without_index(self, client):
        """测试索引未就绪时批量查询"""
        response = client.post(
            "/query/batch",
            json={"questions": ["问题1", "问题2"], "mode": "HYBRID"},
        )
        assert response.status_code in (400, 503)

    def test_invalid_mode_change(self, client):
        """测试无效的检索模式"""
        response = client.post("/mode/change?mode=INVALID")
        assert response.status_code in (400, 503)

    def test_cors_headers(self, client):
        """测试 CORS 头部"""
        response = client.options(
            "/",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers