"""Pydantic 数据模型单元测试"""

from dsm5_rag.models import (
    QueryRequest,
    KeywordSearchRequest,
    BatchQueryRequest,
    SystemStatus,
    IndexStatus,
)


class TestQueryRequest:
    """测试 QueryRequest 模型"""

    def test_required_fields(self):
        """测试必填字段"""
        req = QueryRequest(question="测试问题")
        assert req.question == "测试问题"
        assert req.mode == "HYBRID"
        assert req.similarity_top_k == 5
        assert req.vector_weight == 0.7
        assert req.stream is False

    def test_custom_values(self):
        """测试自定义值"""
        req = QueryRequest(
            question="测试问题",
            mode="VECTOR_ONLY",
            similarity_top_k=10,
            vector_weight=0.5,
            stream=True,
        )
        assert req.mode == "VECTOR_ONLY"
        assert req.similarity_top_k == 10
        assert req.vector_weight == 0.5
        assert req.stream is True

    def test_field_validation(self):
        """测试字段验证"""
        req = QueryRequest(question="测试")
        assert 1 <= req.similarity_top_k <= 20
        assert 0 <= req.vector_weight <= 1


class TestKeywordSearchRequest:
    """测试 KeywordSearchRequest 模型"""

    def test_required_fields(self):
        """测试必填字段"""
        req = KeywordSearchRequest(keyword="depression")
        assert req.keyword == "depression"
        assert req.top_k == 5

    def test_custom_top_k(self):
        """测试自定义 top_k"""
        req = KeywordSearchRequest(keyword="anxiety", top_k=10)
        assert req.top_k == 10


class TestBatchQueryRequest:
    """测试 BatchQueryRequest 模型"""

    def test_required_fields(self):
        """测试必填字段"""
        questions = ["问题1", "问题2"]
        req = BatchQueryRequest(questions=questions)
        assert req.questions == questions
        assert req.mode == "HYBRID"

    def test_custom_mode(self):
        """测试自定义模式"""
        req = BatchQueryRequest(
            questions=["问题1"], mode="VECTOR_ONLY"
        )
        assert req.mode == "VECTOR_ONLY"


class TestSystemStatus:
    """测试 SystemStatus 模型"""

    def test_required_fields(self):
        """测试必填字段"""
        status = SystemStatus(status="running", index_built=True)
        assert status.status == "running"
        assert status.index_built is True
        assert status.documents_path is None

    def test_optional_fields(self):
        """测试可选字段"""
        status = SystemStatus(
            status="running",
            index_built=True,
            documents_path="/docs",
            document_count=5,
            embedding_model="test-model",
            uptime=123.4,
        )
        assert status.documents_path == "/docs"
        assert status.document_count == 5
        assert status.embedding_model == "test-model"
        assert status.uptime == 123.4


class TestIndexStatus:
    """测试 IndexStatus 模型"""

    def test_ready_status(self):
        """测试已就绪状态"""
        status = IndexStatus(status="ready", message="索引已构建完成")
        assert status.status == "ready"
        assert status.progress is None

    def test_building_status(self):
        """测试构建中状态"""
        status = IndexStatus(
            status="building",
            message="正在构建",
            progress=0.5,
            details={"files_processed": 5},
        )
        assert status.status == "building"
        assert status.progress == 0.5
        assert status.details["files_processed"] == 5