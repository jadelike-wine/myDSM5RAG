"""API 端点契约测试。

历史问题：本文件以前用 `TestClient(app)`（没有 `with`），lifespan 根本不执行，
`rag_system` 永远是 None，所有端点都走 503 分支，而断言写成
`assert status_code in (200, 503)` 于是恒真 —— 对检索/端点逻辑的实际覆盖为 0。

现在的做法：
  - 用 `with TestClient(app)` 真正跑 lifespan，并把 lifespan 里构造的
    `DeepSeekRAGSystem` 换成 `FakeRAGSystem`（否则测试会去下载几百 MB 的
    embedding 模型并碰真实索引目录）；
  - 正面断言 200/400/422/425/503 各分支，以及 NDJSON 流式的行格式与事件顺序；
  - "未初始化 → 503" 单独用不进 with 的客户端来测，不再和正常路径混在一个
    "in (200, 503)" 里。
"""

import json

import pytest
from fastapi.testclient import TestClient

from dsm5_rag import api
from tests.fake_system import FakeRAGSystem


@pytest.fixture
def fake() -> FakeRAGSystem:
    return FakeRAGSystem()


@pytest.fixture
def client(fake, monkeypatch):
    """执行完整 lifespan 的客户端，lifespan 内部构造的就是 fake。"""
    monkeypatch.setattr(api, "DeepSeekRAGSystem", lambda *args, **kwargs: fake)
    with TestClient(api.app) as test_client:
        assert api.rag_system is fake, "lifespan 没有执行，rag_system 未被初始化"
        yield test_client


@pytest.fixture
def uninit_client(monkeypatch):
    """rag_system 为 None 的客户端（不进 with，因此不跑 lifespan）。"""
    monkeypatch.setattr(api, "rag_system", None)
    return TestClient(api.app)


def _query_payload(**overrides):
    payload = {
        "question": "What are the diagnostic criteria for Major Depressive Disorder?",
        "mode": "HYBRID",
        "similarity_top_k": 5,
        "vector_weight": 0.7,
        "stream": False,
    }
    payload.update(overrides)
    return payload


class TestLifecycle:
    def test_lifespan_initializes_system(self, client, fake):
        assert api.rag_system is fake
        assert fake.initialized is True

    def test_without_lifespan_system_is_none(self, uninit_client):
        """对照组：不用 with 时 lifespan 确实不执行 —— 这就是旧测试失效的原因。"""
        assert api.rag_system is None
        assert uninit_client.get("/status").status_code == 503


class TestHealthAndStatus:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_while_uninitialized(self, uninit_client):
        response = uninit_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "initializing"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "DSM-5 RAG API"
        assert data["index_status"] == "ready"
        assert data["document_count"] == 1
        # 新增端点必须出现在自描述里
        assert data["endpoints"]["index_rebuild"] == "/index/rebuild"
        assert data["endpoints"]["index_diff"] == "/index/diff"

    def test_status_endpoint(self, client):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["index_built"] is True
        assert data["index_stats"]["keyword_index_type"] == "BM25KeywordIndex"
        assert data["supported_formats"] == [".pdf", ".docx", ".txt"]

    @pytest.mark.parametrize(
        "path", ["/status", "/index/status", "/documents", "/index/diff"]
    )
    def test_status_endpoints_503_when_uninitialized(self, uninit_client, path):
        response = uninit_client.get(path)
        assert response.status_code == 503
        assert response.json()["status_code"] == 503

    def test_index_status_endpoint(self, client):
        response = client.get("/index/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["progress"] == 1.0

    def test_documents_endpoint(self, client):
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json()["total_documents"] == 1

    def test_swagger_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "DSM-5 RAG API"
        assert "/index/rebuild" in schema["paths"]
        assert "/index/diff" in schema["paths"]

    def test_redoc_available(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_cors_headers(self, client):
        response = client.options(
            "/",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestQueryEndpoint:
    def test_query_returns_full_contract(self, client):
        response = client.post("/query", json=_query_payload())
        assert response.status_code == 200
        data = response.json()
        for key in (
            "question",
            "answer",
            "mode",
            "timestamp",
            "response_time",
            "retrieval_ms",
            "llm_ms",
            "sources",
            "retrieval_stats",
        ):
            assert key in data, f"响应缺少字段 {key}"
        # response_time 必须是真实秒数，而不是服务已运行时长
        assert 0 < data["response_time"] < 60
        assert data["retrieval_ms"] > 0 and data["llm_ms"] > 0
        assert data["sources"][0]["retrieval_type"] in (
            "vector",
            "keyword",
            "both",
        )
        assert data["sources"][0]["score"] is not None

    def test_query_omits_full_text_by_default(self, client):
        data = client.post("/query", json=_query_payload()).json()
        assert all(source["full_text"] is None for source in data["sources"])

    def test_query_verbose_returns_full_text(self, client):
        data = client.post("/query", json=_query_payload(verbose=True)).json()
        assert any(source["full_text"] for source in data["sources"])

    def test_query_503_when_index_not_built(self, fake, monkeypatch):
        monkeypatch.setattr(api, "DeepSeekRAGSystem", lambda *a, **k: fake)
        fake.index_built = False
        with TestClient(api.app) as client:
            response = client.post("/query", json=_query_payload())
        assert response.status_code == 503

    def test_query_425_while_building(self, fake, monkeypatch):
        monkeypatch.setattr(api, "DeepSeekRAGSystem", lambda *a, **k: fake)
        fake.index_built = False
        fake.index_building = True
        with TestClient(api.app) as client:
            response = client.post("/query", json=_query_payload())
        assert response.status_code == 425

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": ""},  # 空问题
            {"question": "x" * 4001},  # 超长问题
            {"question": "问题", "similarity_top_k": 0},  # top_k 越界
            {"question": "问题", "similarity_top_k": 21},
            {"question": "问题", "vector_weight": 1.5},
            {},  # 缺 question
        ],
    )
    def test_query_validation_422(self, client, payload):
        assert client.post("/query", json=payload).status_code == 422

    def test_query_500_on_internal_error(self, client, fake):
        fake.query_error = "模拟 LLM 失败"
        response = client.post("/query", json=_query_payload())
        assert response.status_code == 500
        # 注意：当前实现会把内部异常原文放进 detail（评审第 21 条的安全项，
        # 属于本次未纳入的改动范围），所以这里只断言状态码与结构。
        assert response.json()["status_code"] == 500


class TestStreamingContract:
    def _lines(self, client, payload):
        with client.stream("POST", "/query", json=payload) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(
                "application/x-ndjson"
            )
            raw = response.read().decode("utf-8")
        return raw

    def test_ndjson_one_json_object_per_line(self, client):
        raw = self._lines(client, _query_payload(stream=True))
        assert raw.endswith("\n")
        lines = [line for line in raw.split("\n") if line]
        events = []
        for line in lines:
            assert line.lstrip().startswith("{"), f"不是 JSON 行: {line[:40]}"
            payload = json.loads(line)
            assert "event" in payload
            events.append(payload)

        names = [e["event"] for e in events]
        assert names[0] == "query_started"
        assert names[1] == "retrieval_completed"
        assert names[-1] == "query_completed"
        assert "sources" in names

    def test_answer_is_delivered_in_multiple_chunks(self, client, fake):
        raw = self._lines(client, _query_payload(stream=True))
        chunks = [
            json.loads(line)
            for line in raw.split("\n")
            if line.strip() and json.loads(line)["event"] == "answer_chunk"
        ]
        content_chunks = [c for c in chunks if c["chunk"]]
        assert len(content_chunks) >= 2, "流式必须分多块下发，而不是整段一次给完"
        assert sum(1 for c in chunks if c.get("is_complete")) == 1
        final = next(c for c in chunks if c.get("is_complete"))
        assert final["ttft_ms"] <= final["total_ms"]
        answer = "".join(c["chunk"] for c in content_chunks)
        assert answer == "".join(fake.stream_chunks)

    def test_stream_text_format(self, client):
        raw = self._lines(client, _query_payload(stream=True))
        text = client.post("/query?format=text", json=_query_payload(stream=True)).text
        assert "查询开始" in text
        assert "回答:" in text
        assert raw.count("\n") >= 6


class TestKeywordSearchEndpoint:
    def test_contract(self, client):
        response = client.post("/search/keyword", json={"keyword": "296.3x", "top_k": 5})
        assert response.status_code == 200
        data = response.json()
        assert data["keyword"] == "296.3x"
        assert data["count"] == len(data["results"])
        assert data["max_sources"] >= data["count"]
        assert data["similarity_cutoff"] == pytest.approx(0.2)
        assert all(r["full_text"] is None for r in data["results"])

    def test_verbose_returns_full_text(self, client):
        data = client.post(
            "/search/keyword", json={"keyword": "296.3x", "top_k": 5, "verbose": True}
        ).json()
        assert any(r["full_text"] for r in data["results"])

    def test_503_when_index_missing(self, uninit_client):
        assert uninit_client.post(
            "/search/keyword", json={"keyword": "x"}
        ).status_code == 503

    def test_422_on_empty_keyword(self, client):
        assert client.post("/search/keyword", json={"keyword": ""}).status_code == 422

    def test_422_on_top_k_out_of_range(self, client):
        assert client.post(
            "/search/keyword", json={"keyword": "x", "top_k": 99}
        ).status_code == 422


class TestBatchQueryEndpoint:
    def test_contract(self, client):
        response = client.post(
            "/query/batch", json={"questions": ["问题1", "问题2"], "mode": "HYBRID"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2 and data["successful"] == 2
        assert data["concurrency"] >= 1

    def test_rejects_too_many_questions(self, client):
        payload = {"questions": [f"问题{i}" for i in range(21)], "mode": "HYBRID"}
        assert client.post("/query/batch", json=payload).status_code == 422

    def test_rejects_empty_list(self, client):
        assert client.post("/query/batch", json={"questions": []}).status_code == 422

    def test_rejects_overlong_question(self, client):
        payload = {"questions": ["x" * 4001]}
        assert client.post("/query/batch", json=payload).status_code == 422

    def test_503_when_uninitialized(self, uninit_client):
        assert uninit_client.post(
            "/query/batch", json={"questions": ["a"]}
        ).status_code == 503


class TestModeAndIndexEndpoints:
    def test_mode_change_sets_defaults(self, client, fake):
        response = client.post("/mode/change?mode=VECTOR_ONLY&similarity_top_k=8")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "VECTOR_ONLY"
        assert data["similarity_top_k"] == 8
        assert fake.default_mode == "VECTOR_ONLY"
        assert fake.default_top_k == 8

    def test_invalid_mode_is_400(self, client):
        assert client.post("/mode/change?mode=INVALID").status_code == 400

    def test_mode_change_503_when_uninitialized(self, uninit_client):
        assert uninit_client.post("/mode/change?mode=HYBRID").status_code == 503

    def test_rebuild_sync_returns_result(self, client, fake):
        response = client.post("/index/rebuild?force=true&background=false")
        assert response.status_code == 200
        assert response.json()["status"] == "built"
        assert fake.rebuild_calls == [True]

    def test_rebuild_background_is_accepted(self, client):
        response = client.post("/index/rebuild?background=true")
        assert response.status_code == 202
        assert response.json()["status"] == "started"

    def test_rebuild_503_when_uninitialized(self, uninit_client):
        assert uninit_client.post("/index/rebuild").status_code == 503

    def test_index_diff(self, client):
        response = client.get("/index/diff")
        assert response.status_code == 200
        data = response.json()
        assert "up_to_date" in data
        assert "changed_since_index" in data
