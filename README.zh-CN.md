# DSM-5 RAG 诊断标准知识问答系统

[English](README.md) | **简体中文**

基于混合检索（向量检索 + 关键词检索）的 DSM-5 诊断标准知识问答系统，支持中英文多语言查询。

系统提供**两种服务入口**：

- **FastAPI 微服务** (`dsm5-rag-api`) — 完整的 RESTful API，集成 DeepSeek LLM 生成专业回答
- **FastMCP 服务** (`dsm5-rag-mcp`) — MCP 协议纯检索服务，剥离外部 LLM 依赖，适合 IDE 集成

## 特性

- **混合检索** — 向量语义检索（MiniLM 多语言 + Chroma）配合真正的 **BM25 关键词通道**（`llama-index-retrievers-bm25`），支持 `HYBRID` / `VECTOR_ONLY` / `KEYWORD_ONLY` / `FUSION` 四种模式
- **免尺度混淆的融合** — `HYBRID` 用名次融合（Reciprocal Rank Fusion，复用 core 的 `QueryFusionRetriever`），`FUSION` 先对各通道 min-max 归一再按 `vector_weight` 加权；关键词分数统一归一到 0~1，因此相似度阈值、MCP 展示、`retrieval_stats` 在两路上的语义一致
- **术语精确匹配** — DSM-5 里大量场景依赖原词命中（`296.3x`、`Major Depressive Disorder`、`快感缺失`），这正是 BM25 通道擅长而纯向量容易糊掉的部分
- **多语言支持** — 使用 `paraphrase-multilingual-MiniLM-L12-v2` 多语言嵌入模型，支持中英文查询
- **DeepSeek LLM** — 集成 DeepSeek API，基于检索结果生成专业回答（FastAPI 服务）
- **纯检索 MCP** — FastMCP 服务剥离 LLM，仅暴露检索能力，适合 IDE 和工具链集成
- **真正的 token 级流式** — `stream=true` 按 token 下发 NDJSON，并单独上报 `ttft_ms`（首 token 耗时）
- **按请求组装引擎** — `mode` / `similarity_top_k` / `vector_weight` 每次请求现场组装，并发请求互不污染；`/mode/change` 只改服务级默认值
- **可信耗时** — 响应里给出实测的 `retrieval_ms` / `llm_ms` / `total_ms`
- **带变更检测的自动索引管理** — `index_manifest.json` 记录每个文档的 sha256/mtime：语料没变就直接加载（不重复 embedding），有新增/修改才重建；提供 `POST /index/rebuild` 与 `GET /index/diff`
- **RESTful API** — 基于 FastAPI 构建，提供完整的 REST API 接口
- **文档格式支持** — 支持 PDF、DOCX、TXT 格式文档解析
- **ChromaDB 持久化** — 向量索引与 BM25 关键词索引均持久化，重启后无需重新构建

## 快速开始

### 前置条件

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) 包管理器（推荐）或 pip

### 安装

```bash
# 克隆项目
git clone <your-repo-url> && cd myDSM5RAG

# 使用 uv 安装（推荐）
uv sync --group dev

# 或使用 pip
pip install -e .
```

### 配置

复制环境变量模板并编辑：

```bash
cp .example.env .env
```

在 `.env` 中配置 DeepSeek API 密钥（从 [platform.deepseek.com](https://platform.deepseek.com/) 获取）：

```env
DEEPSEEK_API_KEY="sk-your-api-key-here"
```

> **注意**：如不设置 API 密钥，FastAPI 服务将使用 MockLLM 模拟模式运行，MCP 服务不受影响（纯检索无需 LLM）。

将 DSM-5 诊断标准文档（PDF/DOCX/TXT）放入 `dsm5_documents/` 目录。

### 运行

#### FastAPI 服务（含 LLM 回答生成）

```bash
# 使用 uv
uv run dsm5-rag-api

# 或直接使用 uvicorn
uv run uvicorn dsm5_rag.api:app --host 127.0.0.1 --port 8031
```

服务启动后访问：
- API 文档（Swagger UI）：http://127.0.0.1:8031/docs
- ReDoc：http://127.0.0.1:8031/redoc

#### MCP 服务（纯检索，无 LLM）

```bash
# stdio 模式（IDE 集成，如 Claude Desktop、Cursor）
uv run dsm5-rag-mcp

# Streamable HTTP 模式（远程/生产）
uv run python -c "from dsm5_rag.mcp_server import main_http; main_http()"

# 或使用 fastmcp dev 模式调试
uv run fastmcp dev src/dsm5_rag/mcp_server.py
```

#### 运行示例

```bash
# 基本查询（使用 Python SDK）
uv run python examples/basic_query.py

# 批量查询 & 关键词搜索
uv run python examples/batch_query.py

# HTTP API 客户端（需要先启动 API 服务）
uv run python examples/api_client.py
```

## 配置项

所有配置通过环境变量设置（参考 `.env` 文件）：

### API 与服务器

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_HOST` | `localhost` | 服务监听地址 |
| `API_PORT` | `8000` | 服务端口 |
| `API_RELOAD` | `false` | 是否启用热重载 |
| `CORS_ORIGINS` | `*` | 跨域允许的源 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### DeepSeek LLM

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | `""` | API 密钥（不设置则使用 MockLLM） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 基础地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 模型名称 |

### 嵌入模型

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 嵌入模型名称 |
| `EMBEDDING_DEVICE` | `cpu` | 运行设备（cpu / cuda） |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像站 |
| `MODELS_DIR` | `./models` | 本地模型缓存目录 |

### 存储与文档

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PERSIST_DIR` | `./chroma_dsm5_db` | ChromaDB 持久化目录 |
| `DOCUMENTS_PATH` | `./dsm5_documents` | 文档文件路径（支持目录或单文件） |

### 检索

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_MODE` | `HYBRID` | 默认检索模式 |
| `DEFAULT_TOP_K` | `5` | 默认检索数量 |
| `DEFAULT_VECTOR_WEIGHT` | `0.7` | 默认向量权重（FUSION 模式） |
| `SIMILARITY_CUTOFF` | `0.2` | 相似度过滤阈值（两路通用，分数已统一到 0~1） |
| `MAX_SOURCES` | `3` | `/query` 与 `/search/keyword` 最多返回的来源条数 |
| `ENABLE_CONTEXT_REORDER` | `true` | 是否启用上下文重排序（Lost-in-the-Middle 优化） |
| `ENABLE_METADATA_REPLACEMENT` | `false` | 是否启用元数据替换（依赖 sentence-window 切分，配合当前分层切分请保持 false） |
| `BATCH_CONCURRENCY` | `4` | `POST /query/batch` 的单请求并发上限（`asyncio.Semaphore`） |
| `STREAM_CONTEXT_CHAR_LIMIT` | `24000` | 流式回答拼进提示词的上下文长度上限（字符） |

### BM25 关键词索引

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BM25_LANGUAGE` | `en` | 关键词通道停用词语言（`en/de/nl/fr/es/pt/it/ru/sv/no/zh/tr/ko/da`） |
| `BM25_SKIP_STEMMING` | `false` | 关闭词干化（`disorder` 与 `disorders` 严格区分） |

### 文档解析

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | `1024` | 文档分块大小（句子切分器按 token 计） |
| `CHUNK_OVERLAP` | `100` | 分块重叠大小 |
| `MAX_FILE_SIZE_MB` | `50` | 最大文件大小（MB） |

> **关于 `API_PORT`**：代码默认值是 `8000`，而 `.example.env` 里示例值是 `8031`；两个服务同时跑时请显式区分端口。
>
> **关于 Chroma**：`dsm5-rag-api` 与 `dsm5-rag-mcp` 打开的是同一个 `PERSIST_DIR`，请勿同时启动（sqlite 存在锁冲突），或给每个服务配置独立的 `PERSIST_DIR`。

## 项目结构

```
myDSM5RAG/
├── src/dsm5_rag/                # 核心包
│   ├── __init__.py              # 包入口，公开 API 导出
│   ├── api.py                   # FastAPI 微服务（含 LLM）
│   ├── background.py            # 线程池 / run_in_executor（把同步重活挪出事件循环）
│   ├── config.py                # 配置模型（RagConfig）
│   ├── index_manifest.py        # 每个文档的 sha256/mtime 清单与重建判定
│   ├── keyword_index.py         # BM25 关键词索引（构建/持久化/加载/旧索引迁移 + 分数归一）
│   ├── mcp_server.py            # FastMCP 纯检索服务（无 LLM）
│   ├── models.py                # Pydantic 数据模型
│   ├── parser.py                # 文档解析器（PDF/DOCX/TXT）
│   ├── retriever.py             # HybridRetriever（基于 core QueryFusionRetriever 的 RRF / 加权融合）
│   └── system.py                # RAG 系统核心（DeepSeekRAGSystem）
├── scripts/
│   └── repro_keyword_channel.py # 关键词通道诊断脚本（改动前后对比用）
├── tests/                       # 测试套件
│   ├── conftest.py              # fixtures：临时目录隔离 + 测试用嵌入替身（不下载权重）
│   ├── fake_system.py           # 端点契约测试用的假 RAG 系统
│   ├── lexical_embedding.py     # 确定性词袋哈希向量（测试专用）
│   ├── fixtures/corpus/         # 5 个真实 DSM-5 片段 txt
│   ├── test_api.py              # 端点契约测试（含 NDJSON 流式行格式）
│   ├── test_config.py           # 配置测试
│   ├── test_index_manifest.py   # 索引清单/变更检测测试
│   ├── test_keyword_index.py    # 真实语料的 BM25 召回测试
│   ├── test_mcp_server.py       # MCP 注册测试 + 真实工具调用测试
│   ├── test_models.py           # 数据模型测试
│   ├── test_parser.py           # 文档解析测试
│   ├── test_retriever.py        # 融合语义测试（尺度无关、权重、top_k）
│   ├── test_streaming.py        # 真流式 + 事件循环不被堵住
│   └── test_system_integration.py # 用样例语料跑通的端到端检索测试
├── examples/                    # 使用示例
│   ├── basic_query.py           # 基本查询
│   ├── batch_query.py           # 批量查询 & 关键词搜索
│   └── api_client.py            # HTTP API 客户端
├── dsm5_documents/              # 文档目录（已 gitignore）
├── chroma_dsm5_db/              # ChromaDB 数据 + index_manifest.json（已 gitignore）
├── models/                      # 本地模型缓存（已 gitignore）
├── .github/workflows/ci.yml     # CI：uv sync → ruff check → pytest --cov
├── .env                         # 环境变量配置（已 gitignore）
├── .example.env                 # 环境变量模板
├── pyproject.toml               # 项目配置、依赖、ruff/pytest 设置
└── uv.lock                      # uv 锁文件
```

## FastAPI 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| GET | `/status` | 系统状态 |
| GET | `/index/status` | 索引状态 |
| GET | `/index/diff` | 自上次建索引以来文档的变化（新增/修改/删除/配置） |
| POST | `/index/rebuild` | 重建索引（`force`、`background`；默认后台执行并返回 202） |
| GET | `/documents` | 文档列表 |
| POST | `/query` | 执行查询（支持流式） |
| POST | `/search/keyword` | BM25 关键词搜索（同样受 cutoff 与 `max_sources` 截断；`verbose` 才返回全文） |
| POST | `/query/batch` | 批量查询（单次最多 20 条，`BATCH_CONCURRENCY` 并发上限） |
| POST | `/mode/change` | 修改服务级默认检索参数 |
| GET | `/test` | 非流式测试 |
| GET | `/test/stream` | 流式测试 |
| GET | `/test/simple` | 简单流式测试 |

### 流式格式（`POST /query` 且 `"stream": true`）

`application/x-ndjson`，每行一个 JSON 对象：

```
{"event":"query_started","question":"...","mode":"HYBRID","similarity_top_k":5,"timestamp":"..."}
{"event":"retrieval_completed","count":5,"retrieval_ms":12.3}
{"event":"answer_chunk","chunk":"部","is_complete":false}      # 多次，token 级
...
{"event":"answer_chunk","chunk":"","is_complete":true,"answer_chars":512,
 "retrieval_ms":12.3,"llm_ms":820.5,"ttft_ms":210.4,"total_ms":833.1}
{"event":"sources","sources":[{"rank":1,"file_name":"dsm5.txt","retrieval_type":"both","score":1.0}],
 "total_sources":1,"retrieval_stats":{"vector":1,"keyword":1,"both":1}}
{"event":"query_completed","timestamp":"...","total_ms":833.1}
```

出错时以 `{"event":"error","error":"..."}` 结尾。`format=text` 会把同样的事件渲染成纯文本，方便在 Swagger UI 里看。

## FastMCP 服务

> MCP（Model Context Protocol）是一种标准化协议，为 LLM 应用提供安全、统一的上下文和工具访问方式。

### 工具（Tools）

| 工具 | 说明 |
|------|------|
| `build_index` | 构建/重建检索索引，支持 `force_rebuild`（会先比对文档清单） |
| `get_index_status` | 查询索引构建状态（ready / building / not_built） |
| `search` | 混合/向量/BM25 关键词检索，支持四种模式；结果同样按 cutoff 与 `max_sources` 截断 |
| `keyword_search` | 纯 BM25 关键词检索（与 `search` 同一套截断规则） |
| `list_documents` | 列出已索引文档信息 |

### 资源（Resources）

| URI | 说明 |
|-----|------|
| `dsm5://status` | 系统整体状态 |
| `dsm5://index/status` | 索引构建状态 |
| `dsm5://config` | 配置概览（不含密钥） |
| `dsm5://documents` | 文档文件列表 |

### 提示模板（Prompts）

| 模板 | 说明 |
|------|------|
| `dsm5_search_assistant` | 检索助手系统提示，包含工具说明和检索模式指南 |
| `dsm5_differential_diagnosis(symptoms)` | 基于症状描述生成鉴别诊断检索提示 |

## 检索模式

| 模式 | 说明 |
|------|------|
| `HYBRID` | 两路各自检索后按名次融合（Reciprocal Rank Fusion），再把融合分数缩放到 0~1 |
| `VECTOR_ONLY` | 仅向量语义检索 |
| `KEYWORD_ONLY` | 仅 BM25 关键词检索；无词重叠（原始分为 0）的节点直接丢弃 |
| `FUSION` | 各通道先 min-max 归一到 0~1，再按 `vector_weight` / `1 - vector_weight` 加权融合 |

四种模式返回的分数都是非 `None` 的 0~1 值，因此 `SIMILARITY_CUTOFF`、MCP 结果展示和 `retrieval_stats` 的行为在所有模式下一致。

## 放入新文档后如何刷新索引

只把文件丢进 `DOCUMENTS_PATH` 是不够的，运行中的服务需要显式重建：

```bash
# 看索引是否落后于文档目录
curl -s localhost:8000/index/diff

# 后台重建（立即返回 202，用 /index/status 轮询进度）
curl -s -X POST "localhost:8000/index/rebuild?force=false"

# 小语料可以同步等结果
curl -s -X POST "localhost:8000/index/rebuild?force=true&background=false"
```

`build_index()` 会重新扫描文档目录，并把每个文件的 sha256（以及影响切分的配置）与 `PERSIST_DIR/index_manifest.json` 比对：没有任何变化就直接加载现有索引（不重复算 embedding）；有变化才重建并重写清单。注意重建仍是整库重新 embedding，按文档粒度的增量 embedding 尚未实现。

旧版关键词索引目录（`keyword_index/docstore.json`，即原来 `SummaryIndex` 的结构）在首次加载时会就地迁移成 BM25，不需要重新计算 embedding。

## 测试

测试既不下载 embedding 权重，也不会碰 `chroma_dsm5_db/`：`tests/conftest.py` 把文档/持久化/模型目录指向临时目录，并装上确定性的词袋哈希向量替身（无 API Key 时 LLM 自动用 MockLLM）。

```bash
# 运行所有测试
uv run pytest

# 运行指定测试文件
uv run pytest tests/test_api.py -v

# MCP 测试（注册 + 用样例语料真实调用工具）
uv run pytest tests/test_mcp_server.py -v

# 覆盖率（pytest-cov 已在 dev 依赖里）
uv run pytest --cov=dsm5_rag --cov-report=term-missing

# 静态检查 / 格式
uv run ruff check src tests scripts
uv run ruff format --check src/dsm5_rag

# 关键词通道诊断：打印分数、融合贡献、耗时分段
uv run python scripts/repro_keyword_channel.py
```

覆盖范围：真实 `tests/fixtures/corpus` 片段的 BM25 召回、融合语义（名次融合的尺度无关性、`vector_weight`、`top_k`）、含 NDJSON 流式行格式的端点契约、真流式与事件循环响应性、索引变更检测、关键词索引的持久化与旧格式迁移。

CI（`.github/workflows/ci.yml`）执行 `uv sync --frozen` → `ruff check` → `pytest --cov`。

## 技术栈

- **API 框架**: [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/)
- **MCP 框架**: [FastMCP](https://gofastmcp.com/) (v3)
- **RAG**: [LlamaIndex](https://www.llamaindex.ai/)（向量索引、`QueryFusionRetriever`、检索查询引擎）
- **关键词检索**: [llama-index-retrievers-bm25](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/bm25/)（底层 `bm25s` + PyStemmer）
- **向量数据库**: [ChromaDB](https://www.trychroma.com/)
- **嵌入模型**: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **LLM**: [DeepSeek](https://platform.deepseek.com/) API
- **数据验证**: [Pydantic](https://docs.pydantic.dev/)
- **文档解析**: pypdf / python-docx
- **包管理**: [uv](https://docs.astral.sh/uv/)
- **静态检查/测试**: ruff + pytest（含 pytest-cov）、GitHub Actions
- **构建**: Hatchling

## MCP 客户端配置

### stdio 模式（本地/IDE 集成）

用于 Claude Desktop、Cursor、VS Code 等支持 MCP 的 IDE：

```json
{
  "mcpServers": {
    "dsm5-rag-mcp": {
      "command": "uv",
      "args": ["run", "dsm5-rag-mcp"],
      "cwd": "/path/to/myDSM5RAG"
    }
  }
}
```

### Streamable HTTP 模式（远程服务）

```json
{
  "mcpServers": {
    "dsm5-rag-mcp": {
      "url": "http://host:8000/mcp",
      "transport": "streamable-http"
    }
  }
}
```
