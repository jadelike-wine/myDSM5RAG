# DSM-5 RAG 诊断标准知识问答系统

基于混合检索（向量检索 + 关键词检索）的 DSM-5 诊断标准知识问答系统，支持中英文多语言查询。

系统提供**两种服务入口**：

- **FastAPI 微服务** (`dsm5-rag-api`) — 完整的 RESTful API，集成 DeepSeek LLM 生成专业回答
- **FastMCP 服务** (`dsm5-rag-mcp`) — MCP 协议纯检索服务，剥离外部 LLM 依赖，适合 IDE 集成

## 特性

- **混合检索** — 融合向量语义检索与关键词精确匹配，支持 `HYBRID` / `VECTOR_ONLY` / `KEYWORD_ONLY` / `FUSION` 四种模式
- **多语言支持** — 使用 `paraphrase-multilingual-MiniLM-L12-v2` 多语言嵌入模型，支持中英文查询
- **DeepSeek LLM** — 集成 DeepSeek API，基于检索结果生成专业回答（FastAPI 服务）
- **纯检索 MCP** — FastMCP 服务剥离 LLM，仅暴露检索能力，适合 IDE 和工具链集成
- **流式响应** — 支持流式输出，实时展示回答生成过程
- **RESTful API** — 基于 FastAPI 构建，提供完整的 REST API 接口
- **自动索引管理** — 系统启动时自动扫描文档、构建/加载索引，无需手动干预
- **文档格式支持** — 支持 PDF、DOCX、TXT 格式文档解析
- **ChromaDB 持久化** — 索引自动持久化，重启后无需重新构建

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
| `SIMILARITY_CUTOFF` | `0.2` | 相似度过滤阈值 |
| `MAX_SOURCES` | `3` | 最大来源数量 |
| `ENABLE_CONTEXT_REORDER` | `true` | 是否启用上下文重排序（Lost-in-the-Middle 优化） |
| `ENABLE_METADATA_REPLACEMENT` | `false` | 是否启用元数据替换 |

### 文档解析

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | `1024` | 文档分块大小（字符数） |
| `CHUNK_OVERLAP` | `100` | 分块重叠大小 |
| `MAX_FILE_SIZE_MB` | `50` | 最大文件大小（MB） |

## 项目结构

```
myDSM5RAG/
├── src/dsm5_rag/                # 核心包
│   ├── __init__.py              # 包入口，公开 API 导出
│   ├── api.py                   # FastAPI 微服务（含 LLM）
│   ├── config.py                # 配置模型（RagConfig）
│   ├── mcp_server.py            # FastMCP 纯检索服务（无 LLM）
│   ├── models.py                # Pydantic 数据模型
│   ├── parser.py                # 文档解析器（PDF/DOCX/TXT）
│   ├── retriever.py             # 混合检索器（HybridRetriever）
│   └── system.py                # RAG 系统核心（DeepSeekRAGSystem）
├── tests/                       # 测试套件
│   ├── conftest.py              # pytest 共享 fixtures
│   ├── test_api.py              # API 端点测试
│   ├── test_config.py           # 配置测试
│   ├── test_mcp_server.py       # MCP 服务器注册测试
│   ├── test_models.py           # 数据模型测试
│   ├── test_parser.py           # 文档解析测试
│   └── test_retriever.py        # 检索器测试
├── examples/                    # 使用示例
│   ├── basic_query.py           # 基本查询
│   ├── batch_query.py           # 批量查询 & 关键词搜索
│   └── api_client.py            # HTTP API 客户端
├── dsm5_documents/              # 文档目录（已 gitignore）
├── chroma_dsm5_db/              # ChromaDB 数据（已 gitignore）
├── models/                      # 本地模型缓存（已 gitignore）
├── .env                         # 环境变量配置（已 gitignore）
├── .example.env                 # 环境变量模板
├── pyproject.toml               # 项目配置与依赖
└── uv.lock                      # uv 锁文件
```

## FastAPI 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| GET | `/status` | 系统状态 |
| GET | `/index/status` | 索引状态 |
| GET | `/documents` | 文档列表 |
| POST | `/query` | 执行查询（支持流式） |
| POST | `/search/keyword` | 关键词搜索 |
| POST | `/query/batch` | 批量查询 |
| POST | `/mode/change` | 切换检索模式 |
| GET | `/test` | 非流式测试 |
| GET | `/test/stream` | 流式测试 |
| GET | `/test/simple` | 简单流式测试 |

## FastMCP 服务

> MCP（Model Context Protocol）是一种标准化协议，为 LLM 应用提供安全、统一的上下文和工具访问方式。

### 工具（Tools）

| 工具 | 说明 |
|------|------|
| `build_index` | 构建/重建检索索引，支持 `force_rebuild` 强制重建 |
| `get_index_status` | 查询索引构建状态（ready / building / not_built） |
| `search` | 混合/向量/关键词检索，支持四种模式 |
| `keyword_search` | 纯关键词精确检索 |
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
| `HYBRID` | 混合检索，向量检索 + 关键词检索结果合并去重 |
| `VECTOR_ONLY` | 仅向量语义检索 |
| `KEYWORD_ONLY` | 仅关键词精确检索 |
| `FUSION` | 融合检索，加权平均排序（可调向量/关键词权重） |

## 测试

```bash
# 运行所有测试
uv run pytest

# 运行指定测试文件
uv run pytest tests/test_api.py -v

# 运行 MCP 服务测试
uv run pytest tests/test_mcp_server.py -v

# 查看覆盖率
uv run pytest --cov=dsm5_rag
```

## 技术栈

- **API 框架**: [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/)
- **MCP 框架**: [FastMCP](https://gofastmcp.com/) (v3)
- **RAG**: [LlamaIndex](https://www.llamaindex.ai/) (向量索引、关键词索引、检索查询引擎)
- **向量数据库**: [ChromaDB](https://www.trychroma.com/)
- **嵌入模型**: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **LLM**: [DeepSeek](https://platform.deepseek.com/) API
- **数据验证**: [Pydantic](https://docs.pydantic.dev/)
- **文档解析**: pypdf / python-docx
- **包管理**: [uv](https://docs.astral.sh/uv/)
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