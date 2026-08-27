# DSM-5 RAG Diagnostic Criteria Knowledge QA System

**English** | [简体中文](README.zh-CN.md)

A DSM-5 diagnostic criteria knowledge QA system built on hybrid retrieval (vector search + keyword search), supporting queries in both Chinese and English.

The system provides **two service entry points**:

- **FastAPI microservice** (`dsm5-rag-api`) — full RESTful API with integrated DeepSeek LLM for professional answer generation
- **FastMCP service** (`dsm5-rag-mcp`) — pure retrieval service over the MCP protocol, with no external LLM dependency, ideal for IDE integration

## Features

- **Hybrid Retrieval** — combines vector semantic search with exact keyword matching, supporting `HYBRID` / `VECTOR_ONLY` / `KEYWORD_ONLY` / `FUSION` modes
- **Multilingual Support** — uses the `paraphrase-multilingual-MiniLM-L12-v2` multilingual embedding model, supporting both Chinese and English queries
- **DeepSeek LLM** — integrates the DeepSeek API to generate professional answers from retrieved content (FastAPI service)
- **Pure Retrieval MCP** — the FastMCP service strips out the LLM and exposes only retrieval capabilities, ideal for IDE and toolchain integration
- **Streaming Responses** — supports streaming output with real-time answer generation
- **RESTful API** — built on FastAPI with a complete REST API
- **Automatic Index Management** — scans documents and builds/loads the index automatically on startup, no manual intervention required
- **Document Format Support** — parses PDF, DOCX, and TXT documents
- **ChromaDB Persistence** — the index is persisted automatically and survives restarts

## Quick Start

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url> && cd myDSM5RAG

# Install with uv (recommended)
uv sync --group dev

# Or with pip
pip install -e .
```

### Configuration

Copy the environment template and edit it:

```bash
cp .example.env .env
```

Configure your DeepSeek API key in `.env` (get one from [platform.deepseek.com](https://platform.deepseek.com/)):

```env
DEEPSEEK_API_KEY="sk-your-api-key-here"
```

> **Note**: Without an API key, the FastAPI service runs in MockLLM simulation mode. The MCP service is unaffected (pure retrieval, no LLM needed).

Place your DSM-5 diagnostic criteria documents (PDF/DOCX/TXT) into the `dsm5_documents/` directory.

### Running

#### FastAPI Service (with LLM answer generation)

```bash
# With uv
uv run dsm5-rag-api

# Or directly with uvicorn
uv run uvicorn dsm5_rag.api:app --host 127.0.0.1 --port 8031
```

Once the service is running:
- API docs (Swagger UI): http://127.0.0.1:8031/docs
- ReDoc: http://127.0.0.1:8031/redoc

#### MCP Service (pure retrieval, no LLM)

```bash
# stdio mode (IDE integration, e.g. Claude Desktop, Cursor)
uv run dsm5-rag-mcp

# Streamable HTTP mode (remote/production)
uv run python -c "from dsm5_rag.mcp_server import main_http; main_http()"

# Or debug with fastmcp dev mode
uv run fastmcp dev src/dsm5_rag/mcp_server.py
```

#### Running Examples

```bash
# Basic query (using the Python SDK)
uv run python examples/basic_query.py

# Batch queries & keyword search
uv run python examples/batch_query.py

# HTTP API client (requires the API service to be running)
uv run python examples/api_client.py
```

## Configuration

All configuration is set via environment variables (see `.env`):

### API & Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `localhost` | Server listen address |
| `API_PORT` | `8000` | Server port |
| `API_RELOAD` | `false` | Enable hot reload |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Log level |

### DeepSeek LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | `""` | API key (falls back to MockLLM if unset) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model name |

### Embedding Model

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model name |
| `EMBEDDING_DEVICE` | `cpu` | Runtime device (cpu / cuda) |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror endpoint |
| `MODELS_DIR` | `./models` | Local model cache directory |

### Storage & Documents

| Variable | Default | Description |
|----------|---------|-------------|
| `PERSIST_DIR` | `./chroma_dsm5_db` | ChromaDB persistence directory |
| `DOCUMENTS_PATH` | `./dsm5_documents` | Document path (directory or single file) |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODE` | `HYBRID` | Default retrieval mode |
| `DEFAULT_TOP_K` | `5` | Default number of results |
| `DEFAULT_VECTOR_WEIGHT` | `0.7` | Default vector weight (FUSION mode) |
| `SIMILARITY_CUTOFF` | `0.2` | Similarity filter threshold |
| `MAX_SOURCES` | `3` | Maximum number of sources |
| `ENABLE_CONTEXT_REORDER` | `true` | Enable context reordering (Lost-in-the-Middle optimization) |
| `ENABLE_METADATA_REPLACEMENT` | `false` | Enable metadata replacement |

### Document Parsing

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1024` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `100` | Chunk overlap size |
| `MAX_FILE_SIZE_MB` | `50` | Maximum file size (MB) |

## Project Structure

```
myDSM5RAG/
├── src/dsm5_rag/                # Core package
│   ├── __init__.py              # Package entry, public API exports
│   ├── api.py                   # FastAPI microservice (with LLM)
│   ├── config.py                # Configuration model (RagConfig)
│   ├── mcp_server.py            # FastMCP pure retrieval service (no LLM)
│   ├── models.py                # Pydantic data models
│   ├── parser.py                # Document parser (PDF/DOCX/TXT)
│   ├── retriever.py             # Hybrid retriever (HybridRetriever)
│   └── system.py                # RAG system core (DeepSeekRAGSystem)
├── tests/                       # Test suite
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_api.py              # API endpoint tests
│   ├── test_config.py           # Configuration tests
│   ├── test_mcp_server.py       # MCP server registration tests
│   ├── test_models.py           # Data model tests
│   ├── test_parser.py           # Document parser tests
│   └── test_retriever.py        # Retriever tests
├── examples/                    # Usage examples
│   ├── basic_query.py           # Basic query
│   ├── batch_query.py           # Batch queries & keyword search
│   └── api_client.py            # HTTP API client
├── dsm5_documents/              # Documents directory (gitignored)
├── chroma_dsm5_db/              # ChromaDB data (gitignored)
├── models/                      # Local model cache (gitignored)
├── .env                         # Environment variables (gitignored)
├── .example.env                 # Environment template
├── pyproject.toml               # Project config & dependencies
└── uv.lock                      # uv lock file
```

## FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/status` | System status |
| GET | `/index/status` | Index status |
| GET | `/documents` | Document list |
| POST | `/query` | Execute a query (streaming supported) |
| POST | `/search/keyword` | Keyword search |
| POST | `/query/batch` | Batch queries |
| POST | `/mode/change` | Switch retrieval mode |
| GET | `/test` | Non-streaming test |
| GET | `/test/stream` | Streaming test |
| GET | `/test/simple` | Simple streaming test |

## FastMCP Service

> MCP (Model Context Protocol) is a standardized protocol that provides LLM applications with secure, unified access to context and tools.

### Tools

| Tool | Description |
|------|-------------|
| `build_index` | Build/rebuild the retrieval index, supports `force_rebuild` |
| `get_index_status` | Query index build status (ready / building / not_built) |
| `search` | Hybrid/vector/keyword retrieval, supports four modes |
| `keyword_search` | Pure exact keyword retrieval |
| `list_documents` | List indexed documents |

### Resources

| URI | Description |
|-----|-------------|
| `dsm5://status` | Overall system status |
| `dsm5://index/status` | Index build status |
| `dsm5://config` | Configuration overview (secrets excluded) |
| `dsm5://documents` | Document file list |

### Prompts

| Template | Description |
|----------|-------------|
| `dsm5_search_assistant` | Retrieval assistant system prompt with tool descriptions and retrieval mode guidance |
| `dsm5_differential_diagnosis(symptoms)` | Generates a differential diagnosis retrieval prompt from symptom descriptions |

## Retrieval Modes

| Mode | Description |
|------|-------------|
| `HYBRID` | Hybrid retrieval, vector + keyword results merged and deduplicated |
| `VECTOR_ONLY` | Vector semantic search only |
| `KEYWORD_ONLY` | Exact keyword search only |
| `FUSION` | Fusion retrieval with weighted average ranking (adjustable vector/keyword weights) |

## Testing

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_api.py -v

# Run MCP server tests
uv run pytest tests/test_mcp_server.py -v

# View coverage
uv run pytest --cov=dsm5_rag
```

## Tech Stack

- **API framework**: [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/)
- **MCP framework**: [FastMCP](https://gofastmcp.com/) (v3)
- **RAG**: [LlamaIndex](https://www.llamaindex.ai/) (vector index, keyword index, retrieval query engines)
- **Vector database**: [ChromaDB](https://www.trychroma.com/)
- **Embedding model**: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **LLM**: [DeepSeek](https://platform.deepseek.com/) API
- **Data validation**: [Pydantic](https://docs.pydantic.dev/)
- **Document parsing**: pypdf / python-docx
- **Package management**: [uv](https://docs.astral.sh/uv/)
- **Build**: Hatchling

## MCP Client Configuration

### stdio mode (local/IDE integration)

For MCP-capable IDEs such as Claude Desktop, Cursor, and VS Code:

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

### Streamable HTTP mode (remote service)

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
