# DSM-5 RAG Diagnostic Criteria Knowledge QA System

**English** | [简体中文](README.zh-CN.md)

A DSM-5 diagnostic criteria knowledge QA system built on hybrid retrieval (vector search + keyword search), supporting queries in both Chinese and English.

The system provides **two service entry points**:

- **FastAPI microservice** (`dsm5-rag-api`) — full RESTful API with integrated DeepSeek LLM for professional answer generation
- **FastMCP service** (`dsm5-rag-mcp`) — pure retrieval service over the MCP protocol, with no external LLM dependency, ideal for IDE integration

## Features

- **Hybrid Retrieval** — vector semantic search (MiniLM multilingual + Chroma) combined with a real **BM25 keyword channel** (`llama-index-retrievers-bm25`), supporting `HYBRID` / `VECTOR_ONLY` / `KEYWORD_ONLY` / `FUSION` modes
- **Scale-safe fusion** — `HYBRID` fuses the two channels by rank (Reciprocal Rank Fusion from `QueryFusionRetriever`), `FUSION` min-max normalizes each channel and applies `vector_weight`; keyword scores are normalized to 0~1 so the similarity cutoff, MCP rendering and `retrieval_stats` all behave the same for both channels
- **Precise terminology matching** — BM25 handles the DSM-5 cases where exact strings matter (`296.3x`, `Major Depressive Disorder`, `anhedonia`), which pure embedding search tends to blur
- **Multilingual Support** — uses the `paraphrase-multilingual-MiniLM-L12-v2` multilingual embedding model, supporting both Chinese and English queries
- **DeepSeek LLM** — integrates the DeepSeek API to generate professional answers from retrieved content (FastAPI service)
- **Pure Retrieval MCP** — the FastMCP service strips out the LLM and exposes only retrieval capabilities, ideal for IDE and toolchain integration
- **Real token streaming** — `stream=true` streams the answer token by token over NDJSON and reports `ttft_ms` (time to first token) separately from generation time
- **Request-scoped retrieval engine** — `mode` / `similarity_top_k` / `vector_weight` are assembled per request, so concurrent requests cannot pollute each other; `/mode/change` only changes the service-level defaults
- **Timing you can trust** — responses carry `retrieval_ms` / `llm_ms` / `total_ms` measured around the actual work
- **Automatic index management with change detection** — an `index_manifest.json` records each document's sha256/mtime; unchanged corpora are loaded instead of re-embedded, added/edited files trigger a rebuild. Use `POST /index/rebuild` and `GET /index/diff`
- **RESTful API** — built on FastAPI with a complete REST API
- **Document Format Support** — parses PDF, DOCX, and TXT documents
- **ChromaDB Persistence** — the index (vector store + BM25 index) is persisted automatically and survives restarts

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
| `SIMILARITY_CUTOFF` | `0.2` | Similarity filter threshold (applies to both channels; scores are 0~1) |
| `MAX_SOURCES` | `3` | Maximum number of sources returned by `/query` and `/search/keyword` |
| `ENABLE_CONTEXT_REORDER` | `true` | Enable context reordering (Lost-in-the-Middle optimization) |
| `ENABLE_METADATA_REPLACEMENT` | `false` | Enable metadata replacement (requires a sentence-window parser; keep `false` with hierarchical chunking) |
| `BATCH_CONCURRENCY` | `4` | `asyncio.Semaphore` limit for `POST /query/batch` |
| `STREAM_CONTEXT_CHAR_LIMIT` | `24000` | Character cap on the prompt context used for streaming answers |

### BM25 Keyword Index

| Variable | Default | Description |
|----------|---------|-------------|
| `BM25_LANGUAGE` | `en` | Stopword language for the keyword channel (`en/de/nl/fr/es/pt/it/ru/sv/no/zh/tr/ko/da`) |
| `BM25_SKIP_STEMMING` | `false` | Disable stemming (keeps `disorder` and `disorders` strictly distinct) |

### Document Parsing

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1024` | Document chunk size (tokens for the sentence splitter) |
| `CHUNK_OVERLAP` | `100` | Chunk overlap size |
| `MAX_FILE_SIZE_MB` | `50` | Maximum file size (MB) |

> **Note on `API_PORT`**: the code default is `8000`, while `.example.env` ships `8031` as an example. Set it explicitly if you run both services side by side.
>
> **Note on Chroma**: `dsm5-rag-api` and `dsm5-rag-mcp` open the same `PERSIST_DIR`. Run one service at a time per directory, or give each service its own `PERSIST_DIR`, to avoid SQLite lock contention.

## Project Structure

```
myDSM5RAG/
├── src/dsm5_rag/                # Core package
│   ├── __init__.py              # Package entry, public API exports
│   ├── api.py                   # FastAPI microservice (with LLM)
│   ├── background.py            # Thread pool / run_in_executor (keeps the event loop free)
│   ├── config.py                # Configuration model (RagConfig)
│   ├── index_manifest.py        # Per-document sha256/mtime manifest, rebuild decisions
│   ├── keyword_index.py         # BM25 keyword index (build/persist/load/migrate + score normalization)
│   ├── mcp_server.py            # FastMCP pure retrieval service (no LLM)
│   ├── models.py                # Pydantic data models
│   ├── parser.py                # Document parser (PDF/DOCX/TXT)
│   ├── retriever.py             # HybridRetriever (RRF / weighted fusion on top of core's QueryFusionRetriever)
│   └── system.py                # RAG system core (DeepSeekRAGSystem)
├── scripts/
│   └── repro_keyword_channel.py # Keyword-channel diagnostic (runs before/after changes)
├── tests/                       # Test suite
│   ├── conftest.py              # Fixtures: temp-dir isolation + fake embedding (no model download)
│   ├── fake_system.py           # Fake RAG system for API contract tests
│   ├── lexical_embedding.py     # Deterministic hashing embedding used by tests
│   ├── fixtures/corpus/         # 5 real DSM-5 excerpt .txt files
│   ├── test_api.py              # Endpoint contracts incl. NDJSON stream format
│   ├── test_config.py           # Configuration tests
│   ├── test_index_manifest.py   # Manifest / change-detection tests
│   ├── test_keyword_index.py    # BM25 recall tests on the real corpus
│   ├── test_mcp_server.py       # MCP registration + tool behaviour tests
│   ├── test_models.py           # Data model tests
│   ├── test_parser.py           # Document parser tests
│   ├── test_retriever.py        # Fusion semantics tests (scale invariance, weights, top_k)
│   ├── test_streaming.py        # Real streaming + event-loop responsiveness
│   └── test_system_integration.py # End-to-end retrieval with the sample corpus
├── examples/                    # Usage examples
│   ├── basic_query.py           # Basic query
│   ├── batch_query.py           # Batch queries & keyword search
│   └── api_client.py            # HTTP API client
├── dsm5_documents/              # Documents directory (gitignored)
├── chroma_dsm5_db/              # ChromaDB data + index_manifest.json (gitignored)
├── models/                      # Local model cache (gitignored)
├── .github/workflows/ci.yml     # CI: uv sync -> ruff check -> pytest --cov
├── .env                         # Environment variables (gitignored)
├── .example.env                 # Environment template
├── pyproject.toml               # Project config, dependencies, ruff/pytest settings
└── uv.lock                      # uv lock file
```

## FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/status` | System status |
| GET | `/index/status` | Index status |
| GET | `/index/diff` | Documents changed since the last index build (added/changed/removed/config) |
| POST | `/index/rebuild` | Rebuild the index (`force`, `background`; default `background=true` returns 202) |
| GET | `/documents` | Document list |
| POST | `/query` | Execute a query (streaming supported) |
| POST | `/search/keyword` | BM25 keyword search (cutoff + `max_sources` applied; `verbose` for full text) |
| POST | `/query/batch` | Batch queries (max 20 questions, `BATCH_CONCURRENCY` semaphore) |
| POST | `/mode/change` | Change service-level default retrieval parameters |
| GET | `/test` | Non-streaming test |
| GET | `/test/stream` | Streaming test |
| GET | `/test/simple` | Simple streaming test |

### Streaming format (`POST /query` with `"stream": true`)

`application/x-ndjson`, one JSON object per line:

```
{"event":"query_started","question":"...","mode":"HYBRID","similarity_top_k":5,"timestamp":"..."}
{"event":"retrieval_completed","count":5,"retrieval_ms":12.3}
{"event":"answer_chunk","chunk":"部","is_complete":false}      # 一次或多次，token 级
...
{"event":"answer_chunk","chunk":"","is_complete":true,"answer_chars":512,
 "retrieval_ms":12.3,"llm_ms":820.5,"ttft_ms":210.4,"total_ms":833.1}
{"event":"sources","sources":[{"rank":1,"file_name":"dsm5.txt","retrieval_type":"both","score":1.0}],
 "total_sources":1,"retrieval_stats":{"vector":1,"keyword":1,"both":1}}
{"event":"query_completed","timestamp":"...","total_ms":833.1}
```

On failure the stream ends with `{"event":"error","error":"..."}`. `format=text` returns the same events rendered as plain text for Swagger UI.

## FastMCP Service

> MCP (Model Context Protocol) is a standardized protocol that provides LLM applications with secure, unified access to context and tools.

### Tools

| Tool | Description |
|------|-------------|
| `build_index` | Build/rebuild the retrieval index, supports `force_rebuild` (compares the document manifest first) |
| `get_index_status` | Query index build status (ready / building / not_built) |
| `search` | Hybrid/vector/BM25-keyword retrieval, supports four modes; results are cutoff- and `max_sources`-filtered |
| `keyword_search` | Pure BM25 keyword retrieval (same truncation rules as `search`) |
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
| `HYBRID` | Retrieves from both channels and fuses them by rank (Reciprocal Rank Fusion), then rescales the fused score to 0~1 |
| `VECTOR_ONLY` | Vector semantic search only |
| `KEYWORD_ONLY` | BM25 keyword search only; nodes with zero term overlap are dropped |
| `FUSION` | Min-max normalizes each channel, then blends with `vector_weight` / `1 - vector_weight` |

All four modes return non-`None` scores in 0~1, so `SIMILARITY_CUTOFF`, the MCP result renderer and `retrieval_stats` behave consistently.

## Refreshing the index after adding documents

Dropping new files into `DOCUMENTS_PATH` is not enough on its own — the running service has to rebuild:

```bash
# see whether the index is behind the documents
curl -s localhost:8000/index/diff

# rebuild in the background (returns 202 immediately; poll /index/status)
curl -s -X POST "localhost:8000/index/rebuild?force=false"

# small corpora only: wait for the result inline
curl -s -X POST "localhost:8000/index/rebuild?force=true&background=false"
```

`build_index()` re-scans the documents and compares each file's sha256 (plus chunking-related config) against `PERSIST_DIR/index_manifest.json`. Nothing changed → the existing index is loaded (no re-embedding). Something changed → the index is rebuilt and the manifest is rewritten. Note the rebuild re-embeds the whole corpus; per-document incremental embedding is not implemented yet.

Legacy keyword indexes (`keyword_index/docstore.json` from the old `SummaryIndex` layout) are migrated to BM25 in place on first load, without re-computing embeddings.

## Testing

Tests never download embedding weights and never touch `chroma_dsm5_db/`: `tests/conftest.py`
points documents/persist/models at temp directories and installs a deterministic hashing
stand-in for `HuggingFaceEmbedding` (plus `MockLLM` when no API key is set).

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_api.py -v

# Run MCP server tests (registration + real tool calls against the sample corpus)
uv run pytest tests/test_mcp_server.py -v

# Coverage (pytest-cov is part of the dev group)
uv run pytest --cov=dsm5_rag --cov-report=term-missing

# Lint / format
uv run ruff check src tests scripts
uv run ruff format --check src/dsm5_rag

# Keyword-channel diagnostic: prints scores, fusion contributions, timings
uv run python scripts/repro_keyword_channel.py
```

The suite covers: BM25 recall on the real `tests/fixtures/corpus` excerpts, fusion semantics
(rank-based scale invariance, `vector_weight`, `top_k`), the API contracts including the NDJSON
stream shape, real token streaming with an event-loop responsiveness check, index change
detection, and persistence/migration of the keyword index.

CI (`.github/workflows/ci.yml`) runs `uv sync --frozen` → `ruff check` → `pytest --cov`.

## Tech Stack

- **API framework**: [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/)
- **MCP framework**: [FastMCP](https://gofastmcp.com/) (v3)
- **RAG**: [LlamaIndex](https://www.llamaindex.ai/) (vector index, `QueryFusionRetriever`, retrieval query engines)
- **Keyword retrieval**: [llama-index-retrievers-bm25](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/bm25/) on top of `bm25s` + PyStemmer
- **Vector database**: [ChromaDB](https://www.trychroma.com/)
- **Embedding model**: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **LLM**: [DeepSeek](https://platform.deepseek.com/) API
- **Data validation**: [Pydantic](https://docs.pydantic.dev/)
- **Document parsing**: pypdf / python-docx
- **Package management**: [uv](https://docs.astral.sh/uv/)
- **Lint/tests**: ruff + pytest (+ pytest-cov), GitHub Actions
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
