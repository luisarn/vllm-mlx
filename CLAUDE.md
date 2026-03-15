# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vllm-mlx is an Apple Silicon GPU-accelerated inference engine that provides a vLLM-compatible API using Apple's MLX framework. It supports text (LLM), vision (MLLM), audio (TTS/STT), and embeddings.

## Common Commands

### Development Setup

```bash
# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Install with optional audio support
pip install -e ".[audio]"
```

### Running Tests

```bash
# Run all tests (excludes slow and integration tests by default)
pytest

# Run a specific test file
pytest tests/test_paged_cache.py -v

# Run slow tests (requires model loading)
pytest -m slow

# Run integration tests (requires running server)
pytest -m integration

# Run with coverage
pytest --cov=vllm_mlx tests/
```

### Code Quality

```bash
# Format code (uses ruff via pre-commit)
ruff format vllm_mlx/
ruff check --fix vllm_mlx/

# Type checking
mypy vllm_mlx/

# Run pre-commit hooks on all files
pre-commit run --all-files
```

### Running the Server

```bash
# Simple mode (maximum throughput, single user)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching mode (multiple concurrent users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching

# Multimodal (vision)
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000

# With tool calling
vllm-mlx serve mlx-community/Qwen3-4B-4bit --port 8000 --enable-auto-tool-choice --tool-call-parser qwen

# With reasoning extraction
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# With embeddings endpoint
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

### Benchmarking

```bash
# LLM benchmark
vllm-mlx bench mlx-community/Qwen3-0.6B-8bit --num-prompts 10

# Benchmark streaming detokenizer
vllm-mlx bench-detok mlx-community/Qwen3-0.6B-8bit

# Benchmark KV cache quantization
vllm-mlx bench-kv-cache --layers 32 --seq-len 512

# Full benchmark suite
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit
```

## High-Level Architecture

### Core Components

**vLLM Platform Plugin** (`vllm_mlx/plugin.py`, `vllm_mlx/platform.py`)
- Entry point for vLLM's platform detection system
- `MLXPlatform` class provides Apple Silicon-specific configuration
- Auto-detects chip name, memory, and MLX availability

**Server** (`vllm_mlx/server.py`)
- FastAPI-based OpenAI-compatible API server
- Supports both simple mode (single user, max throughput) and batched mode (multiple concurrent users)
- Entry points: `vllm-mlx serve` CLI or `python -m vllm_mlx.server`

**Engines** (`vllm_mlx/engine/`)
- `SimpleEngine`: Direct mlx-lm/mlx-vlm wrapper, maximum throughput for single user
- `BatchedEngine`: Continuous batching via `AsyncEngineCore` for concurrent requests
- `EngineCore`/`AsyncEngineCore`: Low-level async engine with scheduler integration

**Models** (`vllm_mlx/models/`)
- `MLXLanguageModel` (`llm.py`): Wrapper around mlx-lm for text-only inference
- `MLXMultimodalLM` (`mllm.py`): Wrapper around mlx-vlm for vision (images, video)
- Model type auto-detection via `model_registry.py` (checks for vision tags, config.json)

**Schedulers**
- `scheduler.py`: LLM request scheduler with priority queue for continuous batching
- `mllm_scheduler.py`: MLLM-specific scheduler handling vision inputs
- `mllm_batch_generator.py`: Batched generation logic for multimodal requests

**Caching Systems**
- `paged_cache.py`: vLLM-style paged KV cache with block-based memory management
- `prefix_cache.py`: Prefix caching for repeated prompts
- `mllm_cache.py`: MLLM-specific cache for image/video + prompt combinations
- `memory_cache.py`: Memory-aware cache with KV cache quantization support

**API Layer** (`vllm_mlx/api/`)
- `models.py`: Pydantic models for OpenAI-compatible API types
- `anthropic_adapter.py`: Anthropic Messages API adapter for Claude Code compatibility
- `tool_calling.py`: Tool call parsing for JSON/function calling formats
- `streaming.py`: SSE streaming response handling

**Tool Parsers** (`vllm_mlx/tool_parsers/`)
- Model-specific tool call parsers (mistral, qwen, llama, hermes, deepseek, etc.)
- `ToolParserManager` for auto-detection based on model name

**Reasoning Parsers** (`vllm_mlx/reasoning/`)
- Extract thinking content from reasoning models (Qwen3, DeepSeek-R1, GPT-OSS)
- Separates reasoning from final answer in response

**MCP Support** (`vllm_mlx/mcp/`)
- Model Context Protocol client and executor
- Tool sandboxing and security validation
- Server management and configuration

**Audio** (`vllm_mlx/audio/`)
- `stt.py`: Speech-to-text via mlx-audio (Whisper models)
- `tts.py`: Text-to-speech with Kokoro, Chatterbox, VibeVoice models
- `processor.py`: Audio preprocessing

**Embeddings** (`vllm_mlx/embedding.py`)
- OpenAI-compatible `/v1/embeddings` endpoint
- Integration with mlx-embeddings

### Request Flow

1. **API Request** → FastAPI endpoint (`server.py`)
2. **Authentication** → API key check, rate limiting
3. **Engine Selection** → Simple or Batched based on `--continuous-batching` flag
4. **Model Type Routing** → LLM vs MLLM detection via `model_registry.py`
5. **Template Application** → Chat template with optional tool definitions
6. **Generation** → mlx-lm (text), mlx-vlm (vision), mlx-audio (TTS/STT), or mlx-embeddings
7. **Post-processing** → Tool call parsing, reasoning extraction, output cleaning
8. **Streaming** → SSE chunks via `StreamingJSONEncoder`
9. **Caching** → KV cache storage for prefix sharing across requests

### Key Design Patterns

- **Platform Plugin**: Integrates with vLLM's platform system as an out-of-tree (OOT) plugin
- **Engine Abstraction**: `BaseEngine` ABC with `SimpleEngine` and `BatchedEngine` implementations
- **Unified Server**: Single server handles both LLM and MLLM via model type detection
- **Modular API**: API models and utilities in separate `api/` package for reusability

### Testing Notes

- Tests use `pytest-asyncio` for async test support
- Slow tests (marked with `@pytest.mark.slow`) require model downloads
- Integration tests (marked with `@pytest.mark.integration`) require a running server
- Default test run excludes slow and integration tests
