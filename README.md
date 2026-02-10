# 🎯 CTF Attack & Defense — Agentic AI System

Multi-agent AI system for CTF Attack & Defense competitions, built with:

- **A2A Protocol** (Google): inter-agent communication
- **Semantic Kernel** (Microsoft): LLM orchestration & function calling
- **MCP**: agent-to-tool integration
- **ai-chat-protocol** (Microsoft): frontend chat UI

## Quick Start

### Prerequisites

- Python 3.11+
- At least one LLM API key (Anthropic or OpenAI)
- Redis (optional, for team collaboration)

### Option A: Local Development

### Option B: Docker Compose

### Test it

## Project Structure

## LLM Provider Fallback

The system tries providers in order:

1. **Anthropic** (if `ANTHROPIC_API_KEY` set) — best for code analysis
2. **OpenAI** (if `OPENAI_API_KEY` set)
3. **Ollama** (always available) — local, free, lower quality

## License

MIT
