# Deep RAG

Deep RAG enterprise knowledge retrieval system:
- Python orchestration state machine
- MCP tool server (6 tools)
- Offline preprocessor building 3-layer indexes
- Local LLM/embedding via Ollama

## Quick Start

1. Install dependencies
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Configure env
```bash
cp .env.example .env
```

3. Configure model backend (single file)
- Edit `config/deep_rag.yaml`
- `llm.provider: ollama` for local/docker Ollama
- `llm.provider: openai_compatible` for OpenAI-compatible endpoints
- Environment variables still override YAML fields
- For personal keys, create `config/deep_rag.local.yaml` and set `DEEP_RAG_CONFIG=config/deep_rag.local.yaml`
- If your proxy causes cert errors, set `openai_compatible.verify_ssl: false` temporarily (or configure `ca_bundle`)

4. Build indexes
```bash
python -m preprocessor.run_preprocess --kb-dir knowledge_base --index-dir index_store
```

5. Run CLI
```bash
python -m app.main "最近三年的销售趋势如何？" --pretty
python -m app.main --interactive
```

6. Run Web
```bash
streamlit run app/web.py
```

7. Run evaluation
```bash
python -m evaluation.run_eval
```
