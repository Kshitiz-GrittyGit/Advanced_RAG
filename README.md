# Advanced RAG — Financial Document Intelligence

A production-grade Retrieval-Augmented Generation pipeline for SEC financial filings (10-K / 10-Q). Ask natural language questions against real filings and get grounded, cited answers.

> "What was Apple's total net sales revenue in 2025?"
> "What are Apple's main risk factors?"
> "Describe Apple's competitive strategy."

---

## Architecture

```
PDF Files (10-K/10-Q)
        │
        ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────────────────────┐
│ ingestion.py│────▶│ embedding.py│────▶│         retrieval.py         │
│             │     │             │     │  Dense + BM25 (RRF fusion)   │
│ pdfplumber  │     │ BGE-large   │     │  Cross-Encoder Re-Ranking    │
│ tables      │     │ embeddings  │     │  Route-aware table boost     │
│ unstructured│     │             │     └──────────────────────────────┘
│ text        │     │             │                    │
└─────────────┘     └─────────────┘                    ▼
        │                  │                  ┌─────────────────┐
        ▼                  ▼                  │  generation.py  │
  ┌──────────┐      ┌───────────┐             │  LLM providers  │
  │ Postgres │      │  Qdrant   │             │  Groq / Ollama  │
  │ (tables) │      │ (vectors) │             │  Bedrock / HF   │
  └──────────┘      └───────────┘             └─────────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │        app.py          │
                                          │   FastAPI REST Server  │
                                          │   POST /query          │
                                          │   GET  /health         │
                                          ├────────────────────────┤
                                          │     guardrails.py      │
                                          │  Input validation      │
                                          │  Injection detection   │
                                          │  Output grounding      │
                                          └────────────────────────┘
```

---

## Pipeline Stages

### 1. Ingestion (`ingestion.py`)
- **pdfplumber** extracts tables with full header context (column names preserved)
- **unstructured** (fast mode, no hi_res) extracts narrative text with layout signals
- Font-size heading detection + adaptive thresholds for section hierarchy
- Table flattening: each row → one chunk with column headers inline for vector search
- Outputs structured chunks with breadcrumb paths (`ROOT > Item 1 > Revenue`)

### 2. Embedding (`embedding.py`)
- **BGE-large** (BAAI/bge-large-en-v1.5) embeddings — 1024 dimensions
- Metadata attached per chunk: ticker, document type, fiscal year, page, breadcrumb
- Dual storage: Qdrant (vector search) + Postgres (structured table queries)
- Supports local embedding or AWS Bedrock Titan

### 3. Retrieval (`retrieval.py`)
- Hybrid search: dense (Qdrant) + sparse BM25 fused with Reciprocal Rank Fusion (RRF)
- Cross-encoder re-ranking (ms-marco-MiniLM) for precision
- Query routing: `narrative` / `table` / `hybrid` — table chunks get +3.0 score boost on table routes
- Metadata filters: ticker, company name, document type, fiscal year, fiscal quarter

### 4. Generation (`generation.py`)
- Grounded answers using retrieved chunks as context only
- Multi-provider LLM support (set `LLM_PROVIDER` in `.env`):

| Provider | Model | Use case |
|---|---|---|
| `groq` | llama-3.1-8b-instant | Free, fast (default dev) |
| `ollama` | mistral / phi3 | Local, no API key |
| `anthropic` | claude-haiku-4-5 | Paid, high quality |
| `bedrock` | llama3-1-70b | AWS production |
| `huggingface` | Qwen2.5-72B | Free API |

### 5. Guardrails (`guardrails.py`)
All checks are rule-based — no LLM calls, <50ms overhead.

**Input (before retrieval):**
- Query length (5–1000 chars)
- Prompt injection detection (14 regex patterns)
- Off-topic blocking (must be finance-related)

**Output (after generation):**
- Retrieval confidence check (refuses if no relevant chunks found)
- Dollar figure grounding (flags numbers in answer not found in context)
- Financial disclaimer appended to every answer

### 6. Evaluation (`evaluation.py`)
- 15 ground-truth test cases (table, narrative, hybrid query types)
- **RAGAS** metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Giskard** automated vulnerability scan (hallucination, off-topic, injection, etc.)
- Groq (llama-3.3-70b-versatile) as judge LLM for both frameworks

---

## Quick Start

### Prerequisites
- Python 3.12+
- [Qdrant](https://qdrant.tech/) running locally: `docker run -p 6333:6333 qdrant/qdrant`
- PostgreSQL running locally
- (Optional) [Ollama](https://ollama.ai/) for local LLM: `ollama pull mistral`

### Install
```bash
git clone https://github.com/Kshitiz-GrittyGit/Advanced_RAG.git
cd Advanced_RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure
```bash
cp .env.example .env   # edit with your API keys and settings
```

Key `.env` variables:
```
LLM_PROVIDER=groq          # groq | ollama | anthropic | bedrock | huggingface
GROQ_API_KEY=gsk_...
EMBED_PROVIDER=local       # local | bedrock
```

### Ingest documents
```bash
python ingestion.py          # parse PDFs → structured chunks
python embedding.py          # embed chunks → Qdrant + Postgres
```

### Start the API
```bash
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple total revenue in 2025?", "ticker": "AAPL", "fiscal_year": 2025}'
```

Response:
```json
{
  "answer": "Apple's total net sales were $... \n\n---\n*Disclaimer: ...*",
  "sources": ["AAPL 10-K 2025, page 39 — ROOT > Item 8 > Revenue"],
  "route": "table",
  "latency_ms": 1842,
  "warnings": []
}
```

Blocked query response:
```json
{
  "blocked": true,
  "message": "Your query contains patterns that cannot be processed. Please ask a financial question about SEC filings.",
  "reason": "prompt_injection_detected"
}
```

### Health check
```bash
curl http://localhost:8000/health
```

### Run evaluation
```bash
python evaluation.py                   # full RAGAS + Giskard
python evaluation.py --ragas-only      # RAGAS metrics only
python evaluation.py --giskard-only    # Giskard vulnerability scan only
```

---

## Project Structure

```
├── ingestion.py       # PDF parsing, chunking, table extraction
├── embedding.py       # BGE-large embeddings → Qdrant + Postgres
├── retrieval.py       # Hybrid search, RRF, cross-encoder re-ranking
├── generation.py      # Multi-provider LLM answer generation
├── guardrails.py      # Input/output safety checks
├── app.py             # FastAPI server (POST /query, GET /health)
├── evaluation.py      # RAGAS + Giskard evaluation pipeline
└── .env               # API keys and provider config (not committed)
```

---

## Design Decisions

- **pdfplumber for tables, unstructured for text** — each library excels at its domain; using both avoids the quality/speed trade-off of a single tool
- **Table flattening** — rows stored as prose-like chunks so BM25 and dense search both work without a separate table index
- **Route-aware re-ranking** — table queries get a +3.0 cross-encoder score boost to surface structured data over narrative text
- **Free-tier LLMs first** — Groq (daily reset) > HuggingFace (monthly quota) > paid APIs; Ollama for zero-cost local dev
- **Rule-based guardrails** — no LLM calls for safety checks keeps latency under 50ms and avoids cascading API failures

---

## Scale

Designed for nightly batch ingestion of ~1000 PDFs (80–200 pages each) within a 6-hour window (~21s/PDF). Parallelization is built into the ingestion pipeline.

---

*Author: Kshitiz Tiwari — March–April 2026*
