"""
app.py — FastAPI service wrapping the full RAG pipeline.

Endpoints:
  POST /query   — ask a question, get a grounded answer
  GET  /health  — service health check

Start locally:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Example request:
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What was Apple total revenue in 2025?",
         "ticker": "AAPL", "fiscal_year": 2025}'

pip install fastapi uvicorn
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Lazy-load heavy models once at startup, not per request
# ---------------------------------------------------------------------------

_pipeline_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load retrieval models (BGE embedder + cross-encoder + Qdrant client)
    once when the server starts. These are module-level singletons in
    retrieval.py — importing the module triggers the load.
    """
    global _pipeline_ready
    print("[startup] Loading retrieval models...")
    import retrieval   # noqa: F401  — triggers _embed_fn, _reranker, _qdrant init
    import generation  # noqa: F401  — no heavy init, but validates imports
    _pipeline_ready = True
    print("[startup] Ready.")
    yield
    # Nothing to clean up — models are in-process singletons


app = FastAPI(
    title="Financial RAG API",
    description="Retrieval-Augmented Generation over SEC 10-K / 10-Q filings.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:          str            = Field(...,  description="Natural language question")
    ticker:         Optional[str]  = Field(None, description="Stock ticker, e.g. 'AAPL'")
    company_name:   Optional[str]  = Field(None, description="Full company name")
    document_type:  Optional[str]  = Field(None, description="'10-K' or '10-Q'")
    fiscal_year:    Optional[int]  = Field(None, description="Fiscal year, e.g. 2025")
    fiscal_quarter: Optional[str]  = Field(None, description="'Q1'–'Q4' (10-Q only)")
    top_k:          int            = Field(5,    description="Number of chunks to retrieve", ge=1, le=20)


class QueryResponse(BaseModel):
    answer:      str        # Grounded answer from LLM
    sources:     list[str]  # "AAPL 10-K 2025, page 39 — ROOT > Note 2 – Revenue"
    route:       str        # "narrative" | "table" | "hybrid"
    latency_ms:  int        # Total wall-clock latency


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": _pipeline_ready}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Full RAG pipeline:
      1. Hybrid retrieval (dense + BM25 + RRF + cross-encoder re-rank)
      2. LLM generation over top-K chunks
      3. Source citations extracted from chunks
    """
    if not _pipeline_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    from retrieval import retrieve
    from generation import generate

    t0 = time.perf_counter()

    try:
        result = retrieve(
            query          = req.query,
            ticker         = req.ticker,
            company_name   = req.company_name,
            document_type  = req.document_type,
            fiscal_year    = req.fiscal_year,
            fiscal_quarter = req.fiscal_quarter,
            top_k          = req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    try:
        response = generate(req.query, result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    total_ms = int((time.perf_counter() - t0) * 1000)

    return QueryResponse(
        answer     = response.answer,
        sources    = response.sources,
        route      = result.route.value,
        latency_ms = total_ms,
    )
