"""
retrieval.py — Query-time pipeline: hybrid search + re-ranking + query routing.

Pipeline per query:
  1. Route      → classify as NARRATIVE / TABLE / HYBRID (observability only)
  2. Filter     → extract metadata constraints (ticker, year, doc type)
  3. Dense      → Qdrant vector search with BGE embeddings
  4. Sparse     → BM25 over filtered Qdrant payload texts
  5. Fuse       → Reciprocal Rank Fusion (RRF) of dense + sparse results
  6. Re-rank    → cross-encoder scores each (query, chunk) pair precisely
  7. Return     → RetrievalResult with ranked chunks (narrative + table, unified)

Both narrative text and flattened table text live as chunks in Qdrant.
Tables are no longer queried from Postgres at retrieval time.

Install deps:
  pip install rank_bm25 sentence_transformers qdrant-client

Usage:
  from retrieval import retrieve
  result = retrieve("What was Apple's revenue in 2025?", ticker="AAPL")
  for chunk in result.chunks:
      print(chunk.score, chunk.breadcrumb, chunk.text[:200])
  for table in result.tables:
      print(table.page_number, table.table)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Config  (mirrors embedding.py)
# ---------------------------------------------------------------------------

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "financial_docs"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to gather before re-ranking
DENSE_CANDIDATES  = 20
SPARSE_CANDIDATES = 20
RRF_K             = 60    # standard RRF constant
TOP_K_DEFAULT     = 5     # final chunks returned after re-ranking

# BGE query prefix — required at query time, NOT at index time
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# 1. Query Route
# ---------------------------------------------------------------------------

class QueryRoute(Enum):
    NARRATIVE = "narrative"   # semantic text search only (Qdrant)
    TABLE     = "table"       # structured numbers only  (Postgres)
    HYBRID    = "hybrid"      # both stores


# Signals that indicate numerical/table content is needed
_TABLE_PATTERNS = [
    r"\b(revenue|sales|earnings|eps|ebitda|net income|gross profit|operating income)\b",
    r"\b(margin|ratio|growth|decline|increase|decrease|change)\b",
    r"\b(how much|how many|what was|what were|total|amount|figure|number)\b",
    r"\b(compare|vs\.?|versus|year.over.year|yoy|quarter.over.quarter|qoq)\b",
    r"[\$\%]|\b(billion|million|thousand)\b",
]

# Signals that indicate narrative/context content is needed
_NARRATIVE_PATTERNS = [
    r"\b(risk|strategy|outlook|plan|vision|mission|roadmap)\b",
    r"\b(explain|describe|discuss|why|how does|what is the reason)\b",
    r"\b(management|ceo|board|policy|regulation|litigation|lawsuit)\b",
    r"\b(competitive|market|industry|sector|landscape)\b",
    r"\b(overview|summary|background|context|history)\b",
]


def route_query(query: str) -> QueryRoute:
    """
    Classify query into NARRATIVE / TABLE / HYBRID based on signal patterns.
    Both signals present → HYBRID (most informative).
    """
    q = query.lower()
    table_hits     = sum(1 for p in _TABLE_PATTERNS     if re.search(p, q))
    narrative_hits = sum(1 for p in _NARRATIVE_PATTERNS if re.search(p, q))

    if table_hits >= 2 and narrative_hits == 0:
        return QueryRoute.TABLE
    if table_hits >= 1 and narrative_hits >= 1:
        return QueryRoute.HYBRID
    if table_hits >= 1:
        return QueryRoute.HYBRID   # when in doubt, include tables
    return QueryRoute.NARRATIVE


# ---------------------------------------------------------------------------
# 2. Filter Spec
# ---------------------------------------------------------------------------

@dataclass
class FilterSpec:
    """Metadata constraints extracted from the query or passed explicitly."""
    ticker:         Optional[str] = None
    company_name:   Optional[str] = None
    document_type:  Optional[str] = None
    fiscal_year:    Optional[int] = None
    fiscal_quarter: Optional[str] = None
    cik:            Optional[str] = None


def extract_filters(query: str, explicit: FilterSpec) -> FilterSpec:
    """
    Auto-extract fiscal year and document type from query text.
    Explicit filters (passed by caller) always take priority.
    """
    q = query.lower()

    if explicit.fiscal_year is None:
        match = re.search(r"\b(20\d{2})\b", query)
        if match:
            explicit.fiscal_year = int(match.group(1))

    if explicit.document_type is None:
        if re.search(r"\b10-?k\b", q):
            explicit.document_type = "10-K"
        elif re.search(r"\b10-?q\b", q):
            explicit.document_type = "10-Q"

    if explicit.fiscal_quarter is None:
        match = re.search(r"\b(q[1-4])\b", q)
        if match:
            explicit.fiscal_quarter = match.group(1).upper()

    return explicit


def _build_qdrant_filter(spec: FilterSpec) -> Optional[Filter]:
    conditions = []
    if spec.ticker:
        conditions.append(FieldCondition(key="ticker",         match=MatchValue(value=spec.ticker)))
    if spec.company_name:
        conditions.append(FieldCondition(key="company_name",   match=MatchValue(value=spec.company_name)))
    if spec.document_type:
        conditions.append(FieldCondition(key="document_type",  match=MatchValue(value=spec.document_type)))
    if spec.fiscal_year:
        conditions.append(FieldCondition(key="fiscal_year",    match=MatchValue(value=spec.fiscal_year)))
    if spec.fiscal_quarter:
        conditions.append(FieldCondition(key="fiscal_quarter", match=MatchValue(value=spec.fiscal_quarter)))
    if spec.cik:
        conditions.append(FieldCondition(key="cik",            match=MatchValue(value=spec.cik)))
    return Filter(must=conditions) if conditions else None


# ---------------------------------------------------------------------------
# 3. Dense Search  (Qdrant vector search)
# ---------------------------------------------------------------------------

def _dense_search(
    query: str,
    qdrant_filter: Optional[Filter],
    top_k: int,
) -> list[dict]:
    """
    Embed the query with BGE prefix, search Qdrant.
    Returns list of {payload, score, rank}.
    """
    prefixed = BGE_QUERY_PREFIX + query
    vec = _embed_fn([prefixed])[0].tolist()

    result = _qdrant.query_points(
        collection_name = COLLECTION_NAME,
        query           = vec,
        query_filter    = qdrant_filter,
        limit           = top_k,
        with_payload    = True,
    )
    hits = result.points
    return [{"payload": h.payload, "score": h.score, "rank": i + 1}
            for i, h in enumerate(hits)]


# ---------------------------------------------------------------------------
# 4. Sparse Search  (BM25 over Qdrant payload texts)
# ---------------------------------------------------------------------------

def _bm25_search(
    query: str,
    qdrant_filter: Optional[Filter],
    top_k: int,
) -> list[dict]:
    """
    Scroll all matching points from Qdrant → build BM25 index in memory → rank.

    Scale note: fine for dev and up to ~50k chunks. For 200k+ chunks, switch to
    Qdrant sparse vectors (SPLADE/BM25 encoded at index time via embedding.py).
    """
    all_points = []
    offset = None
    while True:
        batch, offset = _qdrant.scroll(
            collection_name = COLLECTION_NAME,
            scroll_filter   = qdrant_filter,
            limit           = 100,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
        )
        all_points.extend(batch)
        if offset is None:
            break

    if not all_points:
        return []

    corpus    = [p.payload.get("text", "") for p in all_points]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(query.lower().split())

    ranked = sorted(
        zip(scores, all_points), key=lambda x: x[0], reverse=True
    )[:top_k]

    return [{"payload": p.payload, "score": float(s), "rank": i + 1}
            for i, (s, p) in enumerate(ranked)]


# ---------------------------------------------------------------------------
# 5. RRF Fusion
# ---------------------------------------------------------------------------

def _rrf_fusion(
    dense:  list[dict],
    sparse: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion: score(d) = Σ  1 / (k + rank_i(d))

    Deduplicates by text prefix (first 128 chars).
    Higher score = appeared high in more lists.
    """
    scores:   dict[str, float] = {}
    payloads: dict[str, dict]  = {}

    for result_list in (dense, sparse):
        for r in result_list:
            key           = r["payload"].get("text", "")[:128]
            scores[key]   = scores.get(key, 0.0) + 1.0 / (k + r["rank"])
            payloads[key] = r["payload"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"payload": payloads[key], "rrf_score": score}
            for key, score in ranked]


# ---------------------------------------------------------------------------
# 6. Cross-Encoder Re-Ranking
# ---------------------------------------------------------------------------

def _rerank(
    query:      str,
    candidates: list[dict],
    top_k:      int,
) -> list[dict]:
    """
    Score each (query, passage) pair with the cross-encoder.
    Cross-encoders use full attention between query and passage —
    much more precise than bi-encoder cosine similarity, but slower.
    Only applied to the top candidates after RRF (typically 20-40).
    """
    if not candidates:
        return []

    pairs  = [(query, c["payload"].get("text", "")) for c in candidates]
    scores = _reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates), key=lambda x: float(x[0]), reverse=True
    )[:top_k]

    return [{"payload": c["payload"], "rerank_score": float(s)}
            for s, c in ranked]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    score:         float          # cross-encoder relevance score
    text:          str            # raw chunk text (narrative or flattened table)
    breadcrumb:    str            # section path or "TABLE > page N"
    page:          Optional[int]  # page number in source PDF
    source:        str            # source filename
    ticker:        str
    fiscal_year:   int
    document_type: str


@dataclass
class RetrievalResult:
    query:  str
    route:  QueryRoute
    chunks: list[ChunkResult]     # re-ranked chunks (narrative + table, unified)


# ---------------------------------------------------------------------------
# 8. Main retrieve() function
# ---------------------------------------------------------------------------

def retrieve(
    query:         str,
    ticker:        Optional[str] = None,
    company_name:  Optional[str] = None,
    document_type: Optional[str] = None,
    fiscal_year:   Optional[int] = None,
    fiscal_quarter: Optional[str] = None,
    cik:           Optional[str] = None,
    top_k:         int = TOP_K_DEFAULT,
) -> RetrievalResult:
    """
    Full retrieval pipeline. All metadata args are optional — auto-extracted
    from query text when not provided.

    Args:
        query:          Natural language question
        ticker:         "AAPL", "MSFT", etc.
        document_type:  "10-K", "10-Q", "8-K"
        fiscal_year:    2025
        fiscal_quarter: "Q1", "Q2", "Q3", "Q4"
        top_k:          Number of chunks to return (after re-ranking)

    Returns:
        RetrievalResult with .chunks (narrative + table chunks, unified ranking)

    Example:
        result = retrieve(
            "What were Apple's main risk factors?",
            ticker="AAPL", document_type="10-K", fiscal_year=2025
        )
    """
    # Step 1 — route (kept for observability, all routes search Qdrant)
    route = route_query(query)

    # Step 2 — build filter (explicit overrides auto-extracted)
    spec = extract_filters(
        query,
        FilterSpec(
            ticker         = ticker,
            company_name   = company_name,
            document_type  = document_type,
            fiscal_year    = fiscal_year,
            fiscal_quarter = fiscal_quarter,
            cik            = cik,
        ),
    )

    # Steps 3-6 — vector + BM25 + RRF + re-rank
    # Both narrative and table chunks live in Qdrant, searched together.
    qdrant_filter = _build_qdrant_filter(spec)

    dense  = _dense_search(query, qdrant_filter, DENSE_CANDIDATES)
    sparse = _bm25_search( query, qdrant_filter, SPARSE_CANDIDATES)
    fused  = _rrf_fusion(dense, sparse)
    ranked = _rerank(query, fused, top_k)

    chunks = [
        ChunkResult(
            score         = r["rerank_score"],
            text          = r["payload"].get("text", ""),
            breadcrumb    = r["payload"].get("breadcrumb", ""),
            page          = r["payload"].get("page"),
            source        = r["payload"].get("source_file", ""),
            ticker        = r["payload"].get("ticker", ""),
            fiscal_year   = r["payload"].get("fiscal_year", 0),
            document_type = r["payload"].get("document_type", ""),
        )
        for r in ranked
    ]

    return RetrievalResult(query=query, route=route, chunks=chunks)


# ---------------------------------------------------------------------------
# Module-level singletons  (loaded once on import)
# ---------------------------------------------------------------------------

def _build_embed_fn():
    """Load BGE locally — model is already cached from embedding.py run."""
    model = SentenceTransformer(EMBED_MODEL)
    print(f"[retrieval] Loaded embedder: {EMBED_MODEL}")

    def _fn(texts: list[str]) -> np.ndarray:
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vecs

    return _fn


print("[retrieval] Loading models...")
_embed_fn  = _build_embed_fn()
_reranker  = CrossEncoder(RERANK_MODEL)
_qdrant    = QdrantClient(url=QDRANT_URL)
print(f"[retrieval] Loaded reranker: {RERANK_MODEL}")
print("[retrieval] Ready.\n")


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    test_queries = [
        # (query, explicit filters)
        ("What were Apple's main risk factors in the 2025 annual report?",
         dict(ticker="AAPL", document_type="10-K", fiscal_year=2025)),

        ("What was Apple's total net sales revenue?",
         dict(ticker="AAPL", fiscal_year=2025)),

        ("Describe Apple's business overview and competitive strategy",
         dict(ticker="AAPL")),
    ]

    for query, filters in test_queries:
        print("=" * 70)
        print(f"QUERY : {query}")
        result = retrieve(query, **filters)
        print(f"ROUTE : {result.route.value}")
        print(f"CHUNKS: {len(result.chunks)}")

        for i, chunk in enumerate(result.chunks, 1):
            is_table = chunk.breadcrumb.startswith("TABLE")
            tag = "[TABLE]" if is_table else "[TEXT]"
            print(f"\n  [{i}] {tag}  score={chunk.score:.4f}  page={chunk.page}")
            print(f"      {chunk.breadcrumb}")
            print(f"      {chunk.text}")

        print()
