"""
embedding.py — Embed RAG chunks and store in Qdrant; hydrate Postgres tables with SLA metadata.

Model  : BAAI/bge-large-en-v1.5  (1024-dim, free, SOTA on financial text retrieval)
Stores : Qdrant  http://localhost:6333  — vectors + full payload per chunk
         Postgres (pdf_tables DB)       — SLA metadata columns added to extracted_tables

Reads  : output/<stem>.json   (produced by ingestion.py)
         sla_registry.json    (repo root — provider-supplied metadata, one entry per PDF)

SLA Registry format (sla_registry.json at repo root):
  {
    "_schema_version": "1.0",
    "documents": {
      "1234567::10-K::2025-09-27": {
        "filename":          "AAPL_10-K_2025-09-27_1234567.pdf",
        "company_name":      "Apple Inc.",
        "ticker":            "AAPL",
        "cik":               "1234567",
        "document_type":     "10-K",
        "fiscal_year":       2025,
        "fiscal_quarter":    null,
        "filing_date":       "2025-11-01",
        "period_of_report":  "2025-09-27",
        "exchange":          "NASDAQ",
        "sic_code":          "3571"
      }
    }
  }
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import time
import requests
import numpy as np
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).resolve().parent
OUTPUT_DIR      = BASE_DIR / "output"
SLA_REGISTRY    = BASE_DIR / "sla_registry.json"   # input, not inside output/

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "financial_docs"

DB_CONFIG = dict(
    host     = "localhost",
    database = "pdf_tables",
    user     = os.getenv("DB_USER", os.getenv("USER", "postgres")),
    password = os.getenv("DB_PASSWORD", ""),
)

EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
EMBED_DIM       = 1024
EMBED_BATCH     = 32    # texts per HF API call (keep low to avoid timeouts)
UPSERT_BATCH    = 256   # points per Qdrant upsert call

# HuggingFace Inference API — free tier, no local download needed.
# Set HF_TOKEN env var: https://huggingface.co/settings/tokens
# Free tier limit: ~1000 requests/day. For production (1000 PDFs/night),
# switch to local: set USE_LOCAL_MODEL=1 to load the model instead.
HF_TOKEN        = os.getenv("HF_TOKEN", "")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "0") == "1"

# BGE: document side needs NO prefix at index time.
# At QUERY time use: "Represent this sentence for searching relevant passages: <query>"


# ---------------------------------------------------------------------------
# SLA Metadata Schema  (10 fields guaranteed by provider agreement)
# ---------------------------------------------------------------------------

@dataclass
class SLAMetadata:
    """
    Provider-guaranteed metadata for every financial filing.
    These 10 fields are the SLA: the provider must supply all of them.
    Stored as Qdrant payload so every query can pre-filter without touching vectors.
    """
    company_name:     str            # "Apple Inc."
    ticker:           str            # "AAPL"
    cik:              str            # "0000320193"  (SEC EDGAR Central Index Key)
    document_type:    str            # "10-K" | "10-Q" | "8-K" | "DEF 14A" | ...
    fiscal_year:      int            # 2023
    fiscal_quarter:   Optional[str]  # "Q1" | "Q2" | "Q3" | "Q4" | null (10-K/8-K)
    filing_date:      str            # "2023-11-03"  ISO-8601 date filed with SEC
    period_of_report: str            # "2023-09-30"  ISO-8601 end of covered period
    exchange:         str            # "NASDAQ" | "NYSE" | "OTC" | ...
    sic_code:         str            # "3571"  Standard Industrial Classification


def load_sla_registry(registry_path: Path = SLA_REGISTRY) -> dict[str, SLAMetadata]:
    """
    Load provider SLA metadata from sla_registry.json (lives at repo root, not output/).

    Supports two key formats:
      Flat   (keyed by filename):           {"aapl_10k_2023.pdf": {...}}
      Stable (keyed by cik::type::period):  {"_schema_version": "1.0",
                                             "documents": {"0000320193::10-K::2023-09-30": {...}}}

    Always returns: {source_filename -> SLAMetadata}
    """
    if not registry_path.exists():
        print(
            f"[WARN] sla_registry.json not found at {registry_path}.\n"
            "       SLA metadata fields will be empty. Create the file per the docstring."
        )
        return {}

    with open(registry_path) as f:
        raw = json.load(f)

    # Detect stable-key format
    entries = raw.get("documents", raw) if isinstance(raw, dict) else {}

    registry: dict[str, SLAMetadata] = {}
    for key, meta in entries.items():
        if key == "_schema_version":
            continue
        filename = meta.get("filename", key)   # fall back to using the key as filename
        registry[filename] = SLAMetadata(
            company_name     = meta.get("company_name", ""),
            ticker           = meta.get("ticker", ""),
            cik              = meta.get("cik", ""),
            document_type    = meta.get("document_type", ""),
            fiscal_year      = int(meta.get("fiscal_year", 0)),
            fiscal_quarter   = meta.get("fiscal_quarter"),
            filing_date      = meta.get("filing_date", ""),
            period_of_report = meta.get("period_of_report", ""),
            exchange         = meta.get("exchange", ""),
            sic_code         = meta.get("sic_code", ""),
        )
    return registry


# ---------------------------------------------------------------------------
# Qdrant — collection + payload indexes
# ---------------------------------------------------------------------------

# Fields we index for fast payload filtering at query time.
# Keyword = exact match / filter; Integer = range queries on fiscal_year.
PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "ticker":         PayloadSchemaType.KEYWORD,
    "company_name":   PayloadSchemaType.KEYWORD,
    "document_type":  PayloadSchemaType.KEYWORD,
    "fiscal_year":    PayloadSchemaType.INTEGER,
    "fiscal_quarter": PayloadSchemaType.KEYWORD,
    "exchange":       PayloadSchemaType.KEYWORD,
    "sic_code":       PayloadSchemaType.KEYWORD,
    "cik":            PayloadSchemaType.KEYWORD,
}


def ensure_collection(client: QdrantClient) -> None:
    """Create collection + payload indexes if not already present."""
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}' (dim={EMBED_DIM}, cosine)")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists — will upsert")

    for field, schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # index already exists; qdrant-client raises on duplicate


# ---------------------------------------------------------------------------
# Stable point ID
# ---------------------------------------------------------------------------

def make_point_id(source: str, chunk_index: int, text: str) -> str:
    """
    Deterministic UUID from (source, chunk_index, first-64-chars-of-text).
    Idempotent: re-running embedding.py upserts rather than duplicates.
    """
    key = f"{source}::{chunk_index}::{text[:64]}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


# ---------------------------------------------------------------------------
# Embedding — HuggingFace Inference API (default) or local model (production)
# ---------------------------------------------------------------------------

def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


def build_embedder():
    """
    Returns a callable: embed_fn(texts: list[str]) -> np.ndarray (N, 1024).

    Development  (default): HuggingFace Serverless Inference API — no download.
    Production            : local SentenceTransformer — set USE_LOCAL_MODEL=1.
    """
    if USE_LOCAL_MODEL:
        from sentence_transformers import SentenceTransformer
        local_model = SentenceTransformer(EMBED_MODEL)
        print(f"Using local model: {EMBED_MODEL}")

        def _local(texts: list[str]) -> np.ndarray:
            return local_model.encode(
                texts, batch_size=EMBED_BATCH,
                show_progress_bar=False, normalize_embeddings=True,
            )
        return _local

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN env var not set.\n"
            "Get a free token at https://huggingface.co/settings/tokens\n"
            "then: export HF_TOKEN=hf_..."
        )

    # Free Serverless Inference API (api-inference.huggingface.co)
    # This bypasses the paid Inference Providers router.
    api_url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    print(f"Using HuggingFace Serverless Inference API: {EMBED_MODEL}")

    def _api(texts: list[str]) -> np.ndarray:
        all_vecs = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            while True:
                resp = requests.post(
                    api_url, headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=60,
                )
                if resp.status_code == 503:
                    # Model loading — HF asks us to wait
                    wait = resp.json().get("estimated_time", 20)
                    tqdm.write(f"  Model loading, waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            vecs = np.array(resp.json(), dtype=np.float32)
            # Some models return (N, seq_len, dim) — mean-pool to (N, dim)
            if vecs.ndim == 3:
                vecs = vecs.mean(axis=1)
            all_vecs.append(vecs)
        return _normalize(np.vstack(all_vecs))

    return _api


def embed_texts(embed_fn, texts: list[str]) -> np.ndarray:
    return embed_fn(texts)


# ---------------------------------------------------------------------------
# Per-file upload
# ---------------------------------------------------------------------------

def upload_file(
    json_path: Path,
    embed_fn,
    client: QdrantClient,
    sla_registry: dict[str, SLAMetadata],
) -> int:
    """
    Embed + upload all chunks from one ingestion JSON.
    Returns number of points upserted.

    Payload per point:
      ├─ SLA fields (10)     — company_name, ticker, cik, document_type,
      │                         fiscal_year, fiscal_quarter, filing_date,
      │                         period_of_report, exchange, sic_code
      ├─ Chunk fields (4)    — source_file, chunk_index, page, breadcrumb
      ├─ Derived field (1)   — chunk_char_len
      └─ Raw text (1)        — text  (for display / re-ranking without DB lookup)
    """
    with open(json_path) as f:
        data = json.load(f)

    source = data["metadata"]["source"]
    chunks = data.get("chunks", [])
    if not chunks:
        tqdm.write(f"  SKIP {source}: no chunks")
        return 0

    sla      = sla_registry.get(source)
    sla_dict = asdict(sla) if sla else {}

    texts   = [c["text"] for c in chunks]
    vectors = embed_texts(embed_fn, texts)   # shape (N, 1024)

    points: list[PointStruct] = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload = {
            # SLA metadata (10 provider-guaranteed fields)
            **sla_dict,
            # Chunk provenance
            "source_file":    source,
            "chunk_index":    i,
            "page":           chunk.get("page"),
            "breadcrumb":     chunk.get("breadcrumb", ""),
            "chunk_char_len": len(chunk["text"]),
            # Raw text stored in payload — avoids a second lookup at serve time
            "text":           chunk["text"],
        }
        points.append(
            PointStruct(
                id      = make_point_id(source, i, chunk["text"]),
                vector  = vec.tolist(),
                payload = payload,
            )
        )

    # Upsert in batches
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + UPSERT_BATCH],
        )

    return len(points)


# ---------------------------------------------------------------------------
# Postgres — add SLA columns + hydrate extracted_tables
# ---------------------------------------------------------------------------

# The 10 SLA columns we add to extracted_tables.
# TEXT for all strings; INT for fiscal_year.
_SLA_COLUMNS: list[tuple[str, str]] = [
    ("company_name",     "TEXT"),
    ("ticker",           "TEXT"),
    ("cik",              "TEXT"),
    ("document_type",    "TEXT"),
    ("fiscal_year",      "INT"),
    ("fiscal_quarter",   "TEXT"),
    ("filing_date",      "TEXT"),
    ("period_of_report", "TEXT"),
    ("exchange",         "TEXT"),
    ("sic_code",         "TEXT"),
]


def ensure_sla_columns(cursor) -> None:
    """Add SLA columns to extracted_tables if they don't already exist."""
    for col, col_type in _SLA_COLUMNS:
        cursor.execute(f"""
            ALTER TABLE extracted_tables
            ADD COLUMN IF NOT EXISTS {col} {col_type}
        """)


def hydrate_postgres(sla_registry: dict[str, SLAMetadata]) -> None:
    """
    For every source file in the registry, UPDATE extracted_tables rows
    that belong to that source with the 10 SLA metadata fields.

    Safe to re-run: UPDATE is idempotent.
    Skips sources that have no rows in extracted_tables (no tables were extracted).
    """
    if not sla_registry:
        print("[WARN] SLA registry empty — skipping Postgres hydration.")
        return

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    ensure_sla_columns(cursor)
    conn.commit()

    col_names = [col for col, _ in _SLA_COLUMNS]
    set_clause = ", ".join(f"{col} = %s" for col in col_names)

    updated_total = 0
    for filename, sla in sla_registry.items():
        values = [
            sla.company_name, sla.ticker, sla.cik, sla.document_type,
            sla.fiscal_year, sla.fiscal_quarter, sla.filing_date,
            sla.period_of_report, sla.exchange, sla.sic_code,
            filename,  # for WHERE clause
        ]
        cursor.execute(
            f"UPDATE extracted_tables SET {set_clause} WHERE source = %s",
            values,
        )
        updated_total += cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Postgres: updated {updated_total} table rows with SLA metadata.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    embed_fn = build_embedder()

    print(f"Connecting to Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client)

    sla_registry = load_sla_registry()
    print(f"SLA registry: {len(sla_registry)} entries\n")

    json_files = sorted(
        p for p in OUTPUT_DIR.glob("*.json")
        if p.name not in ("manifest.json", "sla_registry.json")
    )
    print(f"Found {len(json_files)} ingestion JSON files to embed\n")

    total = 0
    for json_path in tqdm(json_files, desc="Embedding & uploading"):
        n = upload_file(json_path, embed_fn, client, sla_registry)
        total += n
        tqdm.write(f"  {json_path.stem}: {n} points")

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nDone. Uploaded {total} points this run.")
    print(f"Collection '{COLLECTION_NAME}' now has {info.points_count} points total.")

    print("\nHydrating Postgres extracted_tables with SLA metadata...")
    hydrate_postgres(sla_registry)


# ---------------------------------------------------------------------------
# Query helper (for testing / dev use)
# ---------------------------------------------------------------------------

def search(
    query: str,
    ticker: str | None = None,
    document_type: str | None = None,
    fiscal_year: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Example retrieval: semantic search with optional metadata pre-filtering.

    At query time BGE expects the prefix:
      "Represent this sentence for searching relevant passages: <query>"

    Usage:
      results = search("revenue recognition policy", ticker="AAPL",
                       document_type="10-K", fiscal_year=2023)
    """
    embed_fn = build_embedder()
    client   = QdrantClient(url=QDRANT_URL)

    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vec = embed_fn([prefixed])[0].tolist()

    # Build payload filter from whichever fields are specified
    conditions = []
    if ticker:
        conditions.append(FieldCondition(key="ticker",        match=MatchValue(value=ticker)))
    if document_type:
        conditions.append(FieldCondition(key="document_type", match=MatchValue(value=document_type)))
    if fiscal_year:
        conditions.append(FieldCondition(key="fiscal_year",   match=MatchValue(value=fiscal_year)))

    query_filter = Filter(must=conditions) if conditions else None

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score":      h.score,
            "company":    h.payload.get("company_name"),
            "ticker":     h.payload.get("ticker"),
            "doc_type":   h.payload.get("document_type"),
            "year":       h.payload.get("fiscal_year"),
            "page":       h.payload.get("page"),
            "breadcrumb": h.payload.get("breadcrumb"),
            "text":       h.payload.get("text", "")[:300],
        }
        for h in hits
    ]


if __name__ == "__main__":
    main()
