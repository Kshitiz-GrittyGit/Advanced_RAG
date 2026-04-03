"""
embedding.py — Embed RAG chunks and store in Qdrant; hydrate Postgres tables with SLA metadata.

Embedding providers (set EMBED_PROVIDER in .env):
  local   (default) — SentenceTransformer BGE-large-en-v1.5, runs on your machine.
                       Model is downloaded once (~1.3GB) and cached in ~/.cache/huggingface.
                       Use this for development.

  bedrock           — Amazon Titan Embed Text v2 via AWS Bedrock (1024-dim).
                      Requires AWS credentials (aws configure or IAM role).
                      Use this for AWS production. Requires re-indexing when switching
                      from local because the two models produce different vector spaces.

Env vars:
  EMBED_PROVIDER    = local | bedrock          (default: local)
  QDRANT_URL        = http://localhost:6333     (default)
  AWS_REGION        = us-east-1                (bedrock only, default: us-east-1)
  DB_USER / DB_PASSWORD                        (postgres)

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

from dotenv import load_dotenv
load_dotenv()

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

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
SLA_REGISTRY    = BASE_DIR / "sla_registry.json"

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "financial_docs"

DB_CONFIG = dict(
    host     = "localhost",
    database = "pdf_tables",
    user     = os.getenv("DB_USER", os.getenv("USER", "postgres")),
    password = os.getenv("DB_PASSWORD", ""),
)

# Provider selection — drives which embed backend is used
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")   # "local" | "bedrock"

# Shared constants — both providers output 1024-dim vectors
EMBED_DIM    = 1024
EMBED_BATCH  = 32    # texts per batch (kept small for memory and API limits)
UPSERT_BATCH = 256   # points per Qdrant upsert call

# Local provider — BGE-large
BGE_MODEL = "BAAI/bge-large-en-v1.5"
# BGE index-time: no prefix needed.
# BGE query-time: prepend "Represent this sentence for searching relevant passages: "

# Bedrock provider — Titan Embed Text v2
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
AWS_REGION          = os.getenv("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# SLA Metadata Schema
# ---------------------------------------------------------------------------

@dataclass
class SLAMetadata:
    """
    10 provider-guaranteed metadata fields for every financial filing.
    Stored as Qdrant payload so every query can pre-filter without touching vectors.
    """
    company_name:     str
    ticker:           str
    cik:              str
    document_type:    str
    fiscal_year:      int
    fiscal_quarter:   Optional[str]
    filing_date:      str
    period_of_report: str
    exchange:         str
    sic_code:         str


def load_sla_registry(registry_path: Path = SLA_REGISTRY) -> dict[str, SLAMetadata]:
    """Load provider SLA metadata. Supports flat and stable-key formats."""
    if not registry_path.exists():
        print(f"[WARN] sla_registry.json not found at {registry_path}. SLA fields will be empty.")
        return {}

    with open(registry_path) as f:
        raw = json.load(f)

    entries = raw.get("documents", raw) if isinstance(raw, dict) else {}
    registry: dict[str, SLAMetadata] = {}

    for key, meta in entries.items():
        if key == "_schema_version":
            continue
        filename = meta.get("filename", key)
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
            pass


# ---------------------------------------------------------------------------
# Stable point ID — idempotent upserts
# ---------------------------------------------------------------------------

def make_point_id(source: str, chunk_index: int, text: str) -> str:
    key    = f"{source}::{chunk_index}::{text[:64]}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


def _build_local_embedder():
    """
    SentenceTransformer BGE-large-en-v1.5.
    Downloads ~1.3GB on first run, then cached in ~/.cache/huggingface.
    Fastest option for dev — no network calls per batch after first run.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL)
    print(f"[embed] Provider: local  |  Model: {BGE_MODEL}")

    def _embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    return _embed


def _build_bedrock_embedder():
    """
    Amazon Titan Embed Text v2 via AWS Bedrock.
    1024-dim — same as BGE-large, no Qdrant collection changes needed.
    Requires: boto3 installed, AWS credentials configured.

    NOTE: Titan Embed and BGE-large produce DIFFERENT vector spaces.
          If you switch providers, delete the Qdrant collection and re-run
          embedding.py so all vectors come from the same model.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 not installed. Run: pip install boto3")

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    print(f"[embed] Provider: bedrock  |  Model: {BEDROCK_EMBED_MODEL}  |  Region: {AWS_REGION}")

    def _embed(texts: list[str]) -> np.ndarray:
        all_vecs = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            batch_vecs = []
            for text in batch:
                response = client.invoke_model(
                    modelId     = BEDROCK_EMBED_MODEL,
                    contentType = "application/json",
                    accept      = "application/json",
                    body        = json.dumps({
                        "inputText":  text,
                        "dimensions": EMBED_DIM,
                        "normalize":  True,
                    }),
                )
                vec = json.loads(response["body"].read())["embedding"]
                batch_vecs.append(vec)
            all_vecs.append(np.array(batch_vecs, dtype=np.float32))
        return _normalize(np.vstack(all_vecs))

    return _embed


def build_embedder():
    """
    Returns embed_fn(texts: list[str]) -> np.ndarray (N, 1024).
    Provider is selected by EMBED_PROVIDER env var.
    """
    if EMBED_PROVIDER == "bedrock":
        return _build_bedrock_embedder()
    return _build_local_embedder()   # default: local


# ---------------------------------------------------------------------------
# Per-file upload
# ---------------------------------------------------------------------------

def upload_file(
    json_path: Path,
    embed_fn,
    client: QdrantClient,
    sla_registry: dict[str, SLAMetadata],
) -> int:
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
    vectors = embed_fn(texts)

    points: list[PointStruct] = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload = {
            **sla_dict,
            "source_file":    source,
            "chunk_index":    i,
            "page":           chunk.get("page"),
            "breadcrumb":     chunk.get("breadcrumb", ""),
            "chunk_char_len": len(chunk["text"]),
            "text":           chunk["text"],
        }
        points.append(PointStruct(
            id      = make_point_id(source, i, chunk["text"]),
            vector  = vec.tolist(),
            payload = payload,
        ))

    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(
            collection_name = COLLECTION_NAME,
            points          = points[start : start + UPSERT_BATCH],
        )

    return len(points)


# ---------------------------------------------------------------------------
# Postgres — SLA column hydration
# ---------------------------------------------------------------------------

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
    for col, col_type in _SLA_COLUMNS:
        cursor.execute(f"ALTER TABLE extracted_tables ADD COLUMN IF NOT EXISTS {col} {col_type}")


def hydrate_postgres(sla_registry: dict[str, SLAMetadata]) -> None:
    if not sla_registry:
        print("[WARN] SLA registry empty — skipping Postgres hydration.")
        return

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    ensure_sla_columns(cursor)
    conn.commit()

    col_names  = [col for col, _ in _SLA_COLUMNS]
    set_clause = ", ".join(f"{col} = %s" for col in col_names)
    updated    = 0

    for filename, sla in sla_registry.items():
        values = [
            sla.company_name, sla.ticker, sla.cik, sla.document_type,
            sla.fiscal_year, sla.fiscal_quarter, sla.filing_date,
            sla.period_of_report, sla.exchange, sla.sic_code,
            filename,
        ]
        cursor.execute(f"UPDATE extracted_tables SET {set_clause} WHERE source = %s", values)
        updated += cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Postgres: updated {updated} table rows with SLA metadata.")


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


if __name__ == "__main__":
    main()
