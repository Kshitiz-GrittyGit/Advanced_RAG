"""
generation.py — LLM answer generation over retrieved RAG chunks.

LLM providers (set LLM_PROVIDER in .env):
  ollama  (default) — Mistral 7B running locally via Ollama.
                      Free, no API key, OpenAI-compatible server at localhost:11434.
                      Setup: brew install ollama && ollama pull mistral
                      Use this for development.

  bedrock           — Meta Llama 3.1 70B Instruct via AWS Bedrock Converse API.
                      Requires AWS credentials (aws configure or IAM role).
                      Use this for AWS production.

Env vars:
  LLM_PROVIDER   = ollama | bedrock    (default: ollama)
  OLLAMA_URL     = http://localhost:11434   (default)
  OLLAMA_MODEL   = mistral             (default, any model you have pulled)
  AWS_REGION     = us-east-1           (bedrock only, default: us-east-1)

Usage:
  from retrieval import retrieve
  from generation import generate

  result   = retrieve("What was Apple's revenue in 2025?", ticker="AAPL")
  response = generate("What was Apple's revenue in 2025?", result)
  print(response.answer)
  for src in response.sources:
      print(src)
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from dataclasses import dataclass

import requests

from retrieval import RetrievalResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "ollama")    # "ollama" | "bedrock" | "anthropic" | "huggingface" | "groq"
AWS_REGION    = os.getenv("AWS_REGION", "us-east-1")

# Ollama — local dev
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "mistral")

# Bedrock — AWS prod
BEDROCK_LLM_MODEL = "meta.llama3-1-70b-instruct-v1:0"

# Anthropic
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL     = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# HuggingFace Inference API — free tier, open-source models
HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_MODEL    = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# Groq — free, fast, daily reset (better than HF monthly quota)
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_NEW_TOKENS = 1024
TEMPERATURE    = 0.1
TOP_P          = 0.9

# Top-K chunks fed into the context window per query
CONTEXT_CHUNKS = 10


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class AnswerResponse:
    answer:     str
    sources:    list[str]
    latency_ms: int


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a financial analyst assistant that answers questions about SEC filings \
(10-K and 10-Q reports).

Rules:
1. Answer ONLY using the document excerpts provided. Do not use outside knowledge.
2. Quote financial figures exactly as they appear in the source.
3. If the context does not contain enough information, respond with exactly:
   "I cannot find this information in the provided documents."
4. Be concise and precise. Do not pad with filler phrases.
5. When referencing data, mention the section or page it came from.\
"""


def _format_chunk(i: int, chunk) -> str:
    kind   = "TABLE DATA" if chunk.breadcrumb.startswith("TABLE") else "TEXT EXCERPT"
    header = (
        f"[{i}] {kind} | "
        f"{chunk.ticker} {chunk.document_type} {chunk.fiscal_year} | "
        f"Page {chunk.page} | {chunk.breadcrumb}"
    )
    return f"{header}\n{chunk.text.strip()}"


def _build_messages(query: str, result: RetrievalResult) -> tuple[str, str]:
    chunks   = result.chunks[:CONTEXT_CHUNKS]
    context  = "\n\n---\n\n".join(_format_chunk(i + 1, c) for i, c in enumerate(chunks))
    user_msg = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"
    return SYSTEM_PROMPT, user_msg


def _extract_sources(result: RetrievalResult) -> list[str]:
    seen, sources = set(), []
    for chunk in result.chunks[:CONTEXT_CHUNKS]:
        key = (chunk.ticker, chunk.document_type, chunk.fiscal_year, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append(
                f"{chunk.ticker} {chunk.document_type} {chunk.fiscal_year}, "
                f"page {chunk.page} — {chunk.breadcrumb}"
            )
    return sources


# ---------------------------------------------------------------------------
# Ollama provider  (development)
# ---------------------------------------------------------------------------

def _call_ollama(system_prompt: str, user_message: str) -> str:
    """
    Calls a locally running Ollama instance using its OpenAI-compatible
    /v1/chat/completions endpoint. Any model you have pulled works here.

    Start Ollama: ollama serve  (usually auto-starts on Mac after install)
    Pull a model: ollama pull mistral
    """
    url     = f"{OLLAMA_URL}/v1/chat/completions"
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": TEMPERATURE,
        "top_p":       TOP_P,
        "stream":      False,
    }

    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama model '{OLLAMA_MODEL}' not found. "
            f"Run: ollama pull {OLLAMA_MODEL}"
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# AWS Bedrock provider  (production)
# ---------------------------------------------------------------------------

def _call_bedrock(system_prompt: str, user_message: str) -> str:
    """
    Calls AWS Bedrock via the model-agnostic Converse API.
    Swap BEDROCK_LLM_MODEL to change models with no other code changes.

    Requires:
      - pip install boto3
      - AWS credentials configured (aws configure or IAM role on EC2/ECS)
      - Bedrock model access enabled in your AWS account
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 not installed. Run: pip install boto3")

    client   = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    response = client.converse(
        modelId         = BEDROCK_LLM_MODEL,
        system          = [{"text": system_prompt}],
        messages        = [{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig = {
            "maxTokens":   MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "topP":        TOP_P,
        },
    )
    return response["output"]["message"]["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# HuggingFace Inference API provider  (free, open-source)
# ---------------------------------------------------------------------------

def _call_huggingface(system_prompt: str, user_message: str) -> str:
    """
    Calls HuggingFace's free Inference API via huggingface_hub.
    Free tier: no credit card needed, generous rate limits.
    Default model: Qwen2.5-72B-Instruct (high quality, free).
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HF_TOKEN)
    resp = client.chat.completions.create(
        model      = HF_MODEL,
        messages   = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens  = MAX_NEW_TOKENS,
        temperature = TEMPERATURE,
        top_p       = TOP_P,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Groq provider  (free, fast, daily reset)
# ---------------------------------------------------------------------------

def _call_groq(system_prompt: str, user_message: str) -> str:
    """
    Calls Groq's free inference API. Daily rate limits reset every 24 hours —
    much more practical than HuggingFace's monthly quota for eval workloads.
    Default model: llama-3.3-70b-versatile.
    Get a free key at: console.groq.com
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq not installed. Run: pip install groq")

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature = TEMPERATURE,
        max_tokens  = MAX_NEW_TOKENS,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

def _call_anthropic(system_prompt: str, user_message: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model      = ANTHROPIC_MODEL,
        max_tokens = MAX_NEW_TOKENS,
        system     = system_prompt,
        messages   = [{"role": "user", "content": user_message}],
    )
    return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main generate() function
# ---------------------------------------------------------------------------

def generate(query: str, result: RetrievalResult) -> AnswerResponse:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        query:   The original user question.
        result:  RetrievalResult from retrieval.retrieve().

    Returns:
        AnswerResponse(.answer, .sources, .latency_ms)
    """
    if not result.chunks:
        return AnswerResponse(
            answer     = "I cannot find this information in the provided documents.",
            sources    = [],
            latency_ms = 0,
        )

    system_prompt, user_message = _build_messages(query, result)
    sources = _extract_sources(result)

    t0 = time.perf_counter()
    if LLM_PROVIDER == "bedrock":
        answer = _call_bedrock(system_prompt, user_message)
    elif LLM_PROVIDER == "anthropic":
        answer = _call_anthropic(system_prompt, user_message)
    elif LLM_PROVIDER == "huggingface":
        answer = _call_huggingface(system_prompt, user_message)
    elif LLM_PROVIDER == "groq":
        answer = _call_groq(system_prompt, user_message)
    else:
        answer = _call_ollama(system_prompt, user_message)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return AnswerResponse(answer=answer, sources=sources, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from retrieval import retrieve

    tests = [
        ("What was Apple's total net sales revenue in 2025?",
         dict(ticker="AAPL", document_type="10-K", fiscal_year=2025)),
        ("What are Apple's main risk factors?",
         dict(ticker="AAPL", document_type="10-K", fiscal_year=2025)),
        ("Describe Apple's competitive strategy",
         dict(ticker="AAPL", document_type="10-K", fiscal_year=2025)),
    ]

    for query, filters in tests:
        print("=" * 70)
        print(f"QUERY: {query}")
        result   = retrieve(query, **filters)
        response = generate(query, result)
        print(f"ANSWER ({response.latency_ms}ms):\n{response.answer}")
        print("\nSOURCES:")
        for src in response.sources:
            print(f"  • {src}")
        print()
