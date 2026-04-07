"""
guardrails.py — Input and output guardrails for the Financial RAG pipeline.

All checks are rule-based (no LLM calls) to keep overhead under 50ms.

INPUT GUARDRAILS  (run before retrieval):
  1. Query length          — reject too short / too long
  2. Prompt injection      — block known injection patterns
  3. Off-topic detection   — block non-financial queries

OUTPUT GUARDRAILS (run after generation):
  4. Low retrieval confidence — refuse when top chunk score is too low
  5. Answer grounding check  — flag suspiciously specific numbers not in context
  6. Financial disclaimer    — append standard disclaimer to all answers

Usage:
  from guardrails import check_input, check_output, GuardrailError

  # In app.py request handler:
  try:
      check_input(query)
  except GuardrailError as e:
      raise HTTPException(status_code=e.status_code, detail=e.message)

  result   = retrieve(query, ...)
  response = generate(query, result)

  answer, disclaimer = check_output(response.answer, result)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# GuardrailError — raised when a check fails
# ---------------------------------------------------------------------------

@dataclass
class GuardrailError(Exception):
    message:     str
    status_code: int   # HTTP status code to return
    reason:      str   # internal tag for logging/observability


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Query length bounds (characters)
QUERY_MIN_LEN = 5
QUERY_MAX_LEN = 1000

# Minimum cross-encoder score to trust the answer
# Below this → "I cannot find this information"
MIN_RETRIEVAL_SCORE = 0.0   # cross-encoder scores can be negative; 0 = minimal signal

# Minimum number of chunks that must be retrieved
MIN_CHUNKS_REQUIRED = 1

FINANCIAL_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This answer is generated from SEC filing excerpts and is "
    "provided for informational purposes only. It does not constitute financial "
    "advice. Always verify figures directly from the source filing.*"
)


# ---------------------------------------------------------------------------
# 1. Query Length Check
# ---------------------------------------------------------------------------

def _check_length(query: str) -> None:
    if len(query) < QUERY_MIN_LEN:
        raise GuardrailError(
            message     = "Query is too short. Please provide a more specific question.",
            status_code = 422,
            reason      = "query_too_short",
        )
    if len(query) > QUERY_MAX_LEN:
        raise GuardrailError(
            message     = f"Query exceeds maximum length of {QUERY_MAX_LEN} characters.",
            status_code = 422,
            reason      = "query_too_long",
        )


# ---------------------------------------------------------------------------
# 2. Prompt Injection Detection
# ---------------------------------------------------------------------------

# Patterns that indicate an attempt to hijack the system prompt or pipeline
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above|your)\s+instructions?",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+(?!apple|analyst)",  # not "act as an analyst"
    r"(system|user|assistant)\s*:\s*(?!.*\?)",  # role prefix injection
    r"<\s*(system|prompt|instruction|jailbreak)",
    r"\[?\s*(system|inst|instruction)\s*\]",
    r"disregard\s+(your\s+)?(previous|prior|all)\s+(instructions?|rules?|guidelines?)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"jailbreak",
    r"override\s+(your\s+)?(safety|instructions?|guidelines?)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    flags=re.IGNORECASE,
)


def _check_injection(query: str) -> None:
    if _INJECTION_RE.search(query):
        raise GuardrailError(
            message     = "Your query contains patterns that cannot be processed. Please ask a financial question about SEC filings.",
            status_code = 403,
            reason      = "prompt_injection_detected",
        )


# ---------------------------------------------------------------------------
# 3. Off-Topic Detection
# ---------------------------------------------------------------------------

# At least one of these must appear in the query for it to be considered
# financial/business related. This is intentionally broad — we want low
# false-positive rate (don't block valid questions), accepting some off-topic.
_FINANCIAL_SIGNALS = [
    r"\b(revenue|sales|income|earnings|profit|loss|margin|ebitda|eps)\b",
    r"\b(cost|expense|spending|budget|capital|cash|debt|equity|asset|liability)\b",
    r"\b(risk|strategy|competition|market|industry|segment|growth|decline)\b",
    r"\b(apple|aapl|company|business|fiscal|annual|quarterly|report|filing)\b",
    r"\b(10-?k|10-?q|8-?k|sec|edgar|cik|ticker)\b",
    r"\b(quarter|year|2020|2021|2022|2023|2024|2025|2026)\b",
    r"\b(stock|share|dividend|buyback|repurchase|acquisition|merger)\b",
    r"\b(employee|headcount|workforce|executive|ceo|cfo|board|officer)\b",
    r"\b(product|service|iphone|ipad|mac|watch|vision|app\s+store)\b",
    r"\b(tax|depreciation|amortization|goodwill|impairment)\b",
    r"\b(how|what|when|where|why|which|who|did|was|were|is|are)\b",  # question words
    r"\$|\bpercent\b|\bbillion\b|\bmillion\b|\bthousand\b",
]

_FINANCIAL_RE = [re.compile(p, flags=re.IGNORECASE) for p in _FINANCIAL_SIGNALS]

# Hard-block patterns — clearly off-topic regardless of other signals
_OFFTOPIC_PATTERNS = [
    r"\b(recipe|cook|food|restaurant|movie|song|music|game|sport|weather|joke)\b",
    r"\b(write\s+(me\s+)?(a\s+)?(poem|story|essay|code|song|email(?!\s+to\s+investor)))\b",
    r"\b(translate|translation)\b",
    r"\b(celebrity|actor|actress|athlete|politician(?!\s+risk))\b",
]

_OFFTOPIC_RE = re.compile(
    "|".join(_OFFTOPIC_PATTERNS),
    flags=re.IGNORECASE,
)


def _check_offtopic(query: str) -> None:
    # Hard block first
    if _OFFTOPIC_RE.search(query):
        raise GuardrailError(
            message     = "This assistant only answers questions about SEC financial filings (10-K, 10-Q). Please ask a financial or business question.",
            status_code = 422,
            reason      = "off_topic_query",
        )

    # Soft check — require at least one financial signal
    has_signal = any(p.search(query) for p in _FINANCIAL_RE)
    if not has_signal:
        raise GuardrailError(
            message     = "Your question doesn't appear to be related to financial filings. Please ask about revenue, earnings, risk factors, business strategy, or other financial topics from SEC filings.",
            status_code = 422,
            reason      = "no_financial_signal",
        )


# ---------------------------------------------------------------------------
# 4. Low Retrieval Confidence Check
# ---------------------------------------------------------------------------

def _check_retrieval_confidence(result) -> None:
    """
    Refuse if retrieval returned too few chunks or all scores are too low.
    A very low cross-encoder score means the pipeline couldn't find relevant
    content — the LLM would be forced to hallucinate or say "I don't know".
    Better to return a clear error than a weak answer.
    """
    if not result.chunks or len(result.chunks) < MIN_CHUNKS_REQUIRED:
        raise GuardrailError(
            message     = "No relevant information was found in the documents for your query. Please try rephrasing or providing more specific filters (ticker, year, document type).",
            status_code = 404,
            reason      = "no_chunks_retrieved",
        )

    top_score = result.chunks[0].score
    if top_score < MIN_RETRIEVAL_SCORE:
        raise GuardrailError(
            message     = "The retrieved content does not appear relevant enough to answer your question confidently. Please try rephrasing.",
            status_code = 422,
            reason      = f"low_retrieval_confidence_score_{top_score:.3f}",
        )


# ---------------------------------------------------------------------------
# 5. Answer Grounding Check
# ---------------------------------------------------------------------------

# Large dollar/percentage figures in answers should appear in the context.
# This catches cases where the LLM invents specific numbers.
_DOLLAR_FIGURE_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K)?",
    flags=re.IGNORECASE,
)

_SUSPICIOUS_PHRASES = [
    r"as\s+of\s+my\s+(knowledge|training|data)",
    r"based\s+on\s+(my\s+)?(general\s+)?knowledge",
    r"i\s+(believe|think|estimate|estimate)",
    r"approximately\s+\$[\d,]+",  # "approximately $X" — LLM hedging on invented number
    r"(roughly|around|about)\s+\$[\d,]+",
]
_SUSPICIOUS_RE = re.compile(
    "|".join(_SUSPICIOUS_PHRASES),
    flags=re.IGNORECASE,
)


def _check_grounding(answer: str, contexts: list[str]) -> Optional[str]:
    """
    Returns a warning string if grounding issues are detected, else None.
    Does NOT raise — grounding warnings are soft flags, not hard blocks,
    because false positives would block valid answers.
    """
    warnings = []

    # Check for suspicious hedging phrases that suggest LLM is guessing
    if _SUSPICIOUS_RE.search(answer):
        warnings.append("Answer may contain information not sourced from documents.")

    # Check that specific dollar figures in the answer appear in context
    answer_figures = set(_DOLLAR_FIGURE_RE.findall(answer))
    if answer_figures:
        all_context = " ".join(contexts)
        for fig in answer_figures:
            # Normalize: remove spaces, lowercase for comparison
            fig_norm = fig.replace(" ", "").lower()
            ctx_norm = all_context.replace(" ", "").lower()
            if fig_norm not in ctx_norm:
                warnings.append(f"Figure '{fig}' in answer was not found in retrieved context.")

    return " | ".join(warnings) if warnings else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_input(query: str) -> None:
    """
    Run all input guardrails on the raw query string.
    Raises GuardrailError if any check fails.
    Call this BEFORE retrieval.

    Args:
        query: Raw user query string.

    Raises:
        GuardrailError: With .message (user-facing) and .status_code.
    """
    _check_length(query)
    _check_injection(query)
    _check_offtopic(query)


def check_output(answer: str, result, add_disclaimer: bool = True) -> tuple[str, list[str]]:
    """
    Run all output guardrails on the generated answer.
    Raises GuardrailError for hard failures (no chunks).
    Returns (final_answer, warnings) for soft issues.
    Call this AFTER generation.

    Args:
        answer:         Generated answer string.
        result:         RetrievalResult from retrieve().
        add_disclaimer: Whether to append the financial disclaimer.

    Returns:
        (final_answer, warnings) where warnings is a list of grounding issues.
    """
    # Hard check — must have retrieved something
    _check_retrieval_confidence(result)

    # Soft check — grounding
    contexts = [c.text for c in result.chunks]
    grounding_warning = _check_grounding(answer, contexts)
    warnings = [grounding_warning] if grounding_warning else []

    # Append disclaimer
    final_answer = answer + FINANCIAL_DISCLAIMER if add_disclaimer else answer

    return final_answer, warnings
