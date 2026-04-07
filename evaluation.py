"""
evaluation.py — RAG pipeline evaluation with RAGAS and Giskard.

Evaluates the retrieval + generation pipeline on a curated test set of
questions with ground-truth answers derived from the Apple 10-K filing.

RAGAS metrics:
  - Faithfulness:      Does the answer stick to the retrieved context?
  - Answer Relevancy:  Does the answer address the question?
  - Context Precision:  Are relevant chunks ranked higher?
  - Context Recall:     Did retrieval find all necessary context?

Giskard scan:
  - Hallucination detection
  - Robustness to query variations
  - Failure mode identification

Usage:
  .venv/bin/python evaluation.py                    # run full eval
  .venv/bin/python evaluation.py --ragas-only       # RAGAS metrics only
  .venv/bin/python evaluation.py --giskard-only     # Giskard scan only

Requires: pip install ragas giskard langchain-huggingface
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# ---------------------------------------------------------------------------
# Ground-Truth Test Dataset
# ---------------------------------------------------------------------------
# Each entry: question, ground_truth answer, metadata filters for retrieval.
# Ground truths are manually verified from the Apple 10-K 2025 filing.

TEST_CASES = [
    # --- TABLE / NUMERICAL QUERIES ---
    {
        "question": "What was Apple's total net sales revenue in 2025?",
        "ground_truth": "Apple's total net sales in 2025 were $416,161 million.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What was Apple's net income in 2025?",
        "ground_truth": "Apple's net income in 2025 was $112,010 million.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "How much revenue did iPhone generate in 2025?",
        "ground_truth": "iPhone generated $209,586 million in revenue in 2025.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What was Apple's services revenue in 2025?",
        "ground_truth": "Apple's services revenue in 2025 was $109,158 million.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What was Apple's total cost of sales in 2025?",
        "ground_truth": "Apple's total cost of sales in 2025 was $220,960 million.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What was Apple's gross margin in 2025?",
        "ground_truth": "Apple's gross margin in 2025 was $195,201 million.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "How much did Apple spend on research and development in 2025?",
        "ground_truth": "Apple spent $34,550 million on research and development in 2025.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What were Apple's foreign pretax earnings in 2025?",
        "ground_truth": "Apple's foreign pretax earnings were $82.0 billion in 2025.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },

    # --- NARRATIVE QUERIES ---
    {
        "question": "What are Apple's main risk factors?",
        "ground_truth": "Apple's main risk factors include adverse macroeconomic and industry conditions, political and geopolitical events, competition in the technology industry, and the impact of inflation, interest rates, and currency fluctuations on operations and financial condition.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "How does Apple distribute its products?",
        "ground_truth": "Apple sells its products directly to customers through its retail and online stores and its direct sales force. The Company also uses indirect distribution channels including third-party cellular network carriers, wholesalers, retailers, and resellers.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What are the key competitive factors in Apple's industry?",
        "ground_truth": "Key competitive factors include price, product and service features, relative price and performance, quality and reliability, design and technology innovation, a strong third-party software and accessories ecosystem, marketing and distribution capability, service and support, corporate reputation, and the ability to protect intellectual property rights.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What is Apple's depreciation expense for property, plant and equipment?",
        "ground_truth": "Depreciation expense on property, plant and equipment was $8.0 billion during 2025.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },

    # --- HYBRID QUERIES (need both tables and narrative) ---
    {
        "question": "How did Apple's total net sales change from 2024 to 2025?",
        "ground_truth": "Apple's total net sales increased from $391,035 million in 2024 to $416,161 million in 2025, a 6% increase.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What was Apple's revenue from the Greater China region in 2025?",
        "ground_truth": "Apple's revenue from Greater China was $64,377 million in 2025, a 4% decrease from 2024.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
    {
        "question": "What was Apple's total unrecognized tax benefits as of September 2025?",
        "ground_truth": "As of September 27, 2025, Apple's total gross unrecognized tax benefits were $23.2 billion, of which $10.6 billion would affect the effective tax rate if recognized.",
        "filters": {"ticker": "AAPL", "document_type": "10-K", "fiscal_year": 2025},
    },
]


# ---------------------------------------------------------------------------
# Run pipeline on test cases
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question:       str
    ground_truth:   str
    answer:         str
    contexts:       list[str]
    source_pages:   list[str]
    route:          str
    retrieval_ms:   int
    generation_ms:  int


def run_pipeline(test_cases: list[dict]) -> list[EvalSample]:
    """Run retrieval + generation on each test case, collect results."""
    from retrieval import retrieve
    from generation import generate

    samples = []
    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc['question'][:60]}...")

        t0 = time.perf_counter()
        result = retrieve(
            query         = tc["question"],
            ticker        = tc["filters"].get("ticker"),
            document_type = tc["filters"].get("document_type"),
            fiscal_year   = tc["filters"].get("fiscal_year"),
        )
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        response = generate(tc["question"], result)
        generation_ms = int((time.perf_counter() - t1) * 1000)

        contexts = [chunk.text for chunk in result.chunks]
        source_pages = [
            f"page {chunk.page} — {chunk.breadcrumb}"
            for chunk in result.chunks
        ]

        samples.append(EvalSample(
            question      = tc["question"],
            ground_truth  = tc["ground_truth"],
            answer        = response.answer,
            contexts      = contexts,
            source_pages  = source_pages,
            route         = result.route.value,
            retrieval_ms  = retrieval_ms,
            generation_ms = generation_ms,
        ))

    return samples


# ---------------------------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------------------------

def run_ragas(samples: list[EvalSample]):
    """
    Evaluate with RAGAS metrics using Groq (Llama 3.1 70B) as the judge LLM.
    Groq is free, fast, and doesn't exhaust monthly quotas like HF free tier.

    Metrics:
      - Faithfulness:     Is the answer grounded in the retrieved context?
      - AnswerRelevancy:  Does the answer address the user's question?
      - ContextPrecision: Are the relevant chunks ranked at the top?
      - ContextRecall:    Did retrieval find all necessary information?
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env. Get a free key at console.groq.com")

    print("\n[RAGAS] Setting up judge LLM (Llama 3.1 70B via Groq)...")
    chat = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0,
        max_tokens=1024,
    )
    ragas_llm = LangchainLLMWrapper(chat)

    print("[RAGAS] Loading embeddings (BGE-large, same as pipeline)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    # Build RAGAS dataset
    ragas_samples = []
    for s in samples:
        ragas_samples.append(SingleTurnSample(
            user_input          = s.question,
            response            = s.answer,
            retrieved_contexts  = s.contexts,
            reference           = s.ground_truth,
        ))
    dataset = EvaluationDataset(samples=ragas_samples)

    print(f"[RAGAS] Evaluating {len(ragas_samples)} samples...")
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return result


# ---------------------------------------------------------------------------
# Giskard Evaluation
# ---------------------------------------------------------------------------

def run_giskard(samples: list[EvalSample]):
    """
    Run Giskard's automated RAG evaluation scan.

    Detects:
      - Hallucinations (answer not supported by context)
      - Correctness issues (answer contradicts ground truth)
      - Robustness problems (minor query changes flip the answer)
    """
    import giskard
    import pandas as pd

    print("\n[GISKARD] Building model wrapper...")

    # Giskard needs a function that takes a DataFrame and returns predictions
    def predict(df: pd.DataFrame) -> list[str]:
        from retrieval import retrieve
        from generation import generate

        answers = []
        for _, row in df.iterrows():
            result = retrieve(
                query         = row["question"],
                ticker        = "AAPL",
                document_type = "10-K",
                fiscal_year   = 2025,
            )
            response = generate(row["question"], result)
            answers.append(response.answer)
        return answers

    # Build test DataFrame
    df = pd.DataFrame([
        {"question": s.question, "ground_truth": s.ground_truth}
        for s in samples
    ])

    # Wrap as Giskard model
    model = giskard.Model(
        model=predict,
        model_type="text_generation",
        name="Financial RAG Pipeline",
        description="RAG pipeline for SEC 10-K/10-Q filings",
        feature_names=["question"],
    )

    # Wrap as Giskard dataset
    dataset = giskard.Dataset(
        df=df,
        name="Apple 10-K Eval Set",
        target="ground_truth",
    )

    print(f"[GISKARD] Scanning {len(df)} samples for vulnerabilities...")
    scan_result = giskard.scan(model, dataset)

    # Save report
    report_path = "giskard_report.html"
    scan_result.to_html(report_path)
    print(f"[GISKARD] Report saved to {report_path}")

    return scan_result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(samples: list[EvalSample], ragas_result=None):
    """Print formatted evaluation report."""

    print("\n" + "=" * 70)
    print("  RAG PIPELINE EVALUATION REPORT")
    print("=" * 70)

    # Per-query results
    print("\n--- PER-QUERY RESULTS ---\n")
    for i, s in enumerate(samples, 1):
        status = "✗" if "cannot find" in s.answer.lower() else "✓"
        print(f"  [{i}] {status}  route={s.route:<10}  "
              f"retrieval={s.retrieval_ms:>5}ms  generation={s.generation_ms:>5}ms")
        print(f"      Q: {s.question[:70]}")
        print(f"      A: {s.answer[:120]}{'...' if len(s.answer) > 120 else ''}")
        print(f"      GT: {s.ground_truth[:120]}{'...' if len(s.ground_truth) > 120 else ''}")
        print()

    # Summary stats
    total_queries = len(samples)
    answered = sum(1 for s in samples if "cannot find" not in s.answer.lower())
    failed = total_queries - answered
    avg_retrieval = sum(s.retrieval_ms for s in samples) / total_queries
    avg_generation = sum(s.generation_ms for s in samples) / total_queries

    print("--- SUMMARY ---\n")
    print(f"  Total queries:        {total_queries}")
    print(f"  Answered:             {answered} ({100*answered/total_queries:.0f}%)")
    print(f"  Failed (no answer):   {failed} ({100*failed/total_queries:.0f}%)")
    print(f"  Avg retrieval time:   {avg_retrieval:.0f}ms")
    print(f"  Avg generation time:  {avg_generation:.0f}ms")
    print(f"  Avg total latency:    {avg_retrieval + avg_generation:.0f}ms")

    route_counts = {}
    for s in samples:
        route_counts[s.route] = route_counts.get(s.route, 0) + 1
    print(f"  Route distribution:   {route_counts}")

    # RAGAS scores
    if ragas_result is not None:
        print("\n--- RAGAS SCORES ---\n")
        scores = ragas_result.to_pandas()
        meta_cols = {"user_input", "response", "retrieved_contexts", "reference"}
        metric_cols = [c for c in scores.columns if c not in meta_cols]

        for col in metric_cols:
            mean_val = scores[col].mean()
            print(f"  {col:<25} {mean_val:.4f}")

        # Per-query RAGAS breakdown
        print("\n--- RAGAS PER-QUERY BREAKDOWN ---\n")
        for i, row in scores.iterrows():
            q = samples[i].question[:60]
            vals = "  ".join(
                f"{c}={row[c]:.2f}" for c in metric_cols
                if c in row and not pd.isna(row[c])
            )
            print(f"  [{i+1}] {q}...")
            print(f"       {vals}")

    print("\n" + "=" * 70)

    # Save results to JSON
    results_dict = {
        "summary": {
            "total_queries": total_queries,
            "answered": answered,
            "failed": failed,
            "answer_rate": round(100 * answered / total_queries, 1),
            "avg_retrieval_ms": round(avg_retrieval),
            "avg_generation_ms": round(avg_generation),
            "route_distribution": route_counts,
        },
        "per_query": [
            {
                "question": s.question,
                "ground_truth": s.ground_truth,
                "answer": s.answer,
                "route": s.route,
                "retrieval_ms": s.retrieval_ms,
                "generation_ms": s.generation_ms,
                "contexts_count": len(s.contexts),
            }
            for s in samples
        ],
    }

    if ragas_result is not None:
        scores = ragas_result.to_pandas()
        meta_cols = {"user_input", "response", "retrieved_contexts", "reference"}
        metric_cols = [c for c in scores.columns if c not in meta_cols]
        results_dict["ragas"] = {
            "means": {col: round(float(scores[col].mean()), 4) for col in metric_cols},
            "per_query": [
                {col: round(float(row[col]), 4) for col in metric_cols if col in row and not pd.isna(row[col])}
                for _, row in scores.iterrows()
            ],
        }

    with open("eval_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to eval_results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ragas_only   = "--ragas-only" in sys.argv
    giskard_only = "--giskard-only" in sys.argv
    skip_ragas   = "--skip-ragas" in sys.argv
    skip_giskard = "--skip-giskard" in sys.argv

    print("=" * 70)
    print("  RUNNING RAG PIPELINE EVALUATION")
    print("=" * 70)
    print(f"\n  Test cases: {len(TEST_CASES)}")
    print(f"  RAGAS:   {'skip' if skip_ragas or giskard_only else 'enabled'}")
    print(f"  Giskard: {'skip' if skip_giskard or ragas_only else 'enabled'}")

    # Step 1 — Run pipeline on all test cases
    print("\n--- STEP 1: Running pipeline on test cases ---\n")
    samples = run_pipeline(TEST_CASES)

    # Step 2 — RAGAS evaluation
    ragas_result = None
    if not skip_ragas and not giskard_only:
        print("\n--- STEP 2: RAGAS Evaluation ---")
        try:
            ragas_result = run_ragas(samples)
        except Exception as e:
            print(f"[RAGAS] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Step 3 — Giskard scan
    if not skip_giskard and not ragas_only:
        print("\n--- STEP 3: Giskard Scan ---")
        try:
            run_giskard(samples)
        except Exception as e:
            print(f"[GISKARD] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Step 4 — Report
    print("\n--- STEP 4: Report ---")
    print_report(samples, ragas_result)


if __name__ == "__main__":
    main()
