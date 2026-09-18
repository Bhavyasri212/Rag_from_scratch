"""Evaluator — lightweight metrics for retrieval accuracy and answer faithfulness."""

from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

from rag.vector_store import ScoredChunk

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


@dataclass
class EvalResult:
    """Holds all evaluation scores for a single question."""
    question: str
    hit: bool           # did any retrieved chunk contain the answer?
    precision: float    # fraction of retrieved chunks that are relevant
    faithfulness: float # does the answer stay within the context? (0 or 1)
    notes: str = ""


def hit_rate(answer: str, results: list[ScoredChunk]) -> bool:
    """Return True if the expected answer text appears in any retrieved chunk."""
    answer_lower = answer.lower()
    return any(answer_lower in r.chunk.content.lower() for r in results)


def context_precision(relevant_sources: list[str], results: list[ScoredChunk]) -> float:
    """
    Fraction of retrieved chunks whose source is in the relevant_sources list.
    relevant_sources: list of file paths/names you know are relevant.
    """
    if not results:
        return 0.0
    hits = sum(
        1 for r in results
        if any(s in r.chunk.metadata.get("source", "") for s in relevant_sources)
    )
    return hits / len(results)


def faithfulness_check(question: str, answer: str, results: list[ScoredChunk]) -> float:
    """
    LLM-as-judge: ask the LLM whether the answer is fully supported by the context.
    Returns 1.0 (faithful) or 0.0 (not faithful).
    """
    context = "\n\n".join(r.chunk.content for r in results)
    prompt = (
        f"Given the following context, is this answer fully supported by it? "
        f"Reply with only 'yes' or 'no'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )

    try:
        if LLM_BACKEND == "ollama":
            import ollama
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = resp["message"]["content"].strip().lower()
        elif LLM_BACKEND == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            verdict = resp.choices[0].message.content.strip().lower()
        else:
            return -1.0  # unknown backend, skip

        return 1.0 if "yes" in verdict else 0.0

    except Exception as e:
        print(f"[evaluator] faithfulness check failed: {e}")
        return -1.0  # -1 signals "could not evaluate"


def evaluate(
    question: str,
    expected_answer: str,
    generated_answer: str,
    results: list[ScoredChunk],
    relevant_sources: list[str] | None = None,
    check_faithfulness: bool = True,
) -> EvalResult:
    """Run all metrics for one QA pair and return an EvalResult."""
    hit = hit_rate(expected_answer, results)
    precision = context_precision(relevant_sources or [], results)
    faith = faithfulness_check(question, generated_answer, results) if check_faithfulness else -1.0

    return EvalResult(
        question=question,
        hit=hit,
        precision=precision,
        faithfulness=faith,
    )
