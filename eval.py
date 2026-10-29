#!/usr/bin/env python3
"""
Evaluation for the Quebec Newcomer Survival Bot.

Measures retrieval, not vibes:
  - Hit@1 / Hit@5: is the expected doc in the top-1 / top-5 hits?
  - MRR: mean reciprocal rank of the expected doc.
  - Abstention accuracy: for out-of-scope questions, does the bot abstain
    instead of answering from the wrong doc?

Run:
    python eval.py                          # TF-IDF, 512-token chunks
    python eval.py --chunk-tokens 256       # chunk-size comparison
    python eval.py --embedder sbert         # multilingual comparison
    python eval.py --out eval/results.json  # save the table for the README

The question set lives in eval/questions.json. Each entry has:
  question, language ("en" | "fr"), expected_source (or null),
  abstain (true for out-of-scope questions that must NOT be answered).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import answer_question, build_index

ROOT = Path(__file__).parent


def load_questions(path: Path = ROOT / "eval" / "questions.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(index, questions: list[dict], top_k: int = 5, threshold: float = 0.08) -> dict:
    in_scope = [q for q in questions if not q.get("abstain")]
    abstain_qs = [q for q in questions if q.get("abstain")]

    hits1 = hits5 = 0
    mrr_total = 0.0
    per_language: dict[str, dict] = {}
    failures: list[dict] = []

    for q in in_scope:
        result = answer_question(index, q["question"], top_k=top_k, threshold=threshold)
        sources = [h["source"] for h in result["hits"]]
        # An abstained in-scope question delivers no answer, so it counts
        # as a miss even if the right doc was in the retrieved list.
        rank = None
        if not result["abstained"] and q["expected_source"] in sources:
            rank = sources.index(q["expected_source"]) + 1
        ok1, ok5 = rank == 1, rank is not None
        hits1 += int(ok1)
        hits5 += int(ok5)
        mrr_total += 1.0 / rank if rank else 0.0
        lang = q.get("language", "?")
        bucket = per_language.setdefault(lang, {"n": 0, "hit@5": 0})
        bucket["n"] += 1
        bucket["hit@5"] += int(ok5)
        if not ok5:
            failures.append(
                {
                    "question": q["question"],
                    "expected": q["expected_source"],
                    "got": sources,
                    "abstained": result["abstained"],
                    "reason": "abstained (no answer delivered)"
                    if result["abstained"]
                    else "expected doc not in top-5",
                }
            )

    abstain_ok = 0
    for q in abstain_qs:
        result = answer_question(index, q["question"], top_k=top_k, threshold=threshold)
        abstain_ok += int(result["abstained"])

    n = max(1, len(in_scope))
    return {
        "n_in_scope": len(in_scope),
        "n_abstain": len(abstain_qs),
        "hit@1": round(hits1 / n, 3),
        "hit@5": round(hits5 / n, 3),
        "mrr": round(mrr_total / n, 3),
        "abstention_accuracy": round(abstain_ok / max(1, len(abstain_qs)), 3),
        "per_language_hit@5": {
            lang: round(v["hit@5"] / v["n"], 3) for lang, v in per_language.items()
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + abstention")
    parser.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--out", default=None, help="Write results JSON here")
    args = parser.parse_args()

    index = build_index(args.chunk_tokens, 64, args.embedder)
    questions = load_questions()
    results = evaluate(index, questions, top_k=args.top_k, threshold=args.threshold)
    results["embedder"] = args.embedder
    results["chunk_tokens"] = args.chunk_tokens
    results["n_chunks"] = len(index.chunks)

    print(f"Embedder: {args.embedder} | chunk_tokens: {args.chunk_tokens} | chunks: {len(index.chunks)}")
    print(f"In-scope questions: {results['n_in_scope']} | Abstention questions: {results['n_abstain']}")
    print(f"Hit@1: {results['hit@1']} | Hit@5: {results['hit@5']} | MRR: {results['mrr']}")
    print(f"Abstention accuracy: {results['abstention_accuracy']}")
    print(f"Hit@5 by language: {results['per_language_hit@5']}")
    if results["failures"]:
        print("\nFailures (expected doc not in top-5):")
        for f in results["failures"]:
            print(f"  - {f['question']!r} expected {f['expected']}, got {f['got']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
