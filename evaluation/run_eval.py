from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.orchestrator.state_machine import StateMachine
from evaluation.baseline_rag import BaselineRAG


@dataclass
class Metrics:
    coverage: float
    completeness: float
    citation_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Deep RAG evaluation.")
    parser.add_argument(
        "--questions",
        default="evaluation/test_questions.json",
        help="Path to evaluation questions JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory to save evaluation result",
    )
    return parser.parse_args()


def compute_metrics(answer: str, expected_keywords: list[str]) -> Metrics:
    answer_norm = (answer or "").lower()
    if not expected_keywords:
        coverage = 1.0
    else:
        hit = sum(1 for kw in expected_keywords if kw.lower() in answer_norm)
        coverage = hit / len(expected_keywords)
    completeness = 1.0 if coverage >= 0.6 else 0.0
    citation_accuracy = 1.0 if "【来源" in answer else 0.0
    return Metrics(coverage=coverage, completeness=completeness, citation_accuracy=citation_accuracy)


async def _run_deep(question: str) -> dict[str, Any]:
    return await StateMachine().run(question)


def run_evaluation() -> dict[str, Any]:
    args = parse_args()
    questions_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    baseline = BaselineRAG()

    details: list[dict[str, Any]] = []
    for item in questions:
        q = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        base_result = baseline.query(q)
        deep_result = asyncio.run(_run_deep(q))

        base_metrics = compute_metrics(base_result.get("answer", ""), expected_keywords)
        deep_metrics = compute_metrics(deep_result.get("answer", ""), expected_keywords)

        details.append(
            {
                "id": item.get("id"),
                "question": q,
                "baseline": {
                    "answer": base_result.get("answer", ""),
                    "metrics": base_metrics.__dict__,
                },
                "deep_rag": {
                    "answer": deep_result.get("answer", ""),
                    "metrics": deep_metrics.__dict__,
                },
            }
        )

    def _avg(key: str, model: str) -> float:
        values = [float(d[model]["metrics"][key]) for d in details]
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {
        "baseline": {
            "coverage": _avg("coverage", "baseline"),
            "completeness": _avg("completeness", "baseline"),
            "citation_accuracy": _avg("citation_accuracy", "baseline"),
        },
        "deep_rag": {
            "coverage": _avg("coverage", "deep_rag"),
            "completeness": _avg("completeness", "deep_rag"),
            "citation_accuracy": _avg("citation_accuracy", "deep_rag"),
        },
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(details),
        "summary": summary,
        "details": details,
    }
    out_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
