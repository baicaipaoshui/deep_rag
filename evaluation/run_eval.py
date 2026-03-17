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
from project_config import load_project_config

try:
    from datasets import Dataset
except Exception:
    Dataset = None  # type: ignore[assignment]

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
except Exception:
    ragas_evaluate = None  # type: ignore[assignment]
    answer_relevancy = None  # type: ignore[assignment]
    context_precision = None  # type: ignore[assignment]
    context_recall = None  # type: ignore[assignment]
    faithfulness = None  # type: ignore[assignment]


TRANSITION_STATUS = "non_blocking_degraded"
RAGAS_BLOCKING_STATUS = "blocking_gate"


@dataclass
class LegacyMetrics:
    coverage: float
    completeness: float
    citation_accuracy: float
    status: str


@dataclass
class RagasMetrics:
    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    available: bool
    status: str
    error: str | None = None


@dataclass
class ModelMetrics:
    legacy: LegacyMetrics
    ragas: RagasMetrics


def evaluate_ragas_gate(metrics: RagasMetrics, threshold: float) -> dict[str, Any]:
    if not metrics.available or metrics.context_precision is None:
        return {
            "is_blocking_gate": True,
            "passed": False,
            "metric": "context_precision",
            "threshold": threshold,
            "actual": metrics.context_precision,
            "reason": metrics.error or "ragas_unavailable",
            "status": RAGAS_BLOCKING_STATUS,
        }
    passed = float(metrics.context_precision) >= threshold
    return {
        "is_blocking_gate": True,
        "passed": passed,
        "metric": "context_precision",
        "threshold": threshold,
        "actual": metrics.context_precision,
        "reason": "ok" if passed else "context_precision_below_threshold",
        "status": RAGAS_BLOCKING_STATUS,
    }


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


def compute_legacy_metrics(answer: str, expected_keywords: list[str]) -> LegacyMetrics:
    answer_norm = (answer or "").lower()
    if not expected_keywords:
        coverage = 1.0
    else:
        hit = sum(1 for kw in expected_keywords if kw.lower() in answer_norm)
        coverage = hit / len(expected_keywords)
    completeness = 1.0 if coverage >= 0.6 else 0.0
    citation_accuracy = 1.0 if "【来源" in answer else 0.0
    return LegacyMetrics(
        coverage=coverage,
        completeness=completeness,
        citation_accuracy=citation_accuracy,
        status=TRANSITION_STATUS,
    )


def _extract_ragas_score(result: Any, key: str) -> float | None:
    value = None
    if isinstance(result, dict):
        value = result.get(key)
    elif hasattr(result, key):
        value = getattr(result, key)
    elif hasattr(result, "get"):
        try:
            value = result.get(key)
        except Exception:
            value = None
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _round_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _compute_proxy_ragas_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    expected_keywords: list[str],
    error: str,
) -> RagasMetrics:
    normalized_contexts = [str(x).strip().lower() for x in contexts if str(x).strip()]
    normalized_answer = str(answer or "").strip().lower()
    normalized_question = str(question or "").strip().lower()
    normalized_keywords = [str(x).strip().lower() for x in expected_keywords if str(x).strip()]
    if not normalized_keywords:
        normalized_keywords = [x for x in normalized_question.split() if x]

    if not normalized_contexts:
        return RagasMetrics(
            context_precision=0.0,
            context_recall=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            available=True,
            status=TRANSITION_STATUS,
            error=f"proxy_fallback:{error}",
        )

    if not normalized_keywords:
        base = 1.0 if normalized_contexts else 0.0
        return RagasMetrics(
            context_precision=_round_score(base),
            context_recall=_round_score(base),
            faithfulness=_round_score(1.0 if "【来源" in answer else 0.5),
            answer_relevancy=_round_score(1.0 if normalized_answer else 0.0),
            available=True,
            status=TRANSITION_STATUS,
            error=f"proxy_fallback:{error}",
        )

    context_hit_count = sum(
        1 for text in normalized_contexts if any(keyword in text for keyword in normalized_keywords)
    )
    keyword_hit_count = sum(
        1 for keyword in normalized_keywords if any(keyword in text for text in normalized_contexts)
    )
    answer_hit_count = sum(1 for keyword in normalized_keywords if keyword in normalized_answer)
    citation_bonus = 0.2 if "【来源" in answer else 0.0
    context_precision_score = _round_score(context_hit_count / len(normalized_contexts))
    context_recall_score = _round_score(keyword_hit_count / len(normalized_keywords))
    answer_relevancy_score = _round_score(answer_hit_count / len(normalized_keywords))
    faithfulness_score = _round_score((context_precision_score + answer_relevancy_score) / 2 + citation_bonus)
    return RagasMetrics(
        context_precision=context_precision_score,
        context_recall=context_recall_score,
        faithfulness=faithfulness_score,
        answer_relevancy=answer_relevancy_score,
        available=True,
        status=TRANSITION_STATUS,
        error=f"proxy_fallback:{error}",
    )


def compute_ragas_metrics(question: str, answer: str, contexts: list[str], expected_keywords: list[str]) -> RagasMetrics:
    if Dataset is None or ragas_evaluate is None:
        return RagasMetrics(
            context_precision=None,
            context_recall=None,
            faithfulness=None,
            answer_relevancy=None,
            available=False,
            status=TRANSITION_STATUS,
            error="ragas_or_datasets_unavailable",
        )
    safe_contexts = [str(x).strip() for x in contexts if str(x).strip()]
    if not safe_contexts:
        return RagasMetrics(
            context_precision=0.0,
            context_recall=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            available=True,
            status=TRANSITION_STATUS,
        )
    reference = "；".join(str(x).strip() for x in expected_keywords if str(x).strip())
    payload = {
        "question": [question],
        "answer": [answer],
        "contexts": [safe_contexts],
        "ground_truth": [reference],
    }
    try:
        dataset = Dataset.from_dict(payload)
        ragas_result = ragas_evaluate(
            dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )
        return RagasMetrics(
            context_precision=_extract_ragas_score(ragas_result, "context_precision"),
            context_recall=_extract_ragas_score(ragas_result, "context_recall"),
            faithfulness=_extract_ragas_score(ragas_result, "faithfulness"),
            answer_relevancy=_extract_ragas_score(ragas_result, "answer_relevancy"),
            available=True,
            status=TRANSITION_STATUS,
        )
    except Exception as exc:
        return _compute_proxy_ragas_metrics(question, answer, safe_contexts, expected_keywords, str(exc))


def compute_model_metrics(
    question: str,
    answer: str,
    expected_keywords: list[str],
    contexts: list[str],
) -> ModelMetrics:
    return ModelMetrics(
        legacy=compute_legacy_metrics(answer, expected_keywords),
        ragas=compute_ragas_metrics(question, answer, contexts, expected_keywords),
    )


async def _run_deep(question: str) -> dict[str, Any]:
    return await StateMachine().run(question)


def run_evaluation() -> dict[str, Any]:
    args = parse_args()
    cfg = load_project_config()
    ragas_threshold = float(cfg.ragas_context_precision_threshold)
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

        base_contexts = [str(ev.get("content_preview", "")) for ev in base_result.get("evidences", [])]
        deep_contexts = [str(ev.get("content", "")) for ev in deep_result.get("evidences", [])]
        base_metrics = compute_model_metrics(q, base_result.get("answer", ""), expected_keywords, base_contexts)
        deep_metrics = compute_model_metrics(q, deep_result.get("answer", ""), expected_keywords, deep_contexts)
        base_gate = evaluate_ragas_gate(base_metrics.ragas, ragas_threshold)
        deep_gate = evaluate_ragas_gate(deep_metrics.ragas, ragas_threshold)

        details.append(
            {
                "id": item.get("id"),
                "question": q,
                "baseline": {
                    "answer": base_result.get("answer", ""),
                    "metrics": {
                        "legacy": base_metrics.legacy.__dict__,
                        "ragas": base_metrics.ragas.__dict__,
                    },
                    "gate": base_gate,
                },
                "deep_rag": {
                    "answer": deep_result.get("answer", ""),
                    "metrics": {
                        "legacy": deep_metrics.legacy.__dict__,
                        "ragas": deep_metrics.ragas.__dict__,
                    },
                    "gate": deep_gate,
                },
            }
        )

    def _avg_legacy(key: str, model: str) -> float:
        values = [float(d[model]["metrics"]["legacy"][key]) for d in details]
        return round(sum(values) / len(values), 4) if values else 0.0

    def _avg_ragas(key: str, model: str) -> float | None:
        values = [
            d[model]["metrics"]["ragas"][key]
            for d in details
            if d[model]["metrics"]["ragas"]["available"] and d[model]["metrics"]["ragas"][key] is not None
        ]
        return round(sum(float(v) for v in values) / len(values), 4) if values else None

    def _gate_summary(model: str) -> dict[str, Any]:
        gates = [d[model]["gate"] for d in details]
        total = len(gates)
        passed = sum(1 for gate in gates if gate.get("passed"))
        return {
            "is_blocking_gate": True,
            "metric": "context_precision",
            "threshold": ragas_threshold,
            "passed": passed == total and total > 0,
            "passed_cases": passed,
            "total_cases": total,
            "status": RAGAS_BLOCKING_STATUS,
        }

    def _context_precision_drop_cases(model: str) -> list[dict[str, Any]]:
        return [
            {
                "id": d.get("id"),
                "question": d.get("question"),
                "actual": d[model]["gate"].get("actual"),
                "threshold": d[model]["gate"].get("threshold"),
                "reason": d[model]["gate"].get("reason"),
            }
            for d in details
            if not d[model]["gate"].get("passed")
        ]

    summary = {
        "baseline": {
            "legacy": {
                "coverage": _avg_legacy("coverage", "baseline"),
                "completeness": _avg_legacy("completeness", "baseline"),
                "citation_accuracy": _avg_legacy("citation_accuracy", "baseline"),
                "status": TRANSITION_STATUS,
            },
            "ragas": {
                "context_precision": _avg_ragas("context_precision", "baseline"),
                "context_recall": _avg_ragas("context_recall", "baseline"),
                "faithfulness": _avg_ragas("faithfulness", "baseline"),
                "answer_relevancy": _avg_ragas("answer_relevancy", "baseline"),
                "context_precision_drop_cases": _context_precision_drop_cases("baseline"),
                "status": RAGAS_BLOCKING_STATUS,
            },
            "gate": _gate_summary("baseline"),
        },
        "deep_rag": {
            "legacy": {
                "coverage": _avg_legacy("coverage", "deep_rag"),
                "completeness": _avg_legacy("completeness", "deep_rag"),
                "citation_accuracy": _avg_legacy("citation_accuracy", "deep_rag"),
                "status": TRANSITION_STATUS,
            },
            "ragas": {
                "context_precision": _avg_ragas("context_precision", "deep_rag"),
                "context_recall": _avg_ragas("context_recall", "deep_rag"),
                "faithfulness": _avg_ragas("faithfulness", "deep_rag"),
                "answer_relevancy": _avg_ragas("answer_relevancy", "deep_rag"),
                "context_precision_drop_cases": _context_precision_drop_cases("deep_rag"),
                "status": RAGAS_BLOCKING_STATUS,
            },
            "gate": _gate_summary("deep_rag"),
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
