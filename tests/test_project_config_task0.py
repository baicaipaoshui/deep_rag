from __future__ import annotations

from pathlib import Path

from project_config import load_project_config


def test_task0_defaults_when_config_missing() -> None:
    load_project_config.cache_clear()
    cfg = load_project_config()
    assert cfg.judge_route_path == "/judge"
    assert cfg.hyde_route_path == "/judge/HyDE"
    assert cfg.judge_model == "haiku"
    assert cfg.judge_timeout_seconds == 2
    assert cfg.judge_fallback_sufficient is True
    assert cfg.hyde_enabled is True
    assert cfg.hyde_max_retries == 2
    assert cfg.hyde_blacklist == ["忽略以上指令", "system prompt"]
    assert cfg.budget_k_min == 3
    assert cfg.budget_k_max == 30
    assert cfg.rrf_k == 60
    assert cfg.budget_complexity_multipliers == {"simple": 1.5, "medium": 2.5, "complex": 4.0}
    assert cfg.fallback_budget_by_query_type == {
        "fact_lookup": 8,
        "numeric_query": 12,
        "trend_analysis": 16,
        "cross_doc_summary": 24,
    }
    assert cfg.ragas_context_precision_threshold == 0.6


def test_task0_partial_invalid_config_falls_back(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "deep_rag.yaml"
    config_file.write_text(
        """
routing:
  judge:
    path: judge
    timeout_seconds: 0
    fallback_sufficient: no
  hyde:
    path: /custom/hyde
    max_retries: -1
    blacklist: [" ", "custom", 123]
retrieval:
  fusion:
    rrf_k: 0
  budget:
    k_min: 0
    k_max: 1
    complexity_multipliers:
      simple: 0
      medium: bad
      complex: 3.3
  fallback:
    max_results:
      fact_lookup: -2
      numeric_query: 20
evaluation:
  ragas:
    context_precision_threshold: bad
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEP_RAG_CONFIG", str(config_file))
    load_project_config.cache_clear()
    cfg = load_project_config()
    assert cfg.judge_route_path == "/judge"
    assert cfg.hyde_route_path == "/custom/hyde"
    assert cfg.judge_timeout_seconds == 2
    assert cfg.judge_fallback_sufficient is False
    assert cfg.hyde_max_retries == 2
    assert cfg.hyde_blacklist == ["custom", "123"]
    assert cfg.budget_k_min == 3
    assert cfg.budget_k_max == 30
    assert cfg.rrf_k == 60
    assert cfg.budget_complexity_multipliers == {"simple": 1.5, "medium": 2.5, "complex": 3.3}
    assert cfg.fallback_budget_by_query_type == {
        "fact_lookup": 8,
        "numeric_query": 20,
        "trend_analysis": 16,
        "cross_doc_summary": 24,
    }
    assert cfg.ragas_context_precision_threshold == 0.6
