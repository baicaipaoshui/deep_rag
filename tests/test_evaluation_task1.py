from evaluation import run_eval


def test_compute_model_metrics_keeps_legacy_when_ragas_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(run_eval, "Dataset", None)
    monkeypatch.setattr(run_eval, "ragas_evaluate", None)
    metrics = run_eval.compute_model_metrics(
        question="数据库调优建议有哪些",
        answer="建议包含索引优化【来源：database.md-md_001】",
        expected_keywords=["索引"],
        contexts=["索引优化建议"],
    )

    assert metrics.legacy.coverage == 1.0
    assert metrics.legacy.status == run_eval.TRANSITION_STATUS
    assert metrics.ragas.available is False
    assert metrics.ragas.status == run_eval.TRANSITION_STATUS
