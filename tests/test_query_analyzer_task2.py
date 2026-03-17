import asyncio
import json

from app.orchestrator.query_analyzer import QueryAnalyzer


class _FakeLLMSuccess:
    async def call(self, *args, **kwargs):
        payload = {
            "query_type": "cross_doc_summary",
            "keywords": ["年度", "销售", "趋势"],
            "complexity": "complex",
            "sub_intents": ["按区域对比", "总结变化原因"],
            "estimated_evidence_pieces": 12,
            "time_sensitive": True,
            "domain_hints": ["sales"],
            "time_range": "2021-2023",
            "target_formats": ["excel"],
            "expected_dimensions": ["华东", "华南"],
            "initial_file_count": 3,
        }
        return json.dumps(payload, ensure_ascii=False), type("U", (), {"total_tokens": 42})()


class _FakeLLMTimeout:
    async def call(self, *args, **kwargs):
        raise TimeoutError("timeout")


def test_analyze_structured_route_and_dynamic_budget_clamped() -> None:
    analyzer = QueryAnalyzer(_FakeLLMSuccess())  # type: ignore[arg-type]
    plan, _ = asyncio.run(analyzer.analyze("请对比近三年各区域销售趋势并总结原因"))

    assert plan.query_type.value == "cross_doc_summary"
    assert plan.complexity == "complex"
    assert plan.estimated_evidence_pieces == 12
    assert plan.initial_file_count == 30
    assert plan.budget_audit["final_budget"] == 30
    assert plan.time_sensitive is True
    assert plan.sub_intents == ["按区域对比", "总结变化原因"]


def test_analyze_timeout_falls_back_to_heuristic_route() -> None:
    analyzer = QueryAnalyzer(_FakeLLMTimeout())  # type: ignore[arg-type]
    plan, usage = asyncio.run(analyzer.analyze("新员工入职流程是什么"))

    assert usage.total_tokens == 0
    assert plan.route_fallback_reason == "route_timeout"
    assert plan.complexity == "simple"
    assert plan.initial_file_count == plan.budget_audit["final_budget"]
    assert plan.initial_file_count >= 3
