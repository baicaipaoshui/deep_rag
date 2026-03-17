from app.llm_client import LLMClient
from app.orchestrator.query_analyzer import QueryAnalyzer


def test_short_year_range_with_year_suffix_maps_to_years() -> None:
    analyzer = QueryAnalyzer(LLMClient())
    plan = analyzer._heuristic_plan("分析下21-23年的销售业绩", fallback_reason="test")
    assert plan.query_type.value == "trend_analysis"
    assert plan.time_range == "2021-2023"


def test_short_year_range_without_year_suffix_not_treated_as_years() -> None:
    analyzer = QueryAnalyzer(LLMClient())
    plan = analyzer._heuristic_plan("版本 21-23 的接口差异分析", fallback_reason="test")
    assert plan.time_range is None


def test_short_year_range_cross_century_is_reasonable() -> None:
    analyzer = QueryAnalyzer(LLMClient())
    plan = analyzer._heuristic_plan("统计 98-02 年的销量变化", fallback_reason="test")
    assert plan.time_range == "1998-2002"


def test_multi_year_numeric_question_prefers_numeric_query() -> None:
    analyzer = QueryAnalyzer(LLMClient())
    plan = analyzer._heuristic_plan("2022和2023年的销售额分别是多少", fallback_reason="test")
    assert plan.query_type.value == "numeric_query"


def test_keyword_fallback_uses_trimmed_text() -> None:
    tokens = QueryAnalyzer._extract_keywords(" ")
    assert tokens == ["问题"]
