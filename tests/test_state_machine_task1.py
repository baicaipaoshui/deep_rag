import asyncio

from app.evidence_model import QueryPlan, QueryType
from app.llm_client import TokenUsage
from app.orchestrator.state_machine import RetrievalContext, State, StateMachine


class _FakeMCP:
    async def call_tool(self, name: str, arguments: dict):
        return {
            "is_sufficient": False,
            "missing": ["缺少直接数值证据"],
            "suggested_keywords": ["数据", "数值", "销售"],
        }


def test_supplement_keywords_keep_original_and_merge_suggestions() -> None:
    machine = StateMachine()
    machine.mcp = _FakeMCP()
    machine._llm_judge_sufficiency = _fake_judge_insufficient  # type: ignore[method-assign]
    plan = QueryPlan(
        original_question="销售趋势如何",
        query_type=QueryType.NUMERIC_QUERY,
        keywords=["销售", "趋势"],
        initial_file_count=5,
    )
    ctx = RetrievalContext(query_plan=plan, state=State.VERIFYING)

    asyncio.run(machine._do_verifying(ctx))

    assert ctx.supplement_count == 1
    assert ctx.state == State.FILTERING
    assert ctx.query_plan.keywords == ["销售", "趋势", "缺少直接数值证据", "数据", "数值"]


async def _fake_judge_insufficient(ctx) -> tuple[dict, TokenUsage]:
    return (
        {
            "sufficient": False,
            "missing_aspects": ["缺少直接数值证据", "数据", "数值"],
            "confidence": 0.3,
        },
        TokenUsage(),
    )
