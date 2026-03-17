import asyncio

from app.evidence_model import CandidateFile, QueryPlan, QueryType
from app.llm_client import TokenUsage
from app.orchestrator.state_machine import RetrievalContext, State, StateMachine


class _FakeMCPLocateRetry:
    async def call_tool(self, name: str, arguments: dict):
        if name == "query_discovery_index":
            return {"files": [{"file_name": "a.md", "sections": []}]}
        if name == "search_evidence":
            if arguments.get("query") == "hyde 命中":
                return {
                    "results": [
                        {
                            "file_name": "a.md",
                            "section_id": "md_001",
                            "heading": "h",
                            "relevance_score": 0.9,
                        }
                    ]
                }
            return {"results": []}
        return {"error": "unexpected"}


class _FakeMCPVerifySimple:
    async def call_tool(self, name: str, arguments: dict):
        if name == "verify_evidence_coverage":
            return {"is_sufficient": False, "missing": ["缺少直接证据"], "suggested_keywords": ["定义"]}
        return {"error": "unexpected"}


def test_locating_uses_hyde_retry_and_enters_extracting() -> None:
    machine = StateMachine()
    machine.mcp = _FakeMCPLocateRetry()
    machine._llm_locate_sections = _fake_empty_locate  # type: ignore[method-assign]
    machine._generate_hyde_query = _fake_hyde_hit  # type: ignore[method-assign]
    plan = QueryPlan(
        original_question="问题",
        query_type=QueryType.CROSS_DOC_SUMMARY,
        keywords=["问题"],
        complexity="complex",
        initial_file_count=10,
    )
    ctx = RetrievalContext(
        query_plan=plan,
        state=State.LOCATING,
        candidate_files=[CandidateFile(file_name="a.md", file_path="", file_format="markdown")],
    )

    asyncio.run(machine._do_locating(ctx))

    assert ctx.state == State.EXTRACTING
    assert len(ctx.target_locations) == 1
    assert ctx.hyde_retry_count == 1
    assert ctx.locate_failed is False


def test_verify_simple_query_exits_without_supplement_loop() -> None:
    machine = StateMachine()
    machine.mcp = _FakeMCPVerifySimple()
    machine._llm_judge_sufficiency = _fake_judge_insufficient_simple  # type: ignore[method-assign]
    plan = QueryPlan(
        original_question="定义是什么",
        query_type=QueryType.DEFINITION_PROCESS,
        keywords=["定义"],
        complexity="simple",
        initial_file_count=4,
    )
    ctx = RetrievalContext(query_plan=plan, state=State.VERIFYING)

    asyncio.run(machine._do_verifying(ctx))

    assert ctx.state == State.GENERATING
    assert ctx.supplement_count == 0


async def _fake_empty_locate(*args, **kwargs):
    return [], TokenUsage()


async def _fake_hyde_hit(question: str, attempt: int) -> str:
    if attempt == 1:
        return "hyde 命中"
    return ""


async def _fake_judge_insufficient_simple(ctx) -> tuple[dict, TokenUsage]:
    return (
        {"sufficient": False, "missing_aspects": ["缺少直接证据"], "confidence": 0.2},
        TokenUsage(),
    )
