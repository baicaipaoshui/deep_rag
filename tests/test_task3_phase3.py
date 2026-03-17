import asyncio
from pathlib import Path

import pytest

from app.evidence_model import QueryPlan, QueryType
from app.llm_client import TokenUsage
from app.orchestrator.state_machine import RetrievalContext, State, StateMachine
from evaluation.run_eval import RagasMetrics, evaluate_ragas_gate
from mcp_server.core.file_reader import FileReader
from preprocessor.index_builder import IndexBuilder
from preprocessor.parsers.excel_parser import ExcelParser


def test_semantic_split_falls_back_to_50_percent_window() -> None:
    text = "A" * 120
    chunks = IndexBuilder._split_section_content(text, max_chars=40, overlap=0)
    assert len(chunks) >= 2
    assert chunks[0][-20:] == chunks[1][:20]


def test_excel_parser_builds_three_layer_and_covers_tail_rows(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    ws = workbook.active
    ws.title = "数据表"
    ws.append(["年份", "销售额"])
    for i in range(1, 251):
        ws.append([2000 + i, i])
    file_path = tmp_path / "sample.xlsx"
    workbook.save(file_path)
    workbook.close()

    parsed = ExcelParser().parse(file_path)
    section_types = {sec.get("metadata", {}).get("section_type") for sec in parsed["sections"]}
    assert {"summary", "column_chunk", "row_window"}.issubset(section_types)
    row_windows = [sec for sec in parsed["sections"] if sec.get("metadata", {}).get("section_type") == "row_window"]
    assert row_windows
    assert max(int(sec["metadata"]["row_end"]) for sec in row_windows) == 251


def test_file_reader_reads_tail_row_window(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    ws = workbook.active
    ws.title = "数据表"
    ws.append(["年份", "销售额"])
    for i in range(1, 251):
        ws.append([2000 + i, i])
    file_path = tmp_path / "sample.xlsx"
    workbook.save(file_path)
    workbook.close()

    parsed = ExcelParser().parse(file_path)
    tail = max(
        (sec for sec in parsed["sections"] if sec.get("metadata", {}).get("section_type") == "row_window"),
        key=lambda sec: int(sec["metadata"]["row_end"]),
    )

    class _IndexStub:
        def get_section(self, file_name: str, section_id: str):
            return tail if section_id == tail["section_id"] else None

        def get_file_entry(self, file_name: str):
            return {"file_format": "excel"}

        def get_absolute_path(self, file_name: str):
            return file_path

    reader = FileReader(_IndexStub())  # type: ignore[arg-type]
    result = reader.read_section("sample.xlsx", tail["section_id"], max_chars=20000)
    assert "2250 | 250" in result["content"]


def test_verifying_stops_at_third_round_when_judge_still_insufficient() -> None:
    machine = StateMachine()
    machine._llm_judge_sufficiency = _fake_judge_insufficient  # type: ignore[method-assign]
    plan = QueryPlan(
        original_question="请总结所有风险",
        query_type=QueryType.CROSS_DOC_SUMMARY,
        keywords=["风险"],
        complexity="complex",
        initial_file_count=10,
    )
    ctx = RetrievalContext(query_plan=plan, state=State.VERIFYING, supplement_count=2)
    asyncio.run(machine._do_verifying(ctx))
    assert ctx.state == State.GENERATING
    assert ctx.supplement_count == 2


def test_verifying_marks_judge_fallback_when_judge_unavailable() -> None:
    machine = StateMachine()
    machine.cfg = type("Cfg", (), {"judge_fallback_sufficient": True})()  # type: ignore[assignment]
    machine._llm_judge_sufficiency = _fake_judge_none  # type: ignore[method-assign]
    plan = QueryPlan(
        original_question="问题",
        query_type=QueryType.FACT_LOOKUP,
        keywords=["问题"],
        complexity="medium",
        initial_file_count=4,
    )
    ctx = RetrievalContext(query_plan=plan, state=State.VERIFYING)
    asyncio.run(machine._do_verifying(ctx))
    assert ctx.judge_fallback is True
    assert ctx.state == State.GENERATING


def test_ragas_gate_is_only_blocking_decision() -> None:
    metrics = RagasMetrics(
        context_precision=0.55,
        context_recall=0.7,
        faithfulness=0.8,
        answer_relevancy=0.9,
        available=True,
        status="x",
    )
    gate = evaluate_ragas_gate(metrics, threshold=0.6)
    assert gate["is_blocking_gate"] is True
    assert gate["passed"] is False
    assert gate["reason"] == "context_precision_below_threshold"


async def _fake_judge_insufficient(ctx) -> tuple[dict, TokenUsage]:
    return (
        {"sufficient": False, "missing_aspects": ["缺少跨文档对比"], "confidence": 0.2},
        TokenUsage(),
    )


async def _fake_judge_none(ctx) -> tuple[dict | None, TokenUsage]:
    return None, TokenUsage()
