from __future__ import annotations

import re
from typing import Any

from mcp_server.core.index_manager import IndexManager
from mcp_server.tools import BaseTool

try:
    from mcp.types import Tool
except Exception:  # pragma: no cover
    Tool = dict  # type: ignore[assignment]

YEAR_PATTERN = re.compile(r"(20\d{2})(?!\d|万)")


def _parse_years(text: str | None) -> list[str]:
    if not text:
        return []
    years = sorted(set(YEAR_PATTERN.findall(text)))
    if len(years) <= 1:
        return years
    start = int(years[0])
    end = int(years[-1])
    return [str(year) for year in range(start, end + 1)]


class VerifyTool(BaseTool):
    def __init__(self, index_manager: IndexManager) -> None:
        self.index_manager = index_manager

    def get_definition(self) -> Tool | dict[str, Any]:
        return {
            "name": "verify_evidence_coverage",
            "description": "根据查询类型进行证据充分性校验",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query_type": {"type": "string"},
                    "time_range": {"type": ["string", "null"]},
                    "expected_dimensions": {"type": "array", "items": {"type": "string"}},
                    "collected_evidence": {"type": "array", "items": {"type": "object"}},
                    "supplement_round": {"type": "integer", "default": 0},
                },
                "required": ["query_type", "collected_evidence"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query_type = str(arguments.get("query_type", "")).strip()
        time_range = arguments.get("time_range")
        expected_dimensions = [str(x) for x in arguments.get("expected_dimensions", []) or []]
        evidences = arguments.get("collected_evidence", []) or []

        if not query_type:
            return self.error("query_type is required", arguments)

        covered: list[str] = []
        missing: list[str] = []
        suggested_keywords: list[str] = []
        suggested_files: list[str] = []

        if query_type == "trend_analysis":
            expected_years = _parse_years(str(time_range)) if time_range else []
            found_years = sorted(
                {
                    year
                    for ev in evidences
                    for year in _parse_years(str(ev.get("time_period") or ev.get("content")))
                }
            )
            covered = [f"{y}年数据" for y in found_years]
            for y in expected_years:
                if y not in found_years:
                    missing.append(f"{y}年数据")
            suggested_keywords = expected_years if missing else []

        elif query_type == "cross_doc_summary":
            topics = sorted({str(ev.get("topic_category")) for ev in evidences if ev.get("topic_category")})
            covered = topics
            if expected_dimensions:
                for dim in expected_dimensions:
                    if dim not in topics:
                        missing.append(dim)
            elif len({ev.get("source_file") for ev in evidences if ev.get("source_file")}) < 2:
                missing.append("至少两个来源文件")
            suggested_keywords = missing[:]

        elif query_type == "numeric_query":
            has_numeric_direct = any(
                ev.get("evidence_strength") == "direct" and ev.get("numeric_value") is not None
                for ev in evidences
            )
            covered = ["direct_numeric"] if has_numeric_direct else []
            if not has_numeric_direct:
                missing = ["缺少直接数值证据"]
                suggested_keywords = ["数据", "数值"]

        else:  # fact_lookup / definition_process / fallback
            has_direct = any(ev.get("evidence_strength") == "direct" for ev in evidences)
            covered = ["direct_evidence"] if has_direct else []
            if not has_direct:
                missing = ["缺少直接证据"]

        if missing:
            suggested_files = self._suggest_files(missing)
        is_sufficient = len(missing) == 0
        action = "force_generate" if not is_sufficient and int(arguments.get("supplement_round", 0)) >= 2 else (
            "supplement_search" if not is_sufficient else "generate"
        )

        detail_msg = "证据充分" if is_sufficient else f"缺失项: {', '.join(missing)}"
        return {
            "is_sufficient": is_sufficient,
            "covered": covered,
            "missing": missing,
            "missing_detail": detail_msg,
            "action": action,
            "suggested_keywords": suggested_keywords,
            "suggested_files": suggested_files,
        }

    def _suggest_files(self, missing: list[str]) -> list[str]:
        nav_files = self.index_manager.get_navigation_files()
        file_scores: list[tuple[float, str]] = []
        for file_item in nav_files:
            text = " ".join(
                [
                    str(file_item.get("file_name", "")),
                    str(file_item.get("description", "")),
                    " ".join(file_item.get("keywords", [])),
                ]
            )
            score = sum(1.0 for m in missing if m.replace("年数据", "") in text)
            if score > 0:
                file_scores.append((score, str(file_item.get("file_name", ""))))
        file_scores.sort(reverse=True)
        return [name for _, name in file_scores[:5]]
