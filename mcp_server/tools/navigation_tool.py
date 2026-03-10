from __future__ import annotations

import re
from typing import Any

from mcp_server.core.index_manager import IndexManager
from mcp_server.tools import BaseTool

try:
    from mcp.types import Tool
except Exception:  # pragma: no cover
    Tool = dict  # type: ignore[assignment]

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None

TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{1,}")
YEAR_PATTERN = re.compile(r"(20\d{2})(?!\d|万)")


def _tokenize(text: str) -> set[str]:
    if jieba is not None:
        tokens = [tok.strip() for tok in jieba.cut(text or "", cut_all=False) if tok and tok.strip()]
    else:
        tokens = TOKEN_PATTERN.findall(text or "")
    return {t.lower() for t in tokens if t}


def _parse_range(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    years = [int(y) for y in YEAR_PATTERN.findall(text)]
    if not years:
        return None
    return min(years), max(years)


class NavigationTool(BaseTool):
    def __init__(self, index_manager: IndexManager) -> None:
        self.index_manager = index_manager

    def get_definition(self) -> Tool | dict[str, Any]:
        return {
            "name": "query_navigation_index",
            "description": "根据关键词和时间范围从知识库导航索引中筛选候选文件",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "time_range": {"type": ["string", "null"]},
                    "format_preference": {"type": ["array", "null"], "items": {"type": "string"}},
                    "max_results": {"type": "integer", "default": 8},
                },
                "required": ["keywords"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        keywords = arguments.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            return self.error("keywords is required", arguments)
        time_range = arguments.get("time_range")
        format_preference = arguments.get("format_preference") or []
        max_results = int(arguments.get("max_results", 8))

        keyword_tokens = _tokenize(" ".join(str(k) for k in keywords))
        range_q = _parse_range(str(time_range)) if time_range else None

        files = self.index_manager.get_navigation_files()
        ranked: list[dict[str, Any]] = []
        for file_item in files:
            file_range = _parse_range(str(file_item.get("time_range", "")))
            if range_q and file_range and not self._range_overlaps(range_q, file_range):
                continue

            joined = " ".join(
                [
                    str(file_item.get("description", "")),
                    " ".join(file_item.get("keywords", []) or []),
                    str(file_item.get("file_name", "")),
                ]
            )
            candidate_tokens = _tokenize(joined)
            overlap = keyword_tokens & candidate_tokens
            if not keyword_tokens:
                score = 0.0
            else:
                score = len(overlap) / len(keyword_tokens)

            file_format = str(file_item.get("file_format", "")).lower()
            if format_preference and file_format in [str(x).lower() for x in format_preference]:
                score += 0.1
            if score <= 0:
                continue

            ranked.append(
                {
                    "file_name": file_item.get("file_name", ""),
                    "file_path": file_item.get("file_path", ""),
                    "file_format": file_item.get("file_format", ""),
                    "description": file_item.get("description", ""),
                    "time_range": file_item.get("time_range"),
                    "match_score": round(score, 4),
                    "match_reason": f"keyword overlap={len(overlap)}",
                }
            )

        ranked.sort(key=lambda x: x["match_score"], reverse=True)
        candidates = ranked[:max_results]
        return {
            "total_files": len(files),
            "matched": len(candidates),
            "candidates": candidates,
            "all_files": [{"file_name": item.get("file_name", "")} for item in files],
        }

    @staticmethod
    def _range_overlaps(r1: tuple[int, int], r2: tuple[int, int]) -> bool:
        return not (r1[1] < r2[0] or r2[1] < r1[0])
