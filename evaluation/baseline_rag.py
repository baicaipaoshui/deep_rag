from __future__ import annotations

from typing import Any

from mcp_server.core.index_manager import IndexManager
from mcp_server.core.search_engine import SearchEngine


class BaselineRAG:
    def __init__(self) -> None:
        self.index_manager = IndexManager()
        self.search_engine = SearchEngine(self.index_manager)

    def query(self, question: str, max_results: int = 5) -> dict[str, Any]:
        rows = self.search_engine.search(question, search_type="hybrid", max_results=max_results)
        if not rows:
            return {
                "answer": "未检索到相关内容。",
                "evidences": [],
                "retrieval_summary": {"strategy": "baseline_hybrid", "evidence_count": 0},
            }
        lines = ["基于检索内容："]
        for row in rows:
            lines.append(
                f"- {row.get('content_preview', '')}【来源：{row.get('file_name')}-{row.get('section_id')}】"
            )
        return {
            "answer": "\n".join(lines),
            "evidences": rows,
            "retrieval_summary": {"strategy": "baseline_hybrid", "evidence_count": len(rows)},
        }
