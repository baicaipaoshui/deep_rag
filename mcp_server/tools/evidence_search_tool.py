from __future__ import annotations

from typing import Any

from mcp_server.core.index_manager import IndexManager
from mcp_server.core.search_engine import SearchEngine
from mcp_server.tools import BaseTool

try:
    from mcp.types import Tool
except Exception:  # pragma: no cover
    Tool = dict  # type: ignore[assignment]


class EvidenceSearchTool(BaseTool):
    def __init__(self, index_manager: IndexManager) -> None:
        self.engine = SearchEngine(index_manager)

    def get_definition(self) -> Tool | dict[str, Any]:
        return {
            "name": "search_evidence",
            "description": "关键词/向量/混合证据搜索",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_type": {"type": "string", "enum": ["keyword", "vector", "hybrid"], "default": "hybrid"},
                    "file_scope": {"type": ["array", "null"], "items": {"type": "string"}},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return self.error("query is required", arguments)
        results = self.engine.search(
            query=query,
            search_type=str(arguments.get("search_type", "hybrid")),
            file_scope=arguments.get("file_scope"),
            max_results=int(arguments.get("max_results", 5)),
        )
        return {"results": results}
