from __future__ import annotations

from typing import Any

from mcp_server.core.index_manager import IndexManager
from mcp_server.tools import BaseTool

try:
    from mcp.types import Tool
except Exception:  # pragma: no cover
    Tool = dict  # type: ignore[assignment]


class DiscoveryTool(BaseTool):
    def __init__(self, index_manager: IndexManager) -> None:
        self.index_manager = index_manager

    def get_definition(self) -> Tool | dict[str, Any]:
        return {
            "name": "query_discovery_index",
            "description": "读取指定文件的发现索引结构",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_names": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["file_names"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        file_names = arguments.get("file_names")
        if not isinstance(file_names, list) or not file_names:
            return self.error("file_names is required", arguments)

        files = []
        for file_name in file_names:
            discovery = self.index_manager.get_discovery_for_file(str(file_name))
            if discovery is None:
                files.append({"file_name": file_name, "error": "discovery index not found"})
            else:
                files.append(discovery)
        return {"files": files}
