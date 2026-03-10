from __future__ import annotations

from typing import Any

from mcp_server.core.index_manager import IndexManager
from mcp_server.tools import BaseTool

try:
    from mcp.types import Tool
except Exception:  # pragma: no cover
    Tool = dict  # type: ignore[assignment]


class FolderBrowseTool(BaseTool):
    def __init__(self, index_manager: IndexManager) -> None:
        self.index_manager = index_manager

    def get_definition(self) -> Tool | dict[str, Any]:
        return {
            "name": "browse_folder",
            "description": "浏览知识库文件夹摘要",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "folder_path": {"type": "string", "default": ""},
                },
                "required": [],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        folder_path = str(arguments.get("folder_path", "") or "")
        return self.index_manager.browse_folder(folder_path)
