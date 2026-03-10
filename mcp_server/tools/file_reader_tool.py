from __future__ import annotations

from typing import Any

from mcp_server.core.file_reader import FileReader
from mcp_server.core.index_manager import IndexManager
from mcp_server.tools import BaseTool

try:
    from mcp.types import Tool
except Exception:  # pragma: no cover
    Tool = dict  # type: ignore[assignment]


class FileReaderTool(BaseTool):
    def __init__(self, index_manager: IndexManager) -> None:
        self.reader = FileReader(index_manager)

    def get_definition(self) -> Tool | dict[str, Any]:
        return {
            "name": "read_file_section",
            "description": "定向读取文件章节内容",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string"},
                    "section_id": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 2000},
                    "filter": {"type": ["object", "null"]},
                    "max_rows": {"type": ["integer", "null"]},
                },
                "required": ["file_name", "section_id"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        file_name = arguments.get("file_name")
        section_id = arguments.get("section_id")
        if not file_name or not section_id:
            return self.error("file_name and section_id are required", arguments)
        return self.reader.read_section(
            file_name=str(file_name),
            section_id=str(section_id),
            max_chars=int(arguments.get("max_chars", 2000)),
            filter_values=arguments.get("filter"),
            max_rows=arguments.get("max_rows"),
        )
