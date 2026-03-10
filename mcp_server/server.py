from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp_server.core.index_manager import IndexManager
from mcp_server.tools.discovery_tool import DiscoveryTool
from mcp_server.tools.evidence_search_tool import EvidenceSearchTool
from mcp_server.tools.file_reader_tool import FileReaderTool
from mcp_server.tools.folder_browse_tool import FolderBrowseTool
from mcp_server.tools.navigation_tool import NavigationTool
from mcp_server.tools.verify_tool import VerifyTool

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent
except Exception as exc:  # pragma: no cover - optional dependency
    Server = None  # type: ignore[assignment]
    stdio_server = None  # type: ignore[assignment]
    TextContent = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

index_manager = IndexManager()
ALL_TOOLS = {
    "query_navigation_index": NavigationTool(index_manager),
    "query_discovery_index": DiscoveryTool(index_manager),
    "read_file_section": FileReaderTool(index_manager),
    "search_evidence": EvidenceSearchTool(index_manager),
    "verify_evidence_coverage": VerifyTool(index_manager),
    "browse_folder": FolderBrowseTool(index_manager),
}


async def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = ALL_TOOLS.get(name)
    if tool is None:
        return {"error": "unknown_tool", "message": f"Unknown tool: {name}", "detail": None}
    try:
        return await tool.execute(arguments or {})
    except Exception as exc:
        return {"error": "tool_error", "message": str(exc), "detail": {"tool": name}}


def _ensure_sdk() -> None:
    if Server is None or stdio_server is None or TextContent is None:
        raise RuntimeError(f"MCP SDK is required to run server: {_IMPORT_ERROR}")


async def main() -> None:
    _ensure_sdk()
    server = Server("deep-rag")

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return [tool.get_definition() for tool in ALL_TOOLS.values()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        result = await dispatch_tool(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
