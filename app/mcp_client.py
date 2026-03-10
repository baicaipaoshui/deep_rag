from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # pragma: no cover - optional dependency
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]

from mcp_server.server import dispatch_tool
from project_config import load_project_config


class MCPClient:
    def __init__(self) -> None:
        cfg = load_project_config()
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._mode: str = "local"
        self._project_root = Path(__file__).resolve().parents[1]
        self._prefer_stdio = cfg.mcp_transport == "stdio"

    async def connect(self) -> None:
        if self._prefer_stdio and ClientSession is not None and stdio_client is not None and StdioServerParameters is not None:
            try:
                params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "mcp_server.server"],
                    cwd=str(self._project_root),
                )
                stack = AsyncExitStack()
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = ClientSession(read_stream, write_stream)
                await session.initialize()
                self._stack = stack
                self._session = session
                self._mode = "stdio"
                return
            except Exception:
                if self._stack is not None:
                    await self._stack.aclose()
                self._stack = None
                self._session = None

        self._mode = "local"

    async def disconnect(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._mode == "stdio" and self._session is not None:
            result = await self._session.call_tool(name, arguments)
            if not result.content:
                return {"error": "empty_result", "message": f"tool returned empty content: {name}", "detail": None}
            text = result.content[0].text
            return json.loads(text)
        return await dispatch_tool(name, arguments)
