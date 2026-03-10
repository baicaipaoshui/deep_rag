from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    from mcp.types import Tool
except Exception:  # pragma: no cover - optional dependency
    Tool = dict  # type: ignore[assignment]


class BaseTool(ABC):
    @abstractmethod
    def get_definition(self) -> Tool | dict[str, Any]:
        pass

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pass

    @staticmethod
    def error(message: str, detail: Any = None) -> dict[str, Any]:
        return {"error": "tool_error", "message": message, "detail": detail}
