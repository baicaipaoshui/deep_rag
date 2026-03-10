import asyncio

from mcp_server.tools.navigation_tool import NavigationTool


class _FakeIndexManager:
    def get_navigation_files(self):
        return [
            {
                "file_name": "sales_2023.md",
                "file_path": "sales/sales_2023.md",
                "file_format": "markdown",
                "description": "2023 销售 数据",
                "time_range": "2023",
                "keywords": ["销售", "数据"],
            },
            {
                "file_name": "tech.md",
                "file_path": "tech/tech.md",
                "file_format": "markdown",
                "description": "前端 性能",
                "time_range": "2023",
                "keywords": ["前端"],
            },
        ]


def test_navigation_tool_filters_by_keywords() -> None:
    tool = NavigationTool(_FakeIndexManager())  # type: ignore[arg-type]
    result = asyncio.run(tool.execute({"keywords": ["销售"], "max_results": 5}))
    assert result["matched"] >= 1
    assert result["candidates"][0]["file_name"] == "sales_2023.md"
