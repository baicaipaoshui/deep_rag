import asyncio

from mcp_server.tools.verify_tool import VerifyTool


class _FakeIndexManager:
    def get_navigation_files(self):
        return [
            {"file_name": "2021_sales.pdf", "description": "2021 销售 数据", "keywords": ["2021", "销售"]},
            {"file_name": "2022_sales.pdf", "description": "2022 销售 数据", "keywords": ["2022", "销售"]},
        ]


def test_verify_tool_detects_missing_year() -> None:
    tool = VerifyTool(_FakeIndexManager())  # type: ignore[arg-type]
    result = asyncio.run(
        tool.execute(
            {
                "query_type": "trend_analysis",
                "time_range": "2021-2023",
                "expected_dimensions": [],
                "collected_evidence": [
                    {"time_period": "2021", "content": "2021增长", "evidence_strength": "direct"},
                    {"time_period": "2022", "content": "2022增长", "evidence_strength": "direct"},
                ],
                "supplement_round": 0,
            }
        )
    )
    assert result["is_sufficient"] is False
    assert any("2023" in item for item in result["missing"])
