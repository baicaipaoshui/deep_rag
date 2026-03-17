from __future__ import annotations

import json

from mcp_server.core.index_manager import IndexManager
from mcp_server.core.search_engine import SearchEngine


def main() -> None:
    engine = SearchEngine(IndexManager())
    queries = [
        "Aurora 发布窗口是什么时候",
        "2025年Q4销售同比增长多少",
        "PostgreSQL 每周巡检做什么",
    ]
    for query in queries:
        rows = engine.search(query, search_type="hybrid", max_results=3)
        print(f"Q: {query}")
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        print("---")


if __name__ == "__main__":
    main()
