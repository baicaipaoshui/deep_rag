from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
    RealDictCursor = None


class PostgresConnector:
    def __init__(self, dsn: str, query: str) -> None:
        self.dsn = dsn
        self.query = query

    def fetch_documents(self) -> list[dict[str, Any]]:
        if psycopg2 is None or RealDictCursor is None:
            raise ImportError("psycopg2-binary is required for postgres source. Install with: pip install psycopg2-binary")
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(self.query)
                rows = cursor.fetchall()
        finally:
            conn.close()
        docs: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            row_data = dict(row)
            doc_id = str(row_data.get("id")) if row_data.get("id") is not None else f"row_{idx}"
            title = str(row_data.get("title") or f"Document {idx}")
            content = str(row_data.get("content") or "")
            updated_at = str(row_data.get("updated_at") or datetime.now(timezone.utc).isoformat())
            source_table = str(row_data.get("source_table") or "postgres")
            if not content.strip():
                continue
            docs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                    "updated_at": updated_at,
                    "source_table": source_table,
                }
            )
        return docs

    def export_markdown(self, target_dir: str | Path) -> list[Path]:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for doc in self.fetch_documents():
            file_name = self._safe_name(f"{doc['source_table']}_{doc['doc_id']}_{doc['title']}") + ".md"
            path = target / file_name
            markdown = "\n".join(
                [
                    f"# {doc['title']}",
                    "",
                    f"- source_table: {doc['source_table']}",
                    f"- source_id: {doc['doc_id']}",
                    f"- updated_at: {doc['updated_at']}",
                    "",
                    "## Content",
                    "",
                    str(doc["content"]),
                    "",
                ]
            )
            path.write_text(markdown, encoding="utf-8")
            saved.append(path)
        return saved

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", value).strip("_")[:120] or "document"
