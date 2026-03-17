from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SQLiteDocument:
    doc_id: str
    title: str
    content: str
    updated_at: str
    source_table: str


class SQLiteConnector:
    def __init__(self, sqlite_path: str | Path, query: str) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.query = query

    def fetch_documents(self) -> list[SQLiteDocument]:
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.sqlite_path}")
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(self.query).fetchall()
        finally:
            conn.close()
        documents: list[SQLiteDocument] = []
        for idx, row in enumerate(rows, start=1):
            doc_id = str(row["id"]) if "id" in row.keys() and row["id"] is not None else f"row_{idx}"
            title = str(row["title"]) if "title" in row.keys() and row["title"] else f"Document {idx}"
            content = str(row["content"]) if "content" in row.keys() and row["content"] else ""
            updated_at = (
                str(row["updated_at"])
                if "updated_at" in row.keys() and row["updated_at"] is not None
                else datetime.now(timezone.utc).isoformat()
            )
            source_table = str(row["source_table"]) if "source_table" in row.keys() and row["source_table"] else "sqlite"
            if not content.strip():
                continue
            documents.append(
                SQLiteDocument(
                    doc_id=doc_id,
                    title=title,
                    content=content,
                    updated_at=updated_at,
                    source_table=source_table,
                )
            )
        return documents

    def export_markdown(self, target_dir: str | Path) -> list[Path]:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for doc in self.fetch_documents():
            file_name = self._safe_name(f"{doc.source_table}_{doc.doc_id}_{doc.title}") + ".md"
            path = target / file_name
            markdown = "\n".join(
                [
                    f"# {doc.title}",
                    "",
                    f"- source_table: {doc.source_table}",
                    f"- source_id: {doc.doc_id}",
                    f"- updated_at: {doc.updated_at}",
                    "",
                    "## Content",
                    "",
                    doc.content,
                    "",
                ]
            )
            path.write_text(markdown, encoding="utf-8")
            saved.append(path)
        return saved

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", value).strip("_")[:120] or "document"
