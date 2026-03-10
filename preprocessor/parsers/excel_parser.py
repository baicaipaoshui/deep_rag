from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import openpyxl
except Exception:  # pragma: no cover - optional dependency
    openpyxl = None

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency
    jieba = None

_YEAR_PATTERN = re.compile(r"(20\d{2})(?!\d|万)")
_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")


def _extract_keywords(text: str, limit: int = 16) -> list[str]:
    if not text:
        return []
    if jieba is not None:
        tokens = [token.strip() for token in jieba.cut(text) if token.strip()]
    else:
        tokens = _TOKEN_PATTERN.findall(text)
    freq: dict[str, int] = {}
    for token in tokens:
        if len(token) < 2:
            continue
        freq[token] = freq.get(token, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]]


def _row_to_cells(values: tuple[Any, ...]) -> list[str]:
    return ["" if v is None else str(v).strip() for v in values]


class ExcelParser:
    def parse(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        if openpyxl is None:
            fallback = f"[Excel parsing unavailable] {path.name}"
            return {
                "description": fallback,
                "keywords": _extract_keywords(path.stem),
                "time_range": None,
                "sections": [
                    {
                        "section_id": "sh_001",
                        "heading": "Sheet1",
                        "sheet_name": "Sheet1",
                        "columns": [],
                        "preview": fallback,
                        "content": fallback,
                        "has_tables": True,
                        "table_titles": ["Sheet1"],
                        "metadata": {"degraded": True},
                    }
                ],
                "raw_text": fallback,
            }

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        sections: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for i, ws in enumerate(wb.worksheets, start=1):
            rows = ws.iter_rows(values_only=True)
            header_cells: list[str] = []
            row_texts: list[str] = []
            row_samples: list[list[str]] = []
            for row_idx, row in enumerate(rows, start=1):
                cells = _row_to_cells(row)
                if row_idx == 1:
                    header_cells = cells
                if not any(cells):
                    continue
                row_samples.append(cells)
                row_texts.append(" | ".join(cells))
                if len(row_texts) >= 120:
                    break

            content = "\n".join(row_texts).strip() or f"[Empty sheet] {ws.title}"
            text_parts.append(content)
            sections.append(
                {
                    "section_id": f"sh_{i:03d}",
                    "heading": ws.title,
                    "sheet_name": ws.title,
                    "columns": [c for c in header_cells if c],
                    "preview": content[:300],
                    "content": content,
                    "rows": row_samples[:50],
                    "has_tables": True,
                    "table_titles": [ws.title],
                    "metadata": {"sheet_index": i},
                }
            )

        wb.close()
        raw_text = "\n".join(text_parts)
        years = sorted(set(_YEAR_PATTERN.findall(raw_text)))
        time_range = None
        if years:
            time_range = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"

        return {
            "description": f"Workbook {path.name} with {len(sections)} sheets",
            "keywords": _extract_keywords(raw_text),
            "time_range": time_range,
            "sections": sections,
            "raw_text": raw_text,
        }
