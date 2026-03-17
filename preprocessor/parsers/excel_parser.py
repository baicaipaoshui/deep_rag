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
_COLUMN_CHUNK_SIZE = 6
_ROW_WINDOW_SIZE = 100
_ROW_WINDOW_STEP = 80
_SUMMARY_SAMPLE_ROWS = 8
_COLUMN_CHUNK_SAMPLE_ROWS = 20
_MAX_PREVIEW_CHARS = 300


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
                        "section_id": "sh_001_summary",
                        "heading": "Sheet1 摘要",
                        "sheet_name": "Sheet1",
                        "columns": [],
                        "preview": fallback,
                        "content": fallback,
                        "has_tables": True,
                        "table_titles": ["Sheet1"],
                        "metadata": {"degraded": True, "section_type": "summary"},
                    }
                ],
                "raw_text": fallback,
            }

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        sections: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for i, ws in enumerate(wb.worksheets, start=1):
            sheet_sections, sheet_text = self._build_sheet_sections(ws, i)
            sections.extend(sheet_sections)
            text_parts.append(sheet_text)

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

    def _build_sheet_sections(self, ws: Any, sheet_index: int) -> tuple[list[dict[str, Any]], str]:
        rows = [list(row) for row in ws.iter_rows(values_only=True)]
        if not rows:
            empty_content = f"[Empty sheet] {ws.title}"
            return (
                [
                    {
                        "section_id": f"sh_{sheet_index:03d}_summary",
                        "heading": f"{ws.title} 摘要",
                        "sheet_name": ws.title,
                        "columns": [],
                        "preview": empty_content[:_MAX_PREVIEW_CHARS],
                        "content": empty_content,
                        "rows": [],
                        "has_tables": True,
                        "table_titles": [ws.title],
                        "metadata": {"sheet_index": sheet_index, "section_type": "summary", "total_rows": 0},
                    }
                ],
                empty_content,
            )

        header_cells = _row_to_cells(tuple(rows[0]))
        if not any(header_cells):
            width = max((len(row) for row in rows), default=0)
            header_cells = [f"col_{idx + 1}" for idx in range(width)]
        headers = [col if col else f"col_{idx + 1}" for idx, col in enumerate(header_cells)]
        data_rows = [
            _row_to_cells(tuple(row))
            for row in rows[1:]
            if any(cell is not None and str(cell).strip() for cell in row)
        ]
        total_rows = len(data_rows)
        total_cols = len(headers)
        all_row_texts = [" | ".join(row[:total_cols]) for row in data_rows]
        text_payload = "\n".join([" | ".join(headers), *all_row_texts]).strip()

        sections: list[dict[str, Any]] = []
        summary_rows = data_rows[:_SUMMARY_SAMPLE_ROWS]
        summary_lines = [
            f"工作表：{ws.title}",
            f"总列数：{total_cols}",
            f"总行数：{total_rows}",
            f"列名：{' | '.join(headers)}",
        ]
        if summary_rows:
            summary_lines.append("样本行：")
            summary_lines.extend(" | ".join(row[:total_cols]) for row in summary_rows)
        summary_content = "\n".join(summary_lines)
        sections.append(
            {
                "section_id": f"sh_{sheet_index:03d}_summary",
                "heading": f"{ws.title} 摘要",
                "sheet_name": ws.title,
                "columns": headers,
                "preview": summary_content[:_MAX_PREVIEW_CHARS],
                "content": summary_content,
                "rows": summary_rows,
                "has_tables": True,
                "table_titles": [ws.title],
                "metadata": {
                    "sheet_index": sheet_index,
                    "section_type": "summary",
                    "total_rows": total_rows,
                    "total_columns": total_cols,
                },
            }
        )

        col_chunk_idx = 1
        for start in range(0, total_cols, _COLUMN_CHUNK_SIZE):
            end = min(total_cols, start + _COLUMN_CHUNK_SIZE)
            indices = list(range(start, end))
            chunk_headers = [headers[idx] for idx in indices]
            chunk_rows = [[row[idx] if idx < len(row) else "" for idx in indices] for row in data_rows]
            sample_rows = chunk_rows[:_COLUMN_CHUNK_SAMPLE_ROWS]
            lines = [" | ".join(chunk_headers)]
            lines.extend(" | ".join(row) for row in sample_rows)
            content = "\n".join(lines).strip() or f"{ws.title} 列分块 {col_chunk_idx}"
            sections.append(
                {
                    "section_id": f"sh_{sheet_index:03d}_col_{col_chunk_idx:03d}",
                    "heading": f"{ws.title} 列分块 {col_chunk_idx}",
                    "sheet_name": ws.title,
                    "columns": chunk_headers,
                    "preview": content[:_MAX_PREVIEW_CHARS],
                    "content": content,
                    "rows": sample_rows,
                    "has_tables": True,
                    "table_titles": [ws.title],
                    "metadata": {
                        "sheet_index": sheet_index,
                        "section_type": "column_chunk",
                        "column_start": start + 1,
                        "column_end": end,
                        "column_names": chunk_headers,
                    },
                }
            )
            col_chunk_idx += 1

        row_chunk_idx = 1
        if total_rows == 0:
            sections.append(
                {
                    "section_id": f"sh_{sheet_index:03d}_rw_001",
                    "heading": f"{ws.title} 行窗口 1",
                    "sheet_name": ws.title,
                    "columns": headers,
                    "preview": "[No data rows]",
                    "content": " | ".join(headers),
                    "rows": [],
                    "has_tables": True,
                    "table_titles": [ws.title],
                    "metadata": {
                        "sheet_index": sheet_index,
                        "section_type": "row_window",
                        "row_start": 2,
                        "row_end": 1,
                        "window_size": _ROW_WINDOW_SIZE,
                        "step": _ROW_WINDOW_STEP,
                    },
                }
            )
        else:
            for start in range(0, total_rows, _ROW_WINDOW_STEP):
                window_rows = data_rows[start : start + _ROW_WINDOW_SIZE]
                if not window_rows:
                    continue
                lines = [" | ".join(headers)]
                lines.extend(" | ".join(row[:total_cols]) for row in window_rows)
                content = "\n".join(lines).strip()
                row_start = start + 2
                row_end = row_start + len(window_rows) - 1
                sections.append(
                    {
                        "section_id": f"sh_{sheet_index:03d}_rw_{row_chunk_idx:03d}",
                        "heading": f"{ws.title} 行窗口 {row_chunk_idx}",
                        "sheet_name": ws.title,
                        "columns": headers,
                        "preview": content[:_MAX_PREVIEW_CHARS],
                        "content": content,
                        "rows": window_rows[:30],
                        "has_tables": True,
                        "table_titles": [ws.title],
                        "metadata": {
                            "sheet_index": sheet_index,
                            "section_type": "row_window",
                            "row_start": row_start,
                            "row_end": row_end,
                            "window_size": _ROW_WINDOW_SIZE,
                            "step": _ROW_WINDOW_STEP,
                        },
                    }
                )
                row_chunk_idx += 1

        return sections, text_payload
