from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import fitz
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency
    jieba = None

_YEAR_PATTERN = re.compile(r"(20\d{2})(?!\d|万)")
_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")


def _has_table_like(text: str) -> bool:
    lines = text.splitlines()
    table_lines = sum(1 for l in lines if l.count("|") >= 2 or l.count("\t") >= 2)
    return table_lines >= 2


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


class PDFParser:
    def parse(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        if fitz is None:
            fallback = f"[PDF parsing unavailable] {path.name}"
            return {
                "description": fallback,
                "keywords": _extract_keywords(path.stem),
                "time_range": None,
                "sections": [
                    {
                        "section_id": "p001",
                        "heading": "Page 1",
                        "pages": "1",
                        "preview": fallback,
                        "content": fallback,
                        "has_tables": False,
                        "table_titles": [],
                        "metadata": {"degraded": True},
                    }
                ],
                "raw_text": fallback,
            }

        sections: list[dict[str, Any]] = []
        joined_text_parts: list[str] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                cleaned = text.strip() or f"[No extractable text on page {i}]"
                joined_text_parts.append(cleaned)
                sections.append(
                    {
                        "section_id": f"p{i:03d}",
                        "heading": f"Page {i}",
                        "pages": str(i),
                        "preview": cleaned[:300],
                        "content": cleaned,
                        "has_tables": _has_table_like(cleaned),
                        "table_titles": [],
                        "metadata": {"page": i},
                    }
                )

        raw_text = "\n".join(joined_text_parts)
        years = sorted(set(_YEAR_PATTERN.findall(raw_text)))
        time_range = None
        if years:
            time_range = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"

        return {
            "description": (joined_text_parts[0] if joined_text_parts else path.stem)[:220],
            "keywords": _extract_keywords(raw_text),
            "time_range": time_range,
            "sections": sections,
            "raw_text": raw_text,
        }
