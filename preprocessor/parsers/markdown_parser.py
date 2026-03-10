from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency
    jieba = None

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_YEAR_PATTERN = re.compile(r"(20\d{2})(?!\d|万)")
_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "我们",
    "你们",
    "他们",
    "以及",
    "进行",
    "相关",
    "通过",
    "对于",
    "一个",
    "可以",
    "this",
    "that",
    "with",
    "from",
}


def _extract_keywords(text: str, limit: int = 16) -> list[str]:
    if not text:
        return []
    if jieba is not None:
        tokens = [
            token.strip()
            for token in jieba.cut(text, cut_all=False)
            if token and token.strip()
        ]
    else:
        tokens = _TOKEN_PATTERN.findall(text)
    counter: Counter[str] = Counter()
    for token in tokens:
        if len(token) < 2:
            continue
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        counter[token] += 1
    return [token for token, _ in counter.most_common(limit)]


class MarkdownParser:
    def parse(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        heading_points: list[tuple[int, str]] = []
        for idx, line in enumerate(lines, start=1):
            m = _HEADING_PATTERN.match(line)
            if m:
                heading_points.append((idx, m.group(2).strip()))

        sections: list[dict[str, Any]] = []
        if not heading_points:
            preview = "\n".join(lines[:40]).strip()
            sections.append(
                {
                    "section_id": "md_001",
                    "heading": "全文",
                    "line_start": 1,
                    "line_end": max(1, len(lines)),
                    "preview": preview[:300],
                    "content": text,
                    "has_tables": "|" in text,
                    "table_titles": [],
                    "metadata": {},
                }
            )
        else:
            for i, (line_start, heading) in enumerate(heading_points):
                next_line_start = (
                    heading_points[i + 1][0]
                    if i + 1 < len(heading_points)
                    else len(lines) + 1
                )
                line_end = next_line_start - 1
                block = "\n".join(lines[line_start - 1 : line_end]).strip()
                sections.append(
                    {
                        "section_id": f"md_{i + 1:03d}",
                        "heading": heading,
                        "line_start": line_start,
                        "line_end": line_end,
                        "preview": block[:300],
                        "content": block,
                        "has_tables": "|" in block,
                        "table_titles": [],
                        "metadata": {},
                    }
                )

        summary_lines = [line.strip() for line in lines if line.strip()]
        description = " ".join(summary_lines[:3])[:220]
        years = sorted(set(_YEAR_PATTERN.findall(text)))
        time_range = None
        if years:
            time_range = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"

        return {
            "description": description,
            "keywords": _extract_keywords(text),
            "time_range": time_range,
            "sections": sections,
            "raw_text": text,
        }
