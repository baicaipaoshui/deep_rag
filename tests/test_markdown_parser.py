from pathlib import Path

from preprocessor.parsers.markdown_parser import MarkdownParser


def test_markdown_parser_extracts_sections(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text(
        "# 标题\n\n公司成立于2015年。\n\n## 第二节\n\n2023年营收5280万元。",
        encoding="utf-8",
    )
    parsed = MarkdownParser().parse(file_path)
    assert parsed["time_range"] == "2015-2023"
    assert len(parsed["sections"]) >= 2
    assert any(sec["section_id"] == "md_001" for sec in parsed["sections"])
