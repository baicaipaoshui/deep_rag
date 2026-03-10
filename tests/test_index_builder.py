import json
from pathlib import Path

from preprocessor.index_builder import IndexBuilder


def test_index_builder_outputs_navigation_discovery_chunks(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "a.md").write_text("# A\n\n内容 2021 2022", encoding="utf-8")
    index_dir = tmp_path / "index"

    result = IndexBuilder(kb_dir=kb_dir, index_dir=index_dir).build_index()
    assert result["total_files"] == 1
    nav = json.loads((index_dir / "navigation_index.json").read_text(encoding="utf-8"))
    assert nav["total_files"] == 1
    assert (index_dir / "discovery").exists()
    assert (index_dir / "evidence" / "chunks.jsonl").exists()
