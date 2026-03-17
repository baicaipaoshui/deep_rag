from mcp_server.core.search_engine import SearchEngine


class _FakeConfig:
    openai_verify_ssl = True
    openai_ca_bundle = ""
    rrf_k = 1


class _FakeIndexManager:
    config = _FakeConfig()


def test_hybrid_search_uses_configured_rrf_k() -> None:
    engine = SearchEngine(_FakeIndexManager())  # type: ignore[arg-type]

    def _keyword(*args, **kwargs):
        return [
            {
                "chunk_id": "a",
                "file_name": "a.md",
                "section_id": "md_001",
                "heading": "A",
                "content_preview": "A",
                "relevance_score": 1.0,
                "match_type": "keyword",
            },
            {
                "chunk_id": "b",
                "file_name": "b.md",
                "section_id": "md_002",
                "heading": "B",
                "content_preview": "B",
                "relevance_score": 0.8,
                "match_type": "keyword",
            },
        ]

    def _vector(*args, **kwargs):
        return [
            {
                "chunk_id": "b",
                "file_name": "b.md",
                "section_id": "md_002",
                "heading": "B",
                "content_preview": "B",
                "relevance_score": 1.0,
                "match_type": "vector",
            }
        ]

    engine.keyword_search = _keyword  # type: ignore[method-assign]
    engine.vector_search = _vector  # type: ignore[method-assign]

    rows = engine.hybrid_search("query", max_results=2)
    assert rows[0]["chunk_id"] == "b"
    assert rows[0]["match_type"] == "hybrid"
    score_map = {row["chunk_id"]: row["relevance_score"] for row in rows}
    assert score_map["a"] == 0.6
