from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from typing import Any

from mcp_server.core.index_manager import IndexManager
from project_config import build_ssl_context

try:
    from whoosh import index as whoosh_index
    from whoosh.qparser import MultifieldParser
except Exception:  # pragma: no cover - optional dependency
    whoosh_index = None
    MultifieldParser = None

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency
    jieba = None

TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{1,}")


def _tokenize(text: str) -> set[str]:
    if jieba is not None:
        tokens = [tok.strip() for tok in jieba.cut(text or "", cut_all=False) if tok and tok.strip()]
    else:
        tokens = TOKEN_PATTERN.findall(text or "")
    return {tok.lower() for tok in tokens if tok and len(tok) >= 2}


class SearchEngine:
    def __init__(self, index_manager: IndexManager) -> None:
        self.index_manager = index_manager
        self.config = index_manager.config
        self.openai_ssl_context = build_ssl_context(
            self.config.openai_verify_ssl, self.config.openai_ca_bundle
        )

    def search(
        self,
        query: str,
        search_type: str = "hybrid",
        file_scope: list[str] | None = None,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        search_type = (search_type or "hybrid").lower()
        if search_type == "keyword":
            return self.keyword_search(query, file_scope, max_results)
        if search_type == "vector":
            return self.vector_search(query, file_scope, max_results)
        return self.hybrid_search(query, file_scope, max_results)

    def keyword_search(
        self, query: str, file_scope: list[str] | None = None, max_results: int = 5
    ) -> list[dict[str, Any]]:
        if whoosh_index is not None and self.config.whoosh_dir.exists():
            try:
                idx = whoosh_index.open_dir(self.config.whoosh_dir)
                with idx.searcher() as searcher:
                    parser = MultifieldParser(["content", "heading", "file_name"], schema=idx.schema)
                    q = parser.parse(query)
                    hits = searcher.search(q, limit=max_results * 3)
                    results: list[dict[str, Any]] = []
                    for hit in hits:
                        if file_scope and hit["file_name"] not in file_scope:
                            continue
                        results.append(
                            {
                                "chunk_id": hit["chunk_id"],
                                "file_name": hit["file_name"],
                                "section_id": hit["section_id"],
                                "heading": hit["heading"],
                                "content_preview": (hit["content"] or "")[:220],
                                "relevance_score": float(hit.score),
                                "match_type": "keyword",
                            }
                        )
                        if len(results) >= max_results:
                            break
                    if results:
                        return self._normalize_scores(results)
            except Exception:
                pass
        return self._lexical_fallback(query, file_scope, max_results, match_type="keyword")

    def vector_search(
        self, query: str, file_scope: list[str] | None = None, max_results: int = 5
    ) -> list[dict[str, Any]]:
        if chromadb is not None and self.config.chroma_dir.exists():
            vector = self._embed_query(query)
            if vector:
                try:
                    client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
                    collection = client.get_collection("deep_rag_evidence")
                    where = {"file_name": {"$in": file_scope}} if file_scope else None
                    result = collection.query(
                        query_embeddings=[vector],
                        n_results=max_results * 3,
                        where=where,
                        include=["metadatas", "documents", "distances"],
                    )
                    ids = result.get("ids", [[]])[0]
                    metas = result.get("metadatas", [[]])[0]
                    docs = result.get("documents", [[]])[0]
                    dists = result.get("distances", [[]])[0]
                    rows: list[dict[str, Any]] = []
                    for chunk_id, meta, doc, dist in zip(ids, metas, docs, dists):
                        rows.append(
                            {
                                "chunk_id": chunk_id,
                                "file_name": meta.get("file_name", ""),
                                "section_id": meta.get("section_id", ""),
                                "heading": meta.get("heading", ""),
                                "content_preview": (doc or "")[:220],
                                "relevance_score": 1.0 / (1.0 + float(dist)),
                                "match_type": "vector",
                            }
                        )
                    if rows:
                        return self._normalize_scores(rows)[:max_results]
                except Exception:
                    pass
        return self._lexical_fallback(query, file_scope, max_results, match_type="vector")

    def hybrid_search(
        self, query: str, file_scope: list[str] | None = None, max_results: int = 5
    ) -> list[dict[str, Any]]:
        keyword_rows = self.keyword_search(query, file_scope=file_scope, max_results=max_results * 2)
        vector_rows  = self.vector_search(query,  file_scope=file_scope, max_results=max_results * 2)

        rrf_k = max(1, int(getattr(self.config, "rrf_k", 60)))
        rrf_scores: dict[str, float] = {}
        doc_data: dict[str, dict] = {}

        for rank, row in enumerate(keyword_rows, start=1):
            cid = row["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in doc_data:
                doc_data[cid] = {**row, "match_type": "keyword"}

        for rank, row in enumerate(vector_rows, start=1):
            cid = row["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in doc_data:
                doc_data[cid] = {**row, "match_type": "vector"}
            elif doc_data[cid]["match_type"] == "keyword":
                doc_data[cid]["match_type"] = "hybrid"

        ranked = []
        for cid, score in rrf_scores.items():
            entry = dict(doc_data[cid])
            entry["relevance_score"] = score
            ranked.append(entry)

        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return self._normalize_scores(ranked[:max_results])

    def _lexical_fallback(
        self,
        query: str,
        file_scope: list[str] | None,
        max_results: int,
        match_type: str,
    ) -> list[dict[str, Any]]:
        chunks = self.index_manager.load_evidence_chunks()
        q_tokens = _tokenize(query)
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            if file_scope and chunk.get("file_name") not in file_scope:
                continue
            text = f"{chunk.get('heading', '')}\n{chunk.get('content', '')}"
            c_tokens = _tokenize(text)
            if not c_tokens or not q_tokens:
                continue
            overlap = len(q_tokens & c_tokens)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(q_tokens) * len(c_tokens))
            rows.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "file_name": chunk.get("file_name"),
                    "section_id": chunk.get("section_id"),
                    "heading": chunk.get("heading", ""),
                    "content_preview": str(chunk.get("content", ""))[:220],
                    "relevance_score": score,
                    "match_type": match_type,
                }
            )
        rows = sorted(rows, key=lambda x: x["relevance_score"], reverse=True)[:max_results]
        return self._normalize_scores(rows)

    @staticmethod
    def _normalize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        max_score = max(float(r.get("relevance_score", 0.0)) for r in rows)
        if max_score <= 0:
            return rows
        normalized = []
        for row in rows:
            row = dict(row)
            row["relevance_score"] = round(float(row["relevance_score"]) / max_score, 4)
            normalized.append(row)
        return normalized

    def _embed_query(self, text: str) -> list[float] | None:
        if self.config.llm_provider == "openai_compatible":
            payload = {"model": self.config.openai_embed_model, "input": text[:4000]}
            req = urllib.request.Request(
                url=f"{self.config.openai_base_url}/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.openai_api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15, context=self.openai_ssl_context) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    rows = data.get("data", []) or []
                    if rows and isinstance(rows[0], dict) and isinstance(rows[0].get("embedding"), list):
                        return [float(x) for x in rows[0]["embedding"]]
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
                return None
            return None

        payload = {"model": self.config.ollama_embed_model, "prompt": text[:4000]}
        req = urllib.request.Request(
            url=f"{self.config.ollama_base_url}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vec = data.get("embedding")
                if isinstance(vec, list) and vec:
                    return [float(x) for x in vec]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return None
        return None
