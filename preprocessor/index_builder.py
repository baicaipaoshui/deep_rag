from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preprocessor.parsers import ExcelParser, MarkdownParser, PDFParser
from project_config import build_ssl_context, load_project_config

try:
    from whoosh import index as whoosh_index
    from whoosh.fields import ID, STORED, TEXT, Schema
except Exception:  # pragma: no cover - optional dependency
    whoosh_index = None
    Schema = None
    ID = TEXT = STORED = None

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

SUPPORTED_SUFFIXES = {".md", ".pdf", ".xlsx", ".xlsm", ".xls"}
YEAR_PATTERN = re.compile(r"(20\d{2})(?!\d|万)")


@dataclass
class BuildStats:
    total_files: int = 0
    total_sections: int = 0
    total_chunks: int = 0
    whoosh_chunks: int = 0
    chroma_chunks: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total_files": self.total_files,
            "total_sections": self.total_sections,
            "total_chunks": self.total_chunks,
            "whoosh_chunks": self.whoosh_chunks,
            "chroma_chunks": self.chroma_chunks,
        }


class IndexBuilder:
    def __init__(
        self,
        kb_dir: str | Path = "knowledge_base",
        index_dir: str | Path = "index_store",
        ollama_base_url: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.kb_dir = Path(kb_dir)
        self.index_dir = Path(index_dir)
        self.discovery_dir = self.index_dir / "discovery"
        self.evidence_dir = self.index_dir / "evidence"
        self.whoosh_dir = self.evidence_dir / "whoosh_index"
        self.chroma_dir = self.evidence_dir / "chroma_db"
        self.chunks_file = self.evidence_dir / "chunks.jsonl"
        cfg = load_project_config()
        self.llm_provider = cfg.llm_provider
        self.ollama_base_url = (ollama_base_url or cfg.ollama_base_url).rstrip("/")
        self.embedding_model = embedding_model or cfg.ollama_embed_model
        self.openai_base_url = cfg.openai_base_url.rstrip("/")
        self.openai_api_key = cfg.openai_api_key
        self.openai_embedding_model = cfg.openai_embed_model
        self.openai_ssl_context = build_ssl_context(cfg.openai_verify_ssl, cfg.openai_ca_bundle)

        self.chunk_size: int = cfg.chunk_size
        self.chunk_overlap: int = cfg.chunk_overlap

        self.md_parser = MarkdownParser()
        self.pdf_parser = PDFParser()
        self.excel_parser = ExcelParser()

    def build_index(self) -> dict[str, Any]:
        self.discovery_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        navigation_files: list[dict[str, Any]] = []
        evidence_chunks: list[dict[str, Any]] = []
        stats = BuildStats()

        for file_path in sorted(self.kb_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            parsed = self._parse_file(file_path)
            rel_path = file_path.relative_to(self.kb_dir).as_posix()
            file_name = file_path.name
            file_format = self._file_format(file_path)
            file_key = self._safe_key(rel_path)
            discovery_path = self.discovery_dir / f"{file_key}.json"

            sections = parsed.get("sections", [])
            stats.total_files += 1
            stats.total_sections += len(sections)

            navigation_entry = {
                "file_name": file_name,
                "file_path": rel_path,
                "discovery_file": discovery_path.name,
                "file_format": file_format,
                "description": parsed.get("description", ""),
                "time_range": parsed.get("time_range"),
                "keywords": parsed.get("keywords", []),
                "metadata": {
                    "folder": file_path.parent.relative_to(self.kb_dir).as_posix(),
                    "size_bytes": file_path.stat().st_size,
                },
            }
            navigation_files.append(navigation_entry)

            discovery_payload = {
                "file_name": file_name,
                "file_path": rel_path,
                "file_format": file_format,
                "description": parsed.get("description", ""),
                "time_range": parsed.get("time_range"),
                "sections": [
                    {
                        k: v
                        for k, v in section.items()
                        if k
                        in {
                            "section_id",
                            "heading",
                            "line_start",
                            "line_end",
                            "pages",
                            "sheet_name",
                            "columns",
                            "preview",
                            "has_tables",
                            "table_titles",
                            "metadata",
                        }
                    }
                    for section in sections
                ],
            }
            discovery_path.write_text(
                json.dumps(discovery_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            for section in sections:
                section_content = str(section.get("content", "")).strip()
                chunks = self._split_section_content(
                    section_content, max_chars=self.chunk_size, overlap=self.chunk_overlap
                )
                if not chunks:
                    continue
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    chunk_payload = {
                        "chunk_id": f"{file_key}::{section.get('section_id', 'sec')}::{chunk_idx:03d}",
                        "file_name": file_name,
                        "file_path": rel_path,
                        "file_format": file_format,
                        "section_id": section.get("section_id"),
                        "heading": section.get("heading", ""),
                        "content": chunk,
                        "time_range": parsed.get("time_range"),
                        "metadata": {
                            "source_discovery": discovery_path.name,
                            "chunk_index": chunk_idx,
                        },
                    }
                    evidence_chunks.append(chunk_payload)
                    stats.total_chunks += 1

        navigation_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_files": len(navigation_files),
            "files": navigation_files,
        }
        (self.index_dir / "navigation_index.json").write_text(
            json.dumps(navigation_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self.chunks_file.open("w", encoding="utf-8") as fh:
            for chunk in evidence_chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        stats.whoosh_chunks = self._build_whoosh_index(evidence_chunks)
        stats.chroma_chunks = self._build_chroma_index(evidence_chunks)
        return {
            "kb_dir": str(self.kb_dir),
            "index_dir": str(self.index_dir),
            **stats.as_dict(),
        }

    def _parse_file(self, file_path: Path) -> dict[str, Any]:
        suffix = file_path.suffix.lower()
        if suffix == ".md":
            parsed = self.md_parser.parse(file_path)
        elif suffix == ".pdf":
            parsed = self.pdf_parser.parse(file_path)
        elif suffix in {".xlsx", ".xlsm", ".xls"}:
            parsed = self.excel_parser.parse(file_path)
        else:  # pragma: no cover - guarded by suffix filter
            parsed = {"description": "", "keywords": [], "time_range": None, "sections": [], "raw_text": ""}

        if not parsed.get("time_range"):
            years = sorted(set(YEAR_PATTERN.findall(parsed.get("raw_text", ""))))
            if years:
                parsed["time_range"] = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
        return parsed

    @staticmethod
    def _file_format(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".md":
            return "markdown"
        if suffix == ".pdf":
            return "pdf"
        return "excel"

    @staticmethod
    def _safe_key(relative_path: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", relative_path)

    @staticmethod
    def _split_section_content(
        text: str, max_chars: int = 900, overlap: int = 0
    ) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [cleaned]

        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0
        overlap_suffix: str = ""

        for para in cleaned.split("\n"):
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) + 1 > max_chars and current:
                chunk_text = "\n".join(current)
                chunks.append(chunk_text)
                overlap_suffix = chunk_text[-overlap:] if overlap > 0 else ""
                if overlap_suffix:
                    current = [overlap_suffix, para]
                    current_len = len(overlap_suffix) + len(para) + 1
                else:
                    current = [para]
                    current_len = len(para)
            else:
                current.append(para)
                current_len += len(para) + 1

        if current:
            chunks.append("\n".join(current))
        return chunks

    def _build_whoosh_index(self, chunks: list[dict[str, Any]]) -> int:
        if whoosh_index is None or Schema is None:
            return 0
        self.whoosh_dir.mkdir(parents=True, exist_ok=True)
        schema = Schema(
            chunk_id=ID(stored=True, unique=True),
            file_name=ID(stored=True),
            file_path=ID(stored=True),
            section_id=ID(stored=True),
            heading=TEXT(stored=True),
            file_format=ID(stored=True),
            time_range=ID(stored=True),
            content=TEXT(stored=True),
            metadata=STORED,
        )
        if whoosh_index.exists_in(self.whoosh_dir):
            idx = whoosh_index.open_dir(self.whoosh_dir)
        else:
            idx = whoosh_index.create_in(self.whoosh_dir, schema)
        writer = idx.writer()
        for chunk in chunks:
            writer.update_document(
                chunk_id=chunk["chunk_id"],
                file_name=chunk["file_name"],
                file_path=chunk["file_path"],
                section_id=chunk.get("section_id") or "",
                heading=chunk.get("heading") or "",
                file_format=chunk.get("file_format") or "",
                time_range=chunk.get("time_range") or "",
                content=chunk.get("content") or "",
                metadata=chunk.get("metadata") or {},
            )
        writer.commit()
        return len(chunks)

    def _build_chroma_index(self, chunks: list[dict[str, Any]]) -> int:
        if chromadb is None:
            return 0
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            client.delete_collection("deep_rag_evidence")
        except Exception:
            pass
        collection = client.get_or_create_collection("deep_rag_evidence")

        documents = [chunk["content"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadatas.append(
                {
                    "file_name": chunk["file_name"],
                    "file_path": chunk["file_path"],
                    "section_id": chunk.get("section_id") or "",
                    "heading": chunk.get("heading") or "",
                    "file_format": chunk.get("file_format") or "",
                }
            )

        embeddings = self._embed_texts(documents)
        if not embeddings:
            return 0

        batch_size = 64
        for i in range(0, len(documents), batch_size):
            j = i + batch_size
            collection.add(
                ids=ids[i:j],
                documents=documents[i:j],
                metadatas=metadatas[i:j],
                embeddings=embeddings[i:j],
            )
        return len(documents)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            vector = self._embed_single(text)
            if vector is None:
                return []
            embeddings.append(vector)
        return embeddings

    def _embed_single(self, text: str) -> list[float] | None:
        if self.llm_provider == "openai_compatible":
            payload = {"model": self.openai_embedding_model, "input": text[:4000]}
            req = urllib.request.Request(
                url=f"{self.openai_base_url}/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}",
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

        payload = {"model": self.embedding_model, "prompt": text[:4000]}
        req = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vec = data.get("embedding")
                if isinstance(vec, list) and vec:
                    return [float(x) for x in vec]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return None
        return None
