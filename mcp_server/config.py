from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project_config import load_project_config

@dataclass(frozen=True)
class Config:
    project_root: Path
    kb_dir: Path
    index_dir: Path
    navigation_index_file: Path
    discovery_dir: Path
    evidence_dir: Path
    whoosh_dir: Path
    chroma_dir: Path
    chunks_file: Path
    llm_provider: str
    ollama_base_url: str
    ollama_embed_model: str
    openai_base_url: str
    openai_api_key: str
    openai_embed_model: str
    openai_verify_ssl: bool
    openai_ca_bundle: str

    @classmethod
    def from_env(cls) -> "Config":
        cfg = load_project_config()
        project_root = cfg.project_root
        kb_dir = cfg.kb_dir
        index_dir = cfg.index_dir
        evidence_dir = index_dir / "evidence"
        return cls(
            project_root=project_root,
            kb_dir=kb_dir,
            index_dir=index_dir,
            navigation_index_file=index_dir / "navigation_index.json",
            discovery_dir=index_dir / "discovery",
            evidence_dir=evidence_dir,
            whoosh_dir=evidence_dir / "whoosh_index",
            chroma_dir=evidence_dir / "chroma_db",
            chunks_file=evidence_dir / "chunks.jsonl",
            llm_provider=cfg.llm_provider,
            ollama_base_url=cfg.ollama_base_url,
            ollama_embed_model=cfg.ollama_embed_model,
            openai_base_url=cfg.openai_base_url,
            openai_api_key=cfg.openai_api_key,
            openai_embed_model=cfg.openai_embed_model,
            openai_verify_ssl=cfg.openai_verify_ssl,
            openai_ca_bundle=cfg.openai_ca_bundle,
        )
