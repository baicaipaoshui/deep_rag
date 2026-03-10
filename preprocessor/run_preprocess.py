from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessor.index_builder import IndexBuilder
from project_config import load_project_config


def parse_args() -> argparse.Namespace:
    cfg = load_project_config()
    parser = argparse.ArgumentParser(description="Build Deep RAG indexes.")
    parser.add_argument(
        "--kb-dir",
        default=str(cfg.kb_dir),
        help="Knowledge base directory",
    )
    parser.add_argument(
        "--index-dir",
        default=str(cfg.index_dir),
        help="Index output directory",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=cfg.ollama_base_url,
        help="Ollama API base URL for embeddings",
    )
    parser.add_argument(
        "--embed-model",
        default=cfg.ollama_embed_model,
        help="Embedding model name in Ollama",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kb_dir = Path(args.kb_dir)
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    builder = IndexBuilder(
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        ollama_base_url=args.ollama_base_url,
        embedding_model=args.embed_model,
    )
    result = builder.build_index()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
