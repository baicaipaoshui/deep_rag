from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessor.connectors import MySQLConnector, PostgresConnector, SQLiteConnector
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
    parser.add_argument(
        "--source-type",
        choices=["file", "sqlite", "mysql", "postgres"],
        default="file",
        help="Knowledge source type: file, sqlite, mysql, postgres",
    )
    parser.add_argument(
        "--sqlite-path",
        default="",
        help="SQLite database path when source-type=sqlite",
    )
    parser.add_argument(
        "--sqlite-query",
        default="SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents",
        help="SQLite query for source documents",
    )
    parser.add_argument(
        "--sqlite-output-subdir",
        default="db_import",
        help="Sub directory under kb-dir to materialize sqlite rows into markdown",
    )
    parser.add_argument(
        "--mysql-host",
        default="127.0.0.1",
        help="MySQL host when source-type=mysql",
    )
    parser.add_argument(
        "--mysql-port",
        type=int,
        default=3306,
        help="MySQL port when source-type=mysql",
    )
    parser.add_argument(
        "--mysql-user",
        default="",
        help="MySQL user when source-type=mysql",
    )
    parser.add_argument(
        "--mysql-password",
        default="",
        help="MySQL password when source-type=mysql",
    )
    parser.add_argument(
        "--mysql-database",
        default="",
        help="MySQL database when source-type=mysql",
    )
    parser.add_argument(
        "--mysql-query",
        default="SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents",
        help="MySQL query for source documents",
    )
    parser.add_argument(
        "--mysql-output-subdir",
        default="db_import_mysql",
        help="Sub directory under kb-dir for mysql materialized markdown",
    )
    parser.add_argument(
        "--postgres-dsn",
        default="",
        help="PostgreSQL DSN when source-type=postgres",
    )
    parser.add_argument(
        "--postgres-query",
        default="SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents",
        help="PostgreSQL query for source documents",
    )
    parser.add_argument(
        "--postgres-output-subdir",
        default="db_import_postgres",
        help="Sub directory under kb-dir for postgres materialized markdown",
    )
    return parser.parse_args()


def _materialize_source(args: argparse.Namespace, kb_dir: Path) -> dict[str, object]:
    if args.source_type == "file":
        return {"source_type": "file", "materialized_files": 0}
    if args.source_type == "sqlite":
        if not args.sqlite_path:
            raise ValueError("sqlite source requires --sqlite-path")
        connector = SQLiteConnector(sqlite_path=args.sqlite_path, query=args.sqlite_query)
        output_dir = kb_dir / args.sqlite_output_subdir
        exported = connector.export_markdown(output_dir)
        return {"source_type": "sqlite", "materialized_files": len(exported), "materialized_dir": str(output_dir)}
    if args.source_type == "mysql":
        if not args.mysql_user or not args.mysql_database:
            raise ValueError("mysql source requires --mysql-user and --mysql-database")
        connector = MySQLConnector(
            host=args.mysql_host,
            port=args.mysql_port,
            user=args.mysql_user,
            password=args.mysql_password,
            database=args.mysql_database,
            query=args.mysql_query,
        )
        output_dir = kb_dir / args.mysql_output_subdir
        exported = connector.export_markdown(output_dir)
        return {"source_type": "mysql", "materialized_files": len(exported), "materialized_dir": str(output_dir)}
    if args.source_type == "postgres":
        if not args.postgres_dsn:
            raise ValueError("postgres source requires --postgres-dsn")
        connector = PostgresConnector(dsn=args.postgres_dsn, query=args.postgres_query)
        output_dir = kb_dir / args.postgres_output_subdir
        exported = connector.export_markdown(output_dir)
        return {"source_type": "postgres", "materialized_files": len(exported), "materialized_dir": str(output_dir)}
    raise ValueError(f"unsupported source type: {args.source_type}")


def main() -> None:
    args = parse_args()
    kb_dir = Path(args.kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)
    source_info = _materialize_source(args, kb_dir)

    builder = IndexBuilder(
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        ollama_base_url=args.ollama_base_url,
        embedding_model=args.embed_model,
    )
    result = builder.build_index()
    result.update(source_info)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
