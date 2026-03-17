import sqlite3
from pathlib import Path

from preprocessor.connectors.sqlite_connector import SQLiteConnector
from preprocessor.run_preprocess import _materialize_source


def test_sqlite_connector_exports_markdown(tmp_path: Path) -> None:
    db_path = tmp_path / "kb.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT, content TEXT, updated_at TEXT)")
    conn.execute(
        "INSERT INTO documents (title, content, updated_at) VALUES (?, ?, ?)",
        ("季度复盘", "这是数据库里的知识内容。", "2026-03-17T10:00:00Z"),
    )
    conn.commit()
    conn.close()

    connector = SQLiteConnector(
        sqlite_path=db_path,
        query="SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents",
    )
    files = connector.export_markdown(tmp_path / "kb" / "db_import")
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "季度复盘" in content
    assert "source_table: documents" in content
    assert "这是数据库里的知识内容。" in content


def test_materialize_source_reports_count(tmp_path: Path) -> None:
    db_path = tmp_path / "kb.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT, content TEXT, updated_at TEXT)")
    conn.execute(
        "INSERT INTO documents (title, content, updated_at) VALUES (?, ?, ?)",
        ("公告", "增量同步测试", "2026-03-17T11:00:00Z"),
    )
    conn.commit()
    conn.close()

    class Args:
        source_type = "sqlite"
        sqlite_path = str(db_path)
        sqlite_query = "SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents"
        sqlite_output_subdir = "db_import"

    info = _materialize_source(Args(), tmp_path / "knowledge_base")
    assert info["source_type"] == "sqlite"
    assert info["materialized_files"] == 1


def test_materialize_source_mysql_dispatch(tmp_path: Path, monkeypatch) -> None:
    class FakeMySQLConnector:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def export_markdown(self, output_dir: Path):
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "mysql.md"
            p.write_text("# mysql", encoding="utf-8")
            return [p]

    monkeypatch.setattr("preprocessor.run_preprocess.MySQLConnector", FakeMySQLConnector)

    class Args:
        source_type = "mysql"
        mysql_host = "127.0.0.1"
        mysql_port = 3306
        mysql_user = "root"
        mysql_password = "pw"
        mysql_database = "demo"
        mysql_query = "select 1"
        mysql_output_subdir = "db_import_mysql"

    info = _materialize_source(Args(), tmp_path / "knowledge_base")
    assert info["source_type"] == "mysql"
    assert info["materialized_files"] == 1


def test_materialize_source_postgres_dispatch(tmp_path: Path, monkeypatch) -> None:
    class FakePostgresConnector:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def export_markdown(self, output_dir: Path):
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "postgres.md"
            p.write_text("# postgres", encoding="utf-8")
            return [p]

    monkeypatch.setattr("preprocessor.run_preprocess.PostgresConnector", FakePostgresConnector)

    class Args:
        source_type = "postgres"
        postgres_dsn = "postgresql://user:pw@localhost:5432/demo"
        postgres_query = "select 1"
        postgres_output_subdir = "db_import_postgres"

    info = _materialize_source(Args(), tmp_path / "knowledge_base")
    assert info["source_type"] == "postgres"
    assert info["materialized_files"] == 1
