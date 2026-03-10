from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from mcp_server.config import Config


class IndexManager:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._navigation_cache: dict[str, Any] | None = None
        self._discovery_cache: dict[str, dict[str, Any]] = {}
        self._chunks_cache: list[dict[str, Any]] | None = None

    def load_navigation_index(self, refresh: bool = False) -> dict[str, Any]:
        if self._navigation_cache is not None and not refresh:
            return self._navigation_cache
        path = self.config.navigation_index_file
        if not path.exists():
            self._navigation_cache = {"generated_at": None, "total_files": 0, "files": []}
            return self._navigation_cache
        self._navigation_cache = json.loads(path.read_text(encoding="utf-8"))
        return self._navigation_cache

    def get_navigation_files(self) -> list[dict[str, Any]]:
        nav = self.load_navigation_index()
        files = nav.get("files", [])
        if isinstance(files, list):
            return files
        return []

    def get_file_entry(self, file_name: str) -> dict[str, Any] | None:
        for item in self.get_navigation_files():
            if item.get("file_name") == file_name:
                return item
        return None

    def get_discovery_for_file(self, file_name: str) -> dict[str, Any] | None:
        file_entry = self.get_file_entry(file_name)
        if not file_entry:
            return None
        discovery_file = file_entry.get("discovery_file")
        if not discovery_file:
            return None
        if discovery_file in self._discovery_cache:
            return self._discovery_cache[discovery_file]
        path = self.config.discovery_dir / discovery_file
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._discovery_cache[discovery_file] = payload
        return payload

    def get_section(self, file_name: str, section_id: str) -> dict[str, Any] | None:
        discovery = self.get_discovery_for_file(file_name)
        if not discovery:
            return None
        for section in discovery.get("sections", []):
            if section.get("section_id") == section_id:
                return section
        return None

    def get_absolute_path(self, file_name: str) -> Path | None:
        entry = self.get_file_entry(file_name)
        if not entry:
            return None
        rel_path = entry.get("file_path")
        if not rel_path:
            return None
        return self.config.kb_dir / rel_path

    def load_evidence_chunks(self, refresh: bool = False) -> list[dict[str, Any]]:
        if self._chunks_cache is not None and not refresh:
            return self._chunks_cache
        if not self.config.chunks_file.exists():
            self._chunks_cache = []
            return self._chunks_cache
        chunks: list[dict[str, Any]] = []
        with self.config.chunks_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        self._chunks_cache = chunks
        return chunks

    def browse_folder(self, folder_path: str = "") -> dict[str, Any]:
        current = PurePosixPath(folder_path.strip("/")) if folder_path else PurePosixPath(".")
        files = self.get_navigation_files()

        folder_map: dict[str, dict[str, Any]] = {}
        file_items: list[dict[str, Any]] = []
        for item in files:
            rel_path = PurePosixPath(str(item.get("file_path", "")))
            if current != PurePosixPath("."):
                try:
                    relative_to_current = rel_path.relative_to(current)
                except ValueError:
                    continue
            else:
                relative_to_current = rel_path

            parts = relative_to_current.parts
            if len(parts) == 0:
                continue
            if len(parts) == 1:
                file_items.append(
                    {
                        "name": parts[0],
                        "type": "file",
                        "format": item.get("file_format"),
                        "description": item.get("description", ""),
                    }
                )
            else:
                folder_name = parts[0]
                if folder_name not in folder_map:
                    folder_map[folder_name] = {
                        "name": f"{folder_name}/",
                        "type": "folder",
                        "file_count": 0,
                        "description": f"{folder_name} documents",
                    }
                folder_map[folder_name]["file_count"] += 1

        all_items = sorted(folder_map.values(), key=lambda x: x["name"]) + sorted(
            file_items, key=lambda x: x["name"]
        )
        return {
            "current_path": "" if current == PurePosixPath(".") else current.as_posix(),
            "description": "Knowledge base folder view",
            "items": all_items,
        }

    @lru_cache(maxsize=1)
    def list_all_file_names(self) -> list[str]:
        return [f.get("file_name", "") for f in self.get_navigation_files() if f.get("file_name")]
