from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project_config import load_project_config


class ChatMemoryStore:
    """Simple local JSON store for multi-turn chat sessions."""

    def __init__(self, file_path: str | Path | None = None) -> None:
        cfg = load_project_config()
        default_path = cfg.log_file.parent / "chat_sessions.json"
        path = Path(file_path or os.getenv("CHAT_MEMORY_FILE", str(default_path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path = path

    def load_session(self, session_id: str) -> list[dict[str, str]]:
        data = self._read_all()
        rows = data.get(session_id, [])
        normalized: list[dict[str, str]] = []
        for item in rows:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                normalized.append({"question": question, "answer": answer})
        return normalized

    def save_session(self, session_id: str, turns: list[dict[str, str]]) -> None:
        data = self._read_all()
        payload = []
        for turn in turns:
            question = str(turn.get("question", "")).strip()
            answer = str(turn.get("answer", "")).strip()
            if question and answer:
                payload.append(
                    {
                        "question": question,
                        "answer": answer,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        data[session_id] = payload
        self._write_all(data)

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        turns = self.load_session(session_id)
        turns.append({"question": question, "answer": answer})
        self.save_session(session_id, turns)

    def _read_all(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {}
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_all(self, data: dict[str, Any]) -> None:
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

