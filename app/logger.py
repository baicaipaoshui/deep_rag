from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project_config import load_project_config


def get_logger(name: str = "deep_rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        logger.addHandler(handler)
    return logger


class QueryLogger:
    def __init__(self, log_file: str | Path | None = None) -> None:
        cfg = load_project_config()
        file_path = Path(log_file or os.getenv("LOG_FILE", str(cfg.log_file)))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = file_path

    def log(self, payload: dict[str, Any]) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
