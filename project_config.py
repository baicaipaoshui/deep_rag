from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    config_file: Path | None

    llm_provider: str

    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str

    openai_base_url: str
    openai_api_key: str
    openai_chat_model: str
    openai_embed_model: str
    openai_verify_ssl: bool
    openai_ca_bundle: str

    kb_dir: Path
    index_dir: Path
    log_file: Path

    mcp_transport: str
    max_supplements: int
    max_total_tokens: int


def _deep_get(data: dict[str, Any], keys: list[str], default: Any) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            return loaded
    return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def build_ssl_context(verify_ssl: bool, ca_bundle: str = "") -> ssl.SSLContext | None:
    if verify_ssl and not ca_bundle:
        return None
    if verify_ssl:
        return ssl.create_default_context(cafile=ca_bundle)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


@lru_cache(maxsize=1)
def load_project_config() -> ProjectConfig:
    project_root = Path(__file__).resolve().parent
    default_cfg_file = project_root / "config" / "deep_rag.yaml"
    cfg_path_env = os.getenv("DEEP_RAG_CONFIG")
    cfg_path = Path(cfg_path_env).expanduser() if cfg_path_env else default_cfg_file
    yaml_cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}

    # Auto-load and deep-merge *.local.yaml (gitignored, contains secrets/overrides)
    local_cfg_path = cfg_path.with_name(cfg_path.stem + ".local" + cfg_path.suffix)
    if local_cfg_path.exists():
        yaml_cfg = _deep_merge(yaml_cfg, _load_yaml(local_cfg_path))

    llm_provider = str(
        os.getenv(
            "LLM_PROVIDER",
            _deep_get(yaml_cfg, ["llm", "provider"], "ollama"),
        )
    ).strip().lower()
    if llm_provider not in {"ollama", "openai_compatible"}:
        llm_provider = "ollama"

    ollama_base_url = str(
        os.getenv(
            "OLLAMA_BASE_URL",
            _deep_get(yaml_cfg, ["llm", "ollama", "base_url"], "http://127.0.0.1:11434"),
        )
    ).rstrip("/")
    ollama_chat_model = str(
        os.getenv(
            "OLLAMA_CHAT_MODEL",
            _deep_get(yaml_cfg, ["llm", "ollama", "chat_model"], "qwen2.5:7b-instruct"),
        )
    )
    ollama_embed_model = str(
        os.getenv(
            "OLLAMA_EMBED_MODEL",
            _deep_get(yaml_cfg, ["llm", "ollama", "embed_model"], "bge-m3"),
        )
    )

    openai_base_url = str(
        os.getenv(
            "OPENAI_COMPAT_BASE_URL",
            _deep_get(yaml_cfg, ["llm", "openai_compatible", "base_url"], ""),
        )
    ).rstrip("/")
    openai_api_key = str(
        os.getenv(
            "OPENAI_COMPAT_API_KEY",
            _deep_get(yaml_cfg, ["llm", "openai_compatible", "api_key"], ""),
        )
    )
    openai_chat_model = str(
        os.getenv(
            "OPENAI_COMPAT_CHAT_MODEL",
            _deep_get(yaml_cfg, ["llm", "openai_compatible", "chat_model"], "gpt-4o-mini"),
        )
    )
    openai_embed_model = str(
        os.getenv(
            "OPENAI_COMPAT_EMBED_MODEL",
            _deep_get(yaml_cfg, ["llm", "openai_compatible", "embed_model"], "text-embedding-3-small"),
        )
    )
    openai_verify_ssl = _as_bool(
        os.getenv(
            "OPENAI_COMPAT_VERIFY_SSL",
            _deep_get(yaml_cfg, ["llm", "openai_compatible", "verify_ssl"], True),
        ),
        True,
    )
    openai_ca_bundle = str(
        os.getenv(
            "OPENAI_COMPAT_CA_BUNDLE",
            _deep_get(yaml_cfg, ["llm", "openai_compatible", "ca_bundle"], ""),
        )
    )

    kb_raw = os.getenv("KB_DIR", _deep_get(yaml_cfg, ["paths", "kb_dir"], "knowledge_base"))
    index_raw = os.getenv("INDEX_DIR", _deep_get(yaml_cfg, ["paths", "index_dir"], "index_store"))
    log_raw = os.getenv("LOG_FILE", _deep_get(yaml_cfg, ["paths", "log_file"], "logs/query_logs.jsonl"))

    kb_dir = (project_root / kb_raw).resolve() if not Path(kb_raw).is_absolute() else Path(kb_raw)
    index_dir = (project_root / index_raw).resolve() if not Path(index_raw).is_absolute() else Path(index_raw)
    log_file = (project_root / log_raw).resolve() if not Path(log_raw).is_absolute() else Path(log_raw)

    mcp_transport = str(
        os.getenv(
            "MCP_TRANSPORT",
            _deep_get(yaml_cfg, ["mcp", "transport"], "local"),
        )
    ).lower()
    if mcp_transport not in {"local", "stdio"}:
        mcp_transport = "local"

    max_supplements = _as_int(
        os.getenv(
            "MAX_SUPPLEMENTS",
            _deep_get(yaml_cfg, ["retrieval", "max_supplements"], 2),
        ),
        2,
    )
    max_total_tokens = _as_int(
        os.getenv(
            "MAX_TOTAL_TOKENS",
            _deep_get(yaml_cfg, ["retrieval", "max_total_tokens"], 50000),
        ),
        50000,
    )

    return ProjectConfig(
        project_root=project_root,
        config_file=cfg_path if cfg_path.exists() else None,
        llm_provider=llm_provider,
        ollama_base_url=ollama_base_url,
        ollama_chat_model=ollama_chat_model,
        ollama_embed_model=ollama_embed_model,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_chat_model=openai_chat_model,
        openai_embed_model=openai_embed_model,
        openai_verify_ssl=openai_verify_ssl,
        openai_ca_bundle=openai_ca_bundle,
        kb_dir=kb_dir,
        index_dir=index_dir,
        log_file=log_file,
        mcp_transport=mcp_transport,
        max_supplements=max_supplements,
        max_total_tokens=max_total_tokens,
    )
