from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from project_config import build_ssl_context, load_project_config

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    def __init__(self, base_url: str | None = None) -> None:
        cfg = load_project_config()
        self.provider = cfg.llm_provider
        self.ollama_base_url = (base_url or cfg.ollama_base_url).rstrip("/")
        self.ollama_chat_model = cfg.ollama_chat_model
        self.ollama_embed_model = cfg.ollama_embed_model
        self.openai_base_url = cfg.openai_base_url.rstrip("/")
        self.openai_api_key = cfg.openai_api_key
        self.openai_chat_model = cfg.openai_chat_model
        self.openai_embed_model = cfg.openai_embed_model
        self.openai_ssl_context = build_ssl_context(cfg.openai_verify_ssl, cfg.openai_ca_bundle)
        self._availability_checked = False
        self._available = False

    async def call(
        self,
        messages: list[dict[str, str]],
        model: str = "haiku",
        system: str = "",
        max_tokens: int = 2048,
        response_format: str = "text",
    ) -> tuple[str, TokenUsage]:
        if not self.is_available():
            raise RuntimeError(f"{self.provider} server is unavailable")
        actual_model = self._resolve_model_name(model)
        merged_messages = list(messages)
        if system:
            merged_messages = [{"role": "system", "content": system}] + merged_messages
        if response_format == "json":
            merged_messages = merged_messages + [
                {"role": "system", "content": "只输出JSON，不要使用markdown code fence。"}
            ]

        if self.provider == "openai_compatible":
            payload = {
                "model": actual_model,
                "messages": merged_messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            text, usage = self._post_openai_chat(payload)
        else:
            payload = {
                "model": actual_model,
                "messages": merged_messages,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }
            text, usage = self._post_ollama_chat(payload)
        if response_format == "json":
            text = self._strip_json_fence(text)
        return text, usage

    async def embed(self, text: str, model: str | None = None) -> list[float] | None:
        if not self.is_available():
            return None
        if self.provider == "openai_compatible":
            return self._openai_embed(text, model)
        return self._ollama_embed(text, model)

    def is_available(self) -> bool:
        if self._availability_checked:
            return self._available
        if self.provider == "openai_compatible":
            if not self.openai_base_url or not self.openai_api_key:
                self._available = False
            else:
                req = urllib.request.Request(
                    url=f"{self.openai_base_url}/models",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    method="GET",
                )
                try:
                    with urllib.request.urlopen(req, timeout=3, context=self.openai_ssl_context) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        self._available = isinstance(data, dict)
                except Exception:
                    self._available = False
        else:
            req = urllib.request.Request(url=f"{self.ollama_base_url}/api/tags", method="GET")
            try:
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    self._available = isinstance(data, dict)
            except Exception:
                self._available = False
        self._availability_checked = True
        return self._available

    def _post_ollama_chat(self, payload: dict[str, Any]) -> tuple[str, TokenUsage]:
        req = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON: {exc}") from exc

        message = data.get("message", {})
        content = str(message.get("content", "")).strip()
        prompt_eval_count = int(data.get("prompt_eval_count", self._estimate_tokens(json.dumps(payload, ensure_ascii=False))))
        eval_count = int(data.get("eval_count", self._estimate_tokens(content)))
        usage = TokenUsage(input_tokens=prompt_eval_count, output_tokens=eval_count)
        return content, usage

    def _post_openai_chat(self, payload: dict[str, Any]) -> tuple[str, TokenUsage]:
        req = urllib.request.Request(
            url=f"{self.openai_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20, context=self.openai_ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible call failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI-compatible returned invalid JSON: {exc}") from exc

        choices = data.get("choices", []) or []
        message = choices[0].get("message", {}) if choices else {}
        content = str(message.get("content", "")).strip()
        usage_raw = data.get("usage", {}) or {}
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("prompt_tokens", self._estimate_tokens(json.dumps(payload, ensure_ascii=False)))),
            output_tokens=int(usage_raw.get("completion_tokens", self._estimate_tokens(content))),
        )
        return content, usage

    def _ollama_embed(self, text: str, model: str | None = None) -> list[float] | None:
        payload = {"model": model or self.ollama_embed_model, "prompt": text[:4000]}
        req = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/embeddings",
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

    def _openai_embed(self, text: str, model: str | None = None) -> list[float] | None:
        payload = {"model": model or self.openai_embed_model, "input": text[:4000]}
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
            with urllib.request.urlopen(req, timeout=20, context=self.openai_ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("data", []) or []
                if rows and isinstance(rows[0], dict) and isinstance(rows[0].get("embedding"), list):
                    return [float(x) for x in rows[0]["embedding"]]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return None
        return None

    def _resolve_model_name(self, alias_or_name: str) -> str:
        alias = alias_or_name.lower().strip()
        if alias in {"sonnet", "haiku"}:
            if self.provider == "openai_compatible":
                return self.openai_chat_model
            return self.ollama_chat_model
        return alias_or_name

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
        return stripped.strip()
