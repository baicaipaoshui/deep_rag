from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenTracker:
    def __init__(self) -> None:
        self._total = TokenUsage()
        self._by_stage: dict[str, TokenUsage] = defaultdict(TokenUsage)

    def add(self, stage: str, usage: TokenUsage) -> None:
        self._total.input_tokens += usage.input_tokens
        self._total.output_tokens += usage.output_tokens
        self._by_stage[stage].input_tokens += usage.input_tokens
        self._by_stage[stage].output_tokens += usage.output_tokens

    def total_used(self) -> int:
        return self._total.total_tokens

    def summary(self) -> dict[str, object]:
        return {
            "total": {
                "input_tokens": self._total.input_tokens,
                "output_tokens": self._total.output_tokens,
                "total_tokens": self._total.total_tokens,
            },
            "by_stage": {
                k: {
                    "input_tokens": v.input_tokens,
                    "output_tokens": v.output_tokens,
                    "total_tokens": v.total_tokens,
                }
                for k, v in self._by_stage.items()
            },
        }
