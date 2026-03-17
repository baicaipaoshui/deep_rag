from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from app.evidence_model import QueryPlan, QueryType
from app.llm_client import LLMClient, TokenUsage
from project_config import ProjectConfig, load_project_config

try:
    from jinja2 import Environment, FileSystemLoader
except Exception:  # pragma: no cover - optional dependency
    Environment = None  # type: ignore[assignment]
    FileSystemLoader = None  # type: ignore[assignment]

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency
    jieba = None

YEAR_PATTERN = re.compile(r"(20\d{2})(?!\d|万)")
SHORT_YEAR_RANGE_PATTERN = re.compile(r"(?<!\d)(\d{2})\s*[-~到至]\s*(\d{2})(?=\s*年)")
TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")


class QueryAnalyzer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.cfg: ProjectConfig = load_project_config()
        self.prompts_dir = Path(__file__).resolve().parent / "prompts"
        self.env = (
            Environment(loader=FileSystemLoader(str(self.prompts_dir)), autoescape=False)
            if Environment is not None and FileSystemLoader is not None
            else None
        )

    async def analyze(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[QueryPlan, TokenUsage]:
        prompt = self._render_prompt(question, chat_history=chat_history or [])
        messages = [{"role": "user", "content": prompt}]
        try:
            response, usage = await asyncio.wait_for(
                self.llm.call(
                    messages,
                    model=self.cfg.judge_model or "haiku",
                    response_format="json",
                    max_tokens=800,
                ),
                timeout=max(1, self.cfg.judge_timeout_seconds),
            )
            payload = json.loads(response)
            plan = self._to_query_plan(question, payload)
            return plan, usage
        except TimeoutError:
            return self._heuristic_plan(question, fallback_reason="route_timeout"), TokenUsage()
        except Exception:
            return self._heuristic_plan(question, fallback_reason="route_parse_error"), TokenUsage()

    def _render_prompt(self, question: str, chat_history: list[dict[str, str]]) -> str:
        if self.env is None:
            return f"请分析用户问题并输出JSON计划。问题：{question}"
        template = self.env.get_template("analyze_query.j2")
        return template.render(question=question, chat_history=chat_history[-6:])

    def _to_query_plan(self, question: str, payload: dict[str, Any]) -> QueryPlan:
        raw_type = str(payload.get("query_type", "fact_lookup"))
        if raw_type not in {e.value for e in QueryType}:
            raw_type = "fact_lookup"
        query_type = QueryType(raw_type)
        complexity = self._normalize_complexity(payload.get("complexity"), query_type)
        estimated_pieces = self._normalize_estimated_pieces(payload.get("estimated_evidence_pieces"), query_type)
        budget, budget_audit = self._compute_budget(estimated_pieces, complexity)
        normalized_keywords = self._normalize_keywords(payload.get("keywords", []), question)
        question_keywords = self._extract_keywords(question)
        merged_keywords = self._merge_keywords(normalized_keywords, question_keywords)
        return QueryPlan(
            original_question=question,
            query_type=query_type,
            keywords=merged_keywords or question_keywords,
            complexity=complexity,
            sub_intents=self._to_str_list(payload.get("sub_intents")),
            estimated_evidence_pieces=estimated_pieces,
            time_sensitive=self._to_bool(payload.get("time_sensitive"), default=False),
            domain_hints=self._to_str_list(payload.get("domain_hints")),
            budget_audit=budget_audit,
            time_range=payload.get("time_range"),
            target_formats=[str(x) for x in payload.get("target_formats", []) if str(x).strip()],
            expected_dimensions=[str(x) for x in payload.get("expected_dimensions", []) if str(x).strip()],
            initial_file_count=budget,
        )

    def _heuristic_plan(self, question: str, fallback_reason: str) -> QueryPlan:
        q = question.strip()
        q_type = self._classify(q)
        complexity = self._default_complexity_for_type(q_type)
        estimated_pieces = self._default_estimated_pieces_for_type(q_type)
        budget, budget_audit = self._compute_budget(estimated_pieces, complexity)
        years = self._extract_years(q)
        time_range = self._infer_time_range(years, q)

        target_formats = ["excel"] if q_type == QueryType.NUMERIC_QUERY else []
        expected_dimensions: list[str] = []
        if q_type == QueryType.CROSS_DOC_SUMMARY:
            expected_dimensions = [token for token in self._extract_keywords(q)[:5]]
        return QueryPlan(
            original_question=q,
            query_type=q_type,
            keywords=self._extract_keywords(q),
            complexity=complexity,
            estimated_evidence_pieces=estimated_pieces,
            time_sensitive=bool(years),
            route_fallback_reason=fallback_reason,
            budget_audit=budget_audit,
            time_range=time_range,
            target_formats=target_formats,
            expected_dimensions=expected_dimensions,
            initial_file_count=budget,
        )

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        if jieba is not None:
            tokens = [t.strip() for t in jieba.cut(text, cut_all=False) if t and t.strip()]
        else:
            tokens = TOKEN_PATTERN.findall(text)
        uniq: list[str] = []
        for token in tokens:
            if len(token) < 2:
                continue
            if token not in uniq:
                uniq.append(token)
        if uniq:
            return uniq[:8]
        fallback = text.strip()[:10]
        return [fallback] if fallback else ["问题"]

    def _normalize_keywords(self, raw_keywords: object, question: str) -> list[str]:
        if not isinstance(raw_keywords, list):
            return []
        merged = " ".join(str(x) for x in raw_keywords if str(x).strip())
        if not merged.strip():
            return []
        tokens = self._extract_keywords(merged)
        # If model outputs a single combined token (e.g. "销售趋势"), expand from question.
        if len(tokens) <= 1:
            question_tokens = self._extract_keywords(question)
            if question_tokens:
                return question_tokens
        return tokens

    @staticmethod
    def _merge_keywords(primary: list[str], secondary: list[str], max_count: int = 8) -> list[str]:
        merged: list[str] = []
        for group in (primary, secondary):
            for token in group:
                t = str(token).strip()
                if len(t) < 2 or t in merged:
                    continue
                merged.append(t)
                if len(merged) >= max_count:
                    return merged
        return merged

    @staticmethod
    def _default_file_count(query_type: QueryType) -> int:
        if query_type == QueryType.TREND_ANALYSIS:
            return 8
        if query_type == QueryType.CROSS_DOC_SUMMARY:
            return 10
        if query_type == QueryType.NUMERIC_QUERY:
            return 5
        return 3

    def _classify(self, question: str) -> QueryType:
        normalized_question = question.strip().lower()
        years = self._extract_years(question)
        has_multi_year = len(years) >= 2 or bool(SHORT_YEAR_RANGE_PATTERN.search(question))
        if any(word in normalized_question for word in ["分别是多少", "各是多少", "分别为多少"]):
            return QueryType.NUMERIC_QUERY
        if any(word in normalized_question for word in ["多少", "数值", "金额", "占比", "增长率", "%", "万元"]):
            return QueryType.NUMERIC_QUERY
        if any(word in normalized_question for word in ["趋势", "同比", "变化", "近三年", "最近三年"]):
            return QueryType.TREND_ANALYSIS
        if has_multi_year and any(word in normalized_question for word in ["分析", "业绩", "销售", "复盘", "走势"]):
            return QueryType.TREND_ANALYSIS
        if any(word in normalized_question for word in ["总结", "归纳", "对比", "主要", "整体"]):
            return QueryType.CROSS_DOC_SUMMARY
        if any(word in normalized_question for word in ["流程", "步骤", "怎么做", "定义"]):
            return QueryType.DEFINITION_PROCESS
        if "what is" in normalized_question or "when" in normalized_question:
            return QueryType.FACT_LOOKUP
        return QueryType.FACT_LOOKUP

    @staticmethod
    def _expand_short_year(two_digit: str) -> str:
        value = int(two_digit)
        base = 1900 if value >= 70 else 2000
        return str(base + value)

    def _extract_years(self, question: str) -> list[str]:
        years = {y for y in YEAR_PATTERN.findall(question)}
        short_range = SHORT_YEAR_RANGE_PATTERN.search(question)
        if short_range:
            left, right = short_range.group(1), short_range.group(2)
            start = int(self._expand_short_year(left))
            end = int(self._expand_short_year(right))
            if end < start:
                start, end = end, start
            for year in range(start, end + 1):
                years.add(str(year))
        return sorted(years)

    def _infer_time_range(self, years: list[str], question: str) -> str | None:
        if years:
            return years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
        short_range = SHORT_YEAR_RANGE_PATTERN.search(question)
        if not short_range:
            return None
        start = self._expand_short_year(short_range.group(1))
        end = self._expand_short_year(short_range.group(2))
        return start if start == end else f"{min(start, end)}-{max(start, end)}"

    @staticmethod
    def _to_str_list(raw_value: Any) -> list[str]:
        if not isinstance(raw_value, list):
            return []
        return [str(item).strip() for item in raw_value if str(item).strip()]

    @staticmethod
    def _to_bool(raw_value: Any, default: bool) -> bool:
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        return default

    def _normalize_complexity(self, raw_complexity: Any, query_type: QueryType) -> str:
        complexity = str(raw_complexity or "").strip().lower()
        if complexity in {"simple", "medium", "complex"}:
            return complexity
        return self._default_complexity_for_type(query_type)

    def _normalize_estimated_pieces(self, raw_value: Any, query_type: QueryType) -> int:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = self._default_estimated_pieces_for_type(query_type)
        if value <= 0:
            return self._default_estimated_pieces_for_type(query_type)
        return value

    def _compute_budget(self, estimated_pieces: int, complexity: str) -> tuple[int, dict[str, float | int | str]]:
        multiplier = float(
            self.cfg.budget_complexity_multipliers.get(
                complexity,
                self.cfg.budget_complexity_multipliers.get("medium", 2.5),
            )
        )
        raw_budget = estimated_pieces * multiplier
        bounded_budget = int(round(raw_budget))
        bounded_budget = max(self.cfg.budget_k_min, min(self.cfg.budget_k_max, bounded_budget))
        return bounded_budget, {
            "estimated_evidence_pieces": estimated_pieces,
            "complexity": complexity,
            "complexity_multiplier": multiplier,
            "raw_budget": raw_budget,
            "k_min": self.cfg.budget_k_min,
            "k_max": self.cfg.budget_k_max,
            "final_budget": bounded_budget,
        }

    @staticmethod
    def _default_complexity_for_type(query_type: QueryType) -> str:
        if query_type in {QueryType.FACT_LOOKUP, QueryType.DEFINITION_PROCESS}:
            return "simple"
        if query_type in {QueryType.NUMERIC_QUERY, QueryType.TREND_ANALYSIS}:
            return "medium"
        return "complex"

    def _default_estimated_pieces_for_type(self, query_type: QueryType) -> int:
        return self._default_file_count(query_type)
