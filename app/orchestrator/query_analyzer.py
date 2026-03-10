from __future__ import annotations

import json
import re
from pathlib import Path

from app.evidence_model import QueryPlan, QueryType
from app.llm_client import LLMClient, TokenUsage

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
TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")


class QueryAnalyzer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompts_dir = Path(__file__).resolve().parent / "prompts"
        self.env = (
            Environment(loader=FileSystemLoader(str(self.prompts_dir)), autoescape=False)
            if Environment is not None and FileSystemLoader is not None
            else None
        )

    async def analyze(self, question: str) -> tuple[QueryPlan, TokenUsage]:
        prompt = self._render_prompt(question)
        messages = [{"role": "user", "content": prompt}]
        try:
            response, usage = await self.llm.call(messages, model="haiku", response_format="json", max_tokens=800)
            payload = json.loads(response)
            plan = self._to_query_plan(question, payload)
            return plan, usage
        except Exception:
            return self._heuristic_plan(question), TokenUsage()

    def _render_prompt(self, question: str) -> str:
        if self.env is None:
            return f"请分析用户问题并输出JSON计划。问题：{question}"
        template = self.env.get_template("analyze_query.j2")
        return template.render(question=question)

    def _to_query_plan(self, question: str, payload: dict) -> QueryPlan:
        raw_type = str(payload.get("query_type", "fact_lookup"))
        if raw_type not in {e.value for e in QueryType}:
            raw_type = "fact_lookup"
        normalized_keywords = self._normalize_keywords(payload.get("keywords", []), question)
        return QueryPlan(
            original_question=question,
            query_type=QueryType(raw_type),
            keywords=normalized_keywords or self._extract_keywords(question),
            time_range=payload.get("time_range"),
            target_formats=[str(x) for x in payload.get("target_formats", []) if str(x).strip()],
            expected_dimensions=[str(x) for x in payload.get("expected_dimensions", []) if str(x).strip()],
            initial_file_count=int(payload.get("initial_file_count", self._default_file_count(QueryType(raw_type)))),
        )

    def _heuristic_plan(self, question: str) -> QueryPlan:
        q = question.strip()
        q_type = self._classify(q)
        years = sorted(set(YEAR_PATTERN.findall(q)))
        time_range = None
        if years:
            time_range = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"

        target_formats = ["excel"] if q_type == QueryType.NUMERIC_QUERY else []
        expected_dimensions: list[str] = []
        if q_type == QueryType.CROSS_DOC_SUMMARY:
            expected_dimensions = [token for token in self._extract_keywords(q)[:5]]
        return QueryPlan(
            original_question=q,
            query_type=q_type,
            keywords=self._extract_keywords(q),
            time_range=time_range,
            target_formats=target_formats,
            expected_dimensions=expected_dimensions,
            initial_file_count=self._default_file_count(q_type),
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
        return uniq[:8] or [text[:12]]

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
    def _default_file_count(query_type: QueryType) -> int:
        if query_type == QueryType.TREND_ANALYSIS:
            return 8
        if query_type == QueryType.CROSS_DOC_SUMMARY:
            return 10
        if query_type == QueryType.NUMERIC_QUERY:
            return 5
        return 3

    def _classify(self, question: str) -> QueryType:
        q = question.lower()
        if any(word in question for word in ["趋势", "同比", "变化", "近三年", "最近三年"]):
            return QueryType.TREND_ANALYSIS
        if any(word in question for word in ["多少", "数值", "金额", "占比", "增长率", "%", "万元"]):
            return QueryType.NUMERIC_QUERY
        if any(word in question for word in ["总结", "归纳", "对比", "主要", "整体"]):
            return QueryType.CROSS_DOC_SUMMARY
        if any(word in question for word in ["流程", "步骤", "怎么做", "定义"]):
            return QueryType.DEFINITION_PROCESS
        if "what is" in q or "when" in q:
            return QueryType.FACT_LOOKUP
        return QueryType.FACT_LOOKUP
