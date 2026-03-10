from __future__ import annotations

from pathlib import Path

from app.evidence_model import QueryType
from app.llm_client import LLMClient, TokenUsage

try:
    from jinja2 import Environment, FileSystemLoader
except Exception:  # pragma: no cover - optional dependency
    Environment = None  # type: ignore[assignment]
    FileSystemLoader = None  # type: ignore[assignment]


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompts_dir = Path(__file__).resolve().parent / "prompts"
        self.env = (
            Environment(loader=FileSystemLoader(str(self.prompts_dir)), autoescape=False)
            if Environment is not None and FileSystemLoader is not None
            else None
        )

    async def generate(self, ctx: "RetrievalContext") -> tuple[str, TokenUsage]:
        template_name = self._template_for_type(ctx.query_plan.query_type)
        prompt = self._render_prompt(template_name, ctx)
        messages = [{"role": "user", "content": prompt}]
        try:
            answer, usage = await self.llm.call(messages, model="sonnet", response_format="text", max_tokens=1800)
            return answer, usage
        except Exception:
            fallback = self._fallback_answer(ctx)
            return fallback, TokenUsage()

    def _render_prompt(self, template_name: str, ctx: "RetrievalContext") -> str:
        if self.env is None:
            return self._fallback_answer(ctx)
        template = self.env.get_template(template_name)
        return template.render(
            question=ctx.query_plan.original_question,
            query_type=ctx.query_plan.query_type.value,
            evidences=[e.model_dump() for e in ctx.evidences],
            missing_info=ctx.missing_info,
        )

    @staticmethod
    def _template_for_type(query_type: QueryType) -> str:
        if query_type == QueryType.TREND_ANALYSIS:
            return "generate_trend.j2"
        if query_type == QueryType.CROSS_DOC_SUMMARY:
            return "generate_summary.j2"
        return "generate_factual.j2"

    @staticmethod
    def _fallback_answer(ctx: "RetrievalContext") -> str:
        if not ctx.evidences:
            return "未检索到可用证据，无法给出可靠回答。"
        lines = ["基于检索证据的回答："]
        for ev in ctx.evidences[:8]:
            preview = ev.content.replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:180] + "..."
            lines.append(f"- {preview}【来源：{ev.source_file}-{ev.source_location}】")
        if ctx.missing_info:
            lines.append("")
            lines.append("信息差距：")
            for item in ctx.missing_info:
                lines.append(f"- {item}")
        return "\n".join(lines)
