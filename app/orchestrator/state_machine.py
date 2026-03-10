from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.evidence_model import CandidateFile, Evidence, EvidenceStrength, QueryPlan, TargetLocation
from app.llm_client import LLMClient, TokenUsage
from app.mcp_client import MCPClient
from app.token_tracker import TokenTracker
from app.orchestrator.answer_generator import AnswerGenerator
from app.orchestrator.query_analyzer import QueryAnalyzer
from project_config import load_project_config

try:
    from jinja2 import Environment, FileSystemLoader
except Exception:  # pragma: no cover - optional dependency
    Environment = None  # type: ignore[assignment]
    FileSystemLoader = None  # type: ignore[assignment]


class State(str, Enum):
    FILTERING = "filtering"
    LOCATING = "locating"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    GENERATING = "generating"
    FAILED = "failed"


@dataclass
class RetrievalContext:
    query_plan: QueryPlan
    state: State = State.FILTERING
    candidate_files: list[CandidateFile] = field(default_factory=list)
    excluded_files: list[str] = field(default_factory=list)
    target_locations: list[TargetLocation] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    supplement_count: int = 0
    missing_info: list[str] = field(default_factory=list)
    token_tracker: TokenTracker = field(default_factory=TokenTracker)
    decision_log: list[dict[str, Any]] = field(default_factory=list)

    def log_decision(self, stage: str, action: str, detail: str) -> None:
        self.decision_log.append(
            {
                "stage": stage,
                "action": action,
                "detail": detail,
                "tokens_so_far": self.token_tracker.total_used(),
            }
        )


class StateMachine:
    def __init__(self) -> None:
        cfg = load_project_config()
        self.mcp = MCPClient()
        self.llm = LLMClient()
        self.analyzer = QueryAnalyzer(self.llm)
        self.generator = AnswerGenerator(self.llm)
        self.max_supplements = cfg.max_supplements
        self.max_total_tokens = cfg.max_total_tokens
        self.prompts_dir = Path(__file__).resolve().parent / "prompts"
        self.env = (
            Environment(loader=FileSystemLoader(str(self.prompts_dir)), autoescape=False)
            if Environment is not None and FileSystemLoader is not None
            else None
        )

    async def run(self, question: str) -> dict[str, Any]:
        await self.mcp.connect()
        try:
            query_plan, usage = await self.analyzer.analyze(question)
            ctx = RetrievalContext(query_plan=query_plan)
            ctx.token_tracker.add("analyze", usage)
            ctx.log_decision("init", "query_analyzed", f"type={query_plan.query_type.value}")

            while ctx.state not in (State.GENERATING, State.FAILED):
                if ctx.state == State.FILTERING:
                    await self._do_filtering(ctx)
                elif ctx.state == State.LOCATING:
                    await self._do_locating(ctx)
                elif ctx.state == State.EXTRACTING:
                    await self._do_extracting(ctx)
                elif ctx.state == State.VERIFYING:
                    await self._do_verifying(ctx)

                if ctx.token_tracker.total_used() > self.max_total_tokens:
                    ctx.log_decision("global", "budget_exceeded", f"used={ctx.token_tracker.total_used()}")
                    ctx.state = State.GENERATING

            if ctx.state == State.FAILED:
                return self._build_failed_result(ctx)

            answer, usage = await self.generator.generate(ctx)
            ctx.token_tracker.add("generate", usage)
            return self._build_result(ctx, answer)
        finally:
            await self.mcp.disconnect()

    async def _do_filtering(self, ctx: RetrievalContext) -> None:
        args: dict[str, Any] = {
            "keywords": ctx.query_plan.keywords,
            "max_results": ctx.query_plan.initial_file_count,
        }
        if ctx.query_plan.time_range:
            args["time_range"] = ctx.query_plan.time_range
        if ctx.query_plan.target_formats:
            args["format_preference"] = ctx.query_plan.target_formats

        result = await self.mcp.call_tool("query_navigation_index", args)
        if result.get("error"):
            ctx.log_decision("filtering", "error", str(result))
            ctx.state = State.FAILED
            return

        candidates = result.get("candidates", [])
        if not candidates and ctx.query_plan.time_range:
            retry_args = dict(args)
            retry_args.pop("time_range", None)
            result = await self.mcp.call_tool("query_navigation_index", retry_args)
            candidates = result.get("candidates", [])

        if not candidates:
            ctx.log_decision("filtering", "no_candidates", "no files matched")
            ctx.state = State.FAILED
            return

        ctx.candidate_files = [CandidateFile(**item) for item in candidates]
        all_file_names = [x.get("file_name") for x in result.get("all_files", []) if x.get("file_name")]
        matched = {c.file_name for c in ctx.candidate_files}
        ctx.excluded_files = [f for f in all_file_names if f not in matched]
        ctx.log_decision("filtering", "candidates_found", f"{len(ctx.candidate_files)} candidates")
        ctx.state = State.LOCATING

    async def _do_locating(self, ctx: RetrievalContext) -> None:
        file_names = [f.file_name for f in ctx.candidate_files]
        result = await self.mcp.call_tool("query_discovery_index", {"file_names": file_names})
        if result.get("error"):
            ctx.log_decision("locating", "error", str(result))
            ctx.state = State.FAILED
            return
        locations, usage = await self._llm_locate_sections(ctx.query_plan, result.get("files", []))
        ctx.token_tracker.add("locate", usage)
        ctx.target_locations = locations
        if not locations:
            # fallback path: direct evidence search
            search_result = await self.mcp.call_tool(
                "search_evidence",
                {"query": ctx.query_plan.original_question, "max_results": 6, "search_type": "hybrid"},
            )
            fallback_locations = []
            for row in search_result.get("results", []):
                if row.get("file_name") and row.get("section_id"):
                    fallback_locations.append(
                        TargetLocation(
                            file_name=row["file_name"],
                            section_id=row["section_id"],
                            heading=row.get("heading", ""),
                            estimated_relevance=float(row.get("relevance_score", 0.5)),
                        )
                    )
            ctx.target_locations = fallback_locations

        ctx.log_decision("locating", "located", f"{len(ctx.target_locations)} locations")
        ctx.state = State.EXTRACTING if ctx.target_locations else State.FAILED

    async def _do_extracting(self, ctx: RetrievalContext) -> None:
        all_evidence: list[Evidence] = []
        format_map = {c.file_name: c.file_format for c in ctx.candidate_files}
        for loc in ctx.target_locations[:20]:
            read_result = await self.mcp.call_tool(
                "read_file_section",
                {"file_name": loc.file_name, "section_id": loc.section_id, "max_chars": 2200},
            )
            if read_result.get("error"):
                continue
            evs, usage = await self._llm_extract_evidence(
                ctx.query_plan,
                loc,
                read_result,
                source_format=format_map.get(loc.file_name, "unknown"),
            )
            ctx.token_tracker.add("extract", usage)
            all_evidence.extend(evs)

        dedup: dict[str, Evidence] = {}
        for ev in all_evidence:
            key = ev.dedup_key or f"{ev.source_file}:{ev.source_location}:{ev.content[:80]}"
            if key not in dedup:
                dedup[key] = ev
        ctx.evidences = list(dedup.values())
        ctx.log_decision("extracting", "extracted", f"{len(ctx.evidences)} evidences")
        ctx.state = State.VERIFYING

    async def _do_verifying(self, ctx: RetrievalContext) -> None:
        result = await self.mcp.call_tool(
            "verify_evidence_coverage",
            {
                "query_type": ctx.query_plan.query_type.value,
                "time_range": ctx.query_plan.time_range,
                "expected_dimensions": ctx.query_plan.expected_dimensions,
                "collected_evidence": [ev.model_dump() for ev in ctx.evidences],
                "supplement_round": ctx.supplement_count,
            },
        )
        if result.get("error"):
            ctx.log_decision("verify", "error", str(result))
            ctx.state = State.GENERATING
            return
        if result.get("is_sufficient"):
            ctx.log_decision("verify", "sufficient", "coverage passed")
            ctx.state = State.GENERATING
            return

        ctx.missing_info = [str(x) for x in result.get("missing", [])]
        if ctx.supplement_count >= self.max_supplements:
            ctx.log_decision("verify", "force_generate", ",".join(ctx.missing_info))
            ctx.state = State.GENERATING
            return

        ctx.supplement_count += 1
        suggested_keywords = [str(x) for x in result.get("suggested_keywords", []) if str(x).strip()]
        if suggested_keywords:
            ctx.query_plan.keywords = suggested_keywords
        ctx.log_decision("verify", "supplement", f"round={ctx.supplement_count}")
        ctx.state = State.FILTERING

    async def _llm_locate_sections(
        self, query_plan: QueryPlan, files_structure: list[dict[str, Any]]
    ) -> tuple[list[TargetLocation], TokenUsage]:
        if not files_structure:
            return [], TokenUsage()
        prompt = self._render_prompt(
            "locate_sections.j2",
            question=query_plan.original_question,
            query_type=query_plan.query_type.value,
            time_range=query_plan.time_range,
            keywords=query_plan.keywords,
            files=files_structure,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response, usage = await self.llm.call(messages, model="haiku", response_format="json", max_tokens=1400)
            payload = json.loads(response)
            rows = payload if isinstance(payload, list) else payload.get("locations", [])
            locations = [
                TargetLocation(
                    file_name=str(item["file_name"]),
                    section_id=str(item["section_id"]),
                    heading=str(item.get("heading", "")),
                    estimated_relevance=float(item.get("estimated_relevance", 0.5)),
                )
                for item in rows
                if item.get("file_name") and item.get("section_id")
            ]
            if query_plan.query_type.value == "trend_analysis":
                locations = self._augment_trend_locations(locations, files_structure)
            return locations[:20], usage
        except Exception:
            fallback: list[TargetLocation] = []
            for file_item in files_structure:
                file_name = file_item.get("file_name")
                sections = file_item.get("sections", []) or []
                for sec in sections[:2]:
                    if file_name and sec.get("section_id"):
                        fallback.append(
                            TargetLocation(
                                file_name=str(file_name),
                                section_id=str(sec["section_id"]),
                                heading=str(sec.get("heading", "")),
                                estimated_relevance=0.4,
                            )
                        )
            return fallback[:20], TokenUsage()

    def _augment_trend_locations(
        self,
        locations: list[TargetLocation],
        files_structure: list[dict[str, Any]],
    ) -> list[TargetLocation]:
        if len(locations) >= 4:
            return locations
        selected = {(loc.file_name, loc.section_id) for loc in locations}
        boosted: list[TargetLocation] = list(locations)
        keywords = ("年度", "趋势", "同比", "销售", "数据", "合计", "总额", "q1", "q2", "q3", "q4")
        for file_item in files_structure:
            file_name = str(file_item.get("file_name", ""))
            sections = file_item.get("sections", []) or []

            if "annual" in file_name.lower():
                for sec in sections:
                    section_id = str(sec.get("section_id", ""))
                    if section_id == "p001" and (file_name, section_id) not in selected:
                        boosted.append(
                            TargetLocation(
                                file_name=file_name,
                                section_id=section_id,
                                heading=str(sec.get("heading", "")),
                                estimated_relevance=0.7,
                            )
                        )
                        selected.add((file_name, section_id))

            for sec in sections:
                section_id = str(sec.get("section_id", ""))
                heading = str(sec.get("heading", "")).lower()
                if not file_name or not section_id or (file_name, section_id) in selected:
                    continue
                if any(k in heading for k in keywords):
                    boosted.append(
                        TargetLocation(
                            file_name=file_name,
                            section_id=section_id,
                            heading=str(sec.get("heading", "")),
                            estimated_relevance=0.55,
                        )
                    )
                    selected.add((file_name, section_id))
                if len(boosted) >= 8:
                    return boosted
        return boosted

    async def _llm_extract_evidence(
        self,
        query_plan: QueryPlan,
        location: TargetLocation,
        content: dict[str, Any],
        source_format: str,
    ) -> tuple[list[Evidence], TokenUsage]:
        prompt = self._render_prompt(
            "extract_evidence.j2",
            question=query_plan.original_question,
            query_type=query_plan.query_type.value,
            time_range=query_plan.time_range,
            file_name=location.file_name,
            location=location.section_id,
            format=source_format,
            content=content.get("content", ""),
            tables=content.get("tables", []),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response, usage = await self.llm.call(messages, model="sonnet", response_format="json", max_tokens=1600)
            payload = json.loads(response)
            rows = payload if isinstance(payload, list) else payload.get("evidences", [])
            evidences = []
            for i, row in enumerate(rows, start=1):
                raw_strength = str(row.get("evidence_strength") or "direct")
                strength = (
                    EvidenceStrength(raw_strength)
                    if raw_strength in {x.value for x in EvidenceStrength}
                    else EvidenceStrength.DIRECT
                )
                evidences.append(
                    Evidence(
                        evidence_id=str(row.get("evidence_id") or f"ev_{i:03d}"),
                        source_file=location.file_name,
                        source_location=location.section_id,
                        source_format=str(row.get("source_format") or source_format),
                        evidence_type=str(row.get("evidence_type") or "text"),
                        evidence_strength=strength,
                        content=str(row.get("content") or "").strip(),
                        time_period=row.get("time_period"),
                        numeric_value=row.get("numeric_value"),
                        unit=row.get("unit"),
                        topic_category=row.get("topic_category"),
                        dedup_key=row.get("dedup_key"),
                        extraction_quality=str(row.get("extraction_quality") or "good"),
                    )
                )
            evidences = [ev for ev in evidences if ev.content]
            evidences = self._validate_evidence_values(evidences, content)
            if evidences:
                return evidences, usage
        except Exception:
            usage = TokenUsage()

        fallback_content = str(content.get("content", "")).strip()
        if not fallback_content:
            return [], usage
        ev = Evidence(
            evidence_id=f"ev_{location.file_name}_{location.section_id}".replace(" ", "_"),
            source_file=location.file_name,
            source_location=location.section_id,
            source_format=source_format,
            evidence_type="text",
            evidence_strength=EvidenceStrength.DIRECT,
            content=fallback_content[:800],
            extraction_quality="degraded",
        )
        return [ev], usage

    @staticmethod
    def _validate_evidence_values(evidences: list[Evidence], raw_section: dict[str, Any]) -> list[Evidence]:
        section_text = str(raw_section.get("content", "") or "")
        for table in raw_section.get("tables", []) or []:
            section_text += "\n" + " ".join(str(x) for x in table.get("headers", []) or [])
            for row in table.get("rows", []) or []:
                section_text += "\n" + " ".join(str(x) for x in row or [])
        source_numbers = StateMachine._extract_numbers(section_text)

        kept: list[Evidence] = []
        for ev in evidences:
            if ev.evidence_strength == EvidenceStrength.DIRECT and ev.numeric_value is not None:
                if not StateMachine._contains_number(source_numbers, float(ev.numeric_value)):
                    continue
            kept.append(ev)
        return kept

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        values: list[float] = []
        for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
            try:
                values.append(float(token.replace(",", "")))
            except ValueError:
                continue
        return values

    @staticmethod
    def _contains_number(candidates: list[float], target: float, tol: float = 1e-6) -> bool:
        for value in candidates:
            if abs(value - target) <= tol:
                return True
        return False

    def _render_prompt(self, template_name: str, **kwargs: Any) -> str:
        if self.env is None:
            return json.dumps(kwargs, ensure_ascii=False)
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    @staticmethod
    def _build_result(ctx: RetrievalContext, answer: str) -> dict[str, Any]:
        return {
            "answer": answer,
            "retrieval_summary": {
                "query_type": ctx.query_plan.query_type.value,
                "total_candidate_files": len(ctx.candidate_files),
                "target_locations": len(ctx.target_locations),
                "evidence_count": len(ctx.evidences),
                "supplement_rounds": ctx.supplement_count,
                "missing_info": ctx.missing_info,
                "token_usage": ctx.token_tracker.summary(),
                "decision_log": ctx.decision_log,
            },
            "evidences": [ev.model_dump() for ev in ctx.evidences],
        }

    @staticmethod
    def _build_failed_result(ctx: RetrievalContext) -> dict[str, Any]:
        return {
            "answer": "未能完成检索流程，请检查索引与知识库数据。",
            "retrieval_summary": {
                "query_type": ctx.query_plan.query_type.value,
                "total_candidate_files": len(ctx.candidate_files),
                "target_locations": len(ctx.target_locations),
                "evidence_count": len(ctx.evidences),
                "supplement_rounds": ctx.supplement_count,
                "missing_info": ctx.missing_info,
                "token_usage": ctx.token_tracker.summary(),
                "decision_log": ctx.decision_log,
            },
            "evidences": [ev.model_dump() for ev in ctx.evidences],
        }
