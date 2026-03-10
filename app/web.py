from __future__ import annotations

import asyncio
import io
import json as _json
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.orchestrator.state_machine import StateMachine
from preprocessor.index_builder import IndexBuilder
from project_config import load_project_config


def _extract_preview(filename: str, data: bytes) -> str:
    """Extract first ~500 chars of content from uploaded file bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".md":
        return data.decode("utf-8", errors="ignore")[:500]
    if suffix == ".pdf":
        try:
            import fitz
            with fitz.open(stream=data, filetype="pdf") as doc:
                return (doc[0].get_text("text") if doc.page_count > 0 else "")[:500]
        except Exception:
            return ""
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
            ws = wb.worksheets[0]
            lines = []
            for row in ws.iter_rows(values_only=True, max_row=10):
                lines.append(" | ".join("" if v is None else str(v) for v in row))
            wb.close()
            return "\n".join(lines)[:500]
        except Exception:
            return ""
    return ""


async def _classify_file(filename: str, preview: str) -> str:
    """Use LLM to determine a folder name for the file; fall back to 'general'."""
    from app.llm_client import LLMClient
    client = LLMClient()
    prompt = (
        f"文件名：{filename}\n内容摘要：{preview[:400]}\n\n"
        "根据以上信息，为该文件选择一个英文文件夹名（小写，下划线分隔，2~3个单词）。"
        "只输出 JSON，格式：{\"folder\": \"folder_name\"}"
    )
    try:
        result, _ = await client.call(
            messages=[{"role": "user", "content": prompt}],
            response_format="json",
            max_tokens=60,
        )
        return _json.loads(result).get("folder", "general") or "general"
    except Exception:
        return "general"


st.set_page_config(page_title="Deep RAG", layout="wide")
st.title("Deep RAG 知识库检索")
st.caption("Python Orchestrator + MCP Tools + Local Ollama")

cfg = load_project_config()
KB_DIR = Path(cfg.kb_dir)

# ── 侧边栏：上传知识库 ──────────────────────────────────────────────
with st.sidebar:
    st.header("知识库管理")

    # 显示现有文件数
    existing = list(KB_DIR.rglob("*"))
    existing_files = [f for f in existing if f.is_file() and f.suffix.lower() in {".md", ".pdf", ".xlsx", ".xlsm", ".xls"}]
    st.caption(f"当前知识库：{len(existing_files)} 个文件")

    with st.expander("查看已索引文件", expanded=False):
        for f in sorted(existing_files):
            st.caption(str(f.relative_to(KB_DIR)))

    uploaded = st.file_uploader(
        "上传文件（支持 PDF / Excel / Markdown）",
        type=["pdf", "xlsx", "xlsm", "xls", "md"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("上传并重建索引", type="primary"):
        saved = []
        for f in uploaded:
            preview = _extract_preview(f.name, f.getvalue())
            folder = asyncio.run(_classify_file(f.name, preview))
            dest_dir = KB_DIR / folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / f.name).write_bytes(f.getvalue())
            saved.append((f.name, folder))
            st.write(f"✔ `{f.name}` → `{folder}/`")

        with st.spinner("正在重建索引，请稍候..."):
            try:
                builder = IndexBuilder(kb_dir=str(KB_DIR), index_dir=str(cfg.index_dir))
                stats = builder.build_index()
                st.success(
                    f"索引重建完成！共 {stats['total_files']} 个文件，"
                    f"{stats['total_chunks']} 个 chunk"
                )
            except Exception as e:
                st.error(f"索引重建失败：{e}")

    st.divider()
    st.caption("支持格式：`.md` `.pdf` `.xlsx` `.xlsm` `.xls`")

# ── 主区域：检索 ───────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("请输入问题", placeholder="例如：最近三年的销售趋势如何？")
run = st.button("检索")

if run and question.strip():
    with st.spinner("正在执行检索流程..."):
        try:
            result = asyncio.run(StateMachine().run(question.strip()))
        except Exception as e:
            st.error(f"检索失败：{e}")
            st.stop()
    st.session_state["history"].append({"q": question.strip(), "result": result})

if st.session_state["history"]:
    latest = st.session_state["history"][-1]["result"]
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("回答")
        st.write(latest.get("answer", ""))
    with col2:
        st.subheader("统计")
        summary = latest.get("retrieval_summary", {})
        st.json(
            {
                "query_type": summary.get("query_type"),
                "candidate_files": summary.get("total_candidate_files"),
                "target_locations": summary.get("target_locations"),
                "evidences": summary.get("evidence_count"),
                "supplements": summary.get("supplement_rounds"),
                "total_tokens": summary.get("token_usage", {}).get("total", {}).get("total_tokens"),
            }
        )

    with st.expander("决策日志", expanded=False):
        st.json(latest.get("retrieval_summary", {}).get("decision_log", []))
