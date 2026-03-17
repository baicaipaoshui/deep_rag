from __future__ import annotations

import asyncio
import io
import json as _json
import sys
import uuid
from pathlib import Path

import streamlit as st

# Ensure project root is importable when streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chat_memory import ChatMemoryStore
from app.orchestrator.state_machine import StateMachine
from preprocessor.index_builder import IndexBuilder
from project_config import load_project_config


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 20% 20%, #fff6a8 0%, #ffe066 30%, #8bd3ff 100%);
        }
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .sponge-header {
            background: rgba(255, 255, 255, 0.82);
            border: 3px solid #f4a300;
            border-radius: 24px;
            padding: 20px 24px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.15);
            margin-bottom: 14px;
        }
        .sponge-title {
            font-size: 2rem;
            font-weight: 800;
            color: #102a43;
            margin: 0;
        }
        .sponge-subtitle {
            margin-top: 6px;
            color: #334e68;
            font-size: 1rem;
        }
        .bubble-row {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .bubble {
            background: rgba(255, 255, 255, 0.92);
            border: 2px dashed #2f80ed;
            color: #1f2937;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.86rem;
            font-weight: 600;
        }
        .sponge-card {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid #f9a825;
            border-radius: 16px;
            padding: 14px 16px;
            margin-top: 12px;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.12);
        }
        .sponge-card h4 {
            margin: 0 0 8px 0;
            color: #0b3d91;
            font-size: 1rem;
        }
        .history-q {
            margin: 0;
            color: #1f2937;
            font-size: 0.95rem;
            font-weight: 700;
        }
        .history-a {
            margin-top: 8px;
            color: #334155;
            font-size: 0.92rem;
        }
        div.stButton > button {
            border-radius: 999px;
            border: 2px solid #f4a300;
            background: linear-gradient(180deg, #ffe36f 0%, #ffd54f 100%);
            color: #102a43;
            font-weight: 700;
        }
        div.stButton > button:hover {
            border-color: #ff8f00;
            color: #0b2545;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(208, 242, 255, 0.96) 100%);
            border-right: 2px solid rgba(244, 163, 0, 0.45);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="sponge-header">
            <p class="sponge-title">🧽 Deep RAG Bikini Bottom Edition</p>
            <p class="sponge-subtitle">海底总部知识库检索：上传文件、重建索引、快速问答一站完成</p>
            <div class="bubble-row">
                <span class="bubble">🫧 混合检索</span>
                <span class="bubble">🫧 三层索引</span>
                <span class="bubble">🫧 状态机编排</span>
                <span class="bubble">🫧 Token预算守护</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

with st.sidebar:
    st.header("🍍 知识库管理")

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

_inject_theme()
_render_header()

if "history" not in st.session_state:
    st.session_state["history"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"web-{uuid.uuid4().hex[:8]}"

memory = ChatMemoryStore()

with st.sidebar:
    st.subheader("💬 会话记忆")
    session_id_input = st.text_input("Session ID", value=st.session_state["session_id"])
    if session_id_input.strip() != st.session_state["session_id"]:
        st.session_state["session_id"] = session_id_input.strip()
        loaded = memory.load_session(st.session_state["session_id"])
        st.session_state["history"] = [{"q": t["question"], "result": {"answer": t["answer"]}} for t in loaded]
    if st.button("新建会话", key="new_chat_session"):
        st.session_state["session_id"] = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state["history"] = []
    if st.button("加载当前会话记录", key="load_chat_session"):
        loaded = memory.load_session(st.session_state["session_id"])
        st.session_state["history"] = [{"q": t["question"], "result": {"answer": t["answer"]}} for t in loaded]

st.markdown(
    """
    <div class="sponge-card">
      <h4>🔍 海底检索台</h4>
      <p style="margin:0;color:#334155;">输入你的问题，系统会自动执行过滤、定位、抽取、验证和生成流程。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.text_input("请输入问题", placeholder="例如：Aurora 项目发布窗口是什么时候？")
run = st.button("开始检索 🫧")

if run and question.strip():
    with st.spinner("正在执行检索流程..."):
        try:
            chat_history = [
                {"question": row.get("q", ""), "answer": row.get("result", {}).get("answer", "")}
                for row in st.session_state["history"]
                if row.get("q") and row.get("result", {}).get("answer")
            ]
            result = asyncio.run(StateMachine().run(question.strip(), chat_history=chat_history))
        except Exception as e:
            st.error(f"检索失败：{e}")
            st.stop()
    st.session_state["history"].append({"q": question.strip(), "result": result})
    memory.append_turn(st.session_state["session_id"], question.strip(), result.get("answer", ""))

if st.session_state["history"]:
    latest = st.session_state["history"][-1]["result"]
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📝 回答")
        st.write(latest.get("answer", ""))
    with col2:
        st.subheader("📊 统计")
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

    with st.expander("🧭 决策日志", expanded=False):
        st.json(latest.get("retrieval_summary", {}).get("decision_log", []))

    with st.expander("📚 历史问答记录", expanded=False):
        for idx, item in enumerate(reversed(st.session_state["history"]), start=1):
            result = item.get("result", {})
            st.markdown(
                f"""
                <div class="sponge-card">
                    <h4>第 {idx} 次查询</h4>
                    <p class="history-q">Q: {item.get("q", "")}</p>
                    <p class="history-a">A: {result.get("answer", "")[:220]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
