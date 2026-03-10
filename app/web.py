from __future__ import annotations

import asyncio

import streamlit as st

from app.orchestrator.state_machine import StateMachine


st.set_page_config(page_title="Deep RAG", layout="wide")
st.title("Deep RAG 知识库检索")
st.caption("Python Orchestrator + MCP Tools + Local Ollama")

if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("请输入问题", placeholder="例如：最近三年的销售趋势如何？")
run = st.button("检索")

if run and question.strip():
    with st.spinner("正在执行检索流程..."):
        result = asyncio.run(StateMachine().run(question.strip()))
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
