from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone

from app.chat_memory import ChatMemoryStore
from app.llm_client import LLMClient
from app.logger import QueryLogger, get_logger
from app.orchestrator.state_machine import StateMachine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep RAG CLI")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")
    parser.add_argument("--session-id", default="default", help="Conversation session id for chat memory")
    parser.add_argument("--new-session", action="store_true", help="Start a clean session history")
    return parser.parse_args()


async def run_once(question: str, chat_history: list[dict[str, str]] | None = None) -> dict:
    machine = StateMachine()
    return await machine.run(question, chat_history=chat_history)


async def interactive_loop(session_id: str, new_session: bool) -> None:
    logger = get_logger("deep_rag.cli")
    query_logger = QueryLogger()
    memory = ChatMemoryStore()
    history: list[dict[str, str]] = [] if new_session else memory.load_session(session_id)
    print("Deep RAG interactive mode. Type 'exit' to quit.")
    if history:
        print(f"[memory] loaded {len(history)} turns from session '{session_id}'")
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        result = await run_once(question, chat_history=history)
        print("\n" + result.get("answer", ""))
        summary = result.get("retrieval_summary", {})
        print(
            f"\n[summary] evidences={summary.get('evidence_count')} "
            f"supplements={summary.get('supplement_rounds')} "
            f"tokens={summary.get('token_usage', {}).get('total', {}).get('total_tokens')}"
        )
        query_logger.log(
            {
                "mode": "interactive",
                "session_id": session_id,
                "question": question,
                "result": result,
                "timestamp_local": datetime.now(timezone.utc).isoformat(),
            }
        )
        history.append({"question": question, "answer": result.get("answer", "")})
        memory.save_session(session_id, history)
        logger.info("query completed")


async def main_async() -> int:
    args = parse_args()
    llm = LLMClient()
    if not llm.is_available():
        print(
            f"[warn] LLM provider '{llm.provider}' unavailable, running in rule-based fallback mode.",
            file=sys.stderr,
        )
    if args.interactive:
        await interactive_loop(session_id=args.session_id, new_session=args.new_session)
        return 0

    if not args.question:
        print("Question is required unless --interactive is used.", file=sys.stderr)
        return 1

    memory = ChatMemoryStore()
    history = [] if args.new_session else memory.load_session(args.session_id)
    result = await run_once(args.question, chat_history=history)
    QueryLogger().log({"mode": "single", "session_id": args.session_id, "question": args.question, "result": result})
    history.append({"question": args.question, "answer": result.get("answer", "")})
    memory.save_session(args.session_id, history)
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("answer", ""))
        print(json.dumps(result.get("retrieval_summary", {}), ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
