from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone

from app.llm_client import LLMClient
from app.logger import QueryLogger, get_logger
from app.orchestrator.state_machine import StateMachine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep RAG CLI")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")
    return parser.parse_args()


async def run_once(question: str) -> dict:
    machine = StateMachine()
    return await machine.run(question)


async def interactive_loop() -> None:
    logger = get_logger("deep_rag.cli")
    query_logger = QueryLogger()
    print("Deep RAG interactive mode. Type 'exit' to quit.")
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        result = await run_once(question)
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
                "question": question,
                "result": result,
                "timestamp_local": datetime.now(timezone.utc).isoformat(),
            }
        )
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
        await interactive_loop()
        return 0

    if not args.question:
        print("Question is required unless --interactive is used.", file=sys.stderr)
        return 1

    result = await run_once(args.question)
    QueryLogger().log({"mode": "single", "question": args.question, "result": result})
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
