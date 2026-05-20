from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from src.graph import build_chat_app

DEFAULT_QUERY = "Are there any attacks visible in the captured traffic so far?"


async def amain(query: str) -> None:

    thread_id = os.environ.get("THREAD_ID", "cli-session")
    config = {"configurable": {"thread_id": thread_id}}

    async with build_chat_app() as agent:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config,
        )
    print("\n=== FINAL ANSWER ===\n")
    print(result["messages"][-1].content)


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    asyncio.run(amain(query))


if __name__ == "__main__":
    main()
