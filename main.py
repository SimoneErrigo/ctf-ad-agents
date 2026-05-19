from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv


DEFAULT_QUERY = "Are there any attacks visible in the captured traffic so far?"


async def amain(query: str) -> None:
    from src.graph import build_app

    app = await build_app()
    final_state = await app.ainvoke({"query": query})
    print("\n=== FINAL ANSWER ===\n")
    print(final_state.get("final_answer", "<no answer>"))


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
