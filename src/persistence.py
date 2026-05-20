from __future__ import annotations

import os
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# psycopg connections used by AsyncPostgresSaver must be autocommit and must
# not use server-side prepared statements (pgbouncer-safe, pool-safe).
_CONNECTION_KWARGS = {"autocommit": True, "prepare_threshold": 0}


def postgres_dsn() -> str:
    """Build the Postgres DSN from the docker-compose env vars."""
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@asynccontextmanager
async def postgres_checkpointer():
    """Yield a ready AsyncPostgresSaver backed by a connection pool.

    Runs the idempotent ``setup()`` (creates the checkpoint tables / applies
    migrations on first use). The pool is opened on enter and closed on exit,
    so use this as the outermost context around the conversational agent's
    lifetime.
    """
    async with AsyncConnectionPool(
        conninfo=postgres_dsn(),
        max_size=20,
        kwargs=_CONNECTION_KWARGS,
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer
