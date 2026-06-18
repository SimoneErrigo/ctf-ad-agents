from __future__ import annotations

import os
from functools import lru_cache

from botocore.config import Config
from langchain_core.rate_limiters import InMemoryRateLimiter


@lru_cache(maxsize=1)
def bedrock_config() -> Config:
    """
    Bedrock occasionally drops a slow streaming response, surfacing as
    ``ReadTimeoutError(... bedrock-runtime ...)`` and killing the whole run. A
    generous ``read_timeout`` plus adaptive retries makes a transient blip
    recoverable instead of fatal. ``adaptive`` mode also retries throttling
    (``ThrottlingException``) with client-side backoff; the extra attempts give
    that backoff room to clear a residual throttle that slips past the shared
    rate limiter below.
    """
    return Config(
        read_timeout=120,
        connect_timeout=10,
        retries={"max_attempts": 10, "mode": "adaptive"},
    )


@lru_cache(maxsize=1)
def bedrock_rate_limiter() -> InMemoryRateLimiter:
    """A single process-wide, client-side request limiter shared by EVERY
    Bedrock LLM (each agent passes this one cached instance).

    An all-services request ("find the attacks across all services") fans out
    into one sub-agent per live service, all running concurrently in the same
    LangGraph superstep (see ``orchestrator.route_to_agents``). Without a shared
    cap their Converse calls fire at once and burst past the model's per-second
    request quota, so Bedrock returns ``ThrottlingException`` and the run dies.
    One shared token bucket paces the aggregate request rate so the burst is
    spread out instead of hitting Bedrock simultaneously; the adaptive retries
    in ``bedrock_config`` then only have to absorb the rare leftover. Tune with
    ``BEDROCK_MAX_RPS`` (requests/second) to match the account's Bedrock quota.
    """
    rps = float(os.getenv("BEDROCK_MAX_RPS", "4"))
    return InMemoryRateLimiter(
        requests_per_second=rps,
        check_every_n_seconds=0.1,
        # Bucket caps the burst, but must stay >= 1: a sub-1 rps (a 5-req/min
        # Bedrock quota is only ~0.08 rps) would otherwise cap available_tokens
        # below 1.0, never accrue a whole token, and block every call forever.
        max_bucket_size=max(1.0, rps),
    )
