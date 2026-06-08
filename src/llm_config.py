from __future__ import annotations

from functools import lru_cache

from botocore.config import Config


@lru_cache(maxsize=1)
def bedrock_config() -> Config:
    """
    Bedrock occasionally drops a slow streaming response, surfacing as
    ``ReadTimeoutError(... bedrock-runtime ...)`` and killing the whole run. A
    generous ``read_timeout`` plus adaptive retries makes a transient blip
    recoverable instead of fatal.
    """
    return Config(
        read_timeout=120,
        connect_timeout=10,
        retries={"max_attempts": 4, "mode": "adaptive"},
    )
