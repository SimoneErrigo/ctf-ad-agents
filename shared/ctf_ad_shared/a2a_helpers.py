# Reusable helpers for A2A client calls from orchestrator -> remote agent

from __future__ import annotations

import logging
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    TextPart,
)

logger = logging.getLogger(__name__)


async def discover_agent(base_url: str) -> AgentCard:
    """Fetch an agent's card from its .well-known endpoint"""

    async with httpx.AsyncClient() as client:
        resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
        card = await resolver.get_agent_card()
        logger.info("Discovered agent '%s' at %s", card.name, base_url)
        return card


async def send_message_to_agent(
    agent_url: str,
    text: str,
    context_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Send a text message to a remote A2A agent and return the response text. Orchestrator can use this to delegate work"""
    
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        resolver = A2ACardResolver(httpx_client=http_client, base_url=agent_url)
        card = await resolver.get_agent_card()
        a2a_client = A2AClient(httpx_client=http_client, agent_card=card)

        request = SendMessageRequest(
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"type": "text", "text": text}],
                    "messageId": uuid4().hex,
                },
            )
        )

        if context_id:
            request.params.message["contextId"] = context_id

        response = await a2a_client.send_message(request)

        # Extract text from the response
        result = response.result
        if result is None:
            return "[Agent returned no result]"

        # The result could be a Task or a Message
        # Try to extract text from artifacts or parts
        texts: list[str] = []

        if hasattr(result, "artifacts") and result.artifacts:
            for artifact in result.artifacts:
                if hasattr(artifact, "parts") and artifact.parts:
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            texts.append(part.text)

        if hasattr(result, "status") and result.status:
            if hasattr(result.status, "message") and result.status.message:
                msg = result.status.message
                if hasattr(msg, "parts") and msg.parts:
                    for part in msg.parts:
                        if hasattr(part, "text"):
                            texts.append(part.text)

        if not texts:
            return "[Agent returned empty response]"

        return "\n".join(texts)
