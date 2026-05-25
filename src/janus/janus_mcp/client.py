from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from janus_mcp.config import Settings, get_settings

log = logging.getLogger(__name__)


class JanusError(RuntimeError):

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Janus {status}: {message}")
        self.status = status
        self.message = message


class JanusClient:

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._http = httpx.AsyncClient(
            base_url=self._s.janus_url.rstrip("/"),
            timeout=self._s.janus_timeout,
        )
        self._token: str | None = None
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _login(self) -> str:
        async with self._lock:
            if self._token is not None:
                return self._token
            r = await self._http.post(
                "/api/login",
                json={
                    "password": self._s.janus_password,
                    "display_name": self._s.janus_display_name,
                },
            )
            if r.status_code != 200:
                raise JanusError(r.status_code, r.text.strip() or "login failed")
            self._token = r.json()["token"]
            log.info("Logged into Janus as %s", self._s.janus_display_name)
            return self._token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        # Avoid sending None values to Janus, which can cause it to reject the request with 400
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        token = self._token or await self._login()
        headers = {"Authorization": f"Bearer {token}"}
        r = await self._http.request(method, path, params=clean_params, json=json, headers=headers)
        if r.status_code == 401:
            # Token expired or revoked, drop it and retry once
            self._token = None
            token = await self._login()
            headers["Authorization"] = f"Bearer {token}"
            r = await self._http.request(method, path, params=clean_params, json=json, headers=headers)
        if r.status_code >= 400:
            raise JanusError(r.status_code, r.text.strip() or r.reason_phrase)
        return r

    # Traffic tools

    async def list_services(self) -> list[dict[str, Any]]:
        r = await self._request("GET", "/api/services")
        return r.json()

    async def list_packets(self, params: dict[str, Any]) -> dict[str, Any]:
        r = await self._request("GET", "/api/packets", params=params)
        return r.json()

    async def get_packet(self, packet_id: int) -> dict[str, Any]:
        r = await self._request("GET", f"/api/packets/{packet_id}")
        return r.json()

    async def get_flow(self, packet_id: int) -> dict[str, Any]:
        r = await self._request("GET", "/api/packets/flow", params={"packet_id": packet_id})
        return r.json()

    async def validate_filter(self, expression: str) -> dict[str, Any]:
        r = await self._request("POST", "/api/filter/validate", json={"expression": expression})
        return r.json()

    async def get_capture_status(self) -> dict[str, Any]:
        r = await self._request("GET", "/api/traffic/capture")
        return r.json()

    # Rules and alerts

    async def list_rules(self, service_id: str | None = None) -> list[dict[str, Any]]:
        r = await self._request("GET", "/api/rules", params={"service_id": service_id})
        # Janus serializes a nil slice as JSON `null`. Normalize to [] so MCP
        # consumers see a valid array (their schema rejects `null`).
        data = r.json()
        return data if isinstance(data, list) else []

    async def get_rule(self, rule_id: str) -> dict[str, Any]:
        r = await self._request("GET", f"/api/rules/{rule_id}")
        return r.json()

    async def create_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        r = await self._request("POST", "/api/rules", json=rule)
        return r.json()

    async def update_rule(self, rule_id: str, rule: dict[str, Any]) -> dict[str, Any]:
        r = await self._request("PUT", f"/api/rules/{rule_id}", json=rule)
        return r.json()

    async def delete_rule(self, rule_id: str) -> None:
        await self._request("DELETE", f"/api/rules/{rule_id}")


    async def list_alerts(self, params: dict[str, Any]) -> dict[str, Any]:
        r = await self._request("GET", "/api/alerts", params=params)
        data = r.json()
        if isinstance(data, dict) and data.get("alerts") is None:
            data["alerts"] = []
        return data

    async def get_alert(self, alert_id: int) -> dict[str, Any]:
        r = await self._request("GET", f"/api/alerts/{alert_id}")
        return r.json()
