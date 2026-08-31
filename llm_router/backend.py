from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import EndpointConfig
from .messages import is_valid_completion
from .stream import ValidatedStream


class BackendClient:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()

        self._clients.clear()

    def _get_client(self, endpoint: EndpointConfig) -> httpx.AsyncClient:
        key = endpoint.id or "local"
        if key not in self._clients:
            self._clients[key] = httpx.AsyncClient(http2=False)

        return self._clients[key]

    async def probe_local_model(self, endpoint: EndpointConfig) -> str | None:
        timeout = httpx.Timeout(endpoint.health_check_timeout_seconds)
        try:
            client = self._get_client(endpoint)
            response = await client.get(
                f"{endpoint.openai_base_url}/models",
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exception:
            self._logger.info("Local health probe failed: %s", exception)
            return None

        if not isinstance(data, dict):
            return None

        for item in data.get("data", []):
            if (
                isinstance(item, dict)
                and item.get("status", {}).get("value") == "loaded"
            ):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id.strip()

        return None

    async def request_endpoint(
        self,
        endpoint: EndpointConfig,
        model: str,
        body: dict[str, Any],
        read_timeout_override: float | None = None,
    ) -> dict[str, Any] | AsyncIterator[bytes] | None:
        headers = self._headers(endpoint)
        if headers is None:
            return None

        label = endpoint.id or "local"
        read_timeout = read_timeout_override or endpoint.read_timeout_seconds

        return await self._request(
            label=f"{label} {model}",
            endpoint=endpoint,
            headers=headers,
            connect_timeout=endpoint.connect_timeout_seconds,
            read_timeout=read_timeout,
            model=model,
            body=body,
        )

    def _headers(self, endpoint: EndpointConfig) -> dict[str, str] | None:
        headers = {"Content-Type": "application/json"}
        authentication = endpoint.authentication
        if authentication.type == "none":
            return headers

        environment_variable = authentication.environment_variable
        if environment_variable is None:
            return None

        credential = os.getenv(environment_variable, "").strip()
        if not credential:
            self._logger.warning(
                "%s credential environment variable is not configured",
                endpoint.id or "local",
            )

            return None

        if authentication.type == "bearer":
            headers["Authorization"] = f"Bearer {credential}"
        elif authentication.header_name is not None:
            headers[authentication.header_name] = credential

        return headers

    async def _request(
        self,
        label: str,
        endpoint: EndpointConfig,
        headers: dict[str, str],
        connect_timeout: float,
        read_timeout: float,
        model: str,
        body: dict[str, Any],
    ) -> dict[str, Any] | AsyncIterator[bytes] | None:
        url = f"{endpoint.openai_base_url}/chat/completions"
        payload = {**body, "model": model}
        timeout = httpx.Timeout(
            connect=connect_timeout,
            pool=connect_timeout,
            read=read_timeout,
            write=read_timeout,
        )
        if bool(payload.get("stream")):
            return await self._request_stream(
                label, url, headers, timeout, payload, endpoint
            )

        return await self._request_completion(
            label, url, headers, timeout, payload, endpoint
        )

    async def _request_stream(
        self,
        label: str,
        url: str,
        headers: dict[str, str],
        timeout: httpx.Timeout,
        payload: dict[str, Any],
        endpoint: EndpointConfig,
    ) -> AsyncIterator[bytes] | None:
        client = self._get_client(endpoint)
        try:
            request = client.build_request(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exception:
            self._logger.warning("%s request failed: %s", label, exception)
            return None

        if response.is_error:
            self._logger.warning("%s returned HTTP %s", label, response.status_code)
            await response.aclose()

            return None

        return await ValidatedStream(label, response, client).create()

    async def _request_completion(
        self,
        label: str,
        url: str,
        headers: dict[str, str],
        timeout: httpx.Timeout,
        payload: dict[str, Any],
        endpoint: EndpointConfig,
    ) -> dict[str, Any] | None:
        try:
            client = self._get_client(endpoint)
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exception:
            self._logger.warning("%s request failed: %s", label, exception)
            return None

        if not isinstance(data, dict) or not is_valid_completion(data):
            self._logger.warning("%s returned an invalid completion", label)

            return None

        return data