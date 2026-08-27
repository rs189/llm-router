from __future__ import annotations

from collections.abc import AsyncIterator
import logging
import os
from typing import Any

import httpx

from .config import EndpointConfig
from .messages import is_valid_completion
from .stream import ValidatedStream


class BackendClient:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    async def probe_local_model(self, endpoint: EndpointConfig) -> str | None:
        timeout = httpx.Timeout(endpoint.health_check_timeout_seconds)
        try:
            async with httpx.AsyncClient() as client:
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
    ) -> dict[str, Any] | AsyncIterator[bytes] | None:
        headers = self._headers(endpoint)
        if headers is None:
            return None
        label = endpoint.id or "local"
        return await self._request(
            label=f"{label} {model}",
            url=f"{endpoint.openai_base_url}/chat/completions",
            headers=headers,
            connect_timeout=endpoint.connect_timeout_seconds,
            read_timeout=endpoint.read_timeout_seconds,
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
        url: str,
        headers: dict[str, str],
        connect_timeout: float,
        read_timeout: float,
        model: str,
        body: dict[str, Any],
    ) -> dict[str, Any] | AsyncIterator[bytes] | None:
        payload = {**body, "model": model}
        timeout = httpx.Timeout(
            connect=connect_timeout,
            pool=connect_timeout,
            read=read_timeout,
            write=read_timeout,
        )
        if bool(payload.get("stream")):
            return await self._request_stream(label, url, headers, timeout, payload)
        return await self._request_completion(label, url, headers, timeout, payload)

    async def _request_stream(
        self,
        label: str,
        url: str,
        headers: dict[str, str],
        timeout: httpx.Timeout,
        payload: dict[str, Any],
    ) -> AsyncIterator[bytes] | None:
        client = httpx.AsyncClient(http2=False)
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
            await client.aclose()
            self._logger.warning("%s request failed: %s", label, exception)
            return None
        if response.is_error:
            self._logger.warning("%s returned HTTP %s", label, response.status_code)
            await response.aclose()
            await client.aclose()
            return None
        return await ValidatedStream(label, response, client).create()

    async def _request_completion(
        self,
        label: str,
        url: str,
        headers: dict[str, str],
        timeout: httpx.Timeout,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(http2=False) as client:
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
