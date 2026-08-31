from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging

import httpx


SSE_CLEAN_STOP = (
    b"data: {\"id\":\"router-stream-stop\",\"object\":\"chat.completion.chunk\","
    b"\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
)
SSE_DONE = b"data: [DONE]\n\n"


class ValidatedStream:
    def __init__(
        self,
        label: str,
        response: httpx.Response,
        client: httpx.AsyncClient,
    ) -> None:
        self._label = label
        self._response = response
        self._client = client
        self._logger = logging.getLogger(__name__)

    async def create(self) -> AsyncIterator[bytes] | None:
        lines = self._response.aiter_lines()
        try:
            first_line = await anext(lines)
        except (StopAsyncIteration, httpx.HTTPError) as exception:
            await self._close()
            self._logger.warning(
                "%s stream failed before output: %s",
                self._label,
                exception,
            )

            return None

        return self._generate([first_line], lines)

    async def _generate(
        self,
        buffered: list[str],
        lines: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        is_done = False
        try:
            for line in buffered:
                yield f"{line}\n".encode("utf-8")
            async for line in lines:
                _, is_done = _parse_sse_line(line)

                yield f"{line}\n".encode("utf-8")
        except httpx.HTTPError as exception:
            self._logger.warning(
                "%s stream failed after output: %s",
                self._label,
                exception,
            )
        finally:
            if not is_done:
                yield SSE_CLEAN_STOP

                yield SSE_DONE
            await self._close()

    async def _close(self) -> None:
        await self._response.aclose()


def _parse_sse_line(line: str) -> tuple[bool, bool]:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return False, False

    payload = stripped.removeprefix("data:").strip()
    if payload == "[DONE]":
        return False, True

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False, False

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False, False

    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return False, False

    output_fields = ("content", "reasoning", "reasoning_content", "refusal")
    has_text_output = any(delta.get(key) not in (None, "", []) for key in output_fields)
    has_tool_output = bool(delta.get("tool_calls") or delta.get("function_call"))

    return has_text_output or has_tool_output, False