from __future__ import annotations

from typing import Any


ALLOWED_MESSAGE_KEYS = {
    "content",
    "name",
    "role",
    "tool_call_id",
    "tool_calls",
}


def normalize_messages(messages: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)

            continue

        clean = {
            key: value for key, value in message.items() if key in ALLOWED_MESSAGE_KEYS
        }
        has_content = _has_content(clean.get("content"))
        if (
            clean.get("role") == "assistant"
            and not has_content
            and not clean.get("tool_calls")
        ):
            continue

        normalized.append(clean)

    return normalized


def is_valid_completion(data: dict[str, Any]) -> bool:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False

    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return False

    return (
        (_has_content(message.get("content")) or bool(message.get("tool_calls")))
        and choice.get("finish_reason") != "content_filter"
    )


def _has_content(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())

    return content is not None