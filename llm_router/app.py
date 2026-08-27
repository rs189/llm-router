from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .backend import BackendClient
from .config import EndpointConfig, RouterConfig
from .messages import normalize_messages
from .router import ModelRouter
from .state import RoutedModelRegistry


ROUTED_MODEL_HEADER = "x-routed-model"
SWITCHED_BACKEND_HEADER = "x-router-is_switched"


def create_app(config: RouterConfig) -> FastAPI:
    app = FastAPI(title="LLM Router")
    state = RoutedModelRegistry()
    model_router = ModelRouter(
        config,
        BackendClient(config.local_backend),
        state,
    )

    @app.post("/v1/chat/completions")
    async def route_chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "Request body is not valid JSON")
        if not isinstance(body, dict):
            return _error(400, "Request body must be a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return _error(400, "messages must be a non-empty array")
        body["messages"] = normalize_messages(messages)
        if not body["messages"]:
            return _error(400, "No valid messages remain after normalization")
        requested_model = str(
            body.get("model", config.default_model)
        ).strip().lower()
        routed = await model_router.route(body, requested_model)
        if routed is None:
            return _unavailable_response(config, requested_model)
        result, model, is_switched = routed
        return _success_response(config, result, model, is_switched)

    @app.get("/models")
    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return _models_response(config)

    @app.get("/last_routed_model")
    async def last_routed_model() -> JSONResponse:
        return JSONResponse(content={"model": state.read()})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _success_response(
    config: RouterConfig,
    result: dict[str, Any] | AsyncIterator[bytes],
    model: str,
    is_switched: bool,
) -> Response:
    headers = {ROUTED_MODEL_HEADER: config.display_name_for(model)}
    if is_switched:
        headers[SWITCHED_BACKEND_HEADER] = "true"
    if isinstance(result, dict):
        return JSONResponse(content=result, headers=headers)
    return StreamingResponse(result, media_type="text/event-stream", headers=headers)


def _models_response(config: RouterConfig) -> dict[str, Any]:
    records = [
        _model_record(
            preset.id,
            "router",
            config.preset_context_length(preset),
        )
        for preset in config.presets
    ]
    records.extend(
        _model_record(model.id, "local", model.context_length)
        for model in config.local_models
    )
    records.extend(
        _model_record(model.id, "remote", model.context_length)
        for model in config.remote_models()
    )
    return {"object": "list", "data": records}


def _model_record(model_id: str, owner: str, context_length: int) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": owner,
        "context_length": context_length,
        "context_window": context_length,
        "max_model_len": context_length,
    }


def _unavailable_response(config: RouterConfig, requested_model: str) -> JSONResponse:
    if config.find_remote_model(requested_model) is not None:
        message = f"Remote model {requested_model} is unavailable"
    elif config.find_local_model(requested_model) is not None:
        message = f"Local model {requested_model} is unavailable"
    elif config.find_preset(requested_model) is not None:
        message = f"No configured route could serve preset {requested_model}"
    else:
        message = "All configured LLM backends failed before producing a response"
    return _error(503, message)


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "router_error"}},
    )
