from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml


DEFAULT_CONTEXT_LENGTH: Final[int] = 65536


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    id: str
    display_name: str
    context_length: int


@dataclass(frozen=True)
class AuthenticationConfig:
    type: str = "none"
    environment_variable: str | None = None
    header_name: str | None = None


@dataclass(frozen=True)
class EndpointConfig:
    id: str | None = None
    openai_base_url: str = ""
    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 300.0
    attempt_timeout_seconds: float | None = None
    health_check_timeout_seconds: float = 3.0
    models: tuple[ModelConfig, ...] = ()


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class PresetModelConfig:
    id: str
    priority: int


@dataclass(frozen=True)
class PresetRouteConfig:
    endpoint_id: str | None
    priority: int
    models: tuple[PresetModelConfig, ...]


@dataclass(frozen=True)
class PresetConfig:
    id: str
    display_name: str
    system_prompt: str | None
    routes: tuple[PresetRouteConfig, ...]


@dataclass(frozen=True)
class RouterConfig:
    server: ServerConfig
    endpoints: tuple[EndpointConfig, ...]
    global_system_prompt: str | None
    default_model: str
    presets: tuple[PresetConfig, ...]

    def find_endpoint(self, endpoint_id: str | None) -> EndpointConfig | None:
        return next(
            (endpoint for endpoint in self.endpoints if endpoint.id == endpoint_id),
            None,
        )

    def find_endpoint_for_model(
        self,
        model_id: str,
    ) -> tuple[EndpointConfig, ModelConfig] | None:
        for endpoint in self.endpoints:
            model = next(
                (
                    candidate
                    for candidate in endpoint.models
                    if candidate.id == model_id
                ),
                None,
            )
            if model is not None:
                return endpoint, model

        return None

    def endpoint_models(self) -> tuple[ModelConfig, ...]:
        return tuple(
            model
            for endpoint in self.endpoints
            for model in endpoint.models
        )

    def find_preset(self, preset_id: str) -> PresetConfig | None:
        return next(
            (preset for preset in self.presets if preset.id == preset_id),
            None,
        )

    @property
    def local_models(self) -> tuple[ModelConfig, ...]:
        local_endpoint = next(
            (e for e in self.endpoints if e.id is None),
            None,
        )

        return local_endpoint.models if local_endpoint else ()

    def remote_models(self) -> tuple[ModelConfig, ...]:
        return tuple(
            model
            for endpoint in self.endpoints
            if endpoint.id is not None
            for model in endpoint.models
        )

    def find_remote_model(self, model_id: str) -> ModelConfig | None:
        for endpoint in self.endpoints:
            if endpoint.id is not None:
                for model in endpoint.models:
                    if model.id == model_id:
                        return model

        return None

    def find_local_model(self, model_id: str) -> ModelConfig | None:
        for endpoint in self.endpoints:
            if endpoint.id is None:
                for model in endpoint.models:
                    if model.id == model_id:
                        return model

        return None

    def ordered_routes(self, preset: PresetConfig) -> tuple[PresetRouteConfig, ...]:
        return tuple(
            sorted(preset.routes, key=lambda route: route.priority, reverse=True)
        )

    def ordered_route_models(
        self,
        route: PresetRouteConfig,
    ) -> tuple[PresetModelConfig, ...]:
        return tuple(
            sorted(route.models, key=lambda model: model.priority, reverse=True)
        )

    def display_name_for(self, model_id: str) -> str:
        entry = self.find_endpoint_for_model(model_id)
        return entry[1].display_name if entry is not None else model_id

    def context_length_for(self, model_id: str | None) -> int:
        if model_id is None:
            return DEFAULT_CONTEXT_LENGTH

        entry = self.find_endpoint_for_model(model_id)

        return entry[1].context_length if entry is not None else DEFAULT_CONTEXT_LENGTH

    def system_prompt_for(self, requested_model: str) -> str | None:
        preset = self.find_preset(requested_model)
        if preset is not None and preset.system_prompt is not None:
            return preset.system_prompt

        return self.global_system_prompt

    def preset_context_length(self, preset: PresetConfig) -> int:
        return max(
            self.context_length_for(model.id)
            for route in preset.routes
            for model in route.models
        )


class ConfigLoader:
    def load(self, path: Path) -> RouterConfig:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exception:
            raise ConfigError(
                f"Cannot read configuration file {path}: {exception}"
            ) from exception

        try:
            document = yaml.safe_load(content)
        except yaml.YAMLError as exception:
            raise ConfigError(f"Invalid YAML in {path}: {exception}") from exception

        if not isinstance(document, dict):
            raise ConfigError("Configuration root must be a mapping")

        return self._parse(document)

    def _parse(self, document: dict[str, Any]) -> RouterConfig:
        server = self._mapping(document, "server")
        presets = self._presets(self._sequence(document, "presets"))
        router = self._mapping(document, "router")
        default_model = self._string(router, "default_model")
        global_system_prompt = self._system_prompt(router, "system_prompt")

        endpoints = list(self._remote_endpoints(document))

        local_backend_data = document.get("local_backend")
        if local_backend_data is not None:
            local_endpoint = self._local_endpoint(
                self._mapping(document, "local_backend")
            )
            endpoints.insert(0, local_endpoint)

        endpoints_tuple = tuple(endpoints)
        self._validate_presets(presets, endpoints_tuple)
        if default_model not in {preset.id for preset in presets}:
            raise ConfigError("router.default_model must name a configured preset")

        return RouterConfig(
            server=ServerConfig(
                host=self._string(server, "host"),
                port=self._integer(server, "port"),
            ),
            endpoints=endpoints_tuple,
            global_system_prompt=global_system_prompt,
            default_model=default_model,
            presets=presets,
        )

    def _local_endpoint(self, data: dict[str, Any]) -> EndpointConfig:
        return EndpointConfig(
            id=None,
            openai_base_url=self._openai_base_url(data),
            authentication=AuthenticationConfig(),
            connect_timeout_seconds=self._number(data, "connect_timeout_seconds"),
            read_timeout_seconds=self._number(data, "read_timeout_seconds"),
            health_check_timeout_seconds=self._number(
                data, "health_check_timeout_seconds"
            ),
            models=(),
        )

    def _remote_endpoints(self, document: dict[str, Any]) -> tuple[EndpointConfig, ...]:
        data = document.get("remote_endpoints")
        if data is None:
            return ()

        return tuple(
            self._endpoint(self._item_mapping(item))
            for item in self._sequence(document, "remote_endpoints")
        )

    def _endpoints(self, items: list[Any]) -> tuple[EndpointConfig, ...]:
        endpoints = tuple(self._endpoint(self._item_mapping(item)) for item in items)
        if not endpoints:
            raise ConfigError("endpoints cannot be empty")

        endpoint_ids = [endpoint.id for endpoint in endpoints]
        if len(set(endpoint_ids)) != len(endpoints):
            raise ConfigError("Each endpoint id must be unique")

        all_model_ids = [
            model.id
            for endpoint in endpoints
            for model in endpoint.models
        ]
        if len(set(all_model_ids)) != len(all_model_ids):
            raise ConfigError("Model ids must be unique across endpoints")

        return endpoints

    def _endpoint(self, data: dict[str, Any]) -> EndpointConfig:
        authentication_data = data.get("authentication")
        authentication = (
            self._authentication(self._mapping(data, "authentication"))
            if authentication_data is not None
            else AuthenticationConfig()
        )

        return EndpointConfig(
            id=self._string(data, "id"),
            openai_base_url=self._openai_base_url(data),
            authentication=authentication,
            connect_timeout_seconds=self._number(data, "connect_timeout_seconds"),
            read_timeout_seconds=self._number(data, "read_timeout_seconds"),
            attempt_timeout_seconds=self._optional_number(
                data, "attempt_timeout_seconds"
            ),
            health_check_timeout_seconds=self._optional_number(
                data, "health_check_timeout_seconds"
            )
            or 3.0,
            models=self._models(self._sequence(data, "models")),
        )

    def _authentication(self, data: dict[str, Any]) -> AuthenticationConfig:
        authentication_type = self._string(data, "type").lower()
        if authentication_type not in {"none", "bearer", "header"}:
            raise ConfigError("authentication.type must be none, bearer, or header")

        environment_variable = self._optional_string(data, "environment_variable")
        header_name = self._optional_string(data, "header_name")
        if authentication_type == "none":
            return AuthenticationConfig()

        if environment_variable is None:
            raise ConfigError("Authenticated endpoints need an environment_variable")

        if authentication_type == "header" and header_name is None:
            raise ConfigError("Header authentication needs a header_name")

        return AuthenticationConfig(
            type=authentication_type,
            environment_variable=environment_variable,
            header_name=header_name,
        )

    def _openai_base_url(self, data: dict[str, Any]) -> str:
        url = self._string(data, "openai_base_url").rstrip("/")
        if not url.endswith("/v1"):
            raise ConfigError("openai_base_url must end with /v1")

        return url

    def _validate_presets(
        self,
        presets: tuple[PresetConfig, ...],
        endpoints: tuple[EndpointConfig, ...],
    ) -> None:
        endpoint_models_by_id = {
            endpoint.id: {model.id for model in endpoint.models}
            for endpoint in endpoints
        }
        for preset in presets:
            for route in preset.routes:
                if route.endpoint_id is None:
                    if not any(
                        route.models
                        for endpoint in endpoints
                        if endpoint.id is None
                    ):
                        raise ConfigError(
                            f"Preset route in {preset.id} references an "
                            "unknown endpoint"
                        )

                    continue

                if route.endpoint_id not in endpoint_models_by_id:
                    raise ConfigError(
                        f"Preset route in {preset.id} references unknown endpoint "
                        f"{route.endpoint_id}"
                    )

                endpoint_model_ids = endpoint_models_by_id[route.endpoint_id]
                unknown = {
                    model.id
                    for model in route.models
                    if model.id not in endpoint_model_ids
                }
                if unknown:
                    raise ConfigError(
                        f"Preset route in {preset.id} references unknown model(s): "
                        f"{", ".join(sorted(unknown))}"
                    )

    def _validate_model_ids(
        self,
        endpoints: tuple[EndpointConfig, ...],
        presets: tuple[PresetConfig, ...],
    ) -> None:
        base_model_ids = {
            model.id for endpoint in endpoints for model in endpoint.models
        }
        if base_model_ids & {preset.id for preset in presets}:
            raise ConfigError("Preset ids cannot match base model ids")

    def _presets(self, items: list[Any]) -> tuple[PresetConfig, ...]:
        presets = tuple(self._preset(self._item_mapping(item)) for item in items)
        if not presets:
            raise ConfigError("presets cannot be empty")

        if len({preset.id for preset in presets}) != len(presets):
            raise ConfigError("Each preset id must be unique")

        return presets

    def _preset(self, data: dict[str, Any]) -> PresetConfig:
        return PresetConfig(
            id=self._string(data, "id").lower(),
            display_name=self._string(data, "display_name"),
            system_prompt=self._system_prompt(data, "system_prompt"),
            routes=tuple(
                self._preset_route(self._item_mapping(item))
                for item in self._sequence(data, "routes")
            ),
        )

    def _preset_route(self, data: dict[str, Any]) -> PresetRouteConfig:
        models = tuple(
            PresetModelConfig(
                id=self._string(self._item_mapping(item), "id"),
                priority=self._integer(self._item_mapping(item), "priority"),
            )
            for item in self._sequence(data, "models")
        )
        if not models:
            raise ConfigError("Preset routes need at least one model")

        return PresetRouteConfig(
            endpoint_id=self._optional_string(data, "endpoint_id"),
            priority=self._integer(data, "priority"),
            models=models,
        )

    def _models(self, items: list[Any]) -> tuple[ModelConfig, ...]:
        models = tuple(
            ModelConfig(
                id=self._string(self._item_mapping(item), "id"),
                display_name=self._string(self._item_mapping(item), "display_name"),
                context_length=self._integer(
                    self._item_mapping(item),
                    "context_length",
                ),
            )
            for item in items
        )
        if not models:
            raise ConfigError("Model inventories cannot be empty")

        return models

    def _mapping(self, data: dict[str, Any], key: str) -> dict[str, Any]:
        value = data.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"{key} must be a mapping")

        return value

    def _sequence(self, data: dict[str, Any], key: str) -> list[Any]:
        value = data.get(key)
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be a list")

        return value

    def _item_mapping(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ConfigError("Each entry must be a mapping")

        return value

    def _string(self, data: dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")

        return value.strip()

    def _optional_string(self, data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None

        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string when provided")

        return value.strip()

    def _system_prompt(self, data: dict[str, Any], key: str) -> str | None:
        value = data.get(key, "")
        if value is None:
            return None

        if not isinstance(value, str):
            raise ConfigError(f"{key} must be a string")

        return value.strip() or None

    def _integer(self, data: dict[str, Any], key: str) -> int:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{key} must be a positive integer")

        return value

    def _number(self, data: dict[str, Any], key: str) -> float:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{key} must be a positive number")

        return float(value)

    def _optional_number(self, data: dict[str, Any], key: str) -> float | None:
        value = data.get(key)
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{key} must be a positive number when provided")

        return float(value)