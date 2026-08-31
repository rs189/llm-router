from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import ConfigError, ConfigLoader


def run() -> None:
    arguments = _parse_arguments()
    try:
        config = ConfigLoader().load(arguments.config)
    except ConfigError as exception:
        raise SystemExit(f"Configuration error: {exception}") from exception

    uvicorn.run(create_app(config), host=config.server.host, port=config.server.port)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM router")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Path to the router YAML configuration file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run()