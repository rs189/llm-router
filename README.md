# llm-router

### An OpenAI-compatible LLM router.

llm-router is an OpenAI-compatible routing server for LLMs written in Python. It serves a single `/v1/chat/completions` endpoint. Routing is driven by presets and per-route model priorities in a `config.yml` file.

## Requirements

##### Core dependencies:

- [FastAPI](https://github.com/fastapi/fastapi)
- [httpx](https://github.com/encode/httpx)
- [PyYAML](https://github.com/yaml/pyyaml)
- [uvicorn](https://github.com/encode/uvicorn)

## Usage

```bash
# Start the router
llm-router

# List configured presets and base models
curl http://localhost:5000/v1/models

# Route a completion through the auto preset
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'

# Read the last routed model
curl http://localhost:5000/last_routed_model
```

## Installation

Install required packages: 

```bash
python -m pip install -e .
```

Export endpoint credentials: 
```bash
export NIM_API_KEY="..."
```

## License
llm-router is licensed under the [MIT License](LICENSE).