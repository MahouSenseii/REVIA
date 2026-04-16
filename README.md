# REVIA

Local AI assistant with a PySide6 controller UI and a Python core server that handles chat, memory, telemetry, integrations, and live runtime state.

## Primary Runtime

REVIA now treats the Python core as the main runtime:

- `revia_core_py/` is the primary backend
- `revia_controller_py/` is the main desktop controller
- `revia_core_cpp/` remains in the repo as an experimental/secondary core

## Memory

Persistent memory is Redis-first with automatic local JSONL fallback:

- Primary backend: Redis via Docker
- Fallback backend: `revia_core_py/data/memory_<profile>.jsonl`
- Profile-aware memory switching is handled by the Python core

## Quick Start

### 1. Install base dependencies

```powershell
pip install -r requirements.txt
```

Optional extras:

```powershell
pip install -r requirements-optional.txt
pip install -r requirements-sing.txt
```

### 2. Start Redis memory backend

```powershell
docker compose up -d revia-redis
```

Or start the full Docker stack:

```powershell
docker compose up -d
```

### 3. Run the Python core

```powershell
cd revia_core_py
python core_server.py
```

REST defaults to `http://127.0.0.1:8123` and WebSocket defaults to `ws://127.0.0.1:8124`.

### 4. Run the controller

```powershell
cd revia_controller_py
python main.py
```

The controller auto-start path in the UI also launches the Python core directly.

## Environment

Copy `.env.example` into `.env` and set any values you need:

- `REDIS_HOST`
- `REDIS_PORT`
- `REVIA_REST_PORT`
- `REVIA_WS_PORT`
- `REVIA_DISCORD_BOT_TOKEN`
- `REVIA_TWITCH_OAUTH_TOKEN`

Integration secrets are env-aware and should not be committed into source-controlled config files.

## Optional Features

- `requirements-optional.txt`
  Includes `depthai` and `duckduckgo-search`
- `requirements-sing.txt`
  Includes sing-mode dependencies such as `demucs`, `openai-whisper`, `librosa`, and `scipy`

## Main Interfaces

- REST: `/api/status`, `/api/chat`, `/api/profile`, `/api/memory/*`, `/api/integrations/*`
- WebSocket events: `telemetry_update`, `status_update`, `chat_token`, `chat_complete`

## Notes

- The controller now centralizes most tab HTTP work through a shared async client layer to reduce duplicated UI networking code.
- Redis-backed memory status is surfaced directly in the Memory and System tabs.
- The C++ core is still present, but the Python core is the recommended runtime if you want the most complete feature set and persistent memory behavior.
