# REVIA — Neural Assistant Controller

Hybrid C++ core + Python (PySide6) controller with futuristic sci-fi UI, neural network modules, plugin system, telemetry, and safe learning/adaptation.

## Architecture

```
revia_core_cpp/    — C++20 real-time pipeline, REST + WebSocket server
revia_controller_py/ — PySide6 GUI controller, connects to core via HTTP/WS
```

## Quick Start (Windows)

### 1. Build C++ Core

Requirements: CMake 3.20+, C++20 compiler (MSVC 2022 recommended), Git.

```powershell
cd revia_core_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run the core:
```powershell
.\build\Release\revia_core.exe
# REST on 127.0.0.1:8123, WebSocket on 127.0.0.1:8124
```

### 2. Run Python Controller

Requirements: Python 3.10+

```powershell
cd revia_controller_py
pip install -r requirements.txt
python main.py
```

The controller will show **Health: Offline** if the C++ core is not running, and **Health: Online** once connected. The GUI is fully functional either way — it gracefully handles missing core.

## Features

- **Dark + Light themes** — toggle in System tab
- **Neural Modules** — EmotionNet (VAD emotion detection) + RouterClassifier (intent routing)
- **Plugin System** — STT/TTS/Vision/Memory/LLM/Tools with enable/disable
- **Pipeline Telemetry** — per-stage latency tracking with live UI updates
- **Batched Listening** — micro-batch audio frames with partial STT and early routing
- **Safe Learning** — behavior tuning, memory-based RAG, routing feedback export (no autonomous retraining)
- **8 config tabs** — Profile, Model, Memory, Voice, Vision, Filters, Logs, System

## Pipeline Stages (Timed)

| Stage | Description |
|---|---|
| input_capture | Receive user input |
| stt_batch_collect | Micro-batch audio frames |
| stt_decode_partial | Partial STT decode |
| router_classify | RouterClassifier inference |
| emotion_infer | EmotionNet inference |
| rag_retrieve | Memory/RAG retrieval |
| prompt_assemble | Build prompt with context |
| llm_generate_total | LLM token generation |
| tts_synthesize | Text-to-speech synthesis |
| memory_write | Store conversation to memory |

## REST API

| Endpoint | Method | Description |
|---|---|---|
| /api/status | GET | System state + health |
| /api/telemetry | GET | Full telemetry snapshot |
| /api/plugins | GET | List all plugins |
| /api/plugins/{name}/enable | POST | Enable plugin |
| /api/plugins/{name}/disable | POST | Disable plugin |
| /api/neural | GET | Neural module status |
| /api/neural/{name}/enable | POST | Enable neural module |
| /api/neural/{name}/disable | POST | Disable neural module |
| /api/chat | POST | Send chat message (JSON: {"text": "..."}) |
| /api/profile | GET/POST | Get/save profile |

## WebSocket Events (port 8124)

- `telemetry_update` — periodic metrics snapshot
- `status_update` — pipeline state changes
- `chat_token` — streaming token from LLM
- `chat_complete` — full response complete
