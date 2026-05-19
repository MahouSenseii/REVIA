# REVIA How It Works Diagram

Snapshot from the current checkout on 2026-05-14.

## Current Runtime Truth

- `revia_controller_py/` is the desktop controller UI.
- `revia_core_py/` is the main runtime backend.
- REST defaults to `http://127.0.0.1:8123`.
- WebSocket defaults to `ws://127.0.0.1:8124`.
- The normal spoken chat path uses `/api/chat`.
- `/api/agents/chat` is an additive parallel-agent endpoint. It is not the default `ChatPanel` spoken path.
- `revia_core_cpp/` is present as an experimental secondary core, but visible replies still run through the Python path.

## Main System Map

```mermaid
flowchart TD
    User["User<br/>text, mic, camera context"] --> Chat["ChatPanel<br/>chat UI, turn state, TTS queue"]

    subgraph Controller["Controller desktop: revia_controller_py"]
        Main["main.py<br/>EventBus, ControllerClient, MainWindow"]
        Window["MainWindow<br/>services, tabs, startup wiring"]
        Chat
        Client["ControllerClient<br/>REST executor + QWebSocket"]
        EventBus["EventBus<br/>Qt signal bridge"]
        Audio["AudioService<br/>push-to-talk STT"]
        VAD["ContinuousAudioPipeline<br/>always-on VAD + barge-in"]
        Voice["VoiceManager<br/>voice profile + backend owner"]
        TTS["QwenTTSBackend<br/>Qwen/Gradio or pyttsx3 fallback"]
    end

    subgraph Core["Python core: revia_core_py"]
        Flask["Flask REST<br/>/api/status, /api/chat, /api/interrupt"]
        WS["WebSocket broadcaster<br/>status_update, chat_token, chat_sentence, chat_complete"]
        Turn["TurnManager + FSM<br/>request_id, turn_id, Idle/Thinking/Speaking/Cooldown"]
        Pipeline["process_pipeline<br/>default spoken runtime"]
        Perception["Lane 1: Perception<br/>EmotionNet + router + user memory"]
        Prompt["Prompt assembly<br/>persona, memory, emotion, vision, behavior"]
        LLM["LLMBackend.generate_response<br/>local, online, or stub route"]
        Memory["MemoryStore<br/>Redis or local JSONL fallback"]
        Quality["Lane 3: Expression<br/>AVS scoring + RL reward"]
        Hardware["Hardware runtime<br/>profiler, scheduler, provider registry"]
        Agents["/api/agents/chat<br/>AgentOrchestrator + InterfaceRouter"]
    end

    Main --> Window
    Window --> Chat
    Window --> Audio
    Window --> VAD
    Window --> Voice
    Window --> Client

    Audio -->|"speech_recognized"| Chat
    VAD -->|"interruption_detected"| Chat

    Chat -->|"send_chat"| Client
    Client -->|"POST /api/chat"| Flask
    Flask --> Turn
    Turn -->|"daemon thread"| Pipeline

    Pipeline --> Perception
    Perception --> Memory
    Perception --> Prompt
    Prompt --> Hardware
    Hardware --> LLM
    Prompt --> LLM

    LLM -->|"stream tokens"| WS
    Pipeline -->|"sentence chunks"| WS
    Pipeline -->|"final payload"| WS
    Pipeline --> Quality
    Pipeline --> Memory

    WS --> Client
    Client --> EventBus
    EventBus -->|"chat_token and chat_complete"| Chat
    EventBus -->|"chat_sentence"| Chat

    Chat -->|"ordered sentence queue"| Voice
    Voice --> TTS
    TTS -->|"WAV chunks"| Chat
    Chat -->|"ordered playback"| Speaker["Speakers<br/>Revia speaks"]
    Chat -->|"notify_revia_speaking"| VAD

    Chat -->|"barge-in or timeout"| Client
    Client -->|"POST /api/interrupt"| Flask
    Flask -->|"interrupt_ack"| WS

    External["Optional API caller"] --> Agents
    Agents -. "separate from default spoken chat" .-> Memory
    Agents -. "optional output dispatch" .-> WS
```

## Normal Spoken Turn

```mermaid
sequenceDiagram
    participant U as User
    participant CP as ChatPanel
    participant CC as ControllerClient
    participant API as Core /api/chat
    participant PP as process_pipeline
    participant LLM as LLMBackend
    participant WS as Core WebSocket
    participant BUS as EventBus
    participant VM as VoiceManager
    participant TTS as QwenTTSBackend
    participant SPK as Speakers

    U->>CP: Type text or speak through STT
    CP->>CP: Gate with ConversationBehaviorController
    CP->>CC: send_chat(text, image, vision_context)
    CC->>API: POST /api/chat
    API->>API: Build TriggerRequest and readiness decision
    API->>PP: Start _run_pipeline_safe on daemon thread
    API-->>CC: {status: processing, request_id, turn_id}
    CC-->>CP: chat_request_accepted

    PP->>PP: Set runtime state to Thinking
    PP->>PP: Run perception fanout, memory update, Human Feel hints
    PP->>LLM: generate_response(...)
    loop streaming output
        LLM-->>PP: token
        PP-->>WS: chat_token
        PP-->>WS: chat_sentence when sentence boundary is ready
        WS-->>CC: WebSocket message
        CC-->>BUS: Qt signal
        BUS-->>CP: token or sentence
        CP->>VM: synthesize_to_wav(sentence, emotion)
        VM->>TTS: Qwen/Gradio or pyttsx3 fallback
        TTS-->>VM: WAV path or fallback result
        VM->>SPK: play_wav_sync(...)
    end

    PP->>PP: Filter final answer, commit history, update memory
    PP-->>WS: chat_complete
    WS-->>CC: WebSocket message
    CC-->>BUS: chat_complete_payload
    BUS-->>CP: Finalize visible reply and timers
    PP->>PP: Background AVS score and RL reward
```

## Interrupt And Barge-In

```mermaid
flowchart LR
    Speaking["TTS playback active"] --> Notify["ChatPanel marks Revia speaking"]
    Notify --> VAD["ContinuousAudioPipeline<br/>30 ms frames"]
    VAD --> Check{"User speech during TTS?"}
    Check -->|"no"| Continue["Keep listening"]
    Check -->|"yes, after grace/hold"| Barge["interruption_detected"]
    Barge --> Stop["ChatPanel stops backend playback<br/>drains TTS queue"]
    Stop --> Ignore["Current request id is ignored<br/>late tokens are dropped"]
    Stop --> API["POST /api/interrupt"]
    API --> Core["Core sets interrupt flag<br/>broadcasts interrupt_ack"]
    Core --> Idle["Runtime recovers to Idle/ready"]
```

## Parallel-Agent Endpoint

```mermaid
flowchart TD
    Caller["External caller or test<br/>POST /api/agents/chat"] --> Context["AgentContext<br/>text, turn_id, metadata, cancel token"]
    Context --> PreAgents["Parallel pre-agents"]

    subgraph PreAgents["Parallel pre-agents"]
        MemoryAgent["MemoryAgent"]
        EmotionAgent["EmotionAgent"]
        IntentAgent["IntentAgent"]
        VoiceStyleAgent["VoiceStyleAgent"]
        ToolUseAgent["ToolUseAgent"]
        VisionAgent["VisionAgent"]
        ReasoningAgent["ReasoningAgent"]
        HardwareAgent["HardwareAgent"]
    end

    PreAgents --> Candidate["Candidate response + metadata"]
    Candidate --> PostAgents["Post-agents<br/>CriticAgent + ReflectionAgent"]
    PostAgents --> Gate["QualityGate<br/>optional regen of ReasoningAgent"]
    Gate --> Final["FinalResponse"]
    Final --> Interfaces["InterfaceRouter<br/>log, text, voice, vision, notification"]
    Final --> Episode["Episode store + goal detection"]
    Interfaces --> Response["Structured JSON response"]
    Episode --> Response
```

## Ownership Summary

| Area | Owner |
| --- | --- |
| Visible desktop app | `revia_controller_py/gui/main_window.py` |
| Chat turn UI, streaming render, TTS sentence queue, interrupts | `revia_controller_py/gui/widgets/chat_panel.py` |
| REST and WebSocket client bridge | `revia_controller_py/app/controller_client.py` |
| Push-to-talk STT | `revia_controller_py/app/audio_service.py` |
| Continuous VAD and barge-in detection | `revia_controller_py/app/continuous_audio.py` |
| Voice profile and TTS backend ownership | `revia_controller_py/app/voice_manager.py` |
| Qwen/Gradio and pyttsx3 speech backend | `revia_controller_py/app/tts_backend.py` |
| REST, WebSocket, normal chat endpoint, runtime orchestration | `revia_core_py/core_server.py` |
| Conversation state and behavior gates | `revia_core_py/conversation_runtime.py` |
| Turn/request lifecycle | `revia_core_py/runtime_models.py` |
| Persona normalization and prompt assembly | `revia_core_py/persona_manager.py`, `revia_core_py/profile_engine.py`, `revia_core_py/prompt_assembly.py` |
| Hardware detection, scheduling, provider fallback | `revia_core_py/runtime/` |
| Optional parallel agents and output interfaces | `revia_core_py/agents/`, `revia_core_py/interfaces/` |

## Key Takeaway

The default experience is not a single blocking call. The controller posts a chat request, the core returns an immediate request id, generation continues in a background pipeline, and tokens/sentence chunks stream back over WebSocket. The UI speaks sentence chunks as they arrive, while final filtering, memory commits, quality scoring, and reward updates happen around or after the visible response.
