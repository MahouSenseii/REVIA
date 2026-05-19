# REVIA Input-to-Speech Deep Dive

This document describes the current runtime path in this checkout, from a user typing or speaking to Revia generating text, streaming partial speech, playing audio, and handling interruptions.

## Current runtime truth

- The primary user-to-speech path is the Python controller (`revia_controller_py`) talking to the Python core server (`revia_core_py`) over REST on port `8123` and WebSocket on port `8124`.
- The normal chat UI sends to `/api/chat`. It does not currently use `/api/agents/chat`.
- `/api/agents/chat` is an additive parallel-agent endpoint with its own orchestrator and interface router. It is important architecture, but it is not the default ChatPanel path.
- The C++ spine (`revia_core_cpp`) is additive behind feature flags and HTTP bridges. It is not the current main chat-to-speech path.
- `VoiceManager` is the controller-side owner for TTS. Qwen/Gradio is the primary backend when ready, with local fallback behavior.

## End-to-end diagram

```mermaid
flowchart TD
    User["User input\nText box, mic, or continuous VAD"] --> ControllerUI["Controller UI\nMainWindow + ChatPanel"]

    ControllerUI --> InputModes{"Input source"}
    InputModes --> ManualText["Manual text\nChatPanel._send"]
    InputModes --> PushToTalk["Push-to-talk STT\nAudioService.speech_recognized"]
    InputModes --> ContinuousVAD["Always-on VAD\nContinuousAudioPipeline"]

    PushToTalk --> ManualText
    ContinuousVAD --> BargeInCheck{"Revia speaking?"}
    BargeInCheck -->|no| ManualText
    BargeInCheck -->|yes| InterruptPath["Barge-in interrupt\nstop TTS + /api/interrupt"]

    ManualText --> BehaviorGate["ConversationBehaviorController\nreadiness, cooldown, state gates"]
    BehaviorGate -->|blocked| UIBlocked["UI note / no turn"]
    BehaviorGate -->|allowed| ClientSend["ControllerClient.send_chat\nThreadPoolExecutor max_workers=4"]

    ClientSend --> RESTChat["Core REST /api/chat\nFlask request thread"]
    RESTChat --> TurnStart["TurnManager.start_turn\nrequest_id + turn_id"]
    TurnStart --> PipelineThread["Daemon pipeline thread\nprocess_pipeline"]
    RESTChat --> ImmediateAck["Immediate JSON ack\nstatus=processing"]
    ImmediateAck --> ChatPanelAccepted["ChatPanel records request_id\nstarts Thinking timer"]

    PipelineThread --> RuntimeFSM["ConversationRuntime\nListening -> Thinking -> Speaking -> Cooldown"]
    PipelineThread --> Lane1["Lane 1: Perception\nparallel fanout"]
    Lane1 --> Emotion["EmotionNet / emotion context"]
    Lane1 --> Router["Router / intent / search routing hints"]
    Lane1 --> MemoryWrite["Short-term and person memory extraction"]

    Lane1 --> HumanFeel["HumanFeelLayer\nprosody + optional thinking pause"]
    HumanFeel --> SentencePause["chat_sentence thinking pause\noptional"]

    Lane1 --> Lane2["Lane 2: Cognition\nLLMBackend.generate_response"]
    Lane2 --> PromptBuild["Prompt assembly\nprofile, status, memory, emotion, vision, behavior, RL"]
    PromptBuild --> GenerateLock["LLM generate lock\nserializes model generation"]
    GenerateLock --> ModelRoute{"Backend route"}
    ModelRoute --> LocalModel["Local OpenAI-compatible server\nOllama, LM Studio, llama.cpp, koboldcpp, vLLM, TabbyAPI"]
    ModelRoute --> OnlineModel["Online OpenAI-compatible endpoint"]
    ModelRoute --> Stub["Stub response if no model ready"]

    LocalModel --> TokenStream["Streaming tokens"]
    OnlineModel --> TokenStream
    Stub --> CompleteOnly["Non-model complete payload"]

    TokenStream --> PipelineBroadcast["_pipeline_broadcast"]
    PipelineBroadcast --> WSChatToken["WebSocket chat_token"]
    PipelineBroadcast --> SentenceBuffer["Sentence buffer"]
    SentenceBuffer --> WSChatSentence["WebSocket chat_sentence\nsentence-sized speech chunks"]
    CompleteOnly --> WSComplete["WebSocket chat_complete"]

    TokenStream --> FinalText["Final assistant text"]
    FinalText --> ResponseFilter["filter_response + disclaimer checks"]
    ResponseFilter --> CommitHistory["Conversation history + assistant memory"]
    CommitHistory --> WSComplete
    CommitHistory --> Lane3["Lane 3: Expression\nafter output"]
    Lane3 --> AVS["AnswerValidationSystem score"]
    Lane3 --> RL["RL reward update"]

    WSChatToken --> ControllerClientWS["ControllerClient WebSocket"]
    WSChatSentence --> ControllerClientWS
    WSComplete --> ControllerClientWS
    SentencePause --> ControllerClientWS

    ControllerClientWS --> EventBus["Qt EventBus\nqueued signals on UI loop"]
    EventBus --> ChatPanelRender["ChatPanel renders streamed tokens"]
    EventBus --> TTSQueue["ChatPanel TTS queue\nordered sentence chunks"]

    TTSQueue --> TTSWorker["TTS worker / prefetch\nThreadPoolExecutor max_workers=3"]
    TTSWorker --> VoiceManager["VoiceManager\nactive profile + emotion modifiers"]
    VoiceManager --> QwenBackend["QwenTTSBackend\nGradio / local fallback\nsynthesis semaphore=3"]
    QwenBackend --> WavFiles["Generated WAV chunks"]
    WavFiles --> OrderedPlayback["Ordered playback\npreserve sentence order"]
    OrderedPlayback --> Speaker["Revia speaks"]

    OrderedPlayback --> VADNotify["ContinuousAudioPipeline.notify_revia_speaking"]
    VADNotify --> ContinuousVAD
    InterruptPath --> CoreInterrupt["Core /api/interrupt\nLLM interruption flag + interrupt_ack"]
    CoreInterrupt --> ControllerClientWS
```

## Streaming sequence

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

    U->>CP: type message or STT result
    CP->>CP: behavior/readiness gate
    CP->>CC: send_chat(text, image, vision)
    CC->>API: POST /api/chat
    API->>PP: start daemon pipeline thread
    API-->>CC: { status: processing, request_id }
    CC-->>CP: chat_request_accepted

    PP->>PP: perception fanout, memory, human-feel pass
    PP->>LLM: generate_response(...)
    loop streamed tokens
        LLM-->>PP: token
        PP-->>WS: chat_token
        PP-->>WS: chat_sentence when sentence boundary is reached
        WS-->>BUS: chat_token / chat_sentence
        BUS-->>CP: queued Qt signal
        CP-->>CP: render token, enqueue sentence
        CP->>VM: synthesize_to_wav(sentence)
        VM->>TTS: Gradio synthesis
        TTS-->>VM: wav path
        VM->>TTS: play_wav_sync(wav)
        TTS-->>SPK: audio playback
    end

    LLM-->>PP: final assistant text
    PP->>PP: filter, commit history, commit memory
    PP-->>WS: chat_complete
    WS-->>BUS: chat_complete
    BUS-->>CP: clear awaiting, finish timers, continue queued message if any
```

## Thread and executor map

```mermaid
flowchart LR
    QtMain["Qt main thread\nwidgets, signals, rendering"] --> QtSignals["EventBus queued signals"]
    QtMain --> WebSocketClient["QWebSocket callbacks"]
    QtMain --> RestPool["ControllerClient REST pool\nmax_workers=4"]

    MicThread["AudioService mic listener\nrevia-stt-listen"] --> STTWorkers["STT recognizer workers\nBoundedSemaphore(2)"]
    VADThread["ContinuousVAD thread\n30 ms frames"] --> BargeIn["interruption_detected signal"]

    FlaskThread["Core Flask request thread\n/api/chat"] --> PipelineThread2["Daemon pipeline thread\nprocess_pipeline"]
    PipelineThread2 --> LaneExecutor["ParallelPipeline lane executor"]
    PipelineThread2 --> FanoutExecutor["ParallelPipeline fanout executor\nemotion + router"]
    PipelineThread2 --> GenerateLock2["LLMBackend _generate_lock\none model generation at a time"]
    PipelineThread2 --> WSLoop["WebSocket asyncio loop\nbroadcast_json"]

    ChatPanelTTS["ChatPanel TTS prefetch worker"] --> TTSExec["TTS synthesis executor\nmax_workers=3"]
    TTSExec --> SynthSem["QwenTTSBackend semaphore\nup to 3 synthesis calls"]
    ChatPanelTTS --> OrderedAudio["Ordered playback\none spoken stream"]

    AgentsEndpoint["/api/agents/chat"] --> AgentsExec["AgentOrchestrator executor\nmax_workers=4"]
    InterfaceRouter["InterfaceRouter"] --> InterfaceExec["Interface dispatch executor\nparallel output interfaces"]
```

## Main components

### Controller UI

`MainWindow` builds the controller services and wires the chat surface, audio service, continuous VAD, voice tab, behavior controller, status manager, and runtime state sync.

`ChatPanel` owns the visible chat turn. It:

- accepts typed text and STT results;
- gates turns through `ConversationBehaviorController`;
- sends REST chat requests through `ControllerClient`;
- renders `chat_token` streaming output;
- converts `chat_sentence` chunks into a sentence-level TTS queue;
- tracks Thinking, Generating, TTS Gen, and Speaking timing;
- interrupts current output when the user barge-ins or sends another message.

`ControllerClient` keeps the UI responsive by using a small REST executor and a WebSocket push path. `/api/chat` returns quickly with `request_id`; the actual answer arrives over WebSocket as `chat_token`, `chat_sentence`, and `chat_complete`.

`EventBus` is the Qt signal bridge. It lets REST worker threads, WebSocket callbacks, and service threads safely hand updates back to the UI loop.

### Speech input and interruption

There are two speech-related paths:

- `AudioService` handles push-to-talk or explicit mic listening. It records audio, runs speech recognition in background work, and emits `speech_recognized`.
- `ContinuousAudioPipeline` is always-on VAD. It watches 30 ms frames and emits `interruption_detected` only when Revia is currently marked as speaking and the user's speech passes the grace and hold thresholds.

When barge-in fires, `ChatPanel` drains pending TTS chunks, stops backend playback if possible, posts `/api/interrupt`, and ignores stale streamed output for the interrupted request.

### Core `/api/chat` path

The core chat endpoint does not wait for generation. It:

1. Builds a response trigger and checks behavior/readiness gates.
2. Starts a `TurnManager` turn and creates a `request_id`.
3. Starts `process_pipeline` on a daemon thread.
4. Returns `{ status: "processing", request_id, turn_id }`.

`process_pipeline` is the main runtime path:

1. Moves the runtime state through thinking/generating/speaking/cooldown.
2. Runs Lane 1 perception fanout for emotion and routing hints.
3. Adds user memory and extracts person facts.
4. Adds Human Feel prosody and optional thinking-pause output.
5. Calls Lane 2 cognition through `LLMBackend.generate_response`.
6. Streams tokens over WebSocket as they arrive.
7. Buffers sentence chunks and emits `chat_sentence` for low-latency TTS.
8. Filters the final answer, commits conversation history, and saves assistant memory.
9. Broadcasts `chat_complete`.
10. Runs Lane 3 expression work after output, including AVS scoring and RL reward update.

### Prompt construction

`LLMBackend._build_messages` assembles the model input from:

- active character profile;
- runtime status;
- short-term and long-term memory context;
- emotion context;
- optional vision/image context;
- behavior parameters from the profile engine;
- reinforcement-learning behavior parameters;
- human-feel prosody instructions;
- recent conversation history.

The generation lock in `LLMBackend` serializes model generation. Revia can accept a new turn and interrupt a current one, but only one LLM generation is active inside this backend at a time.

### TTS and speaking

The core does not synthesize audio. It streams text and sentence chunks to the controller. The controller speaks.

`ChatPanel` receives `chat_sentence` and pushes each sentence into an ordered TTS queue. The TTS worker can pre-synthesize multiple upcoming chunks with `ThreadPoolExecutor(max_workers=3)`, while still playing the finished WAV chunks in sentence order.

`VoiceManager` owns the selected voice profile and emotion modifiers. `QwenTTSBackend` owns the Gradio/Qwen connection, readiness checks, endpoint selection, WAV generation, playback, and fallback speech. Its synthesis semaphore allows up to three synthesis calls at once, but playback remains ordered so Revia does not speak chunks out of sequence.

During playback, `ChatPanel` calls `ContinuousAudioPipeline.notify_revia_speaking(True)`. That is what lets the VAD distinguish normal listening from barge-in detection.

## Parallel functions and agents

### ParallelPipeline in `/api/chat`

`ParallelPipeline` provides named lanes:

- `perception`: emotion, routing, memory-adjacent signal gathering;
- `cognition`: LLM generation;
- `expression`: answer validation, reward updates, and output-adjacent work.

The current `/api/chat` pipeline uses real parallel fanout inside Lane 1 for independent perception jobs. It then waits for that result before moving into cognition, because the prompt should include emotion, memory, routing, and human-feel context.

Cognition is submitted through the lane executor, but the pipeline waits on its future while streaming tokens. This means LLM decode is asynchronous relative to the REST request and UI, but the turn logic itself still waits for the model result before final commit.

Expression work runs after `chat_complete` is prepared, so scoring and reward updates do not block the user from seeing or hearing the answer.

### `/api/agents/chat` additive endpoint

The parallel-agent system is separate from the normal ChatPanel path.

```mermaid
flowchart TD
    AgentsRequest["POST /api/agents/chat"] --> AgentContext["AgentContext\nuser text, session, metadata"]
    AgentContext --> Orchestrator["AgentOrchestrator.run_turn"]

    Orchestrator --> PreAgents["Phase 1: pre-agents in parallel"]
    PreAgents --> MemoryAgent["MemoryAgent\nshort/long/episodic facts"]
    PreAgents --> EmotionAgent["EmotionAgent\nemotion signal"]
    PreAgents --> IntentAgent["IntentAgent\nintent and route"]
    PreAgents --> VoiceStyleAgent["VoiceStyleAgent\nstyle guidance"]
    PreAgents --> ToolUseAgent["ToolUseAgent\nSkillRegistry dispatch"]
    PreAgents --> VisionAgent["VisionAgent\nvision metadata"]
    PreAgents --> ReasoningAgent["ReasoningAgent\nmodel-router reason_chat"]
    PreAgents --> HardwareAgent["HardwareAgent\nruntime hardware signal"]

    MemoryAgent --> PostAgents["Phase 2: post-agents in parallel"]
    EmotionAgent --> PostAgents
    IntentAgent --> PostAgents
    VoiceStyleAgent --> PostAgents
    ToolUseAgent --> PostAgents
    VisionAgent --> PostAgents
    ReasoningAgent --> PostAgents
    HardwareAgent --> PostAgents

    PostAgents --> CriticAgent["CriticAgent"]
    PostAgents --> ReflectionAgent["ReflectionAgent"]
    CriticAgent --> FinalBuilder["FinalResponseBuilder\nMany agents think, one Revia speaks"]
    ReflectionAgent --> FinalBuilder
    FinalBuilder --> QualityGate["QualityGate"]
    QualityGate -->|accepted| InterfaceRouter["InterfaceRouter"]
    QualityGate -->|rejected and retries left| Regen["Regenerate reasoning\nthen post-agents again"]
    Regen --> FinalBuilder

    InterfaceRouter --> AuditLog["Audit/log interface"]
    InterfaceRouter --> TextChat["Text chat interface"]
    InterfaceRouter --> VoiceInterface["Voice interface\ncurrently disabled unless enabled"]
    InterfaceRouter --> VisionInterface["Vision interface\ncurrently disabled unless enabled"]
    InterfaceRouter --> NotifyInterface["Notification interface\ncurrently disabled unless enabled"]
    InterfaceRouter --> AgentsResponse["Structured JSON response"]
```

The orchestrator runs pre-agents concurrently with a four-worker executor and per-agent timeouts. It then runs post-agents concurrently, builds one final response, applies a quality gate, and may do a limited reasoning-only regeneration loop.

Important current limitation: this endpoint returns structured JSON and dispatches to configured interfaces. It does not drive the current ChatPanel streaming token/TTS path, and it does not mutate the normal `conversation_manager` history path.

## What happens on interrupt

```mermaid
sequenceDiagram
    participant U as User
    participant VAD as ContinuousAudioPipeline
    participant CP as ChatPanel
    participant CC as ControllerClient
    participant API as Core /api/interrupt
    participant LLM as LLMBackend
    participant WS as WebSocket
    participant TTS as VoiceManager/TTS

    CP->>VAD: notify_revia_speaking(true)
    U->>VAD: starts talking while Revia speaks
    VAD->>VAD: grace window + hold-frame checks
    VAD-->>CP: interruption_detected
    CP->>TTS: stop current playback
    CP->>CP: drain queued TTS, mark request ignored
    CP->>CC: send_interrupt()
    CC->>API: POST /api/interrupt
    API->>LLM: request_interrupt()
    API->>API: finish active turn as interrupted
    API-->>WS: chat_complete interrupted + interrupt_ack
    WS-->>CP: ignore stale output, clear speaking state
```

## Notes and caveats

- The normal chat path is streaming and sentence-TTS capable; the additive agent endpoint is not currently the path the GUI uses for spoken chat.
- `LLMBackend` serializes generations with `_generate_lock`. This is deliberate protection around backend calls, but it means true multi-turn parallel model generation is not active in the normal path.
- TTS synthesis is parallelized, but playback is intentionally ordered.
- `ToolUseAgent` can dispatch skills inside the additive agent endpoint, but in the current orchestrator shape it should be verified before assuming its result is injected into the same prompt used by `ReasoningAgent`.
- The Qwen/Gradio TTS backend discovers usable endpoints from the live server. Runtime checks such as `/gradio_api/info` matter before assuming a specific TTS route exists.
- The C++ EventBus/StateManager bridge is useful migration infrastructure, but the current user-input-to-Revia-speaking path is still Python controller plus Python core.

