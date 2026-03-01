from dataclasses import dataclass


@dataclass
class SystemMetrics:
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    ram_mb: float = 0.0
    vram_mb: float = 0.0
    health: str = "Offline"
    model: str = "---"
    backend: str = "---"
    device: str = "---"


@dataclass
class LLMMetrics:
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    context_length: int = 0


@dataclass
class EmotionData:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    label: str = "---"
    confidence: float = 0.0
    inference_ms: float = 0.0


@dataclass
class RouterData:
    mode: str = "---"
    confidence: float = 0.0
    suggested_tool: str = ""
    rag_enable: bool = False
    inference_ms: float = 0.0


@dataclass
class PipelineSpan:
    stage: str = ""
    duration_ms: float = 0.0
    device: str = "CPU"
    error: str = ""


@dataclass
class ProfileData:
    character_name: str = "Revia"
    persona: str = ""
    traits: str = ""
    voice_path: str = ""
    voice_tone: str = "Warm"
    language: str = "English"
    response_style: str = "Conversational"
    verbosity: str = "Normal"
    fallback_msg: str = (
        "Uh... something's wrong. Someone tell my operator he messed up."
    )
    greeting: str = "Hello! I'm Revia, your neural assistant."
    character_prompt: str = ""
