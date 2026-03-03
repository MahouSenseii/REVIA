"""Voice profile data model and persistence."""
import json
import enum
import sys
from pathlib import Path
from datetime import datetime

# Voices root: <repo>/voices/  (two parents above this file's package dir)
_VOICES_ROOT = Path(__file__).resolve().parents[2] / "voices"


def _resolve_wav_path(stored_path: str) -> str:
    """Return a valid local path for *stored_path*, handling cross-platform cases.

    If the stored path exists as-is, it is returned unchanged.
    Otherwise the filename is extracted and searched inside _VOICES_ROOT so that
    Windows paths (e.g. C:/Users/USER/REVIA/voices/audio.wav) resolve correctly
    when running on Linux/macOS.
    """
    if not stored_path:
        return ""
    p = Path(stored_path)
    if p.exists():
        return stored_path
    # Extract filename, normalising backslashes first
    filename = Path(stored_path.replace("\\", "/")).name
    if not filename:
        return stored_path
    # Search voices root directly
    if _VOICES_ROOT.exists():
        candidate = _VOICES_ROOT / filename
        if candidate.exists():
            return str(candidate)
        # Search one level deep (profile subdirectories)
        for subdir in _VOICES_ROOT.iterdir():
            if subdir.is_dir():
                candidate = subdir / filename
                if candidate.exists():
                    return str(candidate)
    return stored_path


class VoiceMode(enum.Enum):
    DESIGN = "design"       # Voice Design: natural language description -> WAV
    CLONE = "clone"         # Voice Clone (Base): ref audio + ref text -> WAV
    CUSTOM = "custom"       # CustomVoice: predefined speaker + style -> WAV


class VoiceProfile:
    """Represents a saved voice identity. Each profile stores the settings used
    to generate the voice and the path to the generated WAV file.
    The WAV can be reused as a clone reference for consistent TTS."""

    def __init__(self, name="Default", mode=VoiceMode.CUSTOM):
        self.name = name
        self.mode = mode
        self.language = "Auto"
        self.model_size = "1.7B"

        # Voice Design fields
        self.voice_description = ""

        # Voice Clone fields
        self.reference_audio = ""   # path to reference WAV for cloning
        self.clone_ref_text = ""    # transcript of the reference audio
        self.x_vector_only = False

        # CustomVoice fields
        self.speaker_id = "Ryan"
        self.style_instruction = ""

        # Generated output
        self.generated_wav = ""     # path to the generated WAV file (reusable)

        # Metadata
        self.created = datetime.now().isoformat()
        self.is_default = False

        # Emotion modulation for speed/pitch (used in pyttsx3 fallback)
        self.base_speed = 1.0
        self.base_pitch = 1.0
        self.emotion_map = {
            "happy": {"speed": 1.08, "pitch": 1.05},
            "sad": {"speed": 0.92, "pitch": 0.96},
            "angry": {"speed": 1.05, "pitch": 0.94},
            "alert": {"speed": 1.10, "pitch": 0.98},
            "calm": {"speed": 0.95, "pitch": 1.0},
            "neutral": {"speed": 1.0, "pitch": 1.0},
        }

    def to_dict(self):
        return {
            "name": self.name,
            "mode": self.mode.value,
            "language": self.language,
            "modelSize": self.model_size,
            "voiceDescription": self.voice_description,
            "referenceAudio": self.reference_audio,
            "cloneRefText": self.clone_ref_text,
            "xVectorOnly": self.x_vector_only,
            "speakerId": self.speaker_id,
            "styleInstruction": self.style_instruction,
            "generatedWav": self.generated_wav,
            "baseSpeed": self.base_speed,
            "basePitch": self.base_pitch,
            "created": self.created,
            "isDefault": self.is_default,
            "emotionMap": self.emotion_map,
        }

    @classmethod
    def from_dict(cls, data):
        p = cls()
        p.name = data.get("name", "Default")
        p.mode = VoiceMode(data.get("mode", "custom"))
        p.language = data.get("language", "Auto")
        p.model_size = data.get("modelSize", "1.7B")
        # Backward compat: old profiles used "description"/"stylePrompt"
        p.voice_description = (
            data.get("voiceDescription", "")
            or data.get("description", "")
            or data.get("stylePrompt", "")
        )
        p.reference_audio = data.get("referenceAudio", "")
        p.clone_ref_text = data.get("cloneRefText", "")
        p.x_vector_only = data.get("xVectorOnly", False)
        p.speaker_id = data.get("speakerId", "Ryan")
        p.style_instruction = (
            data.get("styleInstruction", "")
            or data.get("stylePrompt", "")
        )
        p.generated_wav = _resolve_wav_path(data.get("generatedWav", ""))
        p.base_speed = data.get("baseSpeed", 1.0)
        p.base_pitch = data.get("basePitch", 1.0)
        p.created = data.get("created", datetime.now().isoformat())
        p.is_default = data.get("isDefault", False)
        p.emotion_map = data.get("emotionMap", p.emotion_map)
        return p

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "voice.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory):
        path = Path(directory) / "voice.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def get_modulated(self, emotion_label="neutral"):
        mods = self.emotion_map.get(emotion_label, {})
        return {
            "speed": self.base_speed * mods.get("speed", 1.0),
            "pitch": self.base_pitch * mods.get("pitch", 1.0),
        }

    def has_wav(self):
        resolved = _resolve_wav_path(self.generated_wav)
        return bool(resolved) and Path(resolved).exists()
