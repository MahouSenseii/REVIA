"""Manages saved voice profiles on disk."""
import json
import shutil
from pathlib import Path
from .voice_profile import VoiceProfile, VoiceMode


VOICES_DIR = Path(__file__).resolve().parents[1] / ".." / "voices"


class VoiceLibrary:
    """Persistent voice library stored in /voices/ directory."""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else VOICES_DIR.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.base_dir / "manifest.json"
        self._profiles = {}  # name -> VoiceProfile
        self._default_name = None
        self.load_all()

    def load_all(self):
        """Scan voices directory and load all profiles."""
        self._profiles.clear()
        self._default_name = None
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir():
                p = VoiceProfile.load(d)
                if p:
                    self._profiles[p.name] = p
                    if p.is_default:
                        self._default_name = p.name
        self._save_manifest()

    def list_names(self):
        return list(self._profiles.keys())

    def get(self, name):
        return self._profiles.get(name)

    def get_default(self):
        if self._default_name and self._default_name in self._profiles:
            return self._profiles[self._default_name]
        if self._profiles:
            return next(iter(self._profiles.values()))
        return self._make_fallback()

    def set_default(self, name):
        if name is None or name not in self._profiles:
            return False
        # Snapshot to avoid RuntimeError if another thread mutates _profiles
        # while we are iterating and calling p.save() (I/O can yield the GIL).
        for n, p in list(self._profiles.items()):
            if p is None:
                continue
            p.is_default = (n == name)
            p.save(self.base_dir / self._safe_name(n))
        self._default_name = name
        self._save_manifest()
        return True

    def save_profile(self, profile):
        """Save or overwrite a voice profile."""
        safe = self._safe_name(profile.name)
        voice_dir = self.base_dir / safe
        profile.save(voice_dir)
        self._profiles[profile.name] = profile
        self._save_manifest()
        return voice_dir

    def rename(self, old_name, new_name):
        if old_name not in self._profiles or not new_name.strip():
            return False
        p = self._profiles.pop(old_name)
        old_dir = self.base_dir / self._safe_name(old_name)
        new_dir = self.base_dir / self._safe_name(new_name)
        if old_dir.exists():
            old_dir.rename(new_dir)
        p.name = new_name
        p.save(new_dir)
        self._profiles[new_name] = p
        if self._default_name == old_name:
            self._default_name = new_name
        self._save_manifest()
        return True

    def delete(self, name):
        if name not in self._profiles:
            return False
        self._profiles.pop(name)
        voice_dir = self.base_dir / self._safe_name(name)
        if voice_dir.exists():
            shutil.rmtree(voice_dir, ignore_errors=True)
        if self._default_name == name:
            self._default_name = None
        self._save_manifest()
        return True

    def export_profile(self, name, dest_path):
        """Export a voice profile directory as a zip or copy."""
        p = self._profiles.get(name)
        if not p:
            return False
        src = self.base_dir / self._safe_name(name)
        if src.exists():
            shutil.copytree(src, dest_path, dirs_exist_ok=True)
            return True
        return False

    def import_profile(self, src_dir):
        """Import a voice profile from a directory."""
        p = VoiceProfile.load(src_dir)
        if p:
            self.save_profile(p)
            src = Path(src_dir)
            dest = self.base_dir / self._safe_name(p.name)
            for f in src.iterdir():
                if f.name != "voice.json" and f.is_file():
                    shutil.copy2(f, dest / f.name)
            return p
        return None

    def get_voice_dir(self, name):
        return self.base_dir / self._safe_name(name)

    def open_folder(self):
        """Open the voices folder in the system file explorer."""
        import subprocess, sys
        path = str(self.base_dir)
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                # Linux / BSD — xdg-open is the standard cross-DE launcher
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("[VoiceLibrary] Could not open folder: %s", exc)

    def _safe_name(self, name):
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
        if not safe or not any(c.isalnum() for c in safe):
            raise ValueError("Voice name must contain at least one letter or number")
        # Ensure the resolved path is within base_dir (path traversal protection)
        resolved = (self.base_dir / safe).resolve()
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Path traversal attempt detected: {name}")
        return safe

    def _make_fallback(self):
        p = VoiceProfile("Fallback", VoiceMode.CUSTOM)
        p.style_instruction = "Neutral clear voice"
        p.is_default = True
        return p

    def _save_manifest(self):
        manifest = {
            "default": self._default_name,
            "voices": [
                {
                    "name": p.name,
                    "mode": p.mode.value,
                    "language": p.language,
                    "created": p.created,
                    "modelSize": p.model_size,
                    "isDefault": p.is_default,
                }
                for p in self._profiles.values()
            ],
        }
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
