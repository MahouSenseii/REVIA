"""REVIA backend UI theme ownership.

ThemeManager is the single owner for:
    * built-in and custom theme definitions
    * active theme selection
    * token generation
    * persistence
    * readability validation

Widgets should consume the stylesheet generated from tokens instead of owning
hardcoded colors.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

try:
    from app.style_composer import StyleComposer
except ImportError:
    from .style_composer import StyleComposer


THEME_TOKENS = (
    "PrimaryBackground",
    "SecondaryBackground",
    "Surface",
    "SurfaceAlt",
    "PrimaryText",
    "SecondaryText",
    "Accent",
    "AccentHover",
    "AccentActive",
    "Border",
    "ButtonPrimary",
    "ButtonSecondary",
    "Success",
    "Warning",
    "Error",
    "Info",
    "Disabled",
)


@dataclass(frozen=True)
class ThemeDefinition:
    ThemeId: str
    DisplayName: str
    IsBuiltIn: bool
    PrimaryBackground: str
    SecondaryBackground: str
    Surface: str
    SurfaceAlt: str
    PrimaryText: str
    SecondaryText: str
    Accent: str
    AccentHover: str
    AccentActive: str
    Border: str
    ButtonPrimary: str
    ButtonSecondary: str
    Success: str
    Warning: str
    Error: str
    Info: str
    Disabled: str

    def tokens(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in THEME_TOKENS}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping) -> "ThemeDefinition":
        data = dict(raw)
        for key in ("ThemeId", "DisplayName"):
            if not str(data.get(key, "")).strip():
                raise ValueError(f"theme missing {key}")
        for token in THEME_TOKENS:
            if token not in data:
                raise ValueError(f"theme missing token {token}")
        data["IsBuiltIn"] = bool(data.get("IsBuiltIn", False))
        return cls(**{field: data[field] for field in cls.__dataclass_fields__})


class ThemeManager:
    def __init__(self, app, storage_path: str | Path | None = None):
        self.app = app
        self._repo_root = Path(__file__).resolve().parents[2]
        self._storage_path = Path(storage_path) if storage_path else (
            self._repo_root / "data" / "themes.json"
        )
        self._composer = StyleComposer()
        self._themes: dict[str, ThemeDefinition] = {}
        self._active_theme_id = "dark"
        self._last_good_theme_id = "dark"

        self._load_builtin_themes()
        self._load_custom_themes()

        # Compatibility with old callers.
        self.current_theme = self._active_theme_id

    def available_themes(self) -> list[ThemeDefinition]:
        return sorted(self._themes.values(), key=lambda theme: theme.DisplayName.lower())

    def get_theme(self, theme_id: str) -> ThemeDefinition:
        key = self._normalize_theme_id(theme_id)
        if key not in self._themes:
            raise ValueError(f"theme not found: {theme_id}")
        return self._themes[key]

    def get_active_theme(self) -> ThemeDefinition:
        return self._themes[self._active_theme_id]

    def get_active_tokens(self) -> dict[str, str]:
        return self.get_active_theme().tokens()

    def apply_theme(self, theme_name: str):
        theme_id = self._normalize_theme_id(theme_name)
        if theme_id not in self._themes:
            # Preserve old behavior where unknown themes simply failed soft.
            theme_id = "dark"

        candidate = self._themes[theme_id]
        validation = self.validate_theme(candidate)
        if not validation["valid"]:
            candidate = self._themes[self._last_good_theme_id]
            theme_id = candidate.ThemeId

        self.app.setStyleSheet(self._composer.compose(candidate))
        self._active_theme_id = theme_id
        self._last_good_theme_id = theme_id
        self.current_theme = theme_id
        self._persist()
        return theme_id

    def preview_theme(self, theme: ThemeDefinition) -> str:
        validation = self.validate_theme(theme)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        return self._composer.compose(theme)

    def save_custom_theme(self, theme: ThemeDefinition) -> None:
        if theme.IsBuiltIn:
            raise ValueError("custom save cannot overwrite built-in theme")
        validation = self.validate_theme(theme)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        self._themes[theme.ThemeId] = theme
        self._persist()

    def delete_custom_theme(self, theme_id: str) -> None:
        key = self._normalize_theme_id(theme_id)
        theme = self.get_theme(key)
        if theme.IsBuiltIn:
            raise ValueError("built-in themes cannot be deleted")
        del self._themes[key]
        if self._active_theme_id == key:
            self.apply_theme("dark")
        self._persist()

    def reset_to_default(self) -> str:
        return self.apply_theme("dark")

    def toggle_theme(self):
        new_theme = "light" if self._active_theme_id != "light" else "dark"
        return self.apply_theme(new_theme)

    def validate_theme(self, theme: ThemeDefinition | Mapping) -> dict:
        if not isinstance(theme, ThemeDefinition):
            theme = ThemeDefinition.from_dict(theme)

        errors: list[str] = []
        warnings: list[str] = []
        for token, value in theme.tokens().items():
            if not _is_hex_color(value):
                errors.append(f"{token} must be a #RRGGBB color")

        if not errors:
            pairs = (
                ("PrimaryText", "PrimaryBackground", 4.5),
                ("PrimaryText", "Surface", 4.5),
                ("SecondaryText", "Surface", 3.0),
                ("Accent", "PrimaryBackground", 3.0),
                ("ButtonPrimary", "PrimaryBackground", 3.0),
            )
            tokens = theme.tokens()
            for fg, bg, minimum in pairs:
                ratio = contrast_ratio(tokens[fg], tokens[bg])
                if ratio < minimum:
                    warnings.append(
                        f"{fg} on {bg} contrast {ratio:.2f}:1 below {minimum}:1"
                    )

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _load_builtin_themes(self) -> None:
        self._themes = {
            "dark": ThemeDefinition(
                ThemeId="dark",
                DisplayName="Dark",
                IsBuiltIn=True,
                PrimaryBackground="#0f1115",
                SecondaryBackground="#151923",
                Surface="#1d2330",
                SurfaceAlt="#242c3a",
                PrimaryText="#f3f6fb",
                SecondaryText="#aeb8c7",
                Accent="#62d0ff",
                AccentHover="#86ddff",
                AccentActive="#31b8ef",
                Border="#344052",
                ButtonPrimary="#2f88ff",
                ButtonSecondary="#2d3544",
                Success="#4cc38a",
                Warning="#e0b84f",
                Error="#ee6a6a",
                Info="#62d0ff",
                Disabled="#6b7280",
            ),
            "light": ThemeDefinition(
                ThemeId="light",
                DisplayName="Light",
                IsBuiltIn=True,
                PrimaryBackground="#f5f7fb",
                SecondaryBackground="#e9eef6",
                Surface="#ffffff",
                SurfaceAlt="#eef3fa",
                PrimaryText="#17202e",
                SecondaryText="#526070",
                Accent="#006fba",
                AccentHover="#0b83d8",
                AccentActive="#005c9b",
                Border="#c8d2df",
                ButtonPrimary="#006fba",
                ButtonSecondary="#d7e0ea",
                Success="#217a4d",
                Warning="#8a6400",
                Error="#b93030",
                Info="#006fba",
                Disabled="#8c97a6",
            ),
            "anime": ThemeDefinition(
                ThemeId="anime",
                DisplayName="Anime",
                IsBuiltIn=True,
                PrimaryBackground="#101018",
                SecondaryBackground="#171729",
                Surface="#22223a",
                SurfaceAlt="#2d2d4a",
                PrimaryText="#fff7ff",
                SecondaryText="#c9c3d9",
                Accent="#ff7ac8",
                AccentHover="#ff9bd6",
                AccentActive="#e85cad",
                Border="#4b4564",
                ButtonPrimary="#ff7ac8",
                ButtonSecondary="#343452",
                Success="#6be6a4",
                Warning="#ffd166",
                Error="#ff6b8a",
                Info="#8bd3ff",
                Disabled="#79748a",
            ),
        }

    def _load_custom_themes(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            for raw_theme in data.get("custom_themes", []):
                theme = ThemeDefinition.from_dict(raw_theme)
                if not theme.IsBuiltIn:
                    self._themes[theme.ThemeId] = theme
            active = self._normalize_theme_id(data.get("active_theme", "dark"))
            if active in self._themes:
                self._active_theme_id = active
                self._last_good_theme_id = active
        except (OSError, ValueError, json.JSONDecodeError):
            self._active_theme_id = "dark"
            self._last_good_theme_id = "dark"

    def _persist(self) -> None:
        custom = [
            theme.to_dict()
            for theme in self._themes.values()
            if not theme.IsBuiltIn
        ]
        data = {
            "active_theme": self._active_theme_id,
            "custom_themes": custom,
        }
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_theme_id(theme_id: str) -> str:
        return str(theme_id or "").strip().lower().replace(" ", "_")


def _is_hex_color(value: str) -> bool:
    raw = str(value or "")
    if len(raw) != 7 or not raw.startswith("#"):
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in raw[1:])


def _rgb(color: str) -> tuple[float, float, float]:
    raw = color.lstrip("#")
    return (
        int(raw[0:2], 16) / 255.0,
        int(raw[2:4], 16) / 255.0,
        int(raw[4:6], 16) / 255.0,
    )


def _linear(channel: float) -> float:
    if channel <= 0.03928:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _luminance(color: str) -> float:
    r, g, b = (_linear(c) for c in _rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground: str, background: str) -> float:
    a = _luminance(foreground)
    b = _luminance(background)
    lighter = max(a, b)
    darker = min(a, b)
    return (lighter + 0.05) / (darker + 0.05)
