"""REVIA UI theme ownership.

ThemeManager is the single owner for:
    * built-in and custom theme definitions
    * active theme selection
    * stylesheet generation (StyleComposer for custom themes, QSS files for built-ins)
    * persistence of custom themes
    * readability validation

Widgets should consume the stylesheet generated here instead of owning
hardcoded colors.

API contract (matches main.py and theme_tab.py):
    theme_mgr = ThemeManager(app)
    theme_mgr.apply_theme(theme_mgr.current_theme)
    theme_mgr.available_themes()        -> list[ThemeDefinition]
    theme_mgr.get_theme(theme_id)       -> ThemeDefinition
    theme_mgr.apply_theme(theme_id)     -> str (applied theme_id)
    theme_mgr.reset_to_default()        -> str (applied theme_id)
    theme_mgr.save_custom_theme(draft)
    theme_mgr.delete_custom_theme(theme_id)
    theme_mgr.preview_theme(draft)      -> str (QSS)
    theme_mgr.validate_theme(draft)     -> dict
    theme_mgr.app                       -> QApplication reference
    theme_mgr.current_theme             -> str (active theme_id)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from app.style_composer import StyleComposer
except ImportError:
    from .style_composer import StyleComposer


# Token names (order preserved for UI display)

THEME_TOKENS = (
    "PrimaryBackground",
    "SecondaryBackground",
    "Surface",
    "SurfaceAlt",
    "PrimaryText",
    "SecondaryText",
    "Border",
    "Accent",
    "AccentHover",
    "AccentActive",
    "ButtonPrimary",
    "ButtonSecondary",
    "Success",
    "Warning",
    "Error",
    "Info",
    "Disabled",
)

_DEFAULT_THEME_ID = "anime"
_QSS_DIR = Path(__file__).resolve().parent.parent / "gui" / "qss"
_CUSTOM_FILE = Path(__file__).resolve().parent.parent / "custom_themes.json"


# ThemeDefinition

@dataclass
class ThemeDefinition:
    """A complete theme specification used by both the editor and the compositor.

    Attributes match the API consumed by theme_tab.py (PascalCase).
    """
    ThemeId: str = "custom"
    DisplayName: str = "Custom"
    IsBuiltIn: bool = False
    # Token values - defaults match the Anime (Purple) palette
    PrimaryBackground: str = "#06050f"
    SecondaryBackground: str = "#0d0c1e"
    Surface: str = "#13112a"
    SurfaceAlt: str = "#1c1838"
    PrimaryText: str = "#ede9fe"
    SecondaryText: str = "#8b7ab8"
    Border: str = "#2e2050"
    Accent: str = "#a855f7"
    AccentHover: str = "#c084fc"
    AccentActive: str = "#7c3aed"
    ButtonPrimary: str = "#6c3bf5"
    ButtonSecondary: str = "#1a1535"
    Success: str = "#2dd4bf"
    Warning: str = "#f59e0b"
    Error: str = "#f43f5e"
    Info: str = "#60a5fa"
    Disabled: str = "#4a3a6a"

    # Token access

    def tokens(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in THEME_TOKENS}

    # Serialisation

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ThemeId": self.ThemeId,
            "DisplayName": self.DisplayName,
            "IsBuiltIn": self.IsBuiltIn,
        }
        d.update(self.tokens())
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ThemeDefinition":
        known = {
            "ThemeId", "DisplayName", "IsBuiltIn",
            *THEME_TOKENS,
        }
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(**kwargs)

    # Validation helpers

    def missing_tokens(self) -> list[str]:
        return [t for t in THEME_TOKENS if not str(getattr(self, t, "") or "").strip()]

    def invalid_tokens(self) -> list[str]:
        from PySide6.QtGui import QColor
        return [
            t for t in THEME_TOKENS
            if not QColor(str(getattr(self, t, "") or "")).isValid()
        ]


# Built-in theme catalog

def _builtin(theme_id: str, display: str, **tokens) -> ThemeDefinition:
    return ThemeDefinition(ThemeId=theme_id, DisplayName=display, IsBuiltIn=True, **tokens)


_BUILTIN_THEMES: dict[str, ThemeDefinition] = {
    "dark": _builtin(
        "dark", "Dark (Default)",
        PrimaryBackground="#070b14",
        SecondaryBackground="#0c1020",
        Surface="#101828",
        SurfaceAlt="#161e30",
        PrimaryText="#e0e6f0",
        SecondaryText="#808898",
        Border="#1e2a40",
        Accent="#00d4ff",
        AccentHover="#33ddff",
        AccentActive="#0099cc",
        ButtonPrimary="#004466",
        ButtonSecondary="#0d1520",
        Success="#00dc50",
        Warning="#f59e0b",
        Error="#dc3250",
        Info="#60a5fa",
        Disabled="#404858",
    ),
    "anime": _builtin(
        "anime", "Anime (Purple)",
        PrimaryBackground="#06050f",
        SecondaryBackground="#0d0c1e",
        Surface="#13112a",
        SurfaceAlt="#1c1838",
        PrimaryText="#ede9fe",
        SecondaryText="#8b7ab8",
        Border="#2e2050",
        Accent="#a855f7",
        AccentHover="#c084fc",
        AccentActive="#7c3aed",
        ButtonPrimary="#6c3bf5",
        ButtonSecondary="#1a1535",
        Success="#2dd4bf",
        Warning="#f59e0b",
        Error="#f43f5e",
        Info="#60a5fa",
        Disabled="#4a3a6a",
    ),
    "light": _builtin(
        "light", "Light",
        PrimaryBackground="#f5f5fa",
        SecondaryBackground="#ededf5",
        Surface="#ffffff",
        SurfaceAlt="#f0eef8",
        PrimaryText="#1a1030",
        SecondaryText="#6b6080",
        Border="#d0c8e8",
        Accent="#7c3aed",
        AccentHover="#9340f5",
        AccentActive="#5b21b6",
        ButtonPrimary="#7c3aed",
        ButtonSecondary="#e8e4f8",
        Success="#059669",
        Warning="#d97706",
        Error="#dc2626",
        Info="#2563eb",
        Disabled="#a09ab8",
    ),
    # Nord - the classic arctic palette (https://www.nordtheme.com/).
    # Uses the official 16 Nord colors: polar night for backgrounds,
    # snow storm for text, frost for accents, aurora for status roles.
    "nord": _builtin(
        "nord", "Nord",
        PrimaryBackground="#2e3440",   # nord0
        SecondaryBackground="#3b4252", # nord1
        Surface="#434c5e",             # nord2
        SurfaceAlt="#4c566a",          # nord3
        PrimaryText="#eceff4",         # nord6
        SecondaryText="#d8dee9",       # nord4
        Border="#4c566a",              # nord3
        Accent="#88c0d0",              # nord8 (frost)
        AccentHover="#8fbcbb",         # nord7
        AccentActive="#5e81ac",        # nord10
        ButtonPrimary="#5e81ac",       # nord10
        ButtonSecondary="#3b4252",     # nord1
        Success="#a3be8c",             # nord14
        Warning="#ebcb8b",             # nord13
        Error="#bf616a",               # nord11
        Info="#81a1c1",                # nord9
        Disabled="#4c566a",            # nord3
    ),
    # Red (crimson) - warm dark theme centered on rose/ruby accents.
    "red": _builtin(
        "red", "Crimson (Red)",
        PrimaryBackground="#140707",
        SecondaryBackground="#1d0a0a",
        Surface="#2a1012",
        SurfaceAlt="#361618",
        PrimaryText="#fce4e4",
        SecondaryText="#c89294",
        Border="#5a1e22",
        Accent="#e11d48",       # rose-600
        AccentHover="#f43f5e",  # rose-500
        AccentActive="#be123c", # rose-700
        ButtonPrimary="#be123c",
        ButtonSecondary="#2a1012",
        Success="#10b981",
        Warning="#f59e0b",
        Error="#dc2626",
        Info="#60a5fa",
        Disabled="#5a3a3a",
    ),
    # Sky Blue - cool azure dark theme.
    "sky": _builtin(
        "sky", "Sky Blue",
        PrimaryBackground="#051220",
        SecondaryBackground="#0a1a30",
        Surface="#0e233f",
        SurfaceAlt="#152e55",
        PrimaryText="#e0f0ff",
        SecondaryText="#90b0d0",
        Border="#2a4a70",
        Accent="#38bdf8",       # sky-400
        AccentHover="#7dd3fc",  # sky-300
        AccentActive="#0284c7", # sky-600
        ButtonPrimary="#0284c7",
        ButtonSecondary="#0e233f",
        Success="#22c55e",
        Warning="#f59e0b",
        Error="#ef4444",
        Info="#60a5fa",
        Disabled="#4a6080",
    ),
    # Gold - warm amber dark theme.
    "gold": _builtin(
        "gold", "Gold",
        PrimaryBackground="#1a1408",
        SecondaryBackground="#251c0c",
        Surface="#2e2410",
        SurfaceAlt="#3a2d15",
        PrimaryText="#fff8e0",
        SecondaryText="#d0bc80",
        Border="#5a4820",
        Accent="#fbbf24",       # amber-400
        AccentHover="#fcd34d",  # amber-300
        AccentActive="#d97706", # amber-600
        ButtonPrimary="#d97706",
        ButtonSecondary="#2e2410",
        Success="#65a30d",      # lime-600
        Warning="#f59e0b",
        Error="#dc2626",
        Info="#60a5fa",
        Disabled="#6a5830",
    ),
}


# ThemeManager

class ThemeManager:
    """Single owner of UI theme state for the REVIA controller.

    Strategy for stylesheet generation:
    * Built-in themes: load from the pre-authored QSS files in ``gui/qss/``
      (rich gradients and effects that StyleComposer doesn't replicate).
    * Custom themes: generate via StyleComposer from the token dict so custom
      themes are still fully usable without hand-writing QSS.
    """

    def __init__(self, app, config_dir: Path | None = None):
        self.app = app                       # QApplication — for setStyleSheet
        self._current_theme: str = _DEFAULT_THEME_ID
        self._themes: dict[str, ThemeDefinition] = dict(_BUILTIN_THEMES)
        self._composer = StyleComposer()
        self._config_file = (
            (config_dir or Path(".")) / "custom_themes.json"
            if config_dir
            else _CUSTOM_FILE
        )
        self._load_custom_themes()

    # Public read API

    @property
    def current_theme(self) -> str:
        """ID of the currently active theme (e.g. ``"anime"``, ``"dark"``)."""
        return self._current_theme

    def available_themes(self) -> list[ThemeDefinition]:
        """Return all registered themes (built-in first, then custom)."""
        builtin = [t for tid, t in self._themes.items() if t.IsBuiltIn]
        custom  = [t for tid, t in self._themes.items() if not t.IsBuiltIn]
        return builtin + custom

    def get_theme(self, theme_id: str) -> ThemeDefinition:
        """Return the ThemeDefinition for *theme_id*, or the default if missing."""
        return self._themes.get(theme_id, self._themes[_DEFAULT_THEME_ID])

    # Apply / reset

    def apply_theme(self, theme_id: str) -> str:
        """Apply theme by ID to the Qt application.

        Returns the ID that was actually applied (falls back to default if the
        requested ID is not registered).

        Notes on the "empty setStyleSheet first + polish pass" dance:
        Qt caches parsed QSS state per-widget. When the previous theme used
        gradients/backgrounds on ``QScrollArea`` or ``QTabWidget::pane``,
        switching to a sheet that styles those selectors with a plain color
        (or doesn't style them at all) often leaves the cached brush in
        place until the widget is next polished. Users see "half of the
        UI didn't change colors" until they nudge something else. The
        two-phase ``setStyleSheet("")`` → ``setStyleSheet(qss)`` plus an
        unpolish/polish walk forces Qt to drop that cached state on every
        widget on the spot.
        """
        if theme_id not in self._themes:
            theme_id = _DEFAULT_THEME_ID
        theme = self._themes.get(theme_id, self._themes[_DEFAULT_THEME_ID])
        self._apply_stylesheet(self._load_qss(theme_id), theme)
        self._current_theme = theme_id
        return theme_id

    def apply_preview(self, draft: ThemeDefinition) -> str:
        """Apply a draft theme preview without changing current_theme."""
        qss = self.preview_theme(draft)
        self._apply_stylesheet(qss, draft)
        return qss

    def _apply_stylesheet(self, qss: str, theme: ThemeDefinition) -> None:
        """Apply QSS and theme tokens in one full refresh pass."""
        try:
            self.app.setProperty("reviaThemeId", theme.ThemeId)
            self.app.setProperty("reviaThemeTokens", theme.tokens())
        except Exception:
            pass
        self.app.setStyleSheet("")
        self.app.setStyleSheet(qss)
        self._repolish_all_widgets()

    def _repolish_all_widgets(self) -> None:
        """Force Qt to re-polish every live widget.

        Without this, QScrollArea viewports, QTabWidget::pane, and widgets
        that have their own ``setStyleSheet()`` applied often keep the
        previous theme's painted background until they next get an event.
        """
        try:
            widgets = list(self.app.allWidgets())
        except Exception:
            return
        for widget in widgets:
            try:
                style = widget.style()
                if style is not None:
                    style.unpolish(widget)
                    style.polish(widget)
                widget.update()
            except Exception:
                # Widgets may have been deleted mid-iteration (C++ side);
                # skip them silently - the surviving ones still refresh.
                continue

    def reset_to_default(self) -> str:
        """Reset to the built-in default theme.  Returns the applied theme ID."""
        return self.apply_theme(_DEFAULT_THEME_ID)

    # Preview

    def preview_theme(self, draft: ThemeDefinition) -> str:
        """Return a QSS string for a draft theme without changing the active theme."""
        return self._composer.compose(draft)

    # Custom theme persistence

    def save_custom_theme(self, draft: ThemeDefinition) -> None:
        """Register and persist a custom theme.

        Raises ``ValueError`` if the ThemeId is blank or collides with a
        built-in name.
        """
        tid = str(draft.ThemeId or "").strip()
        if not tid:
            raise ValueError("ThemeId must not be blank.")
        if tid in _BUILTIN_THEMES:
            raise ValueError(f"'{tid}' is a built-in theme and cannot be overwritten.")
        draft.IsBuiltIn = False
        self._themes[tid] = draft
        self._persist_custom_themes()

    def delete_custom_theme(self, theme_id: str) -> None:
        """Remove a custom theme.  Raises ``ValueError`` for built-ins / missing."""
        if theme_id in _BUILTIN_THEMES:
            raise ValueError(f"'{theme_id}' is a built-in theme and cannot be deleted.")
        if theme_id not in self._themes:
            raise ValueError(f"Theme '{theme_id}' not found.")
        del self._themes[theme_id]
        if self._current_theme == theme_id:
            self.apply_theme(_DEFAULT_THEME_ID)
        self._persist_custom_themes()

    # Validation

    def validate_theme(self, draft: ThemeDefinition) -> dict:
        """Validate a ThemeDefinition and return a result dict.

        Returns::

            {
                "valid": bool,
                "errors": [str, ...],
                "warnings": [str, ...],
            }
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not str(draft.ThemeId or "").strip():
            errors.append("ThemeId is required.")

        missing = draft.missing_tokens()
        if missing:
            errors.append(f"Missing tokens: {', '.join(missing)}")

        try:
            invalid = draft.invalid_tokens()
            if invalid:
                errors.append(f"Invalid color values for: {', '.join(invalid)}")
        except Exception:
            pass

        # Soft contrast check (best-effort)
        try:
            ratio = self._contrast_ratio(draft.PrimaryBackground, draft.PrimaryText)
            if ratio < 4.5:
                warnings.append(
                    f"Low contrast ({ratio:.1f}:1) between PrimaryBackground "
                    f"and PrimaryText — WCAG AA requires 4.5:1."
                )
        except Exception:
            pass

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    # Internal: stylesheet loading

    def _load_qss(self, theme_id: str) -> str:
        """Return QSS for *theme_id*.

        For built-in themes, load from the pre-authored QSS file if present.
        For custom themes (or if the file is missing), fall back to
        StyleComposer token generation.
        """
        qss_path = _QSS_DIR / f"theme_{theme_id}.qss"
        if qss_path.exists():
            try:
                return qss_path.read_text(encoding="utf-8")
            except OSError:
                pass
        # Fall back to composer (used for custom themes and missing QSS)
        theme = self._themes.get(theme_id, self._themes[_DEFAULT_THEME_ID])
        return self._composer.compose(theme)

    # Internal: persistence

    def _persist_custom_themes(self) -> None:
        data = {
            tid: t.to_dict()
            for tid, t in self._themes.items()
            if not t.IsBuiltIn
        }
        try:
            self._config_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _load_custom_themes(self) -> None:
        if not self._config_file.exists():
            return
        try:
            raw = json.loads(self._config_file.read_text(encoding="utf-8"))
            for tid, fields in raw.items():
                if tid in _BUILTIN_THEMES:
                    continue
                fields.setdefault("ThemeId", tid)
                fields["IsBuiltIn"] = False
                try:
                    self._themes[tid] = ThemeDefinition.from_dict(fields)
                except TypeError:
                    pass
        except (OSError, json.JSONDecodeError):
            pass

    # Internal: contrast helpers

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return r / 255.0, g / 255.0, b / 255.0

    @staticmethod
    def _relative_luminance(r: float, g: float, b: float) -> float:
        def _s(v: float) -> float:
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        return 0.2126 * _s(r) + 0.7152 * _s(g) + 0.0722 * _s(b)

    def _contrast_ratio(self, hex_a: str, hex_b: str) -> float:
        la = self._relative_luminance(*self._hex_to_rgb(hex_a))
        lb = self._relative_luminance(*self._hex_to_rgb(hex_b))
        light, dark = max(la, lb), min(la, lb)
        return (light + 0.05) / (dark + 0.05)
