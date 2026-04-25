import sys
import os
from pathlib import Path


def _preferred_python():
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",  # Windows
        project_root / ".venv" / "bin" / "python",          # Linux/macOS
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return ""


def _ensure_local_venv_python():
    target = _preferred_python()
    if not target:
        return
    cur = str(Path(sys.executable).resolve())
    if cur.lower() == target.lower():
        return
    # Relaunch this controller under the project's local .venv interpreter.
    print(f"[REVIA Controller] Relaunching under local virtualenv: {target}")
    os.execv(target, [target, str(Path(__file__).resolve()), *sys.argv[1:]])


def _load_local_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def main():
    _ensure_local_venv_python()
    _load_local_env()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont

    from app.event_bus import EventBus
    from app.controller_client import ControllerClient
    from app.theme_manager import ThemeManager
    from gui.main_window import MainWindow

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    app = QApplication(sys.argv)
    app.setApplicationName("REVIA")
    app.setFont(QFont("Segoe UI", 10))

    event_bus = EventBus()
    client = ControllerClient(event_bus)
    theme_mgr = ThemeManager(app, event_bus=event_bus)
    theme_mgr.apply_theme(theme_mgr.current_theme)

    window = MainWindow(event_bus, client, theme_mgr)
    window.showMaximized()

    client.start()

    ret = app.exec()
    client.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
