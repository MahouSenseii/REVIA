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
    os.execv(target, [target, str(Path(__file__).resolve()), *sys.argv[1:]])

def main():
    _ensure_local_venv_python()

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
    theme_mgr = ThemeManager(app)
    theme_mgr.apply_theme("dark")

    window = MainWindow(event_bus, client, theme_mgr)
    window.showMaximized()

    client.start()

    ret = app.exec()
    client.shutdown()   # stop() + executor.shutdown(); ensures clean thread teardown
    sys.exit(ret)


if __name__ == "__main__":
    main()
