import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app.event_bus import EventBus
from app.controller_client import ControllerClient
from app.theme_manager import ThemeManager
from gui.main_window import MainWindow


def main():
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
    client.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
