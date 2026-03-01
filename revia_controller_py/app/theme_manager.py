import os


class ThemeManager:
    def __init__(self, app):
        self.app = app
        self.current_theme = "dark"
        self.qss_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "gui", "qss"
        )

    def apply_theme(self, theme_name):
        qss_path = os.path.join(self.qss_dir, f"theme_{theme_name}.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.app.setStyleSheet(f.read())
            self.current_theme = theme_name

    def toggle_theme(self):
        new_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme(new_theme)
        return new_theme
