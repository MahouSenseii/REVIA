from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QSpinBox, QCheckBox, QPushButton,
)
from PySide6.QtGui import QFont


class MemoryTab(QScrollArea):
    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Memory / RAG")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # --- Memory Backend ---
        backend_group = QGroupBox("Memory Store")
        backend_group.setObjectName("settingsGroup")
        bg = QFormLayout(backend_group)

        self.memory_backend = QComboBox()
        self.memory_backend.addItems(["ChromaDB", "FAISS", "Qdrant", "None"])
        bg.addRow("Backend:", self.memory_backend)

        self.collection = QLineEdit("revia_memories")
        bg.addRow("Collection:", self.collection)

        self.max_results = QSpinBox()
        self.max_results.setRange(1, 50)
        self.max_results.setValue(5)
        bg.addRow("Max Results:", self.max_results)

        self.auto_store = QCheckBox("Auto-store conversations")
        self.auto_store.setChecked(True)
        bg.addRow("", self.auto_store)

        layout.addWidget(backend_group)

        # --- Short-term Memory (Conversation Window) ---
        st_group = QGroupBox("Short-Term Memory (Conversation)")
        st_group.setObjectName("settingsGroup")
        stl = QVBoxLayout(st_group)

        st_info = QLabel(
            "Active conversation context. Recent exchanges are kept in "
            "a sliding window. Older entries promote to long-term automatically."
        )
        st_info.setFont(QFont("Segoe UI", 8))
        st_info.setWordWrap(True)
        stl.addWidget(st_info)

        st_stats = QHBoxLayout()
        self.st_count = QLabel("Entries: 0")
        self.st_count.setFont(QFont("Consolas", 9))
        self.st_count.setObjectName("metricLabel")
        st_stats.addWidget(self.st_count)

        self.st_window = QSpinBox()
        self.st_window.setRange(10, 200)
        self.st_window.setValue(100)
        self.st_window.setSuffix(" messages")
        st_stats.addWidget(QLabel("Window:"))
        st_stats.addWidget(self.st_window)
        stl.addLayout(st_stats)

        self.st_list = QTextEdit()
        self.st_list.setReadOnly(True)
        self.st_list.setMaximumHeight(160)
        self.st_list.setPlaceholderText("No conversation history yet...")
        stl.addWidget(self.st_list)

        st_btn_row = QHBoxLayout()
        st_refresh = QPushButton("Refresh")
        st_refresh.setObjectName("secondaryBtn")
        st_refresh.clicked.connect(self._refresh_short_term)
        st_btn_row.addWidget(st_refresh)

        st_clear = QPushButton("Clear Short-Term")
        st_clear.setObjectName("secondaryBtn")
        st_clear.clicked.connect(self._clear_short_term)
        st_btn_row.addWidget(st_clear)
        stl.addLayout(st_btn_row)

        layout.addWidget(st_group)

        # --- Long-term Memory (Persistent) ---
        lt_group = QGroupBox("Long-Term Memory (Persistent)")
        lt_group.setObjectName("settingsGroup")
        ltl = QVBoxLayout(lt_group)

        lt_info = QLabel(
            "Persistent memory stored on disk. Includes promoted conversation "
            "history, user notes, facts, and preferences. Survives restarts."
        )
        lt_info.setFont(QFont("Segoe UI", 8))
        lt_info.setWordWrap(True)
        ltl.addWidget(lt_info)

        self.lt_count = QLabel("Entries: 0")
        self.lt_count.setFont(QFont("Consolas", 9))
        self.lt_count.setObjectName("metricLabel")
        ltl.addWidget(self.lt_count)

        search_row = QHBoxLayout()
        self.lt_search = QLineEdit()
        self.lt_search.setPlaceholderText("Search long-term memory...")
        search_row.addWidget(self.lt_search, stretch=1)

        search_btn = QPushButton("Search")
        search_btn.setObjectName("secondaryBtn")
        search_btn.clicked.connect(self._search_long_term)
        search_row.addWidget(search_btn)
        ltl.addLayout(search_row)

        self.lt_results = QTextEdit()
        self.lt_results.setReadOnly(True)
        self.lt_results.setMaximumHeight(160)
        self.lt_results.setPlaceholderText("Search results will appear here...")
        ltl.addWidget(self.lt_results)

        # Manual note
        note_row = QHBoxLayout()
        self.lt_note = QLineEdit()
        self.lt_note.setPlaceholderText("Save a note to long-term memory...")
        note_row.addWidget(self.lt_note, stretch=1)

        save_note_btn = QPushButton("Save Note")
        save_note_btn.setObjectName("primaryBtn")
        save_note_btn.clicked.connect(self._save_note)
        note_row.addWidget(save_note_btn)
        ltl.addLayout(note_row)

        lt_btn_row = QHBoxLayout()
        lt_refresh = QPushButton("Refresh Stats")
        lt_refresh.setObjectName("secondaryBtn")
        lt_refresh.clicked.connect(self._refresh_stats)
        lt_btn_row.addWidget(lt_refresh)

        lt_clear = QPushButton("Clear Long-Term")
        lt_clear.setObjectName("secondaryBtn")
        lt_clear.clicked.connect(self._clear_long_term)
        lt_btn_row.addWidget(lt_clear)
        ltl.addLayout(lt_btn_row)

        layout.addWidget(lt_group)
        layout.addStretch()
        self.setWidget(container)

    # --- Actions ---

    def _refresh_short_term(self):
        try:
            import requests
            r = requests.get(
                f"{self.client.BASE_URL}/api/memory/short",
                params={"limit": self.st_window.value()},
                timeout=3,
            )
            if r.ok:
                entries = r.json()
                self.st_count.setText(f"Entries: {len(entries)}")
                lines = []
                for e in entries:
                    role = e.get("role", "?")
                    ts = e.get("timestamp", "")[:19]
                    content = e.get("content", "")[:120]
                    meta = e.get("metadata", {})
                    emo = meta.get("emotion", "")
                    emo_str = f" [{emo}]" if emo else ""
                    lines.append(f"[{ts}] {role}{emo_str}: {content}")
                self.st_list.setPlainText("\n".join(lines))
        except Exception as ex:
            self.st_list.setPlainText(f"Error: {ex}")

    def _clear_short_term(self):
        try:
            import requests
            requests.post(
                f"{self.client.BASE_URL}/api/memory/short/clear", timeout=3
            )
            self.st_list.clear()
            self.st_count.setText("Entries: 0")
        except Exception:
            pass

    def _refresh_stats(self):
        try:
            import requests
            r = requests.get(
                f"{self.client.BASE_URL}/api/memory/stats", timeout=3
            )
            if r.ok:
                stats = r.json()
                prof = stats.get("profile", "default")
                self.st_count.setText(
                    f"Entries: {stats.get('short_term_count', 0)} "
                    f"(Profile: {prof})"
                )
                self.lt_count.setText(
                    f"Entries: {stats.get('long_term_count', 0)} "
                    f"(Profile: {prof})"
                )
        except Exception:
            pass

    def _search_long_term(self):
        query = self.lt_search.text().strip()
        if not query:
            return
        try:
            import requests
            r = requests.get(
                f"{self.client.BASE_URL}/api/memory/long/search",
                params={"q": query, "limit": self.max_results.value()},
                timeout=3,
            )
            if r.ok:
                results = r.json()
                if not results:
                    self.lt_results.setPlainText("No results found.")
                else:
                    lines = []
                    for e in results:
                        ts = e.get("timestamp", "")[:19]
                        cat = e.get("category", e.get("role", "?"))
                        content = e.get("content", "")[:200]
                        lines.append(f"[{ts}] ({cat}) {content}")
                    self.lt_results.setPlainText("\n\n".join(lines))
        except Exception as ex:
            self.lt_results.setPlainText(f"Error: {ex}")

    def _save_note(self):
        note = self.lt_note.text().strip()
        if not note:
            return
        try:
            import requests
            requests.post(
                f"{self.client.BASE_URL}/api/memory/long/save",
                json={"content": note, "category": "user_note"},
                timeout=3,
            )
            self.lt_note.clear()
            self._refresh_stats()
        except Exception:
            pass

    def _clear_long_term(self):
        try:
            import requests
            requests.post(
                f"{self.client.BASE_URL}/api/memory/long/clear", timeout=3
            )
            self.lt_results.clear()
            self.lt_count.setText("Entries: 0")
        except Exception:
            pass
