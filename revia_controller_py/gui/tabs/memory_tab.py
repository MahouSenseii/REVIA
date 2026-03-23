import logging
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QSpinBox, QCheckBox, QPushButton,
)
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont

logger = logging.getLogger(__name__)


class _BgBridge(QObject):
    """Marshals callables from a background thread to the Qt main-thread event loop.

    Usage (from any thread):
        self._bg._call.emit(lambda: <ui update here>)
    The lambda is guaranteed to execute on the main thread.
    """
    _call = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._call.connect(lambda fn: fn())


class MemoryTab(QScrollArea):
    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self._bg = _BgBridge(self)
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Memory / RAG")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        docker_group = QGroupBox("Docker Memory Backend (Long-Term)")
        docker_group.setObjectName("settingsGroup")
        dg = QHBoxLayout(docker_group)

        self.docker_status = QLabel("Status: not checked")
        self.docker_status.setFont(QFont("Consolas", 9))
        self.docker_status.setObjectName("metricLabel")
        dg.addWidget(self.docker_status, stretch=1)

        docker_check_btn = QPushButton("Check")
        docker_check_btn.setObjectName("secondaryBtn")
        docker_check_btn.clicked.connect(self._check_docker_status)
        dg.addWidget(docker_check_btn)
        layout.addWidget(docker_group)

        backend_group = QGroupBox("Memory Store")
        backend_group.setObjectName("settingsGroup")
        bg = QFormLayout(backend_group)

        self.memory_backend = QComboBox()
        self.memory_backend.addItems(["ChromaDB", "FAISS", "Qdrant", "None"])
        bg.addRow("Backend:", self.memory_backend)
        self.memory_backend.setVisible(False)  # Internal: auto-configured

        self.collection = QLineEdit("revia_memories")
        bg.addRow("Collection:", self.collection)
        self.collection.setVisible(False)  # Internal: auto-configured

        self.max_results = QSpinBox()
        self.max_results.setRange(1, 50)
        self.max_results.setValue(5)
        bg.addRow("Max Results:", self.max_results)

        self.auto_store = QCheckBox("Auto-store conversations")
        self.auto_store.setChecked(True)
        bg.addRow("", self.auto_store)
        layout.addWidget(backend_group)

        st_group = QGroupBox("Short-Term Memory (Conversation)")
        st_group.setObjectName("settingsGroup")
        stl = QVBoxLayout(st_group)

        st_info = QLabel(
            "Active conversation context. Recent exchanges are kept in a sliding "
            "window. Older entries are promoted to long-term automatically."
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

        lt_group = QGroupBox("Long-Term Memory (Persistent)")
        lt_group.setObjectName("settingsGroup")
        ltl = QVBoxLayout(lt_group)

        lt_info = QLabel(
            "Persistent memory stored on Redis (if available) or local JSONL fallback. "
            "Includes promoted history, notes, facts, and people profiles."
        )
        lt_info.setFont(QFont("Segoe UI", 8))
        lt_info.setWordWrap(True)
        ltl.addWidget(lt_info)

        self.lt_count = QLabel("Entries: 0")
        self.lt_count.setFont(QFont("Consolas", 9))
        self.lt_count.setObjectName("metricLabel")
        ltl.addWidget(self.lt_count)

        recent_lbl = QLabel("Recent Saved Items")
        recent_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        ltl.addWidget(recent_lbl)

        self.lt_recent = QTextEdit()
        self.lt_recent.setReadOnly(True)
        self.lt_recent.setMaximumHeight(150)
        self.lt_recent.setPlaceholderText("Recent long-term memories will appear here...")
        ltl.addWidget(self.lt_recent)

        recent_btn_row = QHBoxLayout()
        show_recent_btn = QPushButton("Show Recent")
        show_recent_btn.setObjectName("secondaryBtn")
        show_recent_btn.clicked.connect(self._refresh_long_term_recent)
        recent_btn_row.addWidget(show_recent_btn)
        recent_btn_row.addStretch()
        ltl.addLayout(recent_btn_row)

        delete_row = QHBoxLayout()
        self.lt_delete_select = QComboBox()
        self.lt_delete_select.setMinimumWidth(200)
        delete_row.addWidget(self.lt_delete_select, stretch=1)
        delete_btn = QPushButton("Delete Selected")
        delete_btn.setObjectName("secondaryBtn")
        delete_btn.clicked.connect(self._delete_selected_long_term)
        delete_row.addWidget(delete_btn)
        ltl.addLayout(delete_row)

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

        self.event_bus.chat_complete.connect(self._on_chat_complete)
        self.event_bus.connection_changed.connect(self._on_connected)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_chat_complete(self, _text):
        # Each refresh submits its own background task — no blocking on main thread.
        self._refresh_short_term()
        self._refresh_stats()
        self._refresh_long_term_recent()

    def _on_connected(self, connected):
        if connected:
            self._check_docker_status()
            self._refresh_short_term()
            self._refresh_stats()
            self._refresh_long_term_recent()

    # ------------------------------------------------------------------
    # HTTP helpers — all run in client._executor, results marshalled back
    # to the Qt main thread via _bg._call signal.
    # ------------------------------------------------------------------

    def _refresh_short_term(self):
        # Capture UI value on the main thread before handing off to the executor.
        limit = self.st_window.value()

        def _work():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/memory/short",
                    params={"limit": limit},
                    timeout=3,
                )
                if r.ok:
                    entries = r.json()
                    lines = []
                    for e in entries:
                        role = e.get("role", "?")
                        ts = e.get("timestamp", "")[:19]
                        content = e.get("content", "")[:120]
                        meta = e.get("metadata", {}) or {}
                        emo = meta.get("emotion", "")
                        emo_str = f" [{emo}]" if emo else ""
                        lines.append(f"[{ts}] {role}{emo_str}: {content}")
                    text = "\n".join(lines)
                    count = len(entries)

                    def _apply(t=text, c=count):
                        self.st_count.setText(f"Entries: {c}")
                        self.st_list.setPlainText(t)

                    self._bg._call.emit(_apply)
            except Exception as ex:
                err = str(ex)
                self._bg._call.emit(
                    lambda e=err: self.st_list.setPlainText(f"Error: {e}")
                )

        self.client._executor.submit(_work)

    def _clear_short_term(self):
        def _work():
            try:
                self.client._session_post(
                    f"{self.client.BASE_URL}/api/memory/short/clear", timeout=3
                )
            except Exception as e:
                logger.warning(f"Error clearing short-term memory: {e}")

            def _apply():
                self.st_list.clear()
                self.st_count.setText("Entries: 0")

            self._bg._call.emit(_apply)

        self.client._executor.submit(_work)

    def _refresh_stats(self):
        def _work():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/memory/stats", timeout=3
                )
                if r.ok:
                    stats = r.json()
                    prof = stats.get("profile", "default")
                    st_text = (
                        f"Entries: {stats.get('short_term_count', 0)} "
                        f"(Profile: {prof})"
                    )
                    lt_text = (
                        f"Entries: {stats.get('long_term_count', 0)} "
                        f"(Profile: {prof})"
                    )

                    def _apply(s=st_text, lt=lt_text):
                        self.st_count.setText(s)
                        self.lt_count.setText(lt)

                    self._bg._call.emit(_apply)
            except Exception as e:
                logger.warning(f"Error refreshing memory stats: {e}")

        self.client._executor.submit(_work)

    def _refresh_long_term_recent(self):
        limit = max(10, self.max_results.value() * 4)

        def _work():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/memory/long",
                    params={"limit": limit},
                    timeout=3,
                )
                if not r.ok:
                    return
                entries = r.json()
                if not entries:
                    def _apply_empty():
                        self.lt_recent.setPlainText("No long-term entries saved yet.")
                        self.lt_delete_select.clear()
                    self._bg._call.emit(_apply_empty)
                    return

                lines = []
                items = []  # (combo label, userData) pairs
                for e in reversed(entries):
                    entry_id = str(e.get("id", "")).strip()
                    ts = e.get("timestamp", "")[:19]
                    cat = e.get("category", e.get("role", "?"))
                    content = e.get("content", "")[:220]
                    meta = e.get("metadata", {}) or {}
                    preview = content.replace("\n", " ")
                    if len(preview) > 74:
                        preview = preview[:71] + "..."
                    if entry_id:
                        items.append((f"[{ts}] ({cat}) {preview}", entry_id))
                    if cat == "person_profile":
                        pname = meta.get("name", "")
                        rel = meta.get("relation", "contact")
                        imp = meta.get("importance", 0.0)
                        lines.append(
                            f"[{ts}] ({cat}) id={entry_id} {pname} / {rel} / "
                            f"importance={imp}\n{content}"
                        )
                    else:
                        lines.append(f"[{ts}] ({cat}) id={entry_id} {content}")
                text = "\n\n".join(lines)

                def _apply(t=text, it=items):
                    self.lt_delete_select.clear()
                    for label, uid in it:
                        self.lt_delete_select.addItem(label, userData=uid)
                    self.lt_recent.setPlainText(t)

                self._bg._call.emit(_apply)
            except Exception as ex:
                err = str(ex)
                self._bg._call.emit(
                    lambda e=err: self.lt_recent.setPlainText(f"Error: {e}")
                )

        self.client._executor.submit(_work)

    def _search_long_term(self):
        query = self.lt_search.text().strip()
        if not query:
            return
        limit = self.max_results.value()

        def _work():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/memory/long/search",
                    params={"q": query, "limit": limit},
                    timeout=3,
                )
                if r.ok:
                    results = r.json()
                    if not results:
                        self._bg._call.emit(
                            lambda: self.lt_results.setPlainText("No results found.")
                        )
                    else:
                        lines = []
                        for e in results:
                            entry_id = str(e.get("id", "")).strip()
                            ts = e.get("timestamp", "")[:19]
                            cat = e.get("category", e.get("role", "?"))
                            content = e.get("content", "")[:220]
                            lines.append(f"[{ts}] ({cat}) id={entry_id} {content}")
                        text = "\n\n".join(lines)
                        self._bg._call.emit(
                            lambda t=text: self.lt_results.setPlainText(t)
                        )
            except Exception as ex:
                err = str(ex)
                self._bg._call.emit(
                    lambda e=err: self.lt_results.setPlainText(f"Error: {e}")
                )

        self.client._executor.submit(_work)

    def _save_note(self):
        note = self.lt_note.text().strip()
        if not note:
            return

        def _work():
            try:
                self.client._session_post(
                    f"{self.client.BASE_URL}/api/memory/long/save",
                    json={"content": note, "category": "user_note"},
                    timeout=3,
                )
            except Exception as e:
                logger.error(f"Error saving note to memory: {e}")

            # After the save completes, clear the input and kick off refreshes.
            def _apply():
                self.lt_note.clear()
                # These each submit their own background tasks.
                self._refresh_stats()
                self._refresh_long_term_recent()

            self._bg._call.emit(_apply)

        self.client._executor.submit(_work)

    def _clear_long_term(self):
        def _work():
            try:
                self.client._session_post(
                    f"{self.client.BASE_URL}/api/memory/long/clear", timeout=3
                )
            except Exception as e:
                logger.warning(f"Error clearing long-term memory: {e}")

            def _apply():
                self.lt_results.clear()
                self.lt_recent.clear()
                self.lt_delete_select.clear()
                self.lt_count.setText("Entries: 0")

            self._bg._call.emit(_apply)

        self.client._executor.submit(_work)

    def _delete_selected_long_term(self):
        entry_id = str(self.lt_delete_select.currentData() or "").strip()
        if not entry_id:
            return

        def _work():
            try:
                self.client._session_post(
                    f"{self.client.BASE_URL}/api/memory/long/delete",
                    json={"id": entry_id},
                    timeout=3,
                )
            except Exception as ex:
                err = str(ex)
                self._bg._call.emit(
                    lambda e=err: self.lt_results.setPlainText(f"Delete error: {e}")
                )
                return

            def _apply():
                self._refresh_stats()
                self._refresh_long_term_recent()

            self._bg._call.emit(_apply)

        self.client._executor.submit(_work)

    def _check_docker_status(self):
        def _work():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/memory/docker/status", timeout=3
                )
                if not r.ok:
                    return
                d = r.json()
                count = d.get("long_term_count", 0)
                profile = d.get("profile", "default")
                if d.get("redis_available"):
                    host = d.get("redis_host", "127.0.0.1")
                    port = d.get("redis_port", 6379)
                    txt = (
                        f"Redis (Docker): Connected {host}:{port} "
                        f"| entries={count} | profile={profile}"
                    )
                    css = "color: #4ade80;"
                elif d.get("memory_online", False):
                    local_file = d.get("local_file", "")
                    txt = (
                        "Memory backend: ONLINE (local file fallback). "
                        f"entries={count} | profile={profile}\n{local_file}"
                    )
                    css = "color: #4ade80;"
                else:
                    txt = "Memory backend: Unavailable (Redis + local fallback both failed)."
                    css = "color: #f87171;"

                def _apply(t=txt, c=css):
                    self.docker_status.setText(t)
                    self.docker_status.setStyleSheet(c)

                self._bg._call.emit(_apply)
            except Exception as ex:
                err = str(ex)

                def _apply_err(e=err):
                    self.docker_status.setText(f"Docker check error: {e}")
                    self.docker_status.setStyleSheet("color: #f87171;")

                self._bg._call.emit(_apply_err)

        self.client._executor.submit(_work)
