"""
REVIA Sing Tab — GUI for the !sing system.

Provides:
  - Song library browser (add, remove, process songs)
  - Queue display with drag-reorder
  - Now-playing display with lyrics
  - Control buttons (sing, skip, stop)
  - Auto-pick toggle and mood display
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QGroupBox,
    QComboBox, QLineEdit, QCheckBox, QProgressBar, QSplitter,
    QTextEdit, QInputDialog, QMessageBox,
)
from PySide6.QtGui import QFont, QColor

from app.ui_status import apply_status_style, clear_status_role

_log = logging.getLogger(__name__)


class SingTab(QWidget):
    """GUI tab for managing Revia's sing mode."""

    # Emitted when user clicks Sing in GUI
    sing_requested = Signal(str, str)   # (query, "gui_user")
    skip_requested = Signal()
    stop_requested = Signal()

    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client

        # External references set after init
        self._library = None
        self._queue = None
        self._sing_handler = None

        self._build_ui()
        self._connect_signals()

        # Refresh timer
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_ui)
        self._refresh_timer.start(2000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Now Playing ──
        np_group = QGroupBox("Now Playing")
        np_layout = QVBoxLayout(np_group)
        self._now_playing_label = QLabel("Nothing playing")
        self._now_playing_label.setFont(QFont("", 11, QFont.Bold))
        self._now_playing_label.setWordWrap(True)
        np_layout.addWidget(self._now_playing_label)

        self._lyrics_display = QTextEdit()
        self._lyrics_display.setReadOnly(True)
        self._lyrics_display.setMaximumHeight(80)
        self._lyrics_display.setPlaceholderText("Lyrics will appear here...")
        np_layout.addWidget(self._lyrics_display)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m sec")
        self._progress_bar.setValue(0)
        np_layout.addWidget(self._progress_bar)

        # Control buttons
        ctrl_layout = QHBoxLayout()
        self._btn_sing = QPushButton("♪ Sing!")
        self._btn_sing.setObjectName("primaryBtn")
        self._btn_skip = QPushButton("Skip")
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setObjectName("secondaryBtn")
        ctrl_layout.addWidget(self._btn_sing)
        ctrl_layout.addWidget(self._btn_skip)
        ctrl_layout.addWidget(self._btn_stop)
        ctrl_layout.addStretch()

        self._auto_pick_cb = QCheckBox("Auto-pick when queue empty")
        self._auto_pick_cb.setChecked(True)
        ctrl_layout.addWidget(self._auto_pick_cb)
        np_layout.addLayout(ctrl_layout)

        layout.addWidget(np_group)

        # ── Splitter: Queue | Library ──
        splitter = QSplitter(Qt.Horizontal)

        # Queue panel
        queue_group = QGroupBox("Queue")
        q_layout = QVBoxLayout(queue_group)
        self._queue_list = QListWidget()
        self._queue_list.setMaximumHeight(200)
        q_layout.addWidget(self._queue_list)

        q_btn_layout = QHBoxLayout()
        self._btn_clear_queue = QPushButton("Clear Queue")
        q_btn_layout.addWidget(self._btn_clear_queue)
        q_btn_layout.addStretch()
        q_layout.addLayout(q_btn_layout)
        splitter.addWidget(queue_group)

        # Library panel
        lib_group = QGroupBox("Song Library")
        lib_layout = QVBoxLayout(lib_group)

        # Search
        search_layout = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search songs...")
        search_layout.addWidget(self._search_input)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "Prepared", "Unprepared"])
        self._filter_combo.setMinimumWidth(92)
        self._filter_combo.setMaximumWidth(140)
        search_layout.addWidget(self._filter_combo)
        lib_layout.addLayout(search_layout)

        self._library_list = QListWidget()
        lib_layout.addWidget(self._library_list)

        lib_btn_layout = QHBoxLayout()
        self._btn_add_song = QPushButton("Add Song")
        self._btn_add_song.setObjectName("primaryBtn")
        self._btn_remove_song = QPushButton("Remove")
        self._btn_process_song = QPushButton("Process")
        self._btn_process_song.setToolTip("Run karaoke pipeline on selected song")
        self._btn_queue_song = QPushButton("Queue")
        lib_btn_layout.addWidget(self._btn_add_song)
        lib_btn_layout.addWidget(self._btn_remove_song)
        lib_btn_layout.addWidget(self._btn_process_song)
        lib_btn_layout.addWidget(self._btn_queue_song)
        lib_layout.addLayout(lib_btn_layout)

        # Stats
        self._stats_label = QLabel("Library: 0 songs (0 prepared)")
        apply_status_style(self._stats_label, role="muted")
        lib_layout.addWidget(self._stats_label)

        splitter.addWidget(lib_group)
        layout.addWidget(splitter, stretch=1)

        # Processing status
        self._status_label = QLabel("")
        apply_status_style(self._status_label, role="muted")
        layout.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._btn_sing.clicked.connect(self._on_sing_click)
        self._btn_skip.clicked.connect(self._on_skip_click)
        self._btn_stop.clicked.connect(self._on_stop_click)
        self._btn_add_song.clicked.connect(self._on_add_song)
        self._btn_remove_song.clicked.connect(self._on_remove_song)
        self._btn_process_song.clicked.connect(self._on_process_song)
        self._btn_queue_song.clicked.connect(self._on_queue_song)
        self._btn_clear_queue.clicked.connect(self._on_clear_queue)
        self._auto_pick_cb.toggled.connect(self._on_auto_pick_toggled)
        self._search_input.textChanged.connect(self._refresh_library)
        self._filter_combo.currentIndexChanged.connect(self._refresh_library)
        self._library_list.itemDoubleClicked.connect(self._on_library_double_click)

    # ------------------------------------------------------------------
    # External wiring (called after init by MainWindow)
    # ------------------------------------------------------------------

    def set_sing_system(self, library, queue, handler):
        """Connect the sing subsystem components."""
        self._library = library
        self._queue = queue
        self._sing_handler = handler
        self._queue.on_queue_changed = self._refresh_queue
        self._refresh_ui()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @Slot()
    def _on_sing_click(self):
        if not self._sing_handler:
            return
        # If a library item is selected, queue it; otherwise auto-pick
        item = self._library_list.currentItem()
        if item and item.data(Qt.UserRole):
            song_id = item.data(Qt.UserRole)
            reply = self._sing_handler.handle(
                self._library.get_song(song_id).title if self._library.get_song(song_id) else "",
                "gui_user"
            )
        else:
            reply = self._sing_handler.handle("", "gui_user")
        self._status_label.setText(reply or "")

    @Slot()
    def _on_skip_click(self):
        if self._sing_handler:
            reply = self._sing_handler.handle("skip", "gui_user")
            self._status_label.setText(reply or "")

    @Slot()
    def _on_stop_click(self):
        if self._sing_handler:
            reply = self._sing_handler.handle("stop", "gui_user")
            self._status_label.setText(reply or "")

    @Slot()
    def _on_add_song(self):
        if not self._library:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Songs to Library",
            "", "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )
        for path in paths:
            title = Path(path).stem.replace("_", " ").replace("-", " ").title()
            # Ask for title and artist
            title, ok = QInputDialog.getText(
                self, "Song Title", "Enter song title:",
                text=title
            )
            if not ok or not title:
                continue
            artist, _ = QInputDialog.getText(
                self, "Artist", "Enter artist (optional):"
            )
            # Ask for mood/genre tags
            tags_str, _ = QInputDialog.getText(
                self, "Tags",
                "Enter mood/genre tags (comma-separated, e.g. upbeat, pop, happy):"
            )
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
            self._library.add_song(path, title, artist=artist or "", tags=tags)

        self._refresh_library()

    @Slot()
    def _on_remove_song(self):
        item = self._library_list.currentItem()
        if not item or not self._library:
            return
        song_id = item.data(Qt.UserRole)
        entry = self._library.get_song(song_id)
        if not entry:
            return
        confirm = QMessageBox.question(
            self, "Remove Song",
            f"Remove \"{entry.title}\" from the library?\nThis deletes all processed files.",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self._library.remove_song(song_id)
            self._refresh_library()

    @Slot()
    def _on_process_song(self):
        """Run the karaoke pipeline on the selected song."""
        item = self._library_list.currentItem()
        if not item or not self._sing_handler:
            return
        song_id = item.data(Qt.UserRole)
        entry = self._library.get_song(song_id)
        if not entry:
            return
        if entry.prepared:
            self._status_label.setText(f"'{entry.title}' is already prepared.")
            return

        self._status_label.setText(f"Processing '{entry.title}'... (this may take a few minutes)")
        self._btn_process_song.setEnabled(False)

        import threading
        def _process():
            try:
                handler = self._sing_handler
                reply = handler.handle(entry.title, "gui_user")
                self._status_label.setText(reply or "Done!")
            except Exception as exc:
                self._status_label.setText(f"Error: {exc}")
            finally:
                self._btn_process_song.setEnabled(True)
                self._refresh_library()

        threading.Thread(target=_process, daemon=True, name="revia-sing-process").start()

    @Slot()
    def _on_queue_song(self):
        item = self._library_list.currentItem()
        if not item or not self._queue:
            return
        song_id = item.data(Qt.UserRole)
        entry = self._library.get_song(song_id)
        if entry:
            from app.sing_queue import PickMode
            self._queue.add_by_id(song_id, requested_by="gui_user",
                                   pick_mode=PickMode.REQUEST)
            self._status_label.setText(f"Queued: {entry.title}")

    @Slot()
    def _on_clear_queue(self):
        if self._queue:
            self._queue.clear()

    @Slot(bool)
    def _on_auto_pick_toggled(self, checked):
        if self._queue:
            self._queue.auto_pick_enabled = checked

    @Slot(QListWidgetItem)
    def _on_library_double_click(self, item):
        """Double-click to queue and play."""
        if not self._sing_handler:
            return
        song_id = item.data(Qt.UserRole)
        entry = self._library.get_song(song_id) if self._library else None
        if entry:
            reply = self._sing_handler.handle(entry.title, "gui_user")
            self._status_label.setText(reply or "")

    # ------------------------------------------------------------------
    # Refresh / display
    # ------------------------------------------------------------------

    def _refresh_ui(self):
        self._refresh_library()
        self._refresh_queue()
        self._refresh_now_playing()

    def _refresh_library(self):
        self._library_list.clear()
        if not self._library:
            return

        search = self._search_input.text().lower().strip()
        filter_mode = self._filter_combo.currentText()

        songs = self._library.list_songs()
        for song in songs:
            # Apply filter
            if filter_mode == "Prepared" and not song.prepared:
                continue
            if filter_mode == "Unprepared" and song.prepared:
                continue
            # Apply search
            if search and search not in song.title.lower() and search not in song.artist.lower():
                continue

            status = "✓" if song.prepared else "○"
            artist_str = f" — {song.artist}" if song.artist else ""
            tags_str = f"  [{', '.join(song.tags[:3])}]" if song.tags else ""
            label = f"{status} {song.title}{artist_str}{tags_str}"

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, song.song_id)
            if song.prepared:
                item.setForeground(QColor("#88DD88"))
            else:
                item.setForeground(QColor("#AAAAAA"))
            self._library_list.addItem(item)

        self._stats_label.setText(
            f"Library: {self._library.song_count} songs "
            f"({self._library.prepared_count} prepared)"
        )

    def _refresh_queue(self):
        self._queue_list.clear()
        if not self._queue:
            return
        items = self._queue.peek()
        for i, qi in enumerate(items, 1):
            by = f" ({qi.requested_by})" if qi.requested_by else ""
            mode = ""
            if qi.pick_mode.value == "mood":
                mode = " [mood]"
            elif qi.pick_mode.value == "random":
                mode = " [random]"
            label = f"{i}. {qi.title}{by}{mode}"
            self._queue_list.addItem(label)

        if not items:
            placeholder = QListWidgetItem("Queue is empty — use !sing or click Sing!")
            placeholder.setForeground(QColor("#666666"))
            self._queue_list.addItem(placeholder)

    def _refresh_now_playing(self):
        if not self._queue:
            return
        np = self._queue.now_playing
        if np:
            by = f" (requested by {np.requested_by})" if np.requested_by else ""
            self._now_playing_label.setText(f"♪ {np.title}{by}")
        else:
            self._now_playing_label.setText("Nothing playing")

    def update_lyrics(self, line_index: int, lyric_text: str):
        """Called externally when lyrics update during playback."""
        self._lyrics_display.setText(lyric_text)

    def update_state(self, state: str):
        """Called when sing pipeline state changes."""
        self._status_label.setText(f"Status: {state}")
        if state == "playing":
            apply_status_style(self._now_playing_label, role="success")
        elif state == "idle":
            clear_status_role(self._now_playing_label)
            self._refresh_now_playing()
        elif state == "error":
            apply_status_style(self._now_playing_label, role="error")
