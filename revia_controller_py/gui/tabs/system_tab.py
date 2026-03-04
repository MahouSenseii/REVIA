import subprocess, sys, os, signal
from pathlib import Path

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QComboBox, QGroupBox, QCheckBox,
    QPushButton, QTableWidget, QTableWidgetItem, QListWidget,
)
from PySide6.QtCore import QTimer, QProcess
from PySide6.QtGui import QFont


class SystemTab(QScrollArea):
    def __init__(self, event_bus, client, theme_mgr, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.theme_mgr = theme_mgr
        self._core_process = None
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("System")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # --- Core Server (launch / stop) ---
        server_group = QGroupBox("Core Server")
        server_group.setObjectName("settingsGroup")
        svl = QVBoxLayout(server_group)

        sv_info = QLabel(
            "Start the Python core server directly from the UI. "
            "It will auto-connect once ready."
        )
        sv_info.setFont(QFont("Segoe UI", 8))
        sv_info.setWordWrap(True)
        svl.addWidget(sv_info)

        self.server_status = QLabel("Server: Stopped")
        self.server_status.setFont(QFont("Consolas", 9))
        self.server_status.setObjectName("metricLabel")
        self.server_status.setWordWrap(True)
        svl.addWidget(self.server_status)

        sv_btn_row = QHBoxLayout()
        self.start_server_btn = QPushButton("Start Core Server")
        self.start_server_btn.setObjectName("primaryBtn")
        self.start_server_btn.clicked.connect(self._start_core_server)
        sv_btn_row.addWidget(self.start_server_btn)

        self.stop_server_btn = QPushButton("Stop Server")
        self.stop_server_btn.setObjectName("secondaryBtn")
        self.stop_server_btn.clicked.connect(self._stop_core_server)
        self.stop_server_btn.setEnabled(False)
        sv_btn_row.addWidget(self.stop_server_btn)
        svl.addLayout(sv_btn_row)

        layout.addWidget(server_group)

        # Core Connection
        core_group = QGroupBox("Core Connection")
        core_group.setObjectName("settingsGroup")
        cg = QVBoxLayout(core_group)

        addr_form = QFormLayout()
        addr_form.setContentsMargins(0, 0, 0, 0)

        self.core_host = QLineEdit("127.0.0.1")
        addr_form.addRow("Host:", self.core_host)

        ports_row = QHBoxLayout()
        self.rest_port = QSpinBox()
        self.rest_port.setRange(1, 65535)
        self.rest_port.setValue(8123)
        ports_row.addWidget(self.rest_port)
        ws_lbl = QLabel("WS:")
        ws_lbl.setFont(QFont("Segoe UI", 9))
        ports_row.addWidget(ws_lbl)
        self.ws_port = QSpinBox()
        self.ws_port.setRange(1, 65535)
        self.ws_port.setValue(8124)
        ports_row.addWidget(self.ws_port)
        addr_form.addRow("REST / WS:", ports_row)

        cg.addLayout(addr_form)

        self.core_status = QLabel("Status: Not connected")
        self.core_status.setFont(QFont("Consolas", 9))
        self.core_status.setObjectName("metricLabel")
        self.core_status.setWordWrap(True)
        cg.addWidget(self.core_status)

        core_btn_row = QHBoxLayout()
        self.core_connect_btn = QPushButton("Connect")
        self.core_connect_btn.setObjectName("primaryBtn")
        self.core_connect_btn.clicked.connect(self._connect_core)
        core_btn_row.addWidget(self.core_connect_btn)

        self.core_disconnect_btn = QPushButton("Disconnect")
        self.core_disconnect_btn.setObjectName("secondaryBtn")
        self.core_disconnect_btn.clicked.connect(self._disconnect_core)
        self.core_disconnect_btn.setEnabled(False)
        core_btn_row.addWidget(self.core_disconnect_btn)

        self.core_ping_btn = QPushButton("Ping")
        self.core_ping_btn.setObjectName("secondaryBtn")
        self.core_ping_btn.clicked.connect(self._ping_core)
        core_btn_row.addWidget(self.core_ping_btn)
        cg.addLayout(core_btn_row)

        layout.addWidget(core_group)

        # Theme
        theme_group = QGroupBox("Appearance")
        theme_group.setObjectName("settingsGroup")
        tg = QFormLayout(theme_group)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self._change_theme)
        tg.addRow("Theme:", self.theme_combo)
        layout.addWidget(theme_group)

        # Neural Modules
        neural_group = QGroupBox("Neural Modules")
        neural_group.setObjectName("settingsGroup")
        ng = QVBoxLayout(neural_group)

        en_row = QHBoxLayout()
        self.emotion_toggle = QCheckBox("EmotionNet")
        self.emotion_toggle.setChecked(True)
        self.emotion_toggle.toggled.connect(
            lambda on: self.client.toggle_neural("emotion_net", on)
        )
        en_row.addWidget(self.emotion_toggle)
        self.emotion_info = QLabel("Inference: --- ms | Output: ---")
        self.emotion_info.setFont(QFont("Consolas", 8))
        en_row.addWidget(self.emotion_info, stretch=1)
        ng.addLayout(en_row)

        rc_row = QHBoxLayout()
        self.router_toggle = QCheckBox("RouterClassifier")
        self.router_toggle.setChecked(True)
        self.router_toggle.toggled.connect(
            lambda on: self.client.toggle_neural("router_classifier", on)
        )
        rc_row.addWidget(self.router_toggle)
        self.router_info = QLabel("Inference: --- ms | Output: ---")
        self.router_info.setFont(QFont("Consolas", 8))
        rc_row.addWidget(self.router_info, stretch=1)
        ng.addLayout(rc_row)

        # --- Web Search (Internet Access) toggle ---
        ws_row = QHBoxLayout()
        self.websearch_toggle = QCheckBox("Internet Search (Web Access)")
        self.websearch_toggle.setChecked(False)
        self.websearch_toggle.setToolTip(
            "Allow Revia to look up real-time information via DuckDuckGo.\n"
            "When ON, Revia will search the web when you ask about current events.\n"
            "Requires: pip install duckduckgo-search"
        )
        self.websearch_toggle.toggled.connect(self._on_websearch_toggled)
        ws_row.addWidget(self.websearch_toggle)
        self.websearch_info = QLabel("Status: OFF")
        self.websearch_info.setFont(QFont("Consolas", 8))
        self.websearch_info.setObjectName("metricLabel")
        ws_row.addWidget(self.websearch_info, stretch=1)
        ng.addLayout(ws_row)

        layout.addWidget(neural_group)

        # Plugins
        plugin_group = QGroupBox("Plugins")
        plugin_group.setObjectName("settingsGroup")
        pg = QVBoxLayout(plugin_group)

        self.plugin_table = QTableWidget()
        self.plugin_table.setColumnCount(5)
        self.plugin_table.setHorizontalHeaderLabels(
            ["Enabled", "Name", "Category", "Status", "Error"]
        )
        self.plugin_table.horizontalHeader().setStretchLastSection(True)
        self.plugin_table.setMaximumHeight(180)
        pg.addWidget(self.plugin_table)

        refresh_btn = QPushButton("Refresh Plugins")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.clicked.connect(self._refresh_plugins)
        pg.addWidget(refresh_btn)

        layout.addWidget(plugin_group)

        # Batched Listening
        batch_group = QGroupBox("Batched Listening")
        batch_group.setObjectName("settingsGroup")
        bl = QFormLayout(batch_group)
        self.batch_size_label = QLabel("200 ms")
        bl.addRow("Batch Window:", self.batch_size_label)
        self.partial_len_label = QLabel("0 chars")
        bl.addRow("Partial Transcript:", self.partial_len_label)
        layout.addWidget(batch_group)

        # Training Data
        training_group = QGroupBox("Training Data (Safe Learning)")
        training_group.setObjectName("settingsGroup")
        td = QVBoxLayout(training_group)

        info = QLabel(
            "Collected feedback for offline fine-tuning.\n"
            "No autonomous retraining occurs."
        )
        info.setFont(QFont("Segoe UI", 8))
        info.setWordWrap(True)
        td.addWidget(info)

        counts_row = QHBoxLayout()
        self.routing_count = QLabel("Routing events: 0")
        self.emotion_count = QLabel("Emotion events: 0")
        counts_row.addWidget(self.routing_count)
        counts_row.addWidget(self.emotion_count)
        td.addLayout(counts_row)

        export_row = QHBoxLayout()
        export_routing = QPushButton("Export Routing JSONL")
        export_routing.setObjectName("secondaryBtn")
        export_emotion = QPushButton("Export Emotion JSONL")
        export_emotion.setObjectName("secondaryBtn")
        export_row.addWidget(export_routing)
        export_row.addWidget(export_emotion)
        td.addLayout(export_row)

        layout.addWidget(training_group)

        # Skills Library
        skills_group = QGroupBox("Skill Library")
        skills_group.setObjectName("settingsGroup")
        sk = QVBoxLayout(skills_group)
        sk_info = QLabel(
            "User-defined skills (templates/macros) "
            "executed deterministically."
        )
        sk_info.setFont(QFont("Segoe UI", 8))
        sk_info.setWordWrap(True)
        sk.addWidget(sk_info)

        self.skills_list = QListWidget()
        self.skills_list.setMaximumHeight(100)
        sk.addWidget(self.skills_list)

        add_skill_btn = QPushButton("Add Skill")
        add_skill_btn.setObjectName("secondaryBtn")
        sk.addWidget(add_skill_btn)

        layout.addWidget(skills_group)
        layout.addStretch()
        self.setWidget(container)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_neural)
        self.refresh_timer.start(3000)

        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.connection_changed.connect(self._on_core_connection)

    # --- Core server management ---

    def _find_core_script(self):
        candidates = [
            Path(__file__).resolve().parents[2] / ".." / "revia_core_py" / "core_server.py",
            Path(__file__).resolve().parents[3] / "revia_core_py" / "core_server.py",
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())
        home = Path.home() / "REVIA" / "revia_core_py" / "core_server.py"
        if home.exists():
            return str(home.resolve())
        return None

    def _kill_port_holder(self, port):
        """Kill any process already listening on this port (Windows).
        Returns True if anything was killed."""
        if sys.platform != "win32":
            return False
        killed = False
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"], text=True, timeout=5
            )
            for line in out.splitlines():
                if f":{port} " in line and "LISTEN" in line:
                    parts = line.split()
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid():
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", pid],
                            capture_output=True, timeout=5,
                        )
                        self.event_bus.log_entry.emit(
                            f"[Core] Killed stale process PID {pid} on port {port}"
                        )
                        killed = True
        except Exception:
            pass
        return killed

    def _start_core_server(self):
        if self._core_process and self._core_process.state() == QProcess.Running:
            self.server_status.setText("Server: Already running")
            self.server_status.setStyleSheet("color: #ccaa00;")
            return

        script = self._find_core_script()
        if not script:
            self.server_status.setText(
                "Server: core_server.py not found!"
            )
            self.server_status.setStyleSheet("color: #cc3040;")
            return

        rest = self.rest_port.value()
        ws = self.ws_port.value()

        # Check if a core is already running on this port (from terminal etc.)
        already_running = False
        try:
            import requests
            r = requests.get(
                f"http://127.0.0.1:{rest}/api/status", timeout=1
            )
            if r.ok:
                already_running = True
        except Exception:
            pass

        if already_running:
            self.server_status.setText(
                "Server: Already running externally - connecting..."
            )
            self.server_status.setStyleSheet("color: #00aa40;")
            self.event_bus.log_entry.emit(
                f"[Core] Found existing server on port {rest}, connecting"
            )
            self._connect_core()
            return

        # Kill anything already on these ports that isn't responding
        killed = self._kill_port_holder(rest)
        killed |= self._kill_port_holder(ws)
        if killed:
            import time
            time.sleep(1.5)

        self._core_process = QProcess(self)
        self._core_process.setWorkingDirectory(str(Path(script).parent))
        self._core_process.setProcessEnvironment(
            self._build_env(rest, ws)
        )
        self._core_process.readyReadStandardOutput.connect(
            self._on_server_stdout
        )
        self._core_process.readyReadStandardError.connect(
            self._on_server_stderr
        )
        self._core_process.finished.connect(self._on_server_finished)
        self._core_process.errorOccurred.connect(self._on_server_error)

        self.event_bus.log_entry.emit(
            f"[Core] Launching: {sys.executable} {script}"
        )
        self._core_process.start(sys.executable, [script])

        if not self._core_process.waitForStarted(5000):
            err = self._core_process.errorString()
            self.server_status.setText(f"Server: Failed - {err}")
            self.server_status.setStyleSheet("color: #cc3040;")
            self.event_bus.log_entry.emit(f"[Core] Failed to start: {err}")
            self.start_server_btn.setEnabled(True)
            return

        self.server_status.setText("Server: Starting...")
        self.server_status.setStyleSheet("color: #ccaa00;")
        self.start_server_btn.setEnabled(False)
        self.stop_server_btn.setEnabled(True)
        self._connect_attempts = 0

        QTimer.singleShot(2500, self._auto_connect_after_start)

    def _build_env(self, rest_port, ws_port):
        from PySide6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("REVIA_REST_PORT", str(rest_port))
        env.insert("REVIA_WS_PORT", str(ws_port))
        return env

    def _auto_connect_after_start(self):
        self._connect_attempts = getattr(self, '_connect_attempts', 0) + 1
        # Check if process died
        if self._core_process and self._core_process.state() == QProcess.NotRunning:
            self.server_status.setText("Server: Failed to start (check Logs)")
            self.server_status.setStyleSheet("color: #cc3040;")
            self.start_server_btn.setEnabled(True)
            self.stop_server_btn.setEnabled(False)
            return
        if self._connect_attempts > 15:
            self.server_status.setText("Server: Timeout waiting for REST")
            self.server_status.setStyleSheet("color: #cc3040;")
            return
        try:
            import requests
            r = requests.get(
                f"{self.client.BASE_URL}/api/status", timeout=2
            )
            if r.ok:
                data = r.json()
                ver = data.get("version", "?")
                self.server_status.setText(
                    f"Server: Running (v{ver})"
                )
                self.server_status.setStyleSheet("color: #00aa40;")
                self._connect_core()
                return
        except Exception:
            pass
        self.server_status.setText(
            f"Server: Starting ({self._connect_attempts * 2}s...)"
        )
        self.server_status.setStyleSheet("color: #ccaa00;")
        QTimer.singleShot(2000, self._auto_connect_after_start)

    def _on_server_stdout(self):
        if self._core_process:
            data = self._core_process.readAllStandardOutput().data().decode(
                errors="replace"
            ).strip()
            if data:
                self.event_bus.log_entry.emit(f"[Core] {data}")

    def _on_server_error(self, error):
        error_names = {
            0: "FailedToStart (exe not found or no permissions)",
            1: "Crashed",
            2: "Timedout",
            4: "WriteError",
            5: "ReadError",
            3: "UnknownError",
        }
        desc = error_names.get(error, f"Error code {error}")
        self.server_status.setText(f"Server: {desc}")
        self.server_status.setStyleSheet("color: #cc3040;")
        self.event_bus.log_entry.emit(f"[Core] Process error: {desc}")
        self.start_server_btn.setEnabled(True)
        self.stop_server_btn.setEnabled(False)

    def _on_server_stderr(self):
        if self._core_process:
            data = self._core_process.readAllStandardError().data().decode(
                errors="replace"
            ).strip()
            if data:
                for line in data.splitlines():
                    if "DeprecationWarning" not in line:
                        self.event_bus.log_entry.emit(f"[Core] {line}")

    def _on_server_finished(self, exit_code, exit_status):
        last_err = ""
        if self._core_process:
            err = self._core_process.readAllStandardError().data().decode(
                errors="replace"
            ).strip()
            out = self._core_process.readAllStandardOutput().data().decode(
                errors="replace"
            ).strip()
            if err:
                for line in err.splitlines():
                    self.event_bus.log_entry.emit(f"[Core:err] {line}")
                last_err = err.splitlines()[-1]
            if out:
                for line in out.splitlines():
                    self.event_bus.log_entry.emit(f"[Core:out] {line}")

        if exit_code != 0 and not last_err:
            last_err = "port may be in use or missing dependency"

        detail = f" ({last_err})" if last_err else ""
        self.server_status.setText(
            f"Server: Stopped (exit {exit_code}){detail}"
        )
        self.server_status.setStyleSheet("color: #cc3040;")
        self.event_bus.log_entry.emit(
            f"[Core] Process exited with code {exit_code}{detail}"
        )
        self.start_server_btn.setEnabled(True)
        self.stop_server_btn.setEnabled(False)
        self._core_process = None

    def _stop_core_server(self):
        if self._core_process and self._core_process.state() == QProcess.Running:
            self._disconnect_core()
            pid = self._core_process.processId()
            if pid and sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=5,
                    )
                except Exception:
                    self._core_process.kill()
            else:
                self._core_process.kill()
            self._core_process.waitForFinished(3000)
            self.server_status.setText("Server: Stopped")
            self.server_status.setStyleSheet("")
            self.start_server_btn.setEnabled(True)
            self.stop_server_btn.setEnabled(False)
            self._core_process = None
        else:
            self.server_status.setText("Server: Not running")
            self.start_server_btn.setEnabled(True)
            self.stop_server_btn.setEnabled(False)

    # --- Core connection ---

    def _connect_core(self):
        host = self.core_host.text().strip() or "127.0.0.1"
        rest = self.rest_port.value()
        ws = self.ws_port.value()

        self.client.BASE_URL = f"http://{host}:{rest}"
        self.client.WS_URL = f"ws://{host}:{ws}"

        self.core_status.setText("Status: Connecting...")
        self.core_status.setStyleSheet("color: #ccaa00;")
        self.core_connect_btn.setEnabled(False)

        if self.client.connected:
            self.client.ws.close()

        from PySide6.QtCore import QUrl
        self.client.ws.open(QUrl(self.client.WS_URL))

        QTimer.singleShot(300, self._check_core_rest)

    def _check_core_rest(self):
        try:
            import requests
            r = requests.get(
                f"{self.client.BASE_URL}/api/status", timeout=3
            )
            if r.ok:
                data = r.json()
                ver = data.get("version", "?")
                state = data.get("state", "Unknown")
                self.core_status.setText(
                    f"Status: Connected | v{ver} | State: {state}"
                )
                self.core_status.setStyleSheet("color: #00aa40;")
                self.core_disconnect_btn.setEnabled(True)
            else:
                self.core_status.setText(
                    f"Status: REST error ({r.status_code})"
                )
                self.core_status.setStyleSheet("color: #cc3040;")
        except Exception as e:
            self.core_status.setText(f"Status: Unreachable — {e}")
            self.core_status.setStyleSheet("color: #cc3040;")
        self.core_connect_btn.setEnabled(True)

    def _disconnect_core(self):
        self.client.ws.close()
        self.client.connected = False
        self.event_bus.connection_changed.emit(False)
        self.core_status.setText("Status: Disconnected")
        self.core_status.setStyleSheet("")
        self.core_disconnect_btn.setEnabled(False)

    def _ping_core(self):
        self.core_status.setText("Status: Pinging...")
        self.core_status.setStyleSheet("color: #ccaa00;")
        QTimer.singleShot(50, self._do_ping)

    def _do_ping(self):
        import time
        try:
            import requests
            t0 = time.perf_counter()
            r = requests.get(
                f"{self.client.BASE_URL}/api/status", timeout=3
            )
            latency = (time.perf_counter() - t0) * 1000
            if r.ok:
                self.core_status.setText(
                    f"Status: Online | Ping: {latency:.0f} ms"
                )
                self.core_status.setStyleSheet("color: #00aa40;")
            else:
                self.core_status.setText(f"Status: Error ({r.status_code})")
                self.core_status.setStyleSheet("color: #cc3040;")
        except Exception:
            self.core_status.setText("Status: Unreachable")
            self.core_status.setStyleSheet("color: #cc3040;")

    def _on_core_connection(self, connected):
        if connected:
            self.core_status.setText("Status: Connected (WebSocket live)")
            self.core_status.setStyleSheet("color: #00aa40;")
            self.core_disconnect_btn.setEnabled(True)
        else:
            self.core_status.setText("Status: Disconnected")
            self.core_status.setStyleSheet("color: #cc3040;")
            self.core_disconnect_btn.setEnabled(False)

    def _change_theme(self, theme):
        self.theme_mgr.apply_theme(theme.lower())

    def _refresh_plugins(self):
        plugins = self.client.get_plugins()
        self.plugin_table.setRowCount(len(plugins))
        for i, p in enumerate(plugins):
            cb = QCheckBox()
            cb.setChecked(p.get("enabled", False))
            name = p.get("name", "")
            cb.toggled.connect(
                lambda on, n=name: self.client.toggle_plugin(n, on)
            )
            self.plugin_table.setCellWidget(i, 0, cb)
            self.plugin_table.setItem(
                i, 1, QTableWidgetItem(p.get("name", ""))
            )
            self.plugin_table.setItem(
                i, 2, QTableWidgetItem(p.get("category", ""))
            )
            self.plugin_table.setItem(
                i, 3, QTableWidgetItem(p.get("status", ""))
            )
            self.plugin_table.setItem(
                i, 4, QTableWidgetItem(p.get("last_error", ""))
            )

    def _on_websearch_toggled(self, enabled: bool):
        self.client.toggle_websearch(enabled)
        self.websearch_info.setText("Status: ON" if enabled else "Status: OFF")
        self.websearch_info.setStyleSheet(
            "color: #00aa40;" if enabled else "color: #888888;"
        )

    def _refresh_neural(self):
        data = self.client.get_neural()
        if data:
            en = data.get("emotion_net", {})
            self.emotion_info.setText(
                f"Inference: {en.get('last_inference_ms', 0):.1f} ms "
                f"| Output: {en.get('last_output', '---')}"
            )
            rc = data.get("router_classifier", {})
            self.router_info.setText(
                f"Inference: {rc.get('last_inference_ms', 0):.1f} ms "
                f"| Output: {rc.get('last_output', '---')}"
            )
        # Sync web search toggle with actual server state
        ws_data = self.client.get_websearch_status()
        if ws_data:
            enabled = ws_data.get("enabled", False)
            self.websearch_toggle.blockSignals(True)
            self.websearch_toggle.setChecked(enabled)
            self.websearch_toggle.blockSignals(False)
            backend = ws_data.get("backend", "")
            ddg_ok = ws_data.get("ddg_available", False)
            if enabled:
                self.websearch_info.setText(
                    f"Status: ON | Backend: {backend}"
                )
                self.websearch_info.setStyleSheet("color: #00aa40;")
            else:
                self.websearch_info.setText(
                    "Status: OFF"
                    + ("" if ddg_ok else " (install duckduckgo-search for full results)")
                )
                self.websearch_info.setStyleSheet("color: #888888;")

    def _on_telemetry(self, data):
        emotion = data.get("emotion", {})
        router = data.get("router", {})
        self.emotion_info.setText(
            f"Inference: {emotion.get('inference_ms', 0):.1f} ms "
            f"| Output: {emotion.get('label', '---')}"
        )
        self.router_info.setText(
            f"Inference: {router.get('inference_ms', 0):.1f} ms "
            f"| Output: {router.get('mode', '---')}"
        )
