import subprocess, sys, os
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QComboBox, QGroupBox, QCheckBox,
    QPushButton, QTableWidget, QTableWidgetItem, QListWidget,
)
from PySide6.QtCore import QTimer, QProcess
from PySide6.QtGui import QFont

from app.backend_sync_client import BackendSyncClient
from app.ui_async import UiThreadBridge
from app.ui_status import apply_status_style, clear_status_role

logger = logging.getLogger(__name__)


class SystemTab(QScrollArea):
    def __init__(self, event_bus, client, theme_mgr, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.theme_mgr = theme_mgr
        self._core_process = None
        self._server_starting = False
        self._auto_start_attempted = False
        self._neural_refresh_inflight = False
        self._bg = UiThreadBridge(self)
        self._sync_client = BackendSyncClient(event_bus)
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
            "This is the primary REVIA runtime and will auto-connect once ready."
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

        # Architecture monitor
        arch_group = QGroupBox("Architecture Monitor")
        arch_group.setObjectName("settingsGroup")
        ag = QVBoxLayout(arch_group)

        self.arch_overall = QLabel("Overall: Unknown")
        self.arch_overall.setFont(QFont("Consolas", 9, QFont.Bold))
        self.arch_overall.setObjectName("metricLabel")
        ag.addWidget(self.arch_overall)

        self.arch_labels = {}
        module_rows = [
            ("core_reasoning", "Core Reasoning Engine"),
            ("emotion", "Emotion Simulation"),
            ("voice", "Voice I/O"),
            ("vision", "Vision Perception"),
            ("tools", "Tools / Plugin Controller"),
            ("memory", "Long-Term Memory (RAG)"),
            ("personality", "Personality Layer"),
            ("monitoring", "Real-Time Monitoring"),
        ]
        for key, title in module_rows:
            row = QHBoxLayout()
            title_lbl = QLabel(title)
            title_lbl.setFont(QFont("Segoe UI", 9))
            row.addWidget(title_lbl, stretch=1)
            status_lbl = QLabel("Checking...")
            status_lbl.setFont(QFont("Consolas", 8))
            status_lbl.setObjectName("metricLabel")
            row.addWidget(status_lbl)
            self.arch_labels[key] = status_lbl
            ag.addLayout(row)

        layout.addWidget(arch_group)

        # Theme
        theme_group = QGroupBox("Appearance")
        theme_group.setObjectName("settingsGroup")
        tg = QFormLayout(theme_group)
        self.theme_combo = QComboBox()
        self._load_theme_combo()
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
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

        # Neural Refiner status
        nr_row = QHBoxLayout()
        self.refiner_label = QLabel("Neural Refiner: ---")
        self.refiner_label.setFont(QFont("Consolas", 8))
        nr_row.addWidget(self.refiner_label, stretch=1)
        ng.addLayout(nr_row)

        # Parallel Pipeline status
        pp_row = QHBoxLayout()
        self.pipeline_label = QLabel("Parallel Pipeline: ---")
        self.pipeline_label.setFont(QFont("Consolas", 8))
        pp_row.addWidget(self.pipeline_label, stretch=1)
        ng.addLayout(pp_row)

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
        if hasattr(self.event_bus, "ui_theme_changed"):
            self.event_bus.ui_theme_changed.connect(self._on_theme_changed)

    # --- Core server management ---

    def auto_start_on_launch(self):
        if self._auto_start_attempted:
            return
        self._auto_start_attempted = True
        self.event_bus.log_entry.emit("[Core] Auto-starting core server on UI launch.")
        QTimer.singleShot(150, self._start_core_server)

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

    def _resolve_python_exe(self):
        project_root = Path(__file__).resolve().parents[3]
        candidates = [
            project_root / ".venv" / "Scripts" / "python.exe",
            project_root / ".venv" / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return sys.executable

    def _find_port_holders(self, port):
        """Return {pid: {states}} for any process using local `port`. Windows-only."""
        if sys.platform != "win32":
            return {}
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"],
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception as e:
            self.event_bus.log_entry.emit(
                f"[Core] netstat lookup failed: {type(e).__name__}: {e}"
            )
            return {}

        holders = {}
        port_suffix = f":{port}"
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            proto = parts[0].upper()
            if proto not in {"TCP", "UDP"}:
                continue
            local_addr = parts[1]
            if not local_addr.endswith(port_suffix):
                continue
            pid_str = parts[-1]
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == os.getpid():
                continue
            state = parts[-2].upper() if proto == "TCP" and len(parts) >= 5 else "BOUND"
            holders.setdefault(pid, set()).add(state)
        return holders

    @staticmethod
    def _describe_port_holders(holders):
        if not holders:
            return "none"
        details = []
        for pid in sorted(holders):
            states = ", ".join(sorted(holders[pid]))
            details.append(f"PID {pid} [{states}]")
        return "; ".join(details)

    def _find_port_listener_pids(self, port):
        holders = self._find_port_holders(port)
        listeners = set()
        for pid, states in holders.items():
            if "LISTENING" in states or "BOUND" in states:
                listeners.add(pid)
        return listeners

    def _kill_port_holder(self, port):
        """Kill any process using this port and verify the port released.

        Surfaces every failure to the Logs tab so the user can see exactly why
        a previous attempt left a zombie behind. Retries up to 3 times with a
        short wait between each — Windows occasionally needs a second pass to
        actually let go of the socket.
        """
        if sys.platform != "win32":
            return False

        import time as _time
        killed = False
        for attempt in range(1, 4):
            holders = self._find_port_holders(port)
            if not holders:
                if killed:
                    self.event_bus.log_entry.emit(
                        f"[Core] Port {port} confirmed free after {attempt - 1} kill(s)"
                    )
                return killed
            self.event_bus.log_entry.emit(
                f"[Core] Port {port} held by {self._describe_port_holders(holders)} "
                f"(cleanup pass {attempt}/3)"
            )

            for pid in sorted(holders):
                try:
                    result = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        self.event_bus.log_entry.emit(
                            f"[Core] Killed PID {pid} on port {port} (attempt {attempt})"
                        )
                        killed = True
                    else:
                        err = (result.stderr or result.stdout or "").strip()
                        self.event_bus.log_entry.emit(
                            f"[Core] taskkill PID {pid} returned {result.returncode}: {err}"
                        )
                except Exception as e:
                    self.event_bus.log_entry.emit(
                        f"[Core] taskkill PID {pid} failed: {type(e).__name__}: {e}"
                    )

            # Give the OS a moment to release the listening socket before re-checking.
            _time.sleep(0.6)

        # Fell through all attempts — port still held.
        final_holders = self._find_port_holders(port)
        if not final_holders:
            if killed:
                self.event_bus.log_entry.emit(
                    f"[Core] Port {port} confirmed free after 3 kill(s)"
                )
            return killed
        self.event_bus.log_entry.emit(
            f"[Core] Port {port} STILL held by {self._describe_port_holders(final_holders)} "
            "after 3 kill attempts. Manual cleanup required."
        )
        return killed

    def _start_core_server(self):
        if self._core_process and self._core_process.state() == QProcess.Running:
            self.server_status.setText("Server: Already running")
            apply_status_style(self.server_status, "color: #ccaa00;")
            return

        script = self._find_core_script()
        if not script:
            self.server_status.setText("Server: core_server.py not found!")
            apply_status_style(self.server_status, "color: #cc3040;")
            return

        host = self.core_host.text().strip() or "127.0.0.1"
        rest = self.rest_port.value()
        ws = self.ws_port.value()

        # Disable button immediately to prevent double-clicks; show checking state.
        self.start_server_btn.setEnabled(False)
        self.server_status.setText("Server: Checking...")
        apply_status_style(self.server_status, "color: #ccaa00;")

        # Offload the pre-flight "already running?" HTTP check to the background.
        def _check():
            already_running = False
            running_version = ""
            running_has_arch = False
            try:
                r = self.client._session_get(
                    f"http://{host}:{rest}/api/status", timeout=1
                )
                if r.ok:
                    already_running = True
                    data = (
                        r.json()
                        if "application/json" in r.headers.get("content-type", "")
                        else {}
                    )
                    running_version = str(data.get("version", "")).strip()
                    running_has_arch = isinstance(
                        data.get("architecture"), dict
                    ) and bool(data.get("architecture"))
            except Exception as e:
                logger.debug(f"Error checking running core version: {e}")
            self._bg.dispatch(
                lambda: self._launch_core_or_connect(
                    script, host, rest, ws,
                    already_running, running_version, running_has_arch,
                )
            )

        self.client._executor.submit(_check)

    def _launch_core_or_connect(
        self, script, host, rest, ws,
        already_running, running_version, running_has_arch,
    ):
        """Continue _start_core_server on the main thread after the pre-flight check."""
        if already_running:
            rest_listener_pids = self._find_port_listener_pids(rest)
            ws_listener_pids = self._find_port_listener_pids(ws)
            shared_listener = bool(rest_listener_pids & ws_listener_pids)

            # If the running core is legacy (no architecture telemetry),
            # replace it so monitor + module status stay accurate.
            if running_has_arch and shared_listener:
                self.server_status.setText(
                    "Server: Already running externally - connecting..."
                )
                apply_status_style(self.server_status, "color: #00aa40;")
                self.event_bus.log_entry.emit(
                    f"[Core] Found existing server on port {rest} "
                    f"(v{running_version or '?'}), connecting"
                )
                self.start_server_btn.setEnabled(True)
                self._connect_core()
                return

            if running_has_arch:
                self.server_status.setText("Server: Existing core is degraded - restarting...")
                apply_status_style(self.server_status, "color: #ccaa00;")
                rest_detail = self._describe_port_holders(self._find_port_holders(rest))
                ws_detail = self._describe_port_holders(self._find_port_holders(ws))
                self.event_bus.log_entry.emit(
                    f"[Core] Existing server on port {rest} answered REST but is not "
                    f"healthy enough to reuse. REST listeners: {rest_detail}; "
                    f"WS listeners: {ws_detail}. Replacing with current core."
                )
            else:
                self.server_status.setText("Server: Legacy core detected - upgrading...")
                apply_status_style(self.server_status, "color: #ccaa00;")
                self.event_bus.log_entry.emit(
                    f"[Core] Existing server on port {rest} is legacy "
                    f"(v{running_version or '?'}). Replacing with current core."
                )

        # Kill anything already on these ports that isn't responding.
        killed = self._kill_port_holder(rest)
        killed |= self._kill_port_holder(ws)
        if killed:
            self.event_bus.log_entry.emit(
                "[Core] Waiting for port cleanup..."
            )
            # Brief pause so the OS releases the port before Flask tries to bind.
            # Without this, the new process can fail with "Address already in use"
            # immediately after a taskkill, causing a silent crash → timeout.
            import time as _time
            _time.sleep(1.0)

        rest_holders = self._find_port_holders(rest)
        ws_holders = self._find_port_holders(ws)
        if rest_holders or ws_holders:
            if rest_holders:
                self.event_bus.log_entry.emit(
                    f"[Core] REST port {rest} still occupied by "
                    f"{self._describe_port_holders(rest_holders)}"
                )
            if ws_holders:
                self.event_bus.log_entry.emit(
                    f"[Core] WS port {ws} still occupied by "
                    f"{self._describe_port_holders(ws_holders)}"
                )
            self.server_status.setText("Server: Startup blocked by occupied port")
            apply_status_style(self.server_status, "color: #cc3040;")
            self.start_server_btn.setEnabled(True)
            self.stop_server_btn.setEnabled(False)
            return

        self._core_process = QProcess(self)
        self._core_process.setWorkingDirectory(str(Path(script).parent))
        self._core_process.setProcessEnvironment(self._build_env(rest, ws))
        # Merge stderr into stdout so all Python output (including tracebacks)
        # is captured by _on_server_stdout. Without this, crash tracebacks
        # can be lost if stderr is read before stdout.
        from PySide6.QtCore import QProcess as _QProc
        self._core_process.setProcessChannelMode(_QProc.ProcessChannelMode.MergedChannels)
        self._core_process.readyReadStandardOutput.connect(self._on_server_stdout)
        self._core_process.finished.connect(self._on_server_finished)
        self._core_process.errorOccurred.connect(self._on_server_error)
        python_exe = self._resolve_python_exe()

        self.event_bus.log_entry.emit(f"[Core] Python: {python_exe}")
        self.event_bus.log_entry.emit(f"[Core] Script: {script}")
        self.event_bus.log_entry.emit(f"[Core] Launching: {python_exe} {script}")
        # Pass -u flag for unbuffered Python output so startup messages
        # and any tracebacks appear in the Logs tab immediately.
        self._core_process.start(python_exe, ["-u", script])

        if not self._core_process.waitForStarted(5000):
            err = self._core_process.errorString()
            self.server_status.setText(f"Server: Failed - {err}")
            apply_status_style(self.server_status, "color: #cc3040;")
            self.event_bus.log_entry.emit(f"[Core] Failed to start: {err}")
            self.start_server_btn.setEnabled(True)
            return

        self.server_status.setText("Server: Starting...")
        apply_status_style(self.server_status, "color: #ccaa00;")
        self.start_server_btn.setEnabled(False)
        self.stop_server_btn.setEnabled(True)
        self._connect_attempts = 0
        self._server_starting = True

        # Give the server more time before first check — Redis timeout,
        # model loading, and neural refiner init can take 10-15s.
        QTimer.singleShot(5000, self._auto_connect_after_start)

    def _build_env(self, rest_port, ws_port):
        from PySide6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("REVIA_REST_PORT", str(rest_port))
        env.insert("REVIA_WS_PORT", str(ws_port))
        env.insert("PYTHONUNBUFFERED", "1")
        return env

    def _auto_connect_after_start(self):
        self._connect_attempts = getattr(self, "_connect_attempts", 0) + 1
        host = self.core_host.text().strip() or "127.0.0.1"
        rest = self.rest_port.value()
        # If the process we launched has already exited (_server_starting cleared
        # by _on_server_finished), stop polling — the exit status was already
        # shown by _on_server_finished. Without this check, _core_process being
        # None causes the crash to go undetected and we poll all 15 attempts.
        if not self._server_starting:
            return
        # Check if process died before spending time on an HTTP probe.
        if self._core_process and self._core_process.state() == QProcess.NotRunning:
            self.server_status.setText("Server: Failed to start (check Logs)")
            apply_status_style(self.server_status, "color: #cc3040;")
            self.start_server_btn.setEnabled(True)
            self.stop_server_btn.setEnabled(False)
            self._server_starting = False
            return
        if self._connect_attempts > 30:
            self.server_status.setText("Server: Timeout waiting for REST (check Logs)")
            apply_status_style(self.server_status, "color: #cc3040;")
            self._server_starting = False
            return

        attempts = self._connect_attempts  # snapshot for the closure

        def _check():
            ok = False
            ver = "?"
            err_detail = ""
            try:
                # Use a fresh session for health checks to avoid stale
                # Keep-Alive connections from before the server restarted.
                import requests as _requests
                r = _requests.get(
                    f"http://{host}:{rest}/api/status", timeout=2
                )
                if r.ok:
                    ok = True
                    ver = r.json().get("version", "?")
            except Exception as e:
                err_detail = f"{type(e).__name__}: {e}"
                logger.debug(f"Error checking core status: {err_detail}")
                # Emit first attempt, and every 5th after, so the Logs tab
                # shows the real failure reason (e.g. ConnectionRefusedError).
                if attempts <= 1 or attempts % 5 == 0:
                    self._bg.dispatch(
                        lambda d=err_detail, a=attempts:
                            self.event_bus.log_entry.emit(
                                f"[Core] Health check #{a} failed: {d}"
                            )
                    )

            def _apply(ok=ok, ver=ver, a=attempts):
                if ok:
                    self._server_starting = False
                    self.server_status.setText(f"Server: Running (v{ver})")
                    apply_status_style(self.server_status, "color: #00aa40;")
                    self._connect_core()
                else:
                    self.server_status.setText(f"Server: Starting ({a * 2}s...)")
                    apply_status_style(self.server_status, "color: #ccaa00;")
                    QTimer.singleShot(2000, self._auto_connect_after_start)

            self._bg.dispatch(_apply)

        self.client._executor.submit(_check)

    def _on_server_stdout(self):
        if self._core_process:
            data = self._core_process.readAllStandardOutput().data().decode(
                errors="replace"
            ).strip()
            if data:
                for line in data.splitlines():
                    line = line.strip()
                    if line:
                        self.event_bus.log_entry.emit(f"[Core] {line}")
                        # Surface key milestones in the status label
                        if "Ready" in line or "REST server on" in line:
                            self.server_status.setText("Server: Running")
                            apply_status_style(self.server_status, "color: #00aa40;")
                        elif "Traceback" in line or "Error" in line:
                            self.server_status.setText(f"Server: {line[:60]}")
                            apply_status_style(self.server_status, "color: #cc3040;")

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
        apply_status_style(self.server_status, "color: #cc3040;")
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
        apply_status_style(self.server_status, "color: #cc3040;")
        self.event_bus.log_entry.emit(
            f"[Core] Process exited with code {exit_code}{detail}"
        )
        self.start_server_btn.setEnabled(True)
        self.stop_server_btn.setEnabled(False)
        self._core_process = None
        self._server_starting = False

    def _stop_core_server(self):
        stopped = False
        if self._core_process and self._core_process.state() == QProcess.Running:
            self._disconnect_core()
            pid = self._core_process.processId()
            if pid and sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=5,
                    )
                    stopped = True
                except Exception as e:
                    logger.debug(f"taskkill failed for {pid}, using kill(): {e}")
                    self._core_process.kill()
                    stopped = True
            else:
                self._core_process.kill()
                stopped = True
            self._core_process.waitForFinished(3000)
            self._core_process = None
        else:
            # Server was started externally — try to kill by port
            self._disconnect_core()
            rest_port = self.rest_port.value()
            killed = self._kill_port_holder(rest_port)
            if killed:
                stopped = True
            else:
                # Fallback: try the /api/shutdown endpoint
                try:
                    host = self.core_host.text().strip() or "127.0.0.1"
                    r = self.client._session_post(
                        f"http://{host}:{rest_port}/api/shutdown",
                        json={},
                        timeout=3,
                    )
                    if r.ok:
                        stopped = True
                except Exception:
                    pass

        if stopped:
            self.server_status.setText("Server: Stopped")
            clear_status_role(self.server_status)
        else:
            self.server_status.setText("Server: Stop failed (not found)")
            apply_status_style(self.server_status, "color: #cc3040;")
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
        apply_status_style(self.core_status, "color: #ccaa00;")
        self.core_connect_btn.setEnabled(False)

        if self.client.connected:
            self.client.ws.close()

        from PySide6.QtCore import QUrl
        self.client.ws.open(QUrl(self.client.WS_URL))

        QTimer.singleShot(300, self._check_core_rest)

    def _check_core_rest(self):
        def _check():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/status", timeout=3
                )
                if r.ok:
                    ver = r.json().get("version", "?")
                    state = r.json().get("state", "Unknown")

                    def _apply(v=ver, s=state):
                        if self.client.ws_connected:
                            self.core_status.setText(
                                f"Status: Connected | v{v} | State: {s}"
                            )
                            apply_status_style(self.core_status, "color: #00aa40;")
                        else:
                            self.core_status.setText(
                                f"Status: REST online (WebSocket connecting) | v{v} | State: {s}"
                            )
                            apply_status_style(self.core_status, "color: #ccaa00;")
                        self.core_disconnect_btn.setEnabled(True)
                        self.core_connect_btn.setEnabled(True)
                else:
                    sc = r.status_code

                    def _apply(c=sc):
                        self.core_status.setText(f"Status: REST error ({c})")
                        apply_status_style(self.core_status, "color: #cc3040;")
                        self.core_connect_btn.setEnabled(True)

                self._bg.dispatch(_apply)
            except Exception as ex:
                err = str(ex)

                def _apply_err(e=err):
                    self.core_status.setText(f"Status: Unreachable - {e}")
                    apply_status_style(self.core_status, "color: #cc3040;")
                    self.core_connect_btn.setEnabled(True)

                self._bg.dispatch(_apply_err)

        self.client._executor.submit(_check)

    def _disconnect_core(self):
        self.client.ws.close()
        self.client.connected = False
        self.event_bus.connection_changed.emit(False)
        self.core_status.setText("Status: Disconnected")
        clear_status_role(self.core_status)
        self.core_disconnect_btn.setEnabled(False)

    def _ping_core(self):
        self.core_status.setText("Status: Pinging...")
        apply_status_style(self.core_status, "color: #ccaa00;")
        QTimer.singleShot(50, self._do_ping)

    def _do_ping(self):
        import time
        t0 = time.perf_counter()

        def _check():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/status", timeout=3
                )
                latency = (time.perf_counter() - t0) * 1000
                if r.ok:
                    def _apply(lat=latency):
                        self.core_status.setText(
                            f"Status: Online | Ping: {lat:.0f} ms"
                        )
                        apply_status_style(self.core_status, "color: #00aa40;")
                else:
                    sc = r.status_code

                    def _apply(c=sc):
                        self.core_status.setText(f"Status: Error ({c})")
                        apply_status_style(self.core_status, "color: #cc3040;")

                self._bg.dispatch(_apply)
            except Exception as e:
                logger.debug(f"Error checking core status: {e}")
                self._bg.dispatch(lambda: (
                    self.core_status.setText("Status: Unreachable"),
                    apply_status_style(self.core_status, "color: #cc3040;"),
                ))

        self.client._executor.submit(_check)

    def _on_core_connection(self, connected):
        if connected:
            self.core_status.setText("Status: Connected (WebSocket live)")
            apply_status_style(self.core_status, "color: #00aa40;")
            self.core_disconnect_btn.setEnabled(True)
            # Enable stop button even for externally-started servers
            # (the stop handler can kill by port or /api/shutdown)
            self.stop_server_btn.setEnabled(True)
            self._refresh_plugins()
            self._refresh_neural()
        else:
            # Probe REST in the background avoids blocking the main thread
            # on what could be a 1-second timeout during a WS disconnect event.
            def _check():
                rest_ok = False
                try:
                    r = self.client._session_get(
                        f"{self.client.BASE_URL}/api/status", timeout=1
                    )
                    rest_ok = r.ok
                except Exception as e:
                    logger.debug(f"Error checking REST status: {e}")

                def _apply(ok=rest_ok):
                    if ok:
                        self.core_status.setText(
                            "Status: REST online (WebSocket reconnecting)"
                        )
                        apply_status_style(self.core_status, "color: #ccaa00;")
                        self.core_disconnect_btn.setEnabled(False)
                        self.arch_overall.setText("Overall: Degraded (WS offline)")
                        apply_status_style(self.arch_overall, "color: #ccaa00;")
                    else:
                        self.core_status.setText("Status: Disconnected")
                        apply_status_style(self.core_status, "color: #cc3040;")
                        self.core_disconnect_btn.setEnabled(False)
                        self.arch_overall.setText("Overall: Offline")
                        apply_status_style(self.arch_overall, "color: #cc3040;")
                        for lbl in self.arch_labels.values():
                            lbl.setText("OFFLINE")
                            apply_status_style(lbl, "color: #cc3040;")

                self._bg.dispatch(_apply)

            self.client._executor.submit(_check)

    def _load_theme_combo(self):
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        active = self.theme_mgr.current_theme
        for theme in self.theme_mgr.available_themes():
            self.theme_combo.addItem(theme.DisplayName, theme.ThemeId)
        index = self.theme_combo.findData(active)
        self.theme_combo.setCurrentIndex(max(index, 0))
        self.theme_combo.blockSignals(False)

    def _change_theme(self, _index=None):
        theme_id = self.theme_combo.currentData()
        if not theme_id:
            return
        applied = self.theme_mgr.apply_theme(theme_id)
        self._sync_client.publish_config_change(
            {
                "scope": "ui.theme",
                "action": "apply",
                "theme_id": applied,
            }
        )
        if hasattr(self.event_bus, "ui_theme_changed"):
            self.event_bus.ui_theme_changed.emit(applied)
        self.event_bus.log_entry.emit(f"[CoreSync] UI theme apply: {applied}")

    def _on_theme_changed(self, theme_id):
        index = self.theme_combo.findData(theme_id)
        if index < 0:
            self._load_theme_combo()
            index = self.theme_combo.findData(theme_id)
        if index >= 0 and index != self.theme_combo.currentIndex():
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(index)
            self.theme_combo.blockSignals(False)

    def _refresh_plugins(self):
        self.client.get_async(
            "/api/plugins",
            timeout=2,
            default=[],
            on_success=self._apply_plugins,
            on_error=lambda error, _detail=None: logger.warning(
                "Error refreshing plugins: %s", error
            ),
        )

    def _apply_plugins(self, plugins):
        plugins = list(plugins or [])
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
        apply_status_style(self.websearch_info, role="success" if enabled else "muted")

    def _refresh_neural(self):
        if self._neural_refresh_inflight:
            return
        if not (self.client.connected or self.client.rest_reachable):
            return

        self._neural_refresh_inflight = True
        cached_status = self.client.get_status_snapshot()

        def _work(cached_status=cached_status):
            data = self.client.get_neural() or {}
            ws_data = self.client.get_websearch_status() or {}
            status = dict(cached_status or {})
            if not status or not status.get("architecture"):
                status = self.client.get("/api/status", timeout=1) or status

            def _apply():
                self._neural_refresh_inflight = False
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
                    nr = data.get("neural_refiner", {})
                    if nr.get("available", False):
                        self.refiner_label.setText(
                            f"Neural Refiner: Active | Steps: {nr.get('step_count', 0)} "
                            f"| Avg Loss: {nr.get('avg_loss', 0):.4f} "
                            f"| Inference: {nr.get('last_inference_ms', 0):.1f} ms"
                        )
                    else:
                        self.refiner_label.setText(
                            "Neural Refiner: Not available (install PyTorch)"
                        )
                    pp = data.get("parallel_pipeline", {})
                    lanes = pp.get("lanes", {})
                    running = pp.get("any_running", False)
                    lane_status = " | ".join(
                        f"{k}: {v}" for k, v in lanes.items()
                    ) if lanes else "idle"
                    self.pipeline_label.setText(
                        f"Parallel Pipeline: {'ACTIVE' if running else 'Ready'} | {lane_status}"
                    )
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
                        apply_status_style(self.websearch_info, "color: #00aa40;")
                    else:
                        self.websearch_info.setText(
                            "Status: OFF"
                            + ("" if ddg_ok else " (install duckduckgo-search for full results)")
                        )
                        apply_status_style(self.websearch_info, "color: #888888;")
                arch = status.get("architecture", {})
                if arch:
                    self._update_architecture(arch)
                elif status:
                    self.arch_overall.setText("Overall: Partial (legacy core status)")
                    apply_status_style(self.arch_overall, "color: #ccaa00;")
                    core_lbl = self.arch_labels.get("core_reasoning")
                    if core_lbl:
                        core_lbl.setText("ONLINE | Core status endpoint reachable")
                        apply_status_style(core_lbl, "color: #00aa40;")
                    for key, lbl in self.arch_labels.items():
                        if key == "core_reasoning":
                            continue
                        lbl.setText("UNKNOWN | Upgrade core for module telemetry")
                        apply_status_style(lbl, "color: #808898;")

            self._bg.dispatch(_apply)

        try:
            self.client._executor.submit(_work)
        except Exception as exc:
            self._neural_refresh_inflight = False
            logger.debug(f"Failed to submit neural refresh: {exc}")

    def _on_telemetry(self, data):
        state = str(data.get("state", "Unknown"))
        llm = data.get("llm_connection", {}) or {}
        readiness = data.get("conversation_readiness", {}) or {}
        request_lifecycle = data.get("request_lifecycle", {}) or {}
        active_turn = request_lifecycle.get("active_turn", {}) or {}
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
        llm_state = str(llm.get("state", "Disconnected"))
        llm_detail = str(llm.get("detail", "")).strip()
        ready_flag = "Ready" if readiness.get("ready", False) else "Waiting"
        stage = str(active_turn.get("lifecycle_reason", "") or "").strip()
        state_text = f"{state} | Stage: {stage}" if stage else state
        self.core_status.setText(
            f"Status: {state_text} | LLM: {llm_state} | Conversation: {ready_flag}"
            + (f" | {llm_detail}" if llm_detail else "")
        )
        normalized_state = state.strip().lower()
        if llm_state == "Error" or normalized_state == "error":
            apply_status_style(self.core_status, "color: #cc3040;")
        elif normalized_state in {"thinking", "generating", "speaking", "listening", "cooldown"}:
            apply_status_style(self.core_status, "color: #ccaa00;")
        elif readiness.get("ready", False):
            apply_status_style(self.core_status, "color: #00aa40;")
        else:
            apply_status_style(self.core_status, "color: #ccaa00;")
        arch = data.get("architecture", {})
        if arch:
            self._update_architecture(arch)

    def _update_architecture(self, architecture):
        modules = architecture.get("modules", {})
        overall_ready = architecture.get("overall_ready")
        if overall_ready is None:
            self.arch_overall.setText("Overall: Unknown")
            apply_status_style(self.arch_overall, "color: #808898;")
        elif overall_ready:
            self.arch_overall.setText("Overall: Ready")
            apply_status_style(self.arch_overall, "color: #00aa40;")
        else:
            self.arch_overall.setText("Overall: Partial")
            apply_status_style(self.arch_overall, "color: #ccaa00;")

        for key, label in self.arch_labels.items():
            if key not in modules:
                label.setText("UNKNOWN")
                apply_status_style(label, "color: #808898;")
                continue
            module = modules.get(key, {})
            online = bool(module.get("online", False))
            detail = str(module.get("detail", "")).strip()
            if online:
                label.setText("ONLINE" + (f" | {detail}" if detail else ""))
                apply_status_style(label, "color: #00aa40;")
            else:
                label.setText("OFFLINE" + (f" | {detail}" if detail else ""))
                apply_status_style(label, "color: #cc3040;")
