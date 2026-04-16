import sys
import os
import json
import logging
import subprocess
from pathlib import Path
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QFileDialog, QStackedWidget, QCheckBox,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import QProcess, QTimer

from app.ui_status import apply_status_style, clear_status_role

logger = logging.getLogger(__name__)

# Persisted model settings live alongside the top-level config.json
_SETTINGS_FILE = Path(__file__).resolve().parents[3] / "model_settings.json"


class ModelTab(QScrollArea):
    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self._llm_process = None
        self._loading = True  # guard: prevents saving while loading
        self._pending_source = None  # source to push once core connects
        self._llm_server_kind = ""
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Model Settings")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # --- Source type selector ---
        source_group = QGroupBox("Model Source")
        source_group.setObjectName("settingsGroup")
        sg = QFormLayout(source_group)

        self.source_type = QComboBox()
        self.source_type.addItems(["Local Model File", "Online API"])
        self.source_type.currentIndexChanged.connect(self._on_source_changed)
        sg.addRow("Source:", self.source_type)

        # Stacked widget: page 0 = local, page 1 = online
        self.source_stack = QStackedWidget()

        # -- Local model page --
        local_page = QWidget()
        lp = QVBoxLayout(local_page)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(6)

        file_row = QHBoxLayout()
        self.local_path = QLineEdit()
        self.local_path.setPlaceholderText(
            "Path to .gguf, .ggml, .bin, .safetensors, .onnx ..."
        )
        file_row.addWidget(self.local_path, stretch=1)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("browseBtn")
        browse_btn.clicked.connect(self._browse_model)
        file_row.addWidget(browse_btn)
        lp.addLayout(file_row)

        local_form = QFormLayout()
        local_form.setContentsMargins(0, 0, 0, 0)

        self.local_server = QComboBox()
        self.local_server.addItems([
            "Ollama", "LM Studio", "llama.cpp", "koboldcpp",
            "vLLM", "TabbyAPI", "Custom",
        ])
        self.local_server.currentTextChanged.connect(
            self._on_local_server_changed
        )
        local_form.addRow("Server:", self.local_server)

        self.local_server_url = QLineEdit("http://127.0.0.1:11434/v1")
        self.local_server_url.setPlaceholderText(
            "http://127.0.0.1:11434/v1"
        )
        local_form.addRow("Server URL:", self.local_server_url)

        self.local_format = QComboBox()
        self.local_format.addItems([
            "Auto-detect", "GGUF (llama.cpp)", "GGML",
            "ONNX", "SafeTensors", "PyTorch (.bin/.pt)",
        ])
        local_form.addRow("Format:", self.local_format)

        self.local_backend = QComboBox()
        self.local_backend.addItems([
            "CPU", "CUDA", "Vulkan", "DirectML", "Metal", "ROCm",
        ])
        local_form.addRow("Backend:", self.local_backend)

        self.local_loader = QComboBox()
        self.local_loader.addItems([
            "llama.cpp", "vLLM", "Exllamav2", "ctransformers",
            "transformers (HF)", "Custom",
        ])
        local_form.addRow("Loader:", self.local_loader)

        lp.addLayout(local_form)

        # -- Server Controls (start/stop local LLM server) --
        srv_group = QGroupBox("Server Controls")
        srv_group.setObjectName("settingsGroup")
        scl = QVBoxLayout(srv_group)
        scl.setSpacing(6)

        exe_row = QHBoxLayout()
        self.llm_exe_path = QLineEdit()
        self.llm_exe_path.setPlaceholderText(
            "Path to llama-server.exe / ollama.exe ..."
        )
        exe_row.addWidget(self.llm_exe_path, stretch=1)
        browse_exe_btn = QPushButton("Browse")
        browse_exe_btn.setObjectName("browseBtn")
        browse_exe_btn.clicked.connect(self._browse_server_exe)
        exe_row.addWidget(browse_exe_btn)
        scl.addLayout(exe_row)

        srv_params = QFormLayout()
        self.srv_gpu_layers = QSpinBox()
        self.srv_gpu_layers.setRange(-1, 999)
        self.srv_gpu_layers.setValue(-1)
        self.srv_gpu_layers.setToolTip("-1 = all layers on GPU")
        srv_params.addRow("GPU Layers:", self.srv_gpu_layers)

        self.srv_ctx = QSpinBox()
        self.srv_ctx.setRange(512, 131072)
        self.srv_ctx.setValue(4096)
        self.srv_ctx.setSingleStep(512)
        srv_params.addRow("Context Size:", self.srv_ctx)

        self.srv_port = QSpinBox()
        self.srv_port.setRange(1024, 65535)
        self.srv_port.setValue(8080)
        srv_params.addRow("Port:", self.srv_port)
        scl.addLayout(srv_params)

        btn_row = QHBoxLayout()
        self.start_llm_btn = QPushButton("Start Server")
        self.start_llm_btn.setObjectName("connectBtn")
        self.start_llm_btn.clicked.connect(self._start_llm_server)
        btn_row.addWidget(self.start_llm_btn)

        self.stop_llm_btn = QPushButton("Stop Server")
        self.stop_llm_btn.setObjectName("secondaryBtn")
        self.stop_llm_btn.clicked.connect(self._stop_llm_server)
        self.stop_llm_btn.setEnabled(False)
        btn_row.addWidget(self.stop_llm_btn)
        scl.addLayout(btn_row)

        self.auto_start_llm = QCheckBox("Auto-start local LLM server on launch")
        self.auto_start_llm.setChecked(True)
        scl.addWidget(self.auto_start_llm)

        self.llm_server_status = QLabel("Server: Not running")
        self.llm_server_status.setFont(QFont("Segoe UI", 8))
        scl.addWidget(self.llm_server_status)

        lp.addWidget(srv_group)
        self.source_stack.addWidget(local_page)

        # -- Online API page --
        online_page = QWidget()
        op = QVBoxLayout(online_page)
        op.setContentsMargins(0, 0, 0, 0)
        op.setSpacing(6)

        online_form = QFormLayout()
        online_form.setContentsMargins(0, 0, 0, 0)

        self.api_provider = QComboBox()
        self.api_provider.addItems([
            "OpenAI", "Anthropic (Claude)", "Google (Gemini)",
            "Mistral", "Groq", "Together AI", "OpenRouter",
            "Azure OpenAI", "Custom / OpenAI-compatible",
        ])
        self.api_provider.currentTextChanged.connect(
            self._on_provider_changed
        )
        online_form.addRow("Provider:", self.api_provider)

        self.api_endpoint = QLineEdit()
        self.api_endpoint.setPlaceholderText(
            "https://api.openai.com/v1"
        )
        online_form.addRow("API Endpoint:", self.api_endpoint)

        self.api_key = QLineEdit()
        self.api_key.setPlaceholderText("sk-... or API key")
        self.api_key.setEchoMode(QLineEdit.Password)
        online_form.addRow("API Key:", self.api_key)

        key_toggle_row = QHBoxLayout()
        self.show_key_btn = QPushButton("Show Key")
        self.show_key_btn.setObjectName("browseBtn")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_key_visibility)
        key_toggle_row.addWidget(self.show_key_btn)
        key_toggle_row.addStretch()
        online_form.addRow("", key_toggle_row)

        self.api_model = QComboBox()
        self.api_model.setEditable(True)
        self.api_model.addItems([
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
        ])
        online_form.addRow("Model ID:", self.api_model)

        self.api_org = QLineEdit()
        self.api_org.setPlaceholderText("Optional: org ID or project ID")
        online_form.addRow("Organization:", self.api_org)

        op.addLayout(online_form)
        self.source_stack.addWidget(online_page)

        sg.addRow(self.source_stack)
        layout.addWidget(source_group)

        # --- Generation parameters ---
        gen_group = QGroupBox("Generation Parameters")
        gen_group.setObjectName("settingsGroup")
        gf = QFormLayout(gen_group)

        self.ctx_length = QSpinBox()
        self.ctx_length.setRange(512, 131072)
        self.ctx_length.setValue(4096)
        self.ctx_length.setSingleStep(512)
        gf.addRow("Context Length:", self.ctx_length)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setValue(0.7)
        self.temperature.setSingleStep(0.1)
        gf.addRow("Temperature:", self.temperature)

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setValue(0.9)
        self.top_p.setSingleStep(0.05)
        gf.addRow("Top P:", self.top_p)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 32768)
        self.max_tokens.setValue(512)
        gf.addRow("Max Tokens:", self.max_tokens)

        self.repeat_penalty = QDoubleSpinBox()
        self.repeat_penalty.setRange(1.0, 2.0)
        self.repeat_penalty.setValue(1.1)
        self.repeat_penalty.setSingleStep(0.05)
        gf.addRow("Repeat Penalty:", self.repeat_penalty)

        self.fast_mode = QCheckBox("Fast response mode (lower latency)")
        self.fast_mode.setChecked(True)
        gf.addRow("", self.fast_mode)

        layout.addWidget(gen_group)

        # --- GPU settings (local only) ---
        self.gpu_group = QGroupBox("GPU / Quantization  (Local)")
        self.gpu_group.setObjectName("settingsGroup")
        gg = QFormLayout(self.gpu_group)

        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 200)
        self.gpu_layers.setValue(0)
        gg.addRow("GPU Layers:", self.gpu_layers)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 2048)
        self.batch_size.setValue(512)
        gg.addRow("Batch Size:", self.batch_size)

        self.threads = QSpinBox()
        self.threads.setRange(1, 128)
        self.threads.setValue(4)
        gg.addRow("Threads:", self.threads)

        self.quant = QComboBox()
        self.quant.addItems([
            "None / FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
            "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q2_K",
        ])
        gg.addRow("Quantization:", self.quant)

        layout.addWidget(self.gpu_group)

        # --- Connection ---
        conn_group = QGroupBox("Connection")
        conn_group.setObjectName("settingsGroup")
        cg = QVBoxLayout(conn_group)

        self.conn_status = QLabel("Status: Not connected")
        self.conn_status.setFont(QFont("Consolas", 9))
        self.conn_status.setObjectName("metricLabel")
        cg.addWidget(self.conn_status)

        btn_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect / Test")
        self.connect_btn.setObjectName("primaryBtn")
        self.connect_btn.clicked.connect(self._test_connection)
        btn_row.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setObjectName("secondaryBtn")
        self.disconnect_btn.clicked.connect(self._disconnect)
        self.disconnect_btn.setEnabled(False)
        btn_row.addWidget(self.disconnect_btn)
        cg.addLayout(btn_row)

        layout.addWidget(conn_group)
        layout.addStretch()
        self.setWidget(container)

        self.event_bus.connection_changed.connect(self._on_core_connection)
        self.event_bus.telemetry_updated.connect(self._on_runtime_status)
        self._on_source_changed(0)
        self._on_provider_changed(self.api_provider.currentText())
        self._load_settings()  # restore previous session settings
        self._wire_settings_autosave()
        self._loading = False
        if self._pending_source is None:
            self._pending_source = (
                "online" if self.source_type.currentIndex() == 1 else "local"
            )

    def _connect_save_signal(self, signal):
        signal.connect(lambda *_args: self._save_settings())

    def _wire_settings_autosave(self):
        # Local
        self._connect_save_signal(self.local_path.textChanged)
        self._connect_save_signal(self.local_server_url.textChanged)
        self._connect_save_signal(self.local_format.currentTextChanged)
        self._connect_save_signal(self.local_backend.currentTextChanged)
        self._connect_save_signal(self.local_loader.currentTextChanged)
        self._connect_save_signal(self.llm_exe_path.textChanged)
        self._connect_save_signal(self.srv_gpu_layers.valueChanged)
        self._connect_save_signal(self.srv_ctx.valueChanged)
        self._connect_save_signal(self.srv_port.valueChanged)
        self._connect_save_signal(self.auto_start_llm.toggled)

        # Online
        self._connect_save_signal(self.api_provider.currentTextChanged)
        self._connect_save_signal(self.api_endpoint.textChanged)
        self._connect_save_signal(self.api_key.textChanged)
        self._connect_save_signal(self.api_model.currentTextChanged)
        api_model_line_edit = self.api_model.lineEdit()
        if api_model_line_edit:
            self._connect_save_signal(api_model_line_edit.textChanged)
        self._connect_save_signal(self.api_org.textChanged)

        # Generation + GPU
        self._connect_save_signal(self.ctx_length.valueChanged)
        self._connect_save_signal(self.temperature.valueChanged)
        self._connect_save_signal(self.top_p.valueChanged)
        self._connect_save_signal(self.max_tokens.valueChanged)
        self._connect_save_signal(self.repeat_penalty.valueChanged)
        self._connect_save_signal(self.fast_mode.toggled)
        self._connect_save_signal(self.gpu_layers.valueChanged)
        self._connect_save_signal(self.batch_size.valueChanged)
        self._connect_save_signal(self.threads.valueChanged)
        self._connect_save_signal(self.quant.currentTextChanged)

    # --- Slots ---

    def _on_source_changed(self, index):
        self.source_stack.setCurrentIndex(index)
        self.gpu_group.setVisible(index == 0)
        self._save_settings()

    def _on_local_server_changed(self, server):
        urls = {
            "Ollama":   "http://127.0.0.1:11434/v1",
            "LM Studio": "http://127.0.0.1:1234/v1",
            "llama.cpp": "http://127.0.0.1:8080/v1",
            "koboldcpp": "http://127.0.0.1:5001/v1",
            "vLLM":     "http://127.0.0.1:8000/v1",
            "TabbyAPI": "http://127.0.0.1:5000/v1",
        }
        url = urls.get(server, "")
        if url:
            self.local_server_url.setText(url)
        ports = {
            "Ollama": 11434, "LM Studio": 1234, "llama.cpp": 8080,
            "koboldcpp": 5001, "vLLM": 8000, "TabbyAPI": 5000,
        }
        if server in ports:
            self.srv_port.setValue(ports[server])
        self._save_settings()

    def _browse_server_exe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select LLM Server Executable", "",
            "Executables (*.exe);;All Files (*)",
        )
        if path:
            self.llm_exe_path.setText(path)
            self._save_settings()

    def _kill_port_listener(self, port):
        if sys.platform != "win32":
            return
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"], text=True, timeout=5
            )
        except Exception as e:
            logger.debug(f"Error listing port listeners: {e}")
            return
        for line in out.splitlines():
            if f":{port} " not in line or "LISTEN" not in line.upper():
                continue
            parts = line.split()
            if not parts:
                continue
            pid = parts[-1]
            if pid.isdigit() and int(pid) != os.getpid():
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", pid],
                        capture_output=True, timeout=5,
                    )
                    self.event_bus.log_entry.emit(
                        f"[LLM] Killed stale process PID {pid} on port {port}"
                    )
                except Exception as e:
                    logger.debug(f"Error killing process {pid}: {e}")

    def _build_server_command(self, server, model_file, port, ctx, gpu_layers):
        server_l = (server or "").strip().lower()
        env_updates = {}

        if server_l == "ollama":
            env_updates["OLLAMA_HOST"] = f"127.0.0.1:{port}"
            return ["serve"], env_updates, False, "ollama"

        if server_l in {"llama.cpp", "custom"}:
            args = [
                "--port", str(port),
                "-c", str(ctx),
                "-m", model_file,
                "-ngl", str(gpu_layers),
            ]
            return args, env_updates, True, "llama.cpp"

        return None, None, None, None

    def _read_models_from_response(self, response_json):
        if not isinstance(response_json, dict):
            return []
        if isinstance(response_json.get("data"), list):
            return [
                str(item.get("id", "")).strip()
                for item in response_json.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
        if isinstance(response_json.get("models"), list):
            return [
                str(item.get("name", "")).strip()
                for item in response_json.get("models", [])
                if isinstance(item, dict) and item.get("name")
            ]
        return []

    def _probe_local_server(self, server_url):
        base = (server_url or "").strip().rstrip("/")
        if not base:
            return False, []
        candidates = []
        if base.endswith("/v1"):
            candidates.append(base + "/models")
            base_root = base[:-3]
            candidates.append(base_root + "/api/tags")
            candidates.append(base_root + "/health")
        else:
            candidates.append(base + "/models")
            candidates.append(base + "/v1/models")
            candidates.append(base + "/api/tags")
            candidates.append(base + "/health")

        tried = set()
        for url in candidates:
            if url in tried:
                continue
            tried.add(url)
            try:
                import requests
                r = requests.get(url, timeout=2)
                if not r.ok:
                    continue
                try:
                    payload = r.json()
                except Exception as je:
                    logger.debug(f"Error parsing JSON from {url}: {je}")
                    payload = {}
                models = self._read_models_from_response(payload)
                return True, models
            except Exception as e:
                logger.debug(f"Error fetching models from {url}: {e}")
                continue
        return False, []

    def _start_llm_server(self):
        if self._llm_process and self._llm_process.state() == QProcess.Running:
            self.llm_server_status.setText("Server: Already running")
            apply_status_style(self.llm_server_status, "color: #ccaa00;")
            return

        server = self.local_server.currentText()
        exe = self.llm_exe_path.text().strip()
        if not exe:
            self.llm_server_status.setText("Server: Set executable path first")
            apply_status_style(self.llm_server_status, "color: #cc3040;")
            return
        if not Path(exe).exists():
            self.llm_server_status.setText("Server: Executable path not found")
            apply_status_style(self.llm_server_status, "color: #cc3040;")
            return

        model_file = self.local_path.text().strip()
        port = self.srv_port.value()
        gpu_layers = self.srv_gpu_layers.value()
        ctx = self.srv_ctx.value()

        args, env_updates, requires_model, server_kind = self._build_server_command(
            server, model_file, port, ctx, gpu_layers
        )
        if args is None:
            self.llm_server_status.setText(
                f"Server: Auto-launch unsupported for {server}. "
                "Use Ollama/llama.cpp or start externally."
            )
            apply_status_style(self.llm_server_status, "color: #cc3040;")
            return

        if requires_model and not model_file:
            self.llm_server_status.setText("Server: Set model file path first")
            apply_status_style(self.llm_server_status, "color: #cc3040;")
            return
        if requires_model and not Path(model_file).exists():
            self.llm_server_status.setText("Server: Model file path not found")
            apply_status_style(self.llm_server_status, "color: #cc3040;")
            return

        self._kill_port_listener(port)

        self._llm_process = QProcess(self)
        self._llm_process.setWorkingDirectory(str(Path(exe).parent))
        if env_updates:
            from PySide6.QtCore import QProcessEnvironment
            env = QProcessEnvironment.systemEnvironment()
            for key, value in env_updates.items():
                env.insert(key, str(value))
            self._llm_process.setProcessEnvironment(env)
        self._llm_process.readyReadStandardOutput.connect(
            self._on_llm_stdout
        )
        self._llm_process.readyReadStandardError.connect(
            self._on_llm_stderr
        )
        self._llm_process.finished.connect(self._on_llm_finished)
        self._llm_process.start(exe, args)

        self.llm_server_status.setText("Server: Starting...")
        apply_status_style(self.llm_server_status, "color: #ccaa00;")
        self.start_llm_btn.setEnabled(False)
        self.stop_llm_btn.setEnabled(True)
        self._llm_server_kind = server_kind

        cmd_str = Path(exe).name + " " + " ".join(args)
        self.event_bus.log_entry.emit(f"[LLM] Starting: {cmd_str}")
        self._llm_ready_attempts = 0

        # Keep URL aligned with server launch settings.
        self.local_server_url.setText(f"http://127.0.0.1:{port}/v1")
        QTimer.singleShot(3000, self._check_llm_ready)

    def _check_llm_ready(self):
        if not self._llm_process or self._llm_process.state() != QProcess.Running:
            return
        self._llm_ready_attempts = getattr(self, '_llm_ready_attempts', 0) + 1
        if self._llm_ready_attempts > 20:
            self.llm_server_status.setText("Server: Timeout waiting for ready")
            apply_status_style(self.llm_server_status, "color: #cc3040;")
            return
        ready, models = self._probe_local_server(self.local_server_url.text())
        if ready:
            if models:
                shown = ", ".join(models[:2])
                self.llm_server_status.setText(f"Server: Running ({shown})")
                self.event_bus.log_entry.emit(
                    f"[LLM] Server ready with model(s): {shown}"
                )
            else:
                if self._llm_server_kind == "ollama":
                    self.llm_server_status.setText(
                        "Server: Running (Ollama ready)"
                    )
                else:
                    self.llm_server_status.setText(
                        "Server: Running (waiting for model load)"
                    )
            apply_status_style(self.llm_server_status, "color: #00aa40;")
            return
        self.llm_server_status.setText(
            f"Server: Starting ({self._llm_ready_attempts * 3}s...)"
        )
        QTimer.singleShot(3000, self._check_llm_ready)

    def _stop_llm_server(self):
        if self._llm_process and self._llm_process.state() == QProcess.Running:
            pid = self._llm_process.processId()
            if pid and sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=5,
                    )
                except Exception as e:
                    logger.debug(f"taskkill failed, using kill(): {e}")
                    self._llm_process.kill()
            else:
                self._llm_process.kill()
            self._llm_process.waitForFinished(3000)
            self._llm_process = None
        self.llm_server_status.setText("Server: Stopped")
        clear_status_role(self.llm_server_status)
        self.start_llm_btn.setEnabled(True)
        self.stop_llm_btn.setEnabled(False)
        self._llm_server_kind = ""
        self.event_bus.log_entry.emit("[LLM] Server stopped")

    def _on_llm_stdout(self):
        if self._llm_process:
            data = self._llm_process.readAllStandardOutput().data().decode(
                errors="replace"
            ).strip()
            if data:
                self.event_bus.log_entry.emit(f"[LLM] {data}")

    def _on_llm_stderr(self):
        if self._llm_process:
            data = self._llm_process.readAllStandardError().data().decode(
                errors="replace"
            ).strip()
            if data:
                for line in data.splitlines():
                    self.event_bus.log_entry.emit(f"[LLM] {line}")

    def _on_llm_finished(self, exit_code, exit_status):
        self.llm_server_status.setText(
            f"Server: Exited (code {exit_code})"
        )
        apply_status_style(self.llm_server_status, "color: #cc3040;")
        self.start_llm_btn.setEnabled(True)
        self.stop_llm_btn.setEnabled(False)
        self._llm_server_kind = ""
        self._llm_process = None

    def _on_provider_changed(self, provider):
        endpoints = {
            "OpenAI": "https://api.openai.com/v1",
            "Anthropic (Claude)": "https://api.anthropic.com/v1",
            "Google (Gemini)": "https://generativelanguage.googleapis.com/v1beta",
            "Mistral": "https://api.mistral.ai/v1",
            "Groq": "https://api.groq.com/openai/v1",
            "Together AI": "https://api.together.xyz/v1",
            "OpenRouter": "https://openrouter.ai/api/v1",
            "Azure OpenAI": "https://<resource>.openai.azure.com/openai/deployments/<model>",
        }
        models = {
            "OpenAI": [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
                "gpt-3.5-turbo", "o1", "o1-mini", "o3-mini",
            ],
            "Anthropic (Claude)": [
                "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
            ],
            "Google (Gemini)": [
                "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
            ],
            "Mistral": [
                "mistral-large-latest", "mistral-medium-latest",
                "mistral-small-latest", "codestral-latest",
            ],
            "Groq": [
                "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                "mixtral-8x7b-32768", "gemma2-9b-it",
            ],
            "Together AI": [
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
            ],
            "OpenRouter": [
                "openai/gpt-4o", "anthropic/claude-sonnet-4",
                "google/gemini-2.0-flash-001", "meta-llama/llama-3.1-70b-instruct",
            ],
        }

        ep = endpoints.get(provider, "")
        if ep:
            self.api_endpoint.setText(ep)
        else:
            self.api_endpoint.clear()

        self.api_model.clear()
        for m in models.get(provider, []):
            self.api_model.addItem(m)
        self._save_settings()

    def _toggle_key_visibility(self, show):
        if show:
            self.api_key.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("Hide Key")
        else:
            self.api_key.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("Show Key")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Local Model File", "",
            "Model Files (*.gguf *.ggml *.bin *.safetensors *.onnx *.pt);;"
            "GGUF Models (*.gguf);;"
            "SafeTensors (*.safetensors);;"
            "All Files (*)",
        )
        if path:
            self.local_path.setText(path)
            self._save_settings()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _save_settings(self):
        """Write current UI state to model_settings.json."""
        if self._loading:
            return
        data = {
            "source_index": self.source_type.currentIndex(),
            # Local model
            "local_path": self.local_path.text(),
            "local_server": self.local_server.currentText(),
            "local_server_url": self.local_server_url.text(),
            "local_format": self.local_format.currentText(),
            "local_backend": self.local_backend.currentText(),
            "local_loader": self.local_loader.currentText(),
            "llm_exe_path": self.llm_exe_path.text(),
            "srv_gpu_layers": self.srv_gpu_layers.value(),
            "srv_ctx": self.srv_ctx.value(),
            "srv_port": self.srv_port.value(),
            "auto_start_llm": self.auto_start_llm.isChecked(),
            # Online API
            "api_provider": self.api_provider.currentText(),
            "api_endpoint": self.api_endpoint.text(),
            "api_key": self.api_key.text(),
            "api_model": self.api_model.currentText(),
            "api_org": self.api_org.text(),
            # Generation params
            "ctx_length": self.ctx_length.value(),
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "max_tokens": self.max_tokens.value(),
            "repeat_penalty": self.repeat_penalty.value(),
            "fast_mode": self.fast_mode.isChecked(),
            # GPU / quant
            "gpu_layers": self.gpu_layers.value(),
            "batch_size": self.batch_size.value(),
            "threads": self.threads.value(),
            "quant": self.quant.currentText(),
        }
        try:
            _SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            self.event_bus.log_entry.emit(f"[Model] Could not save settings: {e}")

    def _load_settings(self):
        """Restore UI state from model_settings.json (if it exists)."""
        if not _SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Error loading model settings: {e}")
            return

        self._loading = True
        try:
            # --- Source type ---
            src = int(data.get("source_index", 0))
            self.source_type.blockSignals(True)
            self.source_type.setCurrentIndex(src)
            self.source_type.blockSignals(False)

            # --- Local model ---
            self.local_path.setText(data.get("local_path", ""))

            srv = data.get("local_server", "")
            if srv:
                self.local_server.blockSignals(True)
                idx = self.local_server.findText(srv)
                if idx >= 0:
                    self.local_server.setCurrentIndex(idx)
                self.local_server.blockSignals(False)

            # Restore URL after combo (avoids the auto-preset overwriting it)
            url = data.get("local_server_url", "")
            if url:
                self.local_server_url.setText(url)

            fmt = data.get("local_format", "")
            if fmt:
                idx = self.local_format.findText(fmt)
                if idx >= 0:
                    self.local_format.setCurrentIndex(idx)

            backend = data.get("local_backend", "")
            if backend:
                idx = self.local_backend.findText(backend)
                if idx >= 0:
                    self.local_backend.setCurrentIndex(idx)

            loader = data.get("local_loader", "")
            if loader:
                idx = self.local_loader.findText(loader)
                if idx >= 0:
                    self.local_loader.setCurrentIndex(idx)

            self.llm_exe_path.setText(data.get("llm_exe_path", ""))
            self.srv_gpu_layers.setValue(int(data.get("srv_gpu_layers", -1)))
            self.srv_ctx.setValue(int(data.get("srv_ctx", 4096)))
            self.srv_port.setValue(int(data.get("srv_port", 8080)))
            self.auto_start_llm.setChecked(bool(data.get("auto_start_llm", True)))

            # --- Online API ---
            provider = data.get("api_provider", "")
            if provider:
                self.api_provider.blockSignals(True)
                idx = self.api_provider.findText(provider)
                if idx >= 0:
                    self.api_provider.setCurrentIndex(idx)
                self.api_provider.blockSignals(False)
                # Repopulate model list for the saved provider
                self._on_provider_changed(self.api_provider.currentText())

            ep = data.get("api_endpoint", "")
            if ep:
                self.api_endpoint.setText(ep)

            self.api_key.setText(data.get("api_key", ""))

            model = data.get("api_model", "")
            if model:
                idx = self.api_model.findText(model)
                if idx >= 0:
                    self.api_model.setCurrentIndex(idx)
                else:
                    self.api_model.setEditText(model)

            self.api_org.setText(data.get("api_org", ""))

            # --- Generation params ---
            self.ctx_length.setValue(int(data.get("ctx_length", 4096)))
            self.temperature.setValue(float(data.get("temperature", 0.7)))
            self.top_p.setValue(float(data.get("top_p", 0.9)))
            self.max_tokens.setValue(int(data.get("max_tokens", 512)))
            self.repeat_penalty.setValue(float(data.get("repeat_penalty", 1.1)))
            self.fast_mode.setChecked(bool(data.get("fast_mode", True)))

            # --- GPU / quant ---
            self.gpu_layers.setValue(int(data.get("gpu_layers", 0)))
            self.batch_size.setValue(int(data.get("batch_size", 512)))
            self.threads.setValue(int(data.get("threads", 4)))
            quant = data.get("quant", "")
            if quant:
                idx = self.quant.findText(quant)
                if idx >= 0:
                    self.quant.setCurrentIndex(idx)

            # Apply source-page visibility
            self._on_source_changed(src)

        except Exception as e:
            self.event_bus.log_entry.emit(f"[Model] Could not restore settings: {e}")
        finally:
            self._loading = False

        self.event_bus.log_entry.emit("[Model] Previous session settings restored.")
        self._pending_source = (
            "online" if int(data.get("source_index", 0)) == 1 else "local"
        )

    def auto_start_on_launch(self):
        self._pending_source = (
            "online" if self.source_type.currentIndex() == 1 else "local"
        )
        if self.source_type.currentIndex() != 0:
            return
        if not self.auto_start_llm.isChecked():
            return
        if self._llm_process and self._llm_process.state() == QProcess.Running:
            return

        reachable, _ = self._probe_local_server(self.local_server_url.text())
        if reachable:
            self.llm_server_status.setText("Server: Running externally")
            apply_status_style(self.llm_server_status, "color: #00aa40;")
            self.event_bus.log_entry.emit("[LLM] Found existing local LLM server.")
            return

        if not self.llm_exe_path.text().strip():
            self.event_bus.log_entry.emit(
                "[LLM] Auto-start skipped: set an executable path first."
            )
            return
        self._start_llm_server()

    def _test_connection(self):
        self.conn_status.setText("Status: Connecting...")
        apply_status_style(self.conn_status, "color: #ccaa00;")
        self.connect_btn.setEnabled(False)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._do_test)

    def _do_test(self):
        is_online = self.source_type.currentIndex() == 1

        if is_online:
            self._test_online()
        else:
            self._test_local()

        self.connect_btn.setEnabled(True)

    def _push_config_to_core(self, source, verified=False):
        self._save_settings()
        import threading
        cfg = {
            "source": source,
            "temperature": self.temperature.value(),
            "max_tokens": self.max_tokens.value(),
            "top_p": self.top_p.value(),
            "ctx_length": self.ctx_length.value(),
            "fast_mode": self.fast_mode.isChecked(),
            "verified": bool(verified),
        }
        if source == "local":
            cfg["local_path"] = self.local_path.text().strip()
            cfg["local_backend"] = self.local_backend.currentText()
            cfg["local_loader"] = self.local_loader.currentText()
            cfg["local_server"] = self.local_server.currentText()
            cfg["local_server_url"] = self.local_server_url.text().strip()
        else:
            cfg["api_provider"] = self.api_provider.currentText()
            cfg["api_endpoint"] = self.api_endpoint.text().strip()
            cfg["api_key"] = self.api_key.text().strip()
            cfg["api_model"] = self.api_model.currentText().strip()

        def _do():
            try:
                import requests
                r = requests.post(
                    f"{self.client.BASE_URL}/api/model/config",
                    json=cfg, timeout=5,
                )
                if r.ok:
                    self.event_bus.log_entry.emit(
                        f"[Model] Config pushed: {source} / "
                        f"{cfg.get('api_model') or cfg.get('local_path', '?')}"
                    )
            except Exception as e:
                self.event_bus.log_entry.emit(
                    f"[Model] Failed to push config: {e}"
                )
        threading.Thread(target=_do, daemon=True).start()

    def _test_local(self):
        server = self.local_server.currentText()
        server_url = self.local_server_url.text().strip()
        path = self.local_path.text().strip()

        if not server_url:
            self.conn_status.setText("Status: No server URL specified")
            apply_status_style(self.conn_status, "color: #cc3040;")
            return

        llm_ok, models = self._probe_local_server(server_url)
        llm_detail = ", ".join(models[:3]) if models else "connected"
        if not llm_ok:
            llm_detail = "not running"

        # Push config to REVIA core
        self._push_config_to_core("local", verified=llm_ok)

        # Check REVIA core
        core_ok = False
        try:
            import requests
            r = requests.get(
                f"{self.client.BASE_URL}/api/status", timeout=2
            )
            core_ok = r.ok
        except Exception as e:
            logger.debug(f"Error checking core status: {e}")

        if llm_ok:
            file_info = ""
            if path and os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                file_info = f" | File: {size_mb:.0f} MB"
            self.conn_status.setText(
                f"Status: {server} OK ({llm_detail}){file_info} | "
                f"Core: {'Online' if core_ok else 'Offline'}"
            )
            apply_status_style(self.conn_status, "color: #00aa40;")
            self.disconnect_btn.setEnabled(True)
        else:
            self.conn_status.setText(
                f"Status: {server} at {server_url} â€” {llm_detail}"
            )
            apply_status_style(self.conn_status, "color: #cc3040;")

    def _test_online(self):
        endpoint = self.api_endpoint.text().strip()
        key = self.api_key.text().strip()
        model = self.api_model.currentText().strip()
        provider = self.api_provider.currentText()

        if not endpoint:
            self.conn_status.setText("Status: No endpoint specified")
            apply_status_style(self.conn_status, "color: #cc3040;")
            return
        if not key:
            self.conn_status.setText("Status: No API key provided")
            apply_status_style(self.conn_status, "color: #cc3040;")
            return

        try:
            import requests

            if "anthropic" in provider.lower():
                headers = {
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                r = requests.post(
                    f"{endpoint}/messages",
                    headers=headers,
                    json={
                        "model": model or "claude-sonnet-4-20250514",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    timeout=10,
                )
            else:
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                }
                test_url = endpoint.rstrip("/")
                if "/v1" in test_url and not test_url.endswith("/models"):
                    models_url = test_url + "/models"
                else:
                    models_url = test_url + "/models"

                r = requests.get(
                    models_url, headers=headers, timeout=10
                )

            if r.ok:
                self._push_config_to_core("online", verified=True)
                self.conn_status.setText(
                    f"Status: Connected to {provider} | Model: {model}"
                )
                apply_status_style(self.conn_status, "color: #00aa40;")
                self.disconnect_btn.setEnabled(True)
            elif r.status_code == 401:
                self.conn_status.setText("Status: Invalid API key (401)")
                apply_status_style(self.conn_status, "color: #cc3040;")
            elif r.status_code == 403:
                self.conn_status.setText(
                    "Status: Access denied (403) â€” check key permissions"
                )
                apply_status_style(self.conn_status, "color: #cc3040;")
            else:
                self.conn_status.setText(
                    f"Status: API returned {r.status_code}"
                )
                apply_status_style(self.conn_status, "color: #cc8800;")
        except requests.exceptions.Timeout:
            self.conn_status.setText("Status: Connection timed out")
            apply_status_style(self.conn_status, "color: #cc3040;")
        except requests.exceptions.ConnectionError:
            self.conn_status.setText("Status: Cannot reach endpoint")
            apply_status_style(self.conn_status, "color: #cc3040;")
        except Exception as e:
            self.conn_status.setText(f"Status: Error â€” {e}")
            apply_status_style(self.conn_status, "color: #cc3040;")

    def _disconnect(self):
        self.conn_status.setText("Status: Not connected")
        clear_status_role(self.conn_status)
        self.disconnect_btn.setEnabled(False)

    def _on_core_connection(self, connected):
        if connected:
            cur = self.conn_status.text()
            if "Not connected" in cur or "offline" in cur.lower():
                self.conn_status.setText(
                    "Status: Core online (via WebSocket)"
                )
                apply_status_style(self.conn_status, "color: #00aa40;")
                self.disconnect_btn.setEnabled(True)
            if self._pending_source:
                self._push_config_to_core(self._pending_source)
                self._pending_source = None
        else:
            self.conn_status.setText("Status: Waiting for core status...")
            apply_status_style(self.conn_status, "color: #ccaa00;")
            self.disconnect_btn.setEnabled(False)

    def _on_runtime_status(self, data):
        if not isinstance(data, dict):
            return
        llm = data.get("llm_connection", {}) or {}
        if not llm:
            return
        state = str(llm.get("state", "Disconnected"))
        detail = str(llm.get("detail", "")).strip()
        model = str(llm.get("model", "")).strip()
        text = f"Status: {state}"
        if detail:
            text += f" | {detail}"
        elif model and model != "None":
            text += f" | Model: {model}"
        self.conn_status.setText(text)
        if state == "Ready":
            apply_status_style(self.conn_status, "color: #00aa40;")
            self.disconnect_btn.setEnabled(True)
        elif state == "Connecting":
            apply_status_style(self.conn_status, "color: #ccaa00;")
            self.disconnect_btn.setEnabled(True)
        elif state == "Error":
            apply_status_style(self.conn_status, "color: #cc3040;")
            self.disconnect_btn.setEnabled(True)
        else:
            apply_status_style(self.conn_status, "color: #808898;")
            self.disconnect_btn.setEnabled(False)
