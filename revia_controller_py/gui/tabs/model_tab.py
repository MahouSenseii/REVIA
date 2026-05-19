import sys
import os
import json
import logging
import subprocess
import threading
from pathlib import Path
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QFileDialog, QStackedWidget, QCheckBox,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import QProcess, QTimer, Signal

from gui.widgets.settings_card import SettingsCard

from app.hardware_routing import (
    default_hardware_policy,
    detect_nvidia_smi_gpus,
    normalize_hardware_policy,
    read_saved_hardware_policy,
    resolve_gpu_roles,
    write_profile_hardware_policy,
)
from app.ui_status import apply_status_style, clear_status_role

logger = logging.getLogger(__name__)

# Persisted model settings live alongside the top-level config.json
_SETTINGS_FILE = Path(__file__).resolve().parents[3] / "model_settings.json"
_SECRET_SETTINGS_FILE = Path(__file__).resolve().parents[3] / "model_settings.local.json"


class ModelTab(QScrollArea):
    connection_test_completed = Signal(object)

    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self._llm_process = None
        self._loading = True  # guard: prevents saving while loading
        self._pending_source = None  # source to push once core connects
        self._llm_server_kind = ""
        self._hardware_gpus = []
        self._hardware_roles = {}
        self._hardware_policy = read_saved_hardware_policy()
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
        source_card = SettingsCard(
            "Model Source",
            subtitle="Local model or online API",
            icon="M",
        )
        sg = QFormLayout()

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
        srv_card = SettingsCard(
            "Server Controls",
            subtitle="Start/stop local LLM server",
            icon="S",
        )
        scl = QVBoxLayout()
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

        srv_card.add_layout(scl)
        lp.addWidget(srv_card)
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
        source_card.add_layout(sg)
        layout.addWidget(source_card)

        # --- Generation parameters ---
        gen_card = SettingsCard(
            "Generation Parameters",
            subtitle="Sampling and response settings",
            icon="G",
        )
        gf = QFormLayout()

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

        gen_card.add_layout(gf)
        layout.addWidget(gen_card)

        # --- GPU settings (local only) ---
        self.gpu_card = SettingsCard(
            "GPU / Quantization (Local)",
            subtitle="Hardware acceleration and compression",
            icon="Q",
        )
        gg = QFormLayout()

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

        self.hardware_mode = QComboBox()
        self.hardware_mode.addItem("Auto Recommended", "auto_recommended")
        self.hardware_mode.addItem("Single GPU", "single_gpu")
        self.hardware_mode.addItem("Advanced Multi-GPU", "advanced_multi_gpu")
        gg.addRow("GPU Routing:", self.hardware_mode)

        self.main_gpu_combo = QComboBox()
        self.main_gpu_combo.addItem("Auto best", "auto_best")
        gg.addRow("Single GPU:", self.main_gpu_combo)

        self.hardware_status = QLabel("GPU routing: not detected yet")
        self.hardware_status.setWordWrap(True)
        self.hardware_status.setFont(QFont("Consolas", 8))
        self.hardware_status.setObjectName("metricLabel")
        gg.addRow("Detected:", self.hardware_status)

        hw_btn_row = QHBoxLayout()
        detect_hw_btn = QPushButton("Detect / Optimize")
        detect_hw_btn.setObjectName("secondaryBtn")
        detect_hw_btn.clicked.connect(
            lambda _checked=False: self._refresh_hardware_routing(
                apply_recommendations=True
            )
        )
        hw_btn_row.addWidget(detect_hw_btn)
        apply_hw_btn = QPushButton("Apply Routing")
        apply_hw_btn.setObjectName("primaryBtn")
        apply_hw_btn.clicked.connect(self._apply_hardware_routing)
        hw_btn_row.addWidget(apply_hw_btn)
        gg.addRow("", hw_btn_row)

        self.gpu_card.add_layout(gg)
        layout.addWidget(self.gpu_card)

        # --- Connection ---
        conn_card = SettingsCard(
            "Connection",
            subtitle="Test and manage model connection",
            icon="C",
        )
        cg = QVBoxLayout()

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

        conn_card.add_layout(cg)
        layout.addWidget(conn_card)
        layout.addStretch()
        self.setWidget(container)

        self.event_bus.connection_changed.connect(self._on_core_connection)
        self.event_bus.telemetry_updated.connect(self._on_runtime_status)
        self.connection_test_completed.connect(self._apply_connection_test_result)
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
        self._connect_save_signal(self.hardware_mode.currentTextChanged)
        self._connect_save_signal(self.main_gpu_combo.currentTextChanged)

    # --- Slots ---

    def _on_source_changed(self, index):
        self.source_stack.setCurrentIndex(index)
        self.gpu_card.setVisible(index == 0)
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

    def _find_port_holders(self, port):
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
            logger.debug(f"Error listing port listeners: {e}")
            return {}

        holders = {}
        port_suffix = f":{port}"
        for raw_line in out.splitlines():
            line = raw_line.strip()
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
            pid = parts[-1]
            if not pid.isdigit() or int(pid) == os.getpid():
                continue
            state = parts[-2].upper() if proto == "TCP" and len(parts) >= 5 else "BOUND"
            holders.setdefault(int(pid), set()).add(state)
        return holders

    @staticmethod
    def _describe_port_holders(holders):
        if not holders:
            return "none"
        details = []
        for pid in sorted(holders):
            details.append(f"PID {pid} [{', '.join(sorted(holders[pid]))}]")
        return "; ".join(details)

    def _kill_port_listener(self, port):
        if sys.platform != "win32":
            return
        holders = self._find_port_holders(port)
        if not holders:
            return
        self.event_bus.log_entry.emit(
            f"[LLM] Port {port} held by {self._describe_port_holders(holders)}"
        )
        for pid in sorted(holders):
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self.event_bus.log_entry.emit(
                        f"[LLM] Killed stale process PID {pid} on port {port}"
                    )
                else:
                    logger.debug(
                        "taskkill for PID %s returned %s: %s",
                        pid,
                        result.returncode,
                        (result.stderr or result.stdout or "").strip(),
                    )
            except Exception as e:
                try:
                    logger.debug(f"Error killing process {pid}: {e}")
                except Exception:
                    pass
        remaining = self._find_port_holders(port)
        if remaining:
            self.event_bus.log_entry.emit(
                f"[LLM] Port {port} still occupied by "
                f"{self._describe_port_holders(remaining)}"
            )

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

        route_label = ""
        if self.local_backend.currentText().strip().upper() in {"CUDA", "GPU"}:
            route_env, route_label = self._main_llm_gpu_env_updates()
            env_updates.update(route_env)

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
        if route_label:
            self.event_bus.log_entry.emit(
                f"[Hardware] Local LLM routed to {route_label}; model splitting disabled by default."
            )
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
    # Hardware routing
    # ------------------------------------------------------------------

    def _hardware_mode_key(self):
        return self.hardware_mode.currentData() or "auto_recommended"

    def _current_hardware_policy(self):
        gpu_count = len(self._hardware_gpus)
        mode = self._hardware_mode_key()
        if mode == "auto_recommended" and gpu_count == 0:
            saved_mode = str((self._hardware_policy or {}).get("hardware_mode") or "")
            if saved_mode and saved_mode != "cpu_only":
                return normalize_hardware_policy(self._hardware_policy, gpu_count)
        if mode == "single_gpu":
            selected = self.main_gpu_combo.currentData()
            policy = {
                "hardware_mode": "single_gpu",
                "main_llm_gpu": selected if selected is not None else "auto_best",
                "support_gpu": "cpu",
                "allow_model_splitting": False,
                "allow_parallel_agents": True,
                "fallback_to_cpu": True,
            }
        elif mode == "advanced_multi_gpu":
            policy = {
                "hardware_mode": "advanced_multi_gpu",
                "main_llm_gpu": "auto_best",
                "support_gpu": "auto_secondary",
                "allow_model_splitting": True,
                "allow_parallel_agents": True,
                "fallback_to_cpu": True,
            }
        else:
            policy = default_hardware_policy(gpu_count)
            if gpu_count > 1:
                policy["hardware_mode"] = "multi_gpu_auto"
                policy["main_llm_gpu"] = "auto_best"
                policy["support_gpu"] = "auto_secondary"
            elif gpu_count == 1:
                policy["hardware_mode"] = "single_gpu_auto"
                policy["main_llm_gpu"] = "auto_best"
                policy["support_gpu"] = "cpu"
                policy["allow_parallel_agents"] = False
        self._hardware_policy = normalize_hardware_policy(policy, gpu_count)
        return dict(self._hardware_policy)

    def _apply_policy_to_controls(self, policy):
        if not isinstance(policy, dict):
            return
        self._hardware_policy = normalize_hardware_policy(
            policy, len(self._hardware_gpus)
        )
        mode = str(policy.get("hardware_mode") or "").lower()
        target = "auto_recommended"
        if mode == "single_gpu":
            target = "single_gpu"
        elif mode in {"advanced_multi_gpu", "multi_gpu_advanced"}:
            target = "advanced_multi_gpu"
        for i in range(self.hardware_mode.count()):
            if self.hardware_mode.itemData(i) == target:
                self.hardware_mode.blockSignals(True)
                self.hardware_mode.setCurrentIndex(i)
                self.hardware_mode.blockSignals(False)
                break

        main = policy.get("main_llm_gpu")
        if isinstance(main, int) or str(main or "").isdigit():
            main = int(main)
            for i in range(self.main_gpu_combo.count()):
                if self.main_gpu_combo.itemData(i) == main:
                    self.main_gpu_combo.blockSignals(True)
                    self.main_gpu_combo.setCurrentIndex(i)
                    self.main_gpu_combo.blockSignals(False)
                    break

    def _populate_gpu_combo(self, gpus):
        current = self.main_gpu_combo.currentData()
        self.main_gpu_combo.blockSignals(True)
        self.main_gpu_combo.clear()
        self.main_gpu_combo.addItem("Auto best", "auto_best")
        for gpu in gpus:
            try:
                idx = int(gpu.get("index"))
            except (TypeError, ValueError):
                continue
            name = str(gpu.get("name") or f"GPU {idx}")
            total = float(gpu.get("vram_total_mb", 0) or 0) / 1024.0
            used = float(gpu.get("vram_used_mb", 0) or 0) / 1024.0
            load = float(gpu.get("load_percent", gpu.get("gpu_percent", 0.0)) or 0.0)
            self.main_gpu_combo.addItem(
                f"{idx}: {name} ({total:.1f} GB, {used:.1f} used, {load:.0f}% load)",
                idx,
            )
        for i in range(self.main_gpu_combo.count()):
            if self.main_gpu_combo.itemData(i) == current:
                self.main_gpu_combo.setCurrentIndex(i)
                break
        self.main_gpu_combo.blockSignals(False)

    def _describe_hardware_routing(self, gpus, roles, policy):
        if not gpus:
            return "No CUDA GPU detected. REVIA will use CPU fallback."
        lines = [f"REVIA detected {len(gpus)} GPU{'s' if len(gpus) != 1 else ''}:"]
        for gpu in gpus:
            idx = gpu.get("index", "?")
            name = str(gpu.get("name") or f"GPU {idx}")
            total = float(gpu.get("vram_total_mb", 0) or 0) / 1024.0
            used = float(gpu.get("vram_used_mb", 0) or 0) / 1024.0
            load = float(gpu.get("load_percent", gpu.get("gpu_percent", 0.0)) or 0.0)
            lines.append(f"{idx}. {name} | {total:.1f} GB VRAM | {used:.1f} used | {load:.0f}% load")
        main = roles.get("main_llm_gpu_name") or roles.get("main_llm_gpu_index")
        support = roles.get("support_gpu_name") or roles.get("support_gpu_index") or "CPU fallback"
        split = roles.get("model_splitting", {}) or {}
        lines.append(f"Main LLM: {main}")
        lines.append(f"Support: {support}")
        lines.append(
            "Model splitting: "
            + ("allowed, not enabled yet" if split.get("allowed_by_policy") else "disabled")
        )
        lines.append(f"Mode: {policy.get('hardware_mode')}")
        return "\n".join(lines)

    @staticmethod
    def _gpu_float(gpu, key, fallback=0.0):
        try:
            return float(gpu.get(key, fallback) or fallback)
        except (TypeError, ValueError):
            return float(fallback)

    def _selected_main_gpu(self, gpus, roles):
        main_idx = roles.get("main_llm_gpu_index")
        for gpu in gpus:
            try:
                if int(gpu.get("index")) == int(main_idx):
                    return gpu
            except (TypeError, ValueError):
                continue
        if not gpus:
            return None
        return max(
            gpus,
            key=lambda gpu: (
                self._gpu_float(gpu, "vram_total_mb"),
                self._gpu_float(gpu, "vram_free_mb", self._gpu_float(gpu, "vram_total_mb")),
                -self._gpu_float(gpu, "load_percent", self._gpu_float(gpu, "gpu_percent")),
            ),
        )

    def _model_recommendations_for_hardware(self, gpus, roles):
        threads = max(4, min(8, int((os.cpu_count() or 8) - 2)))
        main_gpu = self._selected_main_gpu(gpus, roles)
        if not main_gpu:
            return {
                "profile": "cpu_fallback",
                "local_backend": "CPU",
                "srv_gpu_layers": 0,
                "gpu_layers": 0,
                "srv_ctx": 2048,
                "ctx_length": 2048,
                "batch_size": 64,
                "max_tokens": 160,
                "threads": threads,
                "fast_mode": False,
            }

        vram_mb = self._gpu_float(main_gpu, "vram_total_mb")
        model_path = self.local_path.text().lower()
        quant = self.quant.currentText()
        if "q5_k_s" in model_path:
            quant = "Q5_K_S"
        elif "q4_k_m" in model_path:
            quant = "Q4_K_M"

        if vram_mb < 9 * 1024:
            return {
                "profile": "cuda_8gb_balanced",
                "local_backend": "CUDA",
                "local_loader": "llama.cpp",
                "local_format": "GGUF (llama.cpp)",
                "srv_gpu_layers": 28,
                "gpu_layers": 28,
                "srv_ctx": 3072,
                "ctx_length": 3072,
                "batch_size": 128,
                "max_tokens": 220,
                "threads": threads,
                "quant": quant,
                "fast_mode": False,
            }
        if vram_mb < 17 * 1024:
            return {
                "profile": "cuda_12gb_balanced",
                "local_backend": "CUDA",
                "local_loader": "llama.cpp",
                "local_format": "GGUF (llama.cpp)",
                "srv_gpu_layers": -1,
                "gpu_layers": 40,
                "srv_ctx": 4096,
                "ctx_length": 4096,
                "batch_size": 256,
                "max_tokens": 384,
                "threads": threads,
                "quant": quant,
                "fast_mode": False,
            }
        return {
            "profile": "cuda_high_vram",
            "local_backend": "CUDA",
            "local_loader": "llama.cpp",
            "local_format": "GGUF (llama.cpp)",
            "srv_gpu_layers": -1,
            "gpu_layers": 60,
            "srv_ctx": 8192,
            "ctx_length": 8192,
            "batch_size": 512,
            "max_tokens": 512,
            "threads": threads,
            "quant": quant,
            "fast_mode": False,
        }

    def _set_combo_text_if_present(self, combo, text):
        if not text:
            return
        idx = combo.findText(str(text))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _apply_model_recommendations(self, gpus, roles, *, save=True):
        rec = self._model_recommendations_for_hardware(gpus, roles)
        widgets = [
            self.local_backend, self.local_loader, self.local_format,
            self.srv_gpu_layers, self.gpu_layers, self.srv_ctx, self.ctx_length,
            self.batch_size, self.max_tokens, self.threads, self.quant,
            self.fast_mode,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._set_combo_text_if_present(self.local_backend, rec.get("local_backend"))
            self._set_combo_text_if_present(self.local_loader, rec.get("local_loader"))
            self._set_combo_text_if_present(self.local_format, rec.get("local_format"))
            self.srv_gpu_layers.setValue(int(rec["srv_gpu_layers"]))
            self.gpu_layers.setValue(int(rec["gpu_layers"]))
            self.srv_ctx.setValue(int(rec["srv_ctx"]))
            self.ctx_length.setValue(int(rec["ctx_length"]))
            self.batch_size.setValue(int(rec["batch_size"]))
            self.max_tokens.setValue(int(rec["max_tokens"]))
            self.threads.setValue(int(rec["threads"]))
            if rec.get("quant"):
                self._set_combo_text_if_present(self.quant, rec.get("quant"))
            self.fast_mode.setChecked(bool(rec.get("fast_mode", False)))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        main_gpu = self._selected_main_gpu(gpus, roles)
        label = str(main_gpu.get("name")) if main_gpu else "CPU"
        detail = (
            f"Optimized model tab for {label}: "
            f"{rec['profile']}, ctx={rec['ctx_length']}, "
            f"gpu_layers={rec['srv_gpu_layers']}, batch={rec['batch_size']}, "
            f"max_tokens={rec['max_tokens']}"
        )
        self.event_bus.log_entry.emit(f"[Hardware] {detail}")
        if save:
            self._save_settings()
        return detail

    def _describe_model_recommendations(self, gpus, roles):
        rec = self._model_recommendations_for_hardware(gpus, roles)
        main_gpu = self._selected_main_gpu(gpus, roles)
        label = str(main_gpu.get("name")) if main_gpu else "CPU"
        return (
            f"Recommended for {label}: {rec['profile']}, "
            f"ctx={rec['ctx_length']}, gpu_layers={rec['srv_gpu_layers']}, "
            f"batch={rec['batch_size']}, max_tokens={rec['max_tokens']} "
            "(not applied)"
        )

    def _refresh_hardware_routing(self, _checked=False, *, apply_recommendations=False):
        self.hardware_status.setText("Detecting GPUs...")
        apply_status_style(self.hardware_status, "color: #ccaa00;")

        def _fallback_local(_error=None, _detail=None):
            gpus = detect_nvidia_smi_gpus()
            self._hardware_gpus = gpus
            policy = self._current_hardware_policy() if gpus else default_hardware_policy(0)
            roles = resolve_gpu_roles(gpus, policy)
            self._hardware_policy = policy
            self._hardware_roles = roles
            self._populate_gpu_combo(gpus)
            optimize_detail = (
                self._apply_model_recommendations(gpus, roles, save=True)
                if apply_recommendations
                else self._describe_model_recommendations(gpus, roles)
            )
            self.hardware_status.setText(
                self._describe_hardware_routing(gpus, roles, policy)
                + "\n"
                + optimize_detail
            )
            apply_status_style(
                self.hardware_status,
                "color: #00aa40;" if gpus else "color: #ccaa00;",
            )

        def _on_success(data):
            if not isinstance(data, dict):
                _fallback_local()
                return
            fingerprint = data.get("fingerprint", {}) or {}
            gpus = list(fingerprint.get("cuda_devices") or [])
            if not gpus:
                snapshot = data.get("snapshot", {}) or {}
                gpus = list(snapshot.get("gpus") or [])
            routing = data.get("routing", {}) or {}
            policy = (
                routing.get("policy")
                if gpus
                else default_hardware_policy(0)
            ) or self._current_hardware_policy()
            roles = routing.get("roles") or resolve_gpu_roles(gpus, policy)
            self._hardware_gpus = gpus
            self._hardware_roles = roles
            self._hardware_policy = policy
            self._populate_gpu_combo(gpus)
            self._apply_policy_to_controls(policy)
            optimize_detail = (
                self._apply_model_recommendations(gpus, roles, save=True)
                if apply_recommendations
                else self._describe_model_recommendations(gpus, roles)
            )
            self.hardware_status.setText(
                self._describe_hardware_routing(gpus, roles, policy)
                + "\n"
                + optimize_detail
            )
            apply_status_style(
                self.hardware_status,
                "color: #00aa40;" if gpus else "color: #ccaa00;",
            )

        self.client.get_async(
            "/api/agents/hardware",
            timeout=4,
            default={},
            on_success=_on_success,
            on_error=_fallback_local,
        )

    def _apply_hardware_routing(self):
        if not self._hardware_gpus:
            self._hardware_gpus = detect_nvidia_smi_gpus()
            self._populate_gpu_combo(self._hardware_gpus)
        policy = self._current_hardware_policy()
        roles = resolve_gpu_roles(self._hardware_gpus, policy)
        self._hardware_policy = policy
        self._hardware_roles = roles
        optimize_detail = self._apply_model_recommendations(
            self._hardware_gpus, roles, save=False
        )
        self._save_settings()
        try:
            write_profile_hardware_policy(
                policy,
                roles=roles,
                detected_gpus=self._hardware_gpus,
            )
        except Exception as exc:
            self.event_bus.log_entry.emit(
                f"[Hardware] Could not update profile settings: {exc}"
            )

        self.hardware_status.setText(
            self._describe_hardware_routing(self._hardware_gpus, roles, policy)
            + "\n"
            + optimize_detail
        )
        apply_status_style(self.hardware_status, "color: #00aa40;")

        def _on_success(data):
            if isinstance(data, dict):
                self._hardware_roles = data.get("roles") or self._hardware_roles
            self.event_bus.log_entry.emit(
                f"[Hardware] Routing saved: {policy.get('hardware_mode')}"
            )
            self._refresh_hardware_routing()

        def _on_error(error=None, detail=None):
            self.event_bus.log_entry.emit(
                f"[Hardware] Routing saved locally; core sync pending: {error or detail or 'offline'}"
            )

        self.client.post_async(
            "/api/hardware/config",
            json={"policy": policy},
            timeout=5,
            default={},
            on_success=_on_success,
            on_error=_on_error,
        )

    def _main_llm_gpu_env_updates(self):
        policy = self._current_hardware_policy()
        gpus = self._hardware_gpus or detect_nvidia_smi_gpus()
        roles = resolve_gpu_roles(gpus, policy)
        self._hardware_roles = roles
        visible = roles.get("visible_main_llm_devices") or []
        if not visible:
            return {}, ""
        main = visible[0]
        updates = {
            "CUDA_VISIBLE_DEVICES": str(main),
            "REVIA_MAIN_LLM_GPU": str(main),
        }
        support = roles.get("support_gpu_index")
        if support is not None:
            updates["REVIA_SUPPORT_GPU"] = str(support)
        return updates, f"main GPU {main}"

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _save_settings(self):
        """Write current UI state to model_settings.json.

        API keys are persisted separately in ignored local settings so they
        cannot be committed with normal model configuration.
        """
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
            "hardware_policy": self._current_hardware_policy(),
        }
        try:
            _SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            try:
                write_profile_hardware_policy(
                    data["hardware_policy"],
                    roles=self._hardware_roles,
                    detected_gpus=self._hardware_gpus,
                )
            except Exception:
                pass
            self._save_secret_settings()
        except Exception as e:
            self.event_bus.log_entry.emit(f"[Model] Could not save settings: {e}")

    def _save_secret_settings(self):
        """Persist local-only model secrets outside tracked settings."""
        api_key = self.api_key.text().strip()
        try:
            if api_key:
                _SECRET_SETTINGS_FILE.write_text(
                    json.dumps({"api_key": api_key}, indent=2),
                    encoding="utf-8",
                )
            elif _SECRET_SETTINGS_FILE.exists():
                _SECRET_SETTINGS_FILE.unlink()
        except Exception as e:
            self.event_bus.log_entry.emit(f"[Model] Could not save local secrets: {e}")

    def _load_secret_settings(self) -> dict:
        """Load ignored local model secrets plus environment fallback."""
        data: dict = {}
        if _SECRET_SETTINGS_FILE.exists():
            try:
                raw = json.loads(_SECRET_SETTINGS_FILE.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data.update(raw)
            except Exception as e:
                logger.warning(f"Error loading local model secrets: {e}")
        data.setdefault(
            "api_key",
            os.environ.get("REVIA_API_KEY", "") or os.environ.get("OPENAI_API_KEY", ""),
        )
        return data

    def _load_settings(self):
        """Restore UI state from model_settings.json (if it exists)."""
        env_api_key = (
            os.environ.get("REVIA_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        )
        if (
            not _SETTINGS_FILE.exists()
            and not _SECRET_SETTINGS_FILE.exists()
            and not env_api_key
        ):
            return
        data: dict = {}
        try:
            if _SETTINGS_FILE.exists():
                loaded = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data.update(loaded)
        except Exception as e:
            logger.warning(f"Error loading model settings: {e}")
            return
        data.update(self._load_secret_settings())
        if not isinstance(data.get("hardware_policy"), dict):
            data["hardware_policy"] = read_saved_hardware_policy()

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
            self._apply_policy_to_controls(data.get("hardware_policy") or {})

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
        payload = self._build_connection_test_payload()
        threading.Thread(
            target=self._run_connection_test,
            args=(payload,),
            daemon=True,
            name="revia-model-test",
        ).start()

    def _build_connection_test_payload(self):
        is_online = self.source_type.currentIndex() == 1
        payload = {
            "source": "online" if is_online else "local",
        }
        if is_online:
            payload.update(
                {
                    "endpoint": self.api_endpoint.text().strip(),
                    "key": self.api_key.text().strip(),
                    "model": self.api_model.currentText().strip(),
                    "provider": self.api_provider.currentText(),
                }
            )
        else:
            payload.update(
                {
                    "server": self.local_server.currentText(),
                    "server_url": self.local_server_url.text().strip(),
                    "path": self.local_path.text().strip(),
                    "core_url": self.client.BASE_URL,
                }
            )
        return payload

    def _run_connection_test(self, payload):
        if payload.get("source") == "online":
            result = self._test_online(payload)
        else:
            result = self._test_local(payload)
        self.connection_test_completed.emit(result)

    def _apply_connection_test_result(self, result):
        self.connect_btn.setEnabled(True)
        if not isinstance(result, dict):
            self.conn_status.setText("Status: Connection test failed")
            apply_status_style(self.conn_status, "color: #cc3040;")
            self.disconnect_btn.setEnabled(False)
            return
        if result.get("push_config"):
            self._push_config_to_core(
                result.get("source", "local"),
                verified=bool(result.get("verified")),
            )
        self.conn_status.setText(str(result.get("status_text", "Status: Error")))
        apply_status_style(
            self.conn_status,
            str(result.get("status_style", "color: #cc3040;")),
        )
        self.disconnect_btn.setEnabled(bool(result.get("disconnect_enabled")))

    def _push_config_to_core(self, source, verified=False):
        self._save_settings()
        cfg = {
            "source": source,
            "temperature": self.temperature.value(),
            "max_tokens": self.max_tokens.value(),
            "top_p": self.top_p.value(),
            "ctx_length": self.ctx_length.value(),
            "fast_mode": self.fast_mode.isChecked(),
            "verified": bool(verified),
            "hardware_policy": self._current_hardware_policy(),
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

    def _test_local(self, payload):
        server = str(payload.get("server", "") or "")
        server_url = str(payload.get("server_url", "") or "").strip()
        path = str(payload.get("path", "") or "").strip()
        core_url = str(payload.get("core_url", self.client.BASE_URL) or self.client.BASE_URL).strip()

        if not server_url:
            return {
                "source": "local",
                "push_config": False,
                "verified": False,
                "status_text": "Status: No server URL specified",
                "status_style": "color: #cc3040;",
                "disconnect_enabled": False,
            }

        llm_ok, models = self._probe_local_server(server_url)
        llm_detail = ", ".join(models[:3]) if models else "connected"
        if not llm_ok:
            llm_detail = "not running"

        # Check REVIA core
        core_ok = False
        try:
            import requests
            r = requests.get(
                f"{core_url}/api/status", timeout=2
            )
            core_ok = r.ok
        except Exception as e:
            logger.debug(f"Error checking core status: {e}")

        if llm_ok:
            file_info = ""
            if path and os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                file_info = f" | File: {size_mb:.0f} MB"
            return {
                "source": "local",
                "push_config": True,
                "verified": True,
                "status_text": (
                    f"Status: {server} OK ({llm_detail}){file_info} | "
                    f"Core: {'Online' if core_ok else 'Offline'}"
                ),
                "status_style": "color: #00aa40;",
                "disconnect_enabled": True,
            }
        return {
            "source": "local",
            "push_config": True,
            "verified": False,
            "status_text": f"Status: {server} at {server_url} - {llm_detail}",
            "status_style": "color: #cc3040;",
            "disconnect_enabled": False,
        }

    def _test_online(self, payload):
        endpoint = str(payload.get("endpoint", "") or "").strip()
        key = str(payload.get("key", "") or "").strip()
        model = str(payload.get("model", "") or "").strip()
        provider = str(payload.get("provider", "") or "")

        if not endpoint:
            return {
                "source": "online",
                "push_config": False,
                "verified": False,
                "status_text": "Status: No endpoint specified",
                "status_style": "color: #cc3040;",
                "disconnect_enabled": False,
            }
        if not key:
            return {
                "source": "online",
                "push_config": False,
                "verified": False,
                "status_text": "Status: No API key provided",
                "status_style": "color: #cc3040;",
                "disconnect_enabled": False,
            }

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
                return {
                    "source": "online",
                    "push_config": True,
                    "verified": True,
                    "status_text": f"Status: Connected to {provider} | Model: {model}",
                    "status_style": "color: #00aa40;",
                    "disconnect_enabled": True,
                }
            if r.status_code == 401:
                return {
                    "source": "online",
                    "push_config": False,
                    "verified": False,
                    "status_text": "Status: Invalid API key (401)",
                    "status_style": "color: #cc3040;",
                    "disconnect_enabled": False,
                }
            if r.status_code == 403:
                return {
                    "source": "online",
                    "push_config": False,
                    "verified": False,
                    "status_text": "Status: Access denied (403) - check key permissions",
                    "status_style": "color: #cc3040;",
                    "disconnect_enabled": False,
                }
            return {
                "source": "online",
                "push_config": False,
                "verified": False,
                "status_text": f"Status: API returned {r.status_code}",
                "status_style": "color: #cc8800;",
                "disconnect_enabled": False,
            }
        except requests.exceptions.Timeout:
            return {
                "source": "online",
                "push_config": False,
                "verified": False,
                "status_text": "Status: Connection timed out",
                "status_style": "color: #cc3040;",
                "disconnect_enabled": False,
            }
        except requests.exceptions.ConnectionError:
            return {
                "source": "online",
                "push_config": False,
                "verified": False,
                "status_text": "Status: Cannot reach endpoint",
                "status_style": "color: #cc3040;",
                "disconnect_enabled": False,
            }
        except Exception as e:
            return {
                "source": "online",
                "push_config": False,
                "verified": False,
                "status_text": f"Status: Error - {e}",
                "status_style": "color: #cc3040;",
                "disconnect_enabled": False,
            }

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
            self._refresh_hardware_routing()
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
