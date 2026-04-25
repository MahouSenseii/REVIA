from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QSplitter, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, QProcess

from app.camera_service import CameraService
from app.audio_service import AudioService
from app.continuous_audio import ContinuousAudioPipeline
from app.assistant_status_manager import AssistantStatusManager
from app.conversation_policy import ConversationBehaviorController
from app.conversation_starter import ConversationStarter
from app.runtime_state_sync import RuntimeStateSync
from gui.widgets.sidebar import SidebarWidget
from gui.widgets.topbar import TopBar
from gui.widgets.chat_panel import ChatPanel
from gui.widgets.inference_panel import InferencePanel
from gui.tabs.profile_tab import ProfileTab
from gui.tabs.model_tab import ModelTab
from gui.tabs.memory_tab import MemoryTab
from gui.tabs.voice_tab import VoiceTab
from gui.tabs.vision_tab import VisionTab
from gui.tabs.filters_tab import FiltersTab
from gui.tabs.logs_tab import LogsTab
from gui.tabs.system_tab import SystemTab
from gui.tabs.theme_tab import ThemeTab
from gui.tabs.emotions_tab import EmotionsTab
from gui.tabs.integrations_tab import IntegrationsTab
from gui.tabs.sing_tab import SingTab


class MainWindow(QMainWindow):
    def __init__(self, event_bus, client, theme_mgr):
        super().__init__()
        self.event_bus = event_bus
        self.client = client
        self.theme_mgr = theme_mgr
        self.camera_service = CameraService(self)
        self.audio_service = AudioService(self)
        self.continuous_audio = ContinuousAudioPipeline(parent=self)
        self.sing_library = None
        self.sing_queue = None
        self.sing_handler = None
        self.behavior_controller = None
        self.conversation_starter = None
        self.runtime_state_sync = None
        self.assistant_status_manager = None
        self.setWindowTitle("REVIA \u2014 Neural Assistant Controller")
        self.setMinimumSize(720, 520)
        self._build_ui()
        self._connect_signals()
        self.conversation_starter.enable()
        QTimer.singleShot(600, self._auto_start_services_on_launch)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.shell_splitter = QSplitter(Qt.Horizontal)
        self.shell_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.shell_splitter)

        # Left sidebar
        self.sidebar = SidebarWidget(self.event_bus)
        self.sidebar.setMinimumWidth(118)
        self.sidebar.setMaximumWidth(220)
        self.sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.shell_splitter.addWidget(self.sidebar)

        # Center area
        center = QWidget()
        center.setObjectName("centerPanel")
        center.setMinimumWidth(320)
        center.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(8)

        self.topbar = TopBar(self.event_bus)
        center_layout.addWidget(self.topbar)

        self.chat_panel = ChatPanel(
            self.event_bus, self.client, self.audio_service, self.continuous_audio
        )
        self.chat_panel.set_camera_service(self.camera_service)
        center_layout.addWidget(self.chat_panel, stretch=1)

        self.inference_panel = InferencePanel(self.event_bus)
        center_layout.addWidget(self.inference_panel)

        self.shell_splitter.addWidget(center)

        # Right panel - container gives the tab widget a fixed inset margin so
        # content never bleeds to the splitter edge.
        right_container = QWidget()
        right_container.setObjectName("rightPanel")
        right_container.setMinimumWidth(280)
        right_container.setMaximumWidth(660)
        right_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_container_layout = QVBoxLayout(right_container)
        right_container_layout.setContentsMargins(6, 6, 6, 6)
        right_container_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("rightTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_container_layout.addWidget(self.tabs)

        # ── Tab 1: Personality ────────────────────────────────────
        personality_container = QWidget()
        personality_layout = QVBoxLayout(personality_container)
        personality_layout.setContentsMargins(0, 4, 0, 0)
        personality_layout.setSpacing(0)
        self.personality_tabs = QTabWidget()
        self.personality_tabs.setDocumentMode(True)
        self.personality_tabs.setObjectName("categoryTabs")
        personality_layout.addWidget(self.personality_tabs)
        self.tabs.addTab(personality_container, "Persona")

        self.profile_tab = ProfileTab(self.event_bus, self.client)
        self.personality_tabs.addTab(self.profile_tab, "Profile")

        self.memory_tab = MemoryTab(self.event_bus, self.client)
        self.personality_tabs.addTab(self.memory_tab, "Memory")

        self.tabs.addTab(
            EmotionsTab(self.event_bus, self.client), "Emotions"
        )

        self.sing_tab = SingTab(self.event_bus, self.client)
        self.personality_tabs.addTab(self.sing_tab, "Sing")
        self._init_sing_system()

        # ── Tab 2: System ─────────────────────────────────────────
        system_container = QWidget()
        system_layout = QVBoxLayout(system_container)
        system_layout.setContentsMargins(0, 4, 0, 0)
        system_layout.setSpacing(0)
        self.system_tabs = QTabWidget()
        self.system_tabs.setDocumentMode(True)
        self.system_tabs.setObjectName("categoryTabs")
        system_layout.addWidget(self.system_tabs)
        self.tabs.addTab(system_container, "System")

        self.model_tab = ModelTab(self.event_bus, self.client)
        self.system_tabs.addTab(self.model_tab, "Model")

        self.voice_tab = VoiceTab(
            self.event_bus, self.client, self.audio_service
        )
        self.system_tabs.addTab(self.voice_tab, "Voice")

        self.vision_tab = VisionTab(
            self.event_bus, self.client, self.camera_service
        )
        self.system_tabs.addTab(self.vision_tab, "Vision")

        self.system_tab = SystemTab(
            self.event_bus, self.client, self.theme_mgr
        )
        self.system_tabs.addTab(self.system_tab, "Server")

        self.filters_tab = FiltersTab(self.event_bus, self.client)
        self.system_tabs.addTab(self.filters_tab, "Filters")

        # ── Tab 3: Advanced ────────────────────────────────────────
        advanced_container = QWidget()
        advanced_layout = QVBoxLayout(advanced_container)
        advanced_layout.setContentsMargins(0, 4, 0, 0)
        advanced_layout.setSpacing(0)
        self.advanced_tabs = QTabWidget()
        self.advanced_tabs.setDocumentMode(True)
        self.advanced_tabs.setObjectName("categoryTabs")
        advanced_layout.addWidget(self.advanced_tabs)
        self.tabs.addTab(advanced_container, "Advanced")

        self.advanced_tabs.addTab(LogsTab(self.event_bus), "Logs")

        self.theme_tab = ThemeTab(
            self.event_bus, self.client, self.theme_mgr
        )
        self.advanced_tabs.addTab(self.theme_tab, "Theme")

        self.advanced_tabs.addTab(
            IntegrationsTab(self.event_bus, self.client), "Integrations"
        )

        # Give chat panel access to the voice manager for TTS
        self.chat_panel.set_voice_manager(self.voice_tab.voice_mgr)
        self.chat_panel.set_voice_tab(self.voice_tab)
        self.behavior_controller = ConversationBehaviorController(
            status_provider=self.client.get_status_snapshot,
            log_fn=self.event_bus.log_entry.emit,
            is_user_speaking=self.audio_service.is_listening,
            is_assistant_speaking=self.chat_panel.is_assistant_speaking,
            is_tts_enabled=self.chat_panel.is_tts_enabled,
            is_tts_ready=self.voice_tab.voice_mgr.is_output_ready,
            is_stt_ready=self.audio_service.is_stt_available,
        )
        self.conversation_starter = ConversationStarter(
            self.client,
            self.event_bus,
            self.behavior_controller,
            interval_ms=120_000,
            parent=self,
        )
        self.chat_panel.set_conversation_starter(self.conversation_starter)
        self.chat_panel.set_behavior_controller(self.behavior_controller)

        self.assistant_status_manager = AssistantStatusManager(
            event_bus=self.event_bus,
            model_tab=self.model_tab,
            voice_tab=self.voice_tab,
            vision_tab=self.vision_tab,
            filters_tab=self.filters_tab,
            memory_tab=self.memory_tab,
            system_tab=self.system_tab,
            profile_tab=self.profile_tab,
            chat_panel=self.chat_panel,
            audio_service=self.audio_service,
            parent=self,
        )

        self.runtime_state_sync = RuntimeStateSync(
            client=self.client,
            event_bus=self.event_bus,
            model_tab=self.model_tab,
            voice_tab=self.voice_tab,
            filters_tab=self.filters_tab,
            memory_tab=self.memory_tab,
            system_tab=self.system_tab,
            profile_tab=self.profile_tab,
            chat_panel=self.chat_panel,
            audio_service=self.audio_service,
            assistant_status_manager=self.assistant_status_manager,
            parent=self,
        )

        self.shell_splitter.addWidget(right_container)
        self.shell_splitter.setStretchFactor(0, 0)
        self.shell_splitter.setStretchFactor(1, 4)
        self.shell_splitter.setStretchFactor(2, 1)
        # Give right panel a comfortable default width (380px) so content
        # breathes without feeling cramped.
        self.shell_splitter.setSizes([180, 740, 380])

    def _current_mood_label(self):
        snapshot = self.client.get_status_snapshot()
        emotion = snapshot.get("emotion", {}) if isinstance(snapshot, dict) else {}
        return str(emotion.get("label", "neutral") or "neutral").lower()

    def _init_sing_system(self):
        try:
            self.sing_library, self.sing_queue, self.sing_handler = (
                self.voice_tab.voice_mgr.backend.init_sing_system(
                    event_bus=self.event_bus,
                    get_mood_fn=self._current_mood_label,
                    voice_profile_fn=lambda: self.voice_tab.voice_mgr.active_profile,
                )
            )
            self.sing_tab.set_sing_system(
                self.sing_library,
                self.sing_queue,
                self.sing_handler,
            )
            self.chat_panel.set_sing_handler(self.sing_handler)
            self.event_bus.sing_state_changed.connect(self.sing_tab.update_state)
            self.event_bus.sing_lyrics_update.connect(self.sing_tab.update_lyrics)
        except Exception as exc:
            self.event_bus.log_entry.emit(f"[Sing] Failed to initialize: {exc}")

    def _connect_signals(self):
        self.event_bus.connection_changed.connect(self._on_connection)
        self.camera_service.frame_ready.connect(
            self.inference_panel.on_camera_frame
        )
        if self.runtime_state_sync:
            self.event_bus.connection_changed.connect(
                lambda _connected: self.runtime_state_sync.schedule_sync()
            )

    def _on_connection(self, connected):
        self.topbar.set_health("Online" if connected else "Offline")
        if connected and self.conversation_starter:
            self.conversation_starter.greet_on_startup(delay_ms=6_000)

    def _auto_start_services_on_launch(self):
        if self.continuous_audio:
            self.continuous_audio.start()
        if hasattr(self.system_tab, "auto_start_on_launch"):
            self.system_tab.auto_start_on_launch()
        if hasattr(self.voice_tab, "auto_start_on_launch"):
            QTimer.singleShot(700, self.voice_tab.auto_start_on_launch)
        if hasattr(self.chat_panel, "sync_mic_state_from_audio"):
            QTimer.singleShot(1200, self.chat_panel.sync_mic_state_from_audio)
        if hasattr(self.model_tab, "auto_start_on_launch"):
            QTimer.singleShot(1200, self.model_tab.auto_start_on_launch)
        if self.runtime_state_sync:
            QTimer.singleShot(1400, self.runtime_state_sync.schedule_sync)

    def closeEvent(self, event):
        self.camera_service.disconnect_camera()
        try:
            if self.continuous_audio:
                self.continuous_audio.stop()
        except Exception:
            pass
        try:
            proc = getattr(self.model_tab, '_llm_process', None)
            if proc and proc.state() == QProcess.Running:
                self.model_tab._stop_llm_server()
        except Exception:
            pass
        try:
            proc = getattr(self.system_tab, '_core_process', None)
            if proc and proc.state() == QProcess.Running:
                self.system_tab._stop_core_server()
        except Exception:
            pass
        try:
            proc = getattr(self.voice_tab, '_tts_process', None)
            if proc and proc.state() == QProcess.Running:
                self.voice_tab._stop_tts_server()
        except Exception:
            pass
        if hasattr(self.client, "shutdown"):
            self.client.shutdown()
        super().closeEvent(event)
