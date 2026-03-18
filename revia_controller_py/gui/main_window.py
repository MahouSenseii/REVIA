from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
)
from PySide6.QtCore import Qt, QTimer

from app.camera_service import CameraService
from app.audio_service import AudioService
from app.assistant_status_manager import AssistantStatusManager
from app.conversation_policy import ConversationBehaviorController
from app.conversation_starter import ConversationStarter
from app.runtime_state_sync import RuntimeStateSync
from gui.widgets.sidebar import SidebarWidget
from gui.widgets.topbar import TopBar
from gui.widgets.chat_panel import ChatPanel
from gui.widgets.inference_panel import InferencePanel
from gui.widgets.status_panel import StatusPanel
from gui.tabs.profile_tab import ProfileTab
from gui.tabs.model_tab import ModelTab
from gui.tabs.memory_tab import MemoryTab
from gui.tabs.voice_tab import VoiceTab
from gui.tabs.vision_tab import VisionTab
from gui.tabs.filters_tab import FiltersTab
from gui.tabs.logs_tab import LogsTab
from gui.tabs.system_tab import SystemTab
from gui.tabs.emotions_tab import EmotionsTab
from gui.tabs.integrations_tab import IntegrationsTab


class MainWindow(QMainWindow):
    def __init__(self, event_bus, client, theme_mgr):
        super().__init__()
        self.event_bus = event_bus
        self.client = client
        self.theme_mgr = theme_mgr
        self.camera_service = CameraService(self)
        self.audio_service = AudioService(self)
        self.behavior_controller = None
        self.conversation_starter = None
        self.runtime_state_sync = None
        self.assistant_status_manager = None
        self.setWindowTitle("REVIA \u2014 Neural Assistant Controller")
        self.setMinimumSize(1400, 900)
        self._build_ui()
        self._connect_signals()
        self.conversation_starter.enable()
        self.conversation_starter.greet_on_startup(delay_ms=6_000)
        QTimer.singleShot(600, self._auto_start_services_on_launch)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left sidebar
        self.sidebar = SidebarWidget(self.event_bus)
        self.sidebar.setFixedWidth(200)
        main_layout.addWidget(self.sidebar)

        # Center area
        center = QWidget()
        center.setObjectName("centerPanel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(8)

        self.topbar = TopBar(self.event_bus)
        center_layout.addWidget(self.topbar)

        self.status_panel = StatusPanel(self.event_bus)
        center_layout.addWidget(self.status_panel)

        self.chat_panel = ChatPanel(
            self.event_bus, self.client, self.audio_service
        )
        self.chat_panel.set_camera_service(self.camera_service)
        center_layout.addWidget(self.chat_panel, stretch=1)

        self.inference_panel = InferencePanel(self.event_bus)
        center_layout.addWidget(self.inference_panel)

        main_layout.addWidget(center, stretch=1)

        # Right tabs
        self.tabs = QTabWidget()
        self.tabs.setObjectName("rightTabs")
        self.tabs.setMinimumWidth(360)
        self.tabs.setMaximumWidth(440)

        self.profile_tab = ProfileTab(self.event_bus, self.client)
        self.tabs.addTab(
            self.profile_tab, "Profile"
        )
        self.model_tab = ModelTab(self.event_bus, self.client)
        self.tabs.addTab(self.model_tab, "Model")
        self.memory_tab = MemoryTab(self.event_bus, self.client)
        self.tabs.addTab(
            self.memory_tab, "Memory"
        )
        self.voice_tab = VoiceTab(
            self.event_bus, self.client, self.audio_service
        )
        self.tabs.addTab(self.voice_tab, "Voice")

        # Give chat panel access to the voice manager for TTS
        self.chat_panel.set_voice_manager(self.voice_tab.voice_mgr)
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

        self.vision_tab = VisionTab(
            self.event_bus, self.client, self.camera_service
        )
        self.tabs.addTab(self.vision_tab, "Vision")

        self.tabs.addTab(
            EmotionsTab(self.event_bus, self.client), "Emotions"
        )
        self.filters_tab = FiltersTab(self.event_bus, self.client)
        self.tabs.addTab(
            self.filters_tab, "Filters"
        )
        self.tabs.addTab(LogsTab(self.event_bus), "Logs")
        self.tabs.addTab(
            IntegrationsTab(self.event_bus, self.client), "Integrations"
        )
        self.system_tab = SystemTab(
            self.event_bus, self.client, self.theme_mgr
        )
        self.tabs.addTab(self.system_tab, "System")

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

        main_layout.addWidget(self.tabs)

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
        self.topbar.set_health("Connecting" if connected else "Offline")

    def _auto_start_services_on_launch(self):
        if hasattr(self.system_tab, "auto_start_on_launch"):
            self.system_tab.auto_start_on_launch()
        if hasattr(self.model_tab, "auto_start_on_launch"):
            QTimer.singleShot(1200, self.model_tab.auto_start_on_launch)
        if self.runtime_state_sync:
            QTimer.singleShot(1400, self.runtime_state_sync.schedule_sync)

    def closeEvent(self, event):
        self.camera_service.disconnect_camera()
        if (getattr(self.model_tab, '_llm_process', None)
                and self.model_tab._llm_process.state() != 0):
            self.model_tab._stop_llm_server()
        if (self.system_tab._core_process
                and self.system_tab._core_process.state() != 0):
            self.system_tab._stop_core_server()
        if (self.voice_tab._tts_process
                and self.voice_tab._tts_process.state() != 0):
            self.voice_tab._stop_tts_server()
        if hasattr(self.client, "shutdown"):
            self.client.shutdown()
        super().closeEvent(event)
