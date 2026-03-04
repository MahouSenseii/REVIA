from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
)
from PySide6.QtCore import Qt

from app.camera_service import CameraService
from app.audio_service import AudioService
from app.conversation_starter import ConversationStarter
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
        self.conversation_starter = ConversationStarter(
            client, event_bus, interval_ms=300_000, parent=self
        )
        self.setWindowTitle("REVIA \u2014 Neural Assistant Controller")
        self.setMinimumSize(1400, 900)
        self._build_ui()
        self._connect_signals()

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

        self.tabs.addTab(
            ProfileTab(self.event_bus, self.client), "Profile"
        )
        self.tabs.addTab(ModelTab(self.event_bus, self.client), "Model")
        self.tabs.addTab(
            MemoryTab(self.event_bus, self.client), "Memory"
        )
        self.voice_tab = VoiceTab(
            self.event_bus, self.client, self.audio_service
        )
        self.tabs.addTab(self.voice_tab, "Voice")

        # Give chat panel access to the voice manager for TTS
        self.chat_panel.set_voice_manager(self.voice_tab.voice_mgr)
        # Give chat panel access to the conversation starter for activity tracking
        self.chat_panel.set_conversation_starter(self.conversation_starter)

        self.vision_tab = VisionTab(
            self.event_bus, self.client, self.camera_service
        )
        self.tabs.addTab(self.vision_tab, "Vision")

        self.tabs.addTab(
            EmotionsTab(self.event_bus, self.client), "Emotions"
        )
        self.tabs.addTab(
            FiltersTab(self.event_bus, self.client), "Filters"
        )
        self.tabs.addTab(LogsTab(self.event_bus), "Logs")
        self.tabs.addTab(
            IntegrationsTab(self.event_bus, self.client), "Integrations"
        )
        self.system_tab = SystemTab(
            self.event_bus, self.client, self.theme_mgr
        )
        self.tabs.addTab(self.system_tab, "System")

        main_layout.addWidget(self.tabs)

    def _connect_signals(self):
        self.event_bus.connection_changed.connect(self._on_connection)
        self.camera_service.frame_ready.connect(
            self.inference_panel.on_camera_frame
        )

    def _on_connection(self, connected):
        self.topbar.set_health("Online" if connected else "Offline")
        if connected:
            if not self.conversation_starter.is_enabled:
                self.conversation_starter.enable()
                self.conversation_starter.greet_on_startup(delay_ms=4_000)
        else:
            self.conversation_starter.disable()

    def closeEvent(self, event):
        self.camera_service.disconnect_camera()
        model_tab = self.tabs.widget(1)
        if (getattr(model_tab, '_llm_process', None)
                and model_tab._llm_process.state() != 0):
            model_tab._stop_llm_server()
        if (self.system_tab._core_process
                and self.system_tab._core_process.state() != 0):
            self.system_tab._stop_core_server()
        if (self.voice_tab._tts_process
                and self.voice_tab._tts_process.state() != 0):
            self.voice_tab._stop_tts_server()
        super().closeEvent(event)
