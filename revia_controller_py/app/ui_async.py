import logging

from PySide6.QtCore import QObject, Signal

_log = logging.getLogger(__name__)


class UiThreadBridge(QObject):
    """Marshal callbacks from worker threads back onto the Qt UI thread."""

    _call = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._call.connect(self._invoke)

    def dispatch(self, fn):
        if callable(fn):
            self._call.emit(fn)

    @staticmethod
    def _invoke(fn):
        try:
            fn()
        except Exception:
            _log.exception("[UiThreadBridge] UI callback failed")
