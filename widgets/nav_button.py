from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton

class NavButton(QPushButton):
    clicked_with_index = Signal(int)

    def __init__(self, label: str, index: int) -> None:
        super().__init__(label)

        self._index = index
        self.setObjectName("navButton")
        self.setCursor(self._pointer_cursor())

        self.setProperty("active", False)

        self.clicked.connect(self._on_clicked)
    
    def set_active(self, is_active: bool) -> None:
        self.setProperty("active", is_active)
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_clicked(self) -> None:
        self.clicked_with_index.emit(self._index)

    @staticmethod
    def _pointer_cursor():
        from PySide6.QtCore import Qt
        return Qt.CursorShape.PointingHandCursor