import sys

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from theme import apply_theme
from windows.main_window import MainWindow

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("HECTOR-AI")
    app.setApplicationDisplayName("HECTOR-AI")
    app.setOrganizationName("Karri")
    icon_path = Path(__file__).parent / "assets" / "logo.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
