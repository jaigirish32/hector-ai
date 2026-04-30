import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from paths import resource_path
from theme import apply_theme
from windows.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("HECTOR-AI")
    app.setApplicationDisplayName("HECTOR-AI")
    app.setOrganizationName("Karri")

    # Icon is optional — silently skip if the asset isn't present (dev
    # checkouts where assets/ hasn't been added, or stripped builds).
    icon_path = resource_path("assets/logo.jpeg")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()