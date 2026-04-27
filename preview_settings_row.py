"""Preview — render a couple of SettingsRows to eyeball them."""
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from theme import apply_theme
from widgets.settings_row import SettingsRow


def main():
    app = QApplication(sys.argv)
    apply_theme(app)

    window = QMainWindow()
    window.setWindowTitle("SettingsRow preview")
    window.setMinimumSize(640, 420)
    window.setStyleSheet("background-color: #0F0F0F;")

    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(14)

    # Empty row (no value yet)
    row1 = SettingsRow(
        secret_key="openai_api_key",
        label_text="OpenAI API Key",
        placeholder="sk-proj-...",
        helper_text="Get your key at platform.openai.com → API keys.",
    )
    row1.save_requested.connect(
        lambda key, value: print(f"[would save] {key} = {value[:8]}...")
    )
    layout.addWidget(row1)

    # Row with a pre-loaded value — see the "Saved" status
    row2 = SettingsRow(
        secret_key="anthropic_api_key",
        label_text="Anthropic API Key",
        placeholder="sk-ant-...",
        helper_text="Get your key at console.anthropic.com.",
    )
    row2.set_value("sk-ant-example-preloaded-value-for-preview")
    row2.save_requested.connect(
        lambda key, value: print(f"[would save] {key}")
    )
    layout.addWidget(row2)

    # Row showing a simulated error state
    row3 = SettingsRow(
        secret_key="google_api_key",
        label_text="Google AI Key",
        placeholder="AIza...",
    )
    row3.show_save_error("Keyring refused the write")
    layout.addWidget(row3)

    layout.addStretch()

    window.setCentralWidget(central)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()