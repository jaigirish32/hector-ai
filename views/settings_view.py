"""
The Settings view — where users enter and manage their provider API keys.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from models import DEFAULT_MODELS, Provider
from settings_manager import SecretKey, SettingsError, SettingsManager
from widgets.settings_row import SettingsRow


@dataclass(frozen=True)
class FieldSpec:
    secret_key: str
    label: str
    placeholder: str
    helper: str = ""


PROVIDER_SECTIONS: list[tuple[str, list[FieldSpec]]] = [
    (
        "OpenAI",
        [
            FieldSpec(
                secret_key=SecretKey.OPENAI_API_KEY,
                label="API key",
                placeholder="sk-proj-...",
                helper="Generate at platform.openai.com → API keys. "
                       "Scoped keys (project-bound) are safer than account-wide keys.",
            ),
        ],
    ),
    (
        "Azure OpenAI",
        [
            FieldSpec(
                secret_key=SecretKey.AZURE_OPENAI_API_KEY,
                label="API key",
                placeholder="Paste your Azure OpenAI key",
                helper="Found in Azure Portal → your OpenAI resource → Keys and Endpoint.",
            ),
            FieldSpec(
                secret_key=SecretKey.AZURE_OPENAI_ENDPOINT,
                label="Endpoint URL",
                placeholder="https://your-resource.openai.azure.com/",
                helper="The full resource URL from the same Azure page.",
            ),
            FieldSpec(
                secret_key=f"{SecretKey.AZURE_OPENAI_DEPLOYMENT_PREFIX}gpt-4.1-azure",
                label="Deployment name for GPT-4.1",
                placeholder="e.g. gpt-4.1",
                helper="The deployment name you configured in Azure Portal for this model. "
                       "Case-sensitive. Required — there is no default.",
            ),
        ],
    ),
    (
        "Anthropic (Claude)",
        [
            FieldSpec(
                secret_key=SecretKey.ANTHROPIC_API_KEY,
                label="API key",
                placeholder="sk-ant-...",
                helper="Generate at console.anthropic.com → Settings → API keys.",
            ),
        ],
    ),
    (
        "Google (Gemini)",
        [
            FieldSpec(
                secret_key=SecretKey.GOOGLE_API_KEY,
                label="API key",
                placeholder="AIza...",
                helper="Free tier available at aistudio.google.com → Get API key.",
            ),
        ],
    ),
]


class SettingsView(QWidget):
    """Form-based view for managing provider credentials."""

    configuration_changed = Signal()

    def __init__(self, settings: SettingsManager | None = None) -> None:
        super().__init__()

        self._settings = settings or SettingsManager()
        self._rows: dict[str, SettingsRow] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(28, 18, 28, 28)
        form_layout.setSpacing(20)

        for section_title, fields in PROVIDER_SECTIONS:
            form_layout.addWidget(self._build_section(section_title, fields))

        form_layout.addStretch()
        self._scroll.setWidget(form_container)
        root.addWidget(self._scroll, stretch=1)

        self._load_saved_values()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("settingsHeader")

        layout = QVBoxLayout(header)
        layout.setContentsMargins(28, 22, 28, 18)
        layout.setSpacing(6)

        title = QLabel("Settings")
        title.setObjectName("settingsTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "API keys are stored in your operating system's encrypted credential vault — "
            "never in plain-text files or source code."
        )
        subtitle.setObjectName("settingsSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self._readiness_label = QLabel("")
        self._readiness_label.setObjectName("readinessLabel")
        layout.addWidget(self._readiness_label)

        self._refresh_readiness()
        return header

    def _build_section(self, section_title: str, fields: list[FieldSpec]) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(section_title.upper())
        title.setObjectName("sectionLabel")
        layout.addWidget(title)

        for field in fields:
            row = SettingsRow(
                secret_key=field.secret_key,
                label_text=field.label,
                placeholder=field.placeholder,
                helper_text=field.helper,
            )
            row.save_requested.connect(self._on_save_requested)
            self._rows[field.secret_key] = row
            layout.addWidget(row)

        return container

    def _load_saved_values(self) -> None:
        for secret_key, row in self._rows.items():
            existing = self._settings.get_secret(secret_key)
            row.set_value(existing)

    def _on_save_requested(self, secret_key: str, value: str) -> None:
        row = self._rows.get(secret_key)
        if row is None:
            return

        try:
            if value:
                self._settings.set_secret(secret_key, value)
                row.show_save_success()
            else:
                self._settings.delete_secret(secret_key)
                row.show_cleared()
        except SettingsError as exc:
            row.show_save_error(str(exc))
            return

        self._refresh_readiness()
        self.configuration_changed.emit()

    def _refresh_readiness(self) -> None:
        configured = self._settings.configured_providers()
        expected: set[Provider] = {
            m.provider for m in DEFAULT_MODELS if m.provider != Provider.LOCAL
        }
        ready_count = len(configured & expected)
        total = len(expected)

        if ready_count == total:
            text = f"✓  All {total} providers configured — you're ready to run comparisons."
            self._readiness_label.setObjectName("readinessLabelGood")
        elif ready_count == 0:
            text = "No providers configured yet. Add at least one key to start using HECTOR-AI."
            self._readiness_label.setObjectName("readinessLabel")
        else:
            missing = total - ready_count
            text = (
                f"{ready_count} of {total} providers configured. "
                f"{missing} still missing — the corresponding model chips will be disabled."
            )
            self._readiness_label.setObjectName("readinessLabel")

        self._readiness_label.setText(text)
        self._readiness_label.style().unpolish(self._readiness_label)
        self._readiness_label.style().polish(self._readiness_label)