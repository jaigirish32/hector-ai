"""
The dispatcher — orchestrates multi-provider LLM calls in parallel.

Uses QThreadPool + QRunnable for worker execution. Runnables have no
event loop of their own — their run() method executes directly on a
pool thread. Communication back to the main thread goes through a
separate QObject that lives on the main thread and emits signals.

File handling: when callers pass `file_paths` to dispatch(), the
dispatcher resolves them per-provider via the FileOrchestrator BEFORE
fanning out to workers. Each model gets its own ChatRequest with the
correct file_refs for its provider.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from attachments.orchestrator import FileOrchestrator
from attachments.registry import FileRegistry
from models import Provider, get_model
from providers.anthropic_client import AnthropicClient
from providers.azure_openai_client import AzureOpenAIClient
from providers.base import (
    BaseProviderClient,
    ChatRequest,
    ChatResponse,
    ProviderError,
)
from providers.gemini_client import GeminiClient
from providers.openai_client import OpenAIClient
from settings_manager import SettingsManager


# Maps Provider enum values to the string keys used by the orchestrator.
# These strings must match the `provider_name` declared on each uploader.
_PROVIDER_TO_ORCHESTRATOR_KEY: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.AZURE_OPENAI: "azure_openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE: "gemini",
}


# ---------------------------------------------------------------------------
# Worker signals — a separate QObject because QRunnable can't emit signals
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    """Signals that QRunnable workers emit.

    QRunnable isn't a QObject, so it can't define signals itself.
    We give each worker a signals object owned by the dispatcher.
    """

    succeeded = Signal(str, ChatResponse)
    failed = Signal(str, str)
    finished = Signal(str)


# ---------------------------------------------------------------------------
# Worker — one provider call on one pool thread
# ---------------------------------------------------------------------------

class _ProviderWorker(QRunnable):
    """Runs a single ChatRequest on a QThreadPool thread."""

    def __init__(
        self,
        client: BaseProviderClient,
        request: ChatRequest,
        signals: _WorkerSignals,
    ) -> None:
        super().__init__()
        self._client = client
        self._request = request
        self._signals = signals

    def run(self) -> None:
        """Entry point — runs on a QThreadPool worker thread."""
        model_id = self._request.model.id
        try:
            response = self._client.complete(self._request)
            self._signals.succeeded.emit(model_id, response)
        except ProviderError as exc:
            self._signals.failed.emit(model_id, str(exc))
        except Exception as exc:
            self._signals.failed.emit(model_id, f"Unexpected error: {exc}")
        finally:
            self._signals.finished.emit(model_id)


# ---------------------------------------------------------------------------
# Dispatcher — fans out requests and collects results
# ---------------------------------------------------------------------------

class Dispatcher(QObject):
    """Orchestrates parallel LLM calls across multiple providers."""

    response_received = Signal(str, ChatResponse)
    response_failed = Signal(str, str)
    all_complete = Signal()

    def __init__(
        self,
        settings: SettingsManager | None = None,
        registry: FileRegistry | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings or SettingsManager()

        self._clients: dict[Provider, BaseProviderClient] = {
            Provider.OPENAI: OpenAIClient(self._settings),
            Provider.AZURE_OPENAI: AzureOpenAIClient(self._settings),
            Provider.ANTHROPIC: AnthropicClient(self._settings),
            Provider.GOOGLE: GeminiClient(self._settings),
        }

        # File orchestrator handles upload-and-cache per provider. We give
        # it the same registry instance the rest of the app uses (or let
        # it create its own default).
        self._registry = registry or FileRegistry()
        self._orchestrator = FileOrchestrator(
            registry=self._registry,
            settings=self._settings,
        )

        self._pool = QThreadPool.globalInstance()
        self._pending_count = 0

        # Create a single signals object for all workers. Lives on main thread,
        # so any signal emission from worker threads crosses into main thread.
        self._signals = _WorkerSignals()
        self._signals.succeeded.connect(self._on_worker_succeeded)
        self._signals.failed.connect(self._on_worker_failed)

    def dispatch(
        self,
        prompt: str,
        model_ids: list[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str = "",
        file_paths: list[Path] | None = None,
    ) -> None:
        """Fan out one prompt (with optional files) to every selected model.

        Files are resolved synchronously per-provider before the chat
        workers fan out. This means the call returns once all uploads
        complete; the chat calls themselves still run in parallel.

        For multiple models on the same provider, the file is uploaded
        once and the cached file_id is reused for each model.
        """
        if self._pending_count > 0:
            # Already dispatching — ignore duplicate clicks.
            return

        file_paths = file_paths or []

        # Step 1: collect the model+client pairs we'll actually be calling.
        # If a model is unknown or its client isn't configured, we emit
        # a per-card failure now and exclude it from the fan-out.
        runnable_jobs: list[tuple[str, "ChatRequest", BaseProviderClient]] = []
        for model_id in model_ids:
            model = get_model(model_id)
            if model is None:
                self.response_failed.emit(
                    model_id, f"Unknown model: {model_id}"
                )
                continue

            client = self._clients.get(model.provider)
            if client is None:
                self.response_failed.emit(
                    model_id,
                    f"No client registered for provider: {model.provider.value}. "
                    "This provider is pending integration.",
                )
                continue

            # Step 2: resolve files for this model's provider, if any.
            file_refs_tuple: tuple = ()
            if file_paths:
                provider_key = _PROVIDER_TO_ORCHESTRATOR_KEY.get(model.provider)
                if provider_key is not None:
                    refs, errors = self._orchestrator.resolve_for_provider(
                        file_paths, provider_key
                    )
                    if errors:
                        # Any file-resolution failure for this model means
                        # we can't reliably answer about the file. Surface
                        # the error per-card and skip this model's chat call.
                        message = self._format_file_errors(errors)
                        self.response_failed.emit(model_id, message)
                        continue
                    file_refs_tuple = tuple(refs)

            request = ChatRequest(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or None,
                file_paths=tuple(file_paths),
                file_refs=file_refs_tuple,
            )
            runnable_jobs.append((model_id, request, client))

        if not runnable_jobs:
            self.all_complete.emit()
            return

        self._pending_count = len(runnable_jobs)

        # Step 3: fan out chat calls to the thread pool.
        for model_id, request, client in runnable_jobs:
            worker = _ProviderWorker(client, request, self._signals)
            self._pool.start(worker)

    # ---------- Worker callbacks ----------

    def _on_worker_succeeded(self, model_id: str, response: ChatResponse) -> None:
        self.response_received.emit(model_id, response)
        self._decrement_pending()

    def _on_worker_failed(self, model_id: str, message: str) -> None:
        self.response_failed.emit(model_id, message)
        self._decrement_pending()

    def _decrement_pending(self) -> None:
        self._pending_count -= 1
        if self._pending_count <= 0:
            self._pending_count = 0
            self.all_complete.emit()

    # ---------- Helpers ----------

    @staticmethod
    def _format_file_errors(errors: list) -> str:
        """Turn a list of FileResolutionError into a single human message."""
        if len(errors) == 1:
            err = errors[0]
            return f"File error ({err.file_path.name}): {err.message}"
        # Multiple files failed — summarize.
        lines = [f"{e.file_path.name}: {e.message}" for e in errors]
        return "File errors:\n" + "\n".join(lines)