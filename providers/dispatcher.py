"""
The dispatcher — orchestrates multi-provider LLM calls in parallel.

Two dispatch paths:
- dispatch(): files are resolved at run time via FileOrchestrator (legacy).
- dispatch_with_resolved_refs(): files are pre-resolved by FileLibrary
  at attach time; we just hand off ready-made refs to chat clients.

Native-only routing (Phase 2e), permissive policy:
    Before fanning out, dispatch_with_resolved_refs counts how many of
    the user's attached files each model's provider has refs for.
      - Zero refs (and files were attached) → skip the model with a
        structured response_failed. The model would otherwise have
        nothing to work with and would hallucinate.
      - Some but not all refs → DISPATCH with the subset, attach a
        caveat to the response so the user sees that partial coverage.
        Lets a PDF-supporting provider answer about the PDF even when
        an attached xlsx isn't supported.
      - All refs → dispatch normally, no caveat.

v0.2.0 streaming
----------------
Each worker now consumes the provider's complete_stream() event
iterator instead of calling complete(). Streaming events
(StreamStarted, TextDelta, Usage) are re-emitted as Qt signals during
the stream, so the UI can update live. The terminal events
(StreamCompleted, StreamFailed, StreamCancelled) drive the existing
_pending_count countdown — exactly one terminal per worker, same as
before, just with a third option (cancellation) added.

Cancellation: each worker is given its own threading.Event before it
starts. The dispatcher exposes cancel(model_id) which sets that
worker's flag. The provider client's complete_stream() implementation
is required to check the flag between events and yield StreamCancelled
when set, then close its underlying SDK stream cleanly.

During the migration: providers that have not yet implemented
complete_stream() inherit the default body from BaseProviderClient
which raises NotImplementedError. The worker catches this and emits a
clean failure to the UI. So an un-migrated provider shows "not yet
migrated to streaming" in its card while migrated ones stream live.
"""
from __future__ import annotations

import threading                                                          # NEW: per-worker cancel flags
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
    FileRef,
    ProviderError,
)
from providers.gemini_client import GeminiClient
from providers.openai_client import OpenAIClient
# NEW: streaming event types for the worker's match dispatch
from providers.streaming import (
    StreamCancelled,
    StreamCompleted,
    StreamFailed,
    StreamStarted,
    TextDelta,
    Usage,
)
from settings_manager import SettingsManager


_PROVIDER_TO_ORCHESTRATOR_KEY: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.AZURE_OPENAI: "azure_openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE: "gemini",
}


class _WorkerSignals(QObject):
    # Existing terminal signals — unchanged. Each worker emits exactly ONE
    # of (succeeded, failed, stream_cancelled) before exiting; that one
    # drives the dispatcher's _pending_count countdown. `finished` always
    # follows in the worker's finally block, but does not itself decrement
    # the counter — it exists for any future cleanup that should happen
    # regardless of how the worker exited.
    succeeded = Signal(str, ChatResponse)            # (model_id, response) — terminal happy path
    failed = Signal(str, str)                        # (model_id, friendly_message) — terminal error
    finished = Signal(str)                           # (model_id) — always fired

    # NEW streaming signals — fired DURING the stream, before the terminal.
    # These never decrement _pending_count; they only drive live UI updates.
    stream_started = Signal(str, str)                # (model_id, served_model_name)
    stream_text_delta = Signal(str, str)             # (model_id, text_chunk)
    stream_usage = Signal(str, int, int)             # (model_id, input_tokens, output_tokens)

    # NEW terminal signal — user-cancelled stream. Mirrors succeeded/failed
    # in shape (one per worker, drives countdown via _on_worker_cancelled).
    stream_cancelled = Signal(str)                   # (model_id)


class _ProviderWorker(QRunnable):
    """Runs one provider call. Optional pre_caveats are merged into the
    response's own caveats on success — used by the dispatcher to surface
    routing-layer notes (e.g. partial file coverage) alongside any caveats
    the provider client itself may have produced.

    v0.2.0: consumes complete_stream() event iterator and re-emits each
    event as a Qt signal. Holds a per-worker threading.Event used by the
    dispatcher to request cancellation; the provider's complete_stream()
    is responsible for observing the flag and yielding StreamCancelled.
    """

    def __init__(
        self,
        client: BaseProviderClient,
        request: ChatRequest,
        signals: _WorkerSignals,
        cancel_flag: threading.Event,           # NEW: per-worker cancel flag
        pre_caveats: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._client = client
        self._request = request
        self._signals = signals
        self._cancel_flag = cancel_flag         # NEW
        self._pre_caveats = pre_caveats

    def run(self) -> None:
        model_id = self._request.model.id
        try:
            # Iterate the provider's stream. Each event is matched to the
            # appropriate Qt signal. Terminal events (StreamCompleted,
            # StreamFailed, StreamCancelled) emit their signal then return
            # immediately, ensuring exactly one terminal is emitted even
            # if (against contract) the provider yields more events after.
            for event in self._client.complete_stream(self._request, self._cancel_flag):
                if isinstance(event, StreamStarted):
                    self._signals.stream_started.emit(model_id, event.model)

                elif isinstance(event, TextDelta):
                    self._signals.stream_text_delta.emit(model_id, event.text)

                elif isinstance(event, Usage):
                    self._signals.stream_usage.emit(
                        model_id, event.input_tokens, event.output_tokens
                    )

                elif isinstance(event, StreamCompleted):
                    # Apply dispatcher-level pre_caveats to the final
                    # ChatResponse, exactly as the legacy non-streaming
                    # path did. The provider client doesn't know about
                    # pre_caveats; merging happens at the dispatcher
                    # layer so provider clients stay focused on the API.
                    final = event.final_response
                    if self._pre_caveats:
                        final = ChatResponse(
                            text=final.text,
                            latency_seconds=final.latency_seconds,
                            input_tokens=final.input_tokens,
                            output_tokens=final.output_tokens,
                            cost_usd=final.cost_usd,
                            served_model=final.served_model,
                            caveats=tuple(self._pre_caveats) + tuple(final.caveats),
                        )
                    self._signals.succeeded.emit(model_id, final)
                    return  # terminal — stop processing

                elif isinstance(event, StreamFailed):
                    self._signals.failed.emit(model_id, str(event.error))
                    return  # terminal

                elif isinstance(event, StreamCancelled):
                    self._signals.stream_cancelled.emit(model_id)
                    return  # terminal

                # Unknown event type: ignore, keep iterating. Future event
                # types (e.g. v0.3.0 tool events) added here as needed.

        except NotImplementedError as exc:
            # Provider hasn't been migrated to streaming yet. Per Step 2's
            # design, BaseProviderClient.complete_stream() raises this on
            # un-migrated providers. Surface a clean message to the UI
            # rather than letting the exception escape the worker.
            self._signals.failed.emit(
                model_id,
                "This provider is not yet migrated to streaming. Coming soon.",
            )
        except ProviderError as exc:
            # Defensive: the streaming contract says errors should be
            # StreamFailed events. If a provider client violates the
            # contract and raises instead, treat it as a normal failure.
            self._signals.failed.emit(model_id, str(exc))
        except Exception as exc:
            # Defensive: any other unexpected exception escaping the
            # iterator. Don't let it kill the worker silently.
            self._signals.failed.emit(model_id, f"Unexpected error: {exc}")
        finally:
            # Always fire finished. Currently used only as a generic
            # "worker done" notification; _pending_count is decremented
            # by the terminal-signal slots, not by this.
            self._signals.finished.emit(model_id)


class Dispatcher(QObject):
    # Existing public signals — unchanged. UI code already connected to
    # these continues to work without any change.
    response_received = Signal(str, ChatResponse)
    response_failed = Signal(str, str)
    all_complete = Signal()

    # NEW public streaming signals. UI code that wants live updates
    # connects to these; existing code that only handles completion can
    # stay on response_received / response_failed.
    stream_started = Signal(str, str)                   # (model_id, served_model_name)
    stream_text_delta = Signal(str, str)                # (model_id, text_chunk)
    stream_usage = Signal(str, int, int)                # (model_id, input_tokens, output_tokens)
    stream_cancelled = Signal(str)                      # (model_id) — terminal

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

        self._registry = registry or FileRegistry()
        self._orchestrator = FileOrchestrator(
            registry=self._registry,
            settings=self._settings,
        )

        self._pool = QThreadPool.globalInstance()
        self._pending_count = 0

        # NEW: per-Run map of model_id → cancel flag. Populated when each
        # worker is created in dispatch(); cleared when the Run completes
        # (in _decrement_pending). cancel(model_id) sets the matching flag.
        self._cancel_flags: dict[str, threading.Event] = {}

        self._signals = _WorkerSignals()
        self._signals.succeeded.connect(self._on_worker_succeeded)
        self._signals.failed.connect(self._on_worker_failed)
        # NEW: streaming-event signals re-emitted upward to the UI.
        # Intermediate events pass straight through (no counter changes).
        # The cancellation terminal goes through _on_worker_cancelled so
        # it can both re-emit and decrement _pending_count, mirroring the
        # pattern of _on_worker_succeeded / _on_worker_failed.
        self._signals.stream_started.connect(self.stream_started.emit)
        self._signals.stream_text_delta.connect(self.stream_text_delta.emit)
        self._signals.stream_usage.connect(self.stream_usage.emit)
        self._signals.stream_cancelled.connect(self._on_worker_cancelled)

    # ---------- Public cancellation API ----------

    def cancel(self, model_id: str) -> None:
        """Tell the worker for the given model to stop streaming.

        Sets that worker's threading.Event. The provider client's
        complete_stream() implementation is required to check the flag
        between events and yield StreamCancelled when set, then close
        the underlying SDK stream cleanly. Idempotent — calling twice
        on the same model is a no-op (Event.set() is idempotent).

        Has no effect if no worker for this model is currently running
        (e.g. cancel called after the stream already completed). The
        UI should not rely on the worker being still alive.
        """
        flag = self._cancel_flags.get(model_id)
        if flag is not None:
            flag.set()

    # ---------- Modern dispatch: pre-resolved file refs ----------

    def dispatch_with_resolved_refs(
        self,
        prompt: str,
        model_ids: list[str],
        per_model_refs: dict[str, tuple[FileRef, ...]] | None = None,
        attached_file_ids: list[int] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str = "",
    ) -> None:
        """Fan out using file_refs that the caller has already resolved.

        attached_file_ids is the list of file_ids the user has selected
        in the sidebar for this Run. The dispatcher uses it to detect:
          - Zero coverage: model gets a structured response_failed.
          - Partial coverage: model is dispatched with a caveat that
            will appear under its answer.
          - Full coverage: model is dispatched normally.

        If attached_file_ids is None or empty, the pre-flight check is
        skipped — preserves existing behaviour for text-only Runs.
        """
        if self._pending_count > 0:
            return

        per_model_refs = per_model_refs or {}
        attached_file_ids = attached_file_ids or []
        expected_file_count = len(attached_file_ids)

        runnable_jobs: list[
            tuple[str, ChatRequest, BaseProviderClient, tuple[str, ...]]
        ] = []
        skipped_jobs: list[tuple[str, str]] = []

        for model_id in model_ids:
            model = get_model(model_id)
            if model is None:
                skipped_jobs.append((model_id, f"Unknown model: {model_id}"))
                continue

            client = self._clients.get(model.provider)
            if client is None:
                skipped_jobs.append((
                    model_id,
                    f"No client for provider: {model.provider.value}",
                ))
                continue

            refs_for_model = per_model_refs.get(model_id, ())
            n_refs = len(refs_for_model)

            pre_caveats: tuple[str, ...] = ()

            # Pre-flight check only matters when files are attached.
            if expected_file_count > 0:
                if n_refs == 0:
                    # Zero coverage. The model would have nothing to look
                    # at and any answer would be a hallucination — skip.
                    skipped_jobs.append((
                        model_id,
                        f"This provider doesn't natively support any of the "
                        f"{expected_file_count} attached file(s). "
                        f"Try a different model for these files, "
                        f"or use a different file type.",
                    ))
                    continue

                if n_refs < expected_file_count:
                    # Partial coverage. Dispatch with what we have, and
                    # attach a caveat so the user sees that this provider
                    # only saw a subset of the files.
                    missing = expected_file_count - n_refs
                    pre_caveats = (
                        f"This provider only saw {n_refs} of "
                        f"{expected_file_count} attached file(s); "
                        f"{missing} file(s) were skipped because the type "
                        f"isn't natively supported here.",
                    )

            request = ChatRequest(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or None,
                file_refs=refs_for_model,
            )
            runnable_jobs.append((model_id, request, client, pre_caveats))

        # Emit skip signals first — synchronous, immediate UI feedback —
        # then start workers for the rest.
        for model_id, message in skipped_jobs:
            self.response_failed.emit(model_id, message)

        if not runnable_jobs:
            self.all_complete.emit()
            return

        self._pending_count = len(runnable_jobs)
        # Fresh per-Run cancel flags. Old ones (from the previous Run)
        # were cleared in _decrement_pending when that Run finished.
        self._cancel_flags = {}
        for model_id, request, client, pre_caveats in runnable_jobs:
            cancel_flag = threading.Event()                 # NEW: one Event per worker
            self._cancel_flags[model_id] = cancel_flag      # NEW: track by model_id for cancel()
            worker = _ProviderWorker(
                client, request, self._signals, cancel_flag, pre_caveats
            )
            self._pool.start(worker)

    # ---------- Legacy dispatch: file paths resolved at run time ----------

    def dispatch(
        self,
        prompt: str,
        model_ids: list[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str = "",
        file_paths: list[Path] | None = None,
    ) -> None:
        """Fan out by resolving file paths at run time via the orchestrator.

        Kept for any caller that still passes raw file_paths. The new
        attach-at-upload-time flow uses dispatch_with_resolved_refs.
        """
        if self._pending_count > 0:
            return

        file_paths = file_paths or []

        runnable_jobs: list[tuple[str, ChatRequest, BaseProviderClient]] = []
        for model_id in model_ids:
            model = get_model(model_id)
            if model is None:
                self.response_failed.emit(model_id, f"Unknown model: {model_id}")
                continue

            client = self._clients.get(model.provider)
            if client is None:
                self.response_failed.emit(
                    model_id,
                    f"No client for provider: {model.provider.value}",
                )
                continue

            file_refs_tuple: tuple = ()
            if file_paths:
                provider_key = _PROVIDER_TO_ORCHESTRATOR_KEY.get(model.provider)
                if provider_key is not None:
                    refs, errors = self._orchestrator.resolve_for_provider(
                        file_paths, provider_key
                    )
                    if errors:
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
        # Fresh per-Run cancel flags (same pattern as dispatch_with_resolved_refs).
        self._cancel_flags = {}
        for model_id, request, client in runnable_jobs:
            cancel_flag = threading.Event()                 # NEW
            self._cancel_flags[model_id] = cancel_flag      # NEW
            worker = _ProviderWorker(
                client, request, self._signals, cancel_flag
            )
            self._pool.start(worker)

    # ---------- Worker callbacks ----------

    def _on_worker_succeeded(self, model_id: str, response: ChatResponse) -> None:
        self.response_received.emit(model_id, response)
        self._decrement_pending()

    def _on_worker_failed(self, model_id: str, message: str) -> None:
        self.response_failed.emit(model_id, message)
        self._decrement_pending()

    # NEW: third terminal slot. Mirrors _on_worker_succeeded/_on_worker_failed —
    # re-emits the public signal upward to the UI, then decrements the
    # countdown. Without this, cancellations would leak _pending_count
    # and the dispatcher would refuse all subsequent Runs.
    def _on_worker_cancelled(self, model_id: str) -> None:
        self.stream_cancelled.emit(model_id)
        self._decrement_pending()

    def _decrement_pending(self) -> None:
        self._pending_count -= 1
        if self._pending_count <= 0:
            self._pending_count = 0
            # Run is fully done — drop all per-worker cancel flags.
            # If we don't clear here, stale flags from this Run hang
            # around in memory until the next Run replaces the dict.
            # Cheap to clear, safer to clear.
            self._cancel_flags.clear()                      # NEW
            self.all_complete.emit()

    @staticmethod
    def _format_file_errors(errors: list) -> str:
        if len(errors) == 1:
            err = errors[0]
            return f"File error ({err.file_path.name}): {err.message}"
        lines = [f"{e.file_path.name}: {e.message}" for e in errors]
        return "File errors:\n" + "\n".join(lines)