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
    FileRef,
    ProviderError,
)
from providers.gemini_client import GeminiClient
from providers.openai_client import OpenAIClient
from settings_manager import SettingsManager


_PROVIDER_TO_ORCHESTRATOR_KEY: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.AZURE_OPENAI: "azure_openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE: "gemini",
}


class _WorkerSignals(QObject):
    succeeded = Signal(str, ChatResponse)
    failed = Signal(str, str)
    finished = Signal(str)


class _ProviderWorker(QRunnable):
    """Runs one provider call. Optional pre_caveats are merged into the
    response's own caveats on success — used by the dispatcher to surface
    routing-layer notes (e.g. partial file coverage) alongside any caveats
    the provider client itself may have produced."""

    def __init__(
        self,
        client: BaseProviderClient,
        request: ChatRequest,
        signals: _WorkerSignals,
        pre_caveats: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._client = client
        self._request = request
        self._signals = signals
        self._pre_caveats = pre_caveats

    def run(self) -> None:
        model_id = self._request.model.id
        try:
            response = self._client.complete(self._request)
            if self._pre_caveats:
                # Prepend dispatcher-level caveats so they appear first;
                # any caveats the provider client added stay after them.
                merged = tuple(self._pre_caveats) + tuple(response.caveats)
                response = ChatResponse(
                    text=response.text,
                    latency_seconds=response.latency_seconds,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    served_model=response.served_model,
                    caveats=merged,
                )
            self._signals.succeeded.emit(model_id, response)
        except ProviderError as exc:
            self._signals.failed.emit(model_id, str(exc))
        except Exception as exc:
            self._signals.failed.emit(model_id, f"Unexpected error: {exc}")
        finally:
            self._signals.finished.emit(model_id)


class Dispatcher(QObject):
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

        self._registry = registry or FileRegistry()
        self._orchestrator = FileOrchestrator(
            registry=self._registry,
            settings=self._settings,
        )

        self._pool = QThreadPool.globalInstance()
        self._pending_count = 0

        self._signals = _WorkerSignals()
        self._signals.succeeded.connect(self._on_worker_succeeded)
        self._signals.failed.connect(self._on_worker_failed)

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
        for model_id, request, client, pre_caveats in runnable_jobs:
            worker = _ProviderWorker(client, request, self._signals, pre_caveats)
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

    @staticmethod
    def _format_file_errors(errors: list) -> str:
        if len(errors) == 1:
            err = errors[0]
            return f"File error ({err.file_path.name}): {err.message}"
        lines = [f"{e.file_path.name}: {e.message}" for e in errors]
        return "File errors:\n" + "\n".join(lines)