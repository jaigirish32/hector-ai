"""
routing/capability_matrix.py

Declarative table of what each provider supports natively, per canonical
file type. The router reads this table and applies policy. No logic lives
here — only data. Adding a provider or a file type is an edit to this file
plus, at most, a new MIME alias in mime_canonical.py.

Strategy values:
    NATIVE
        Provider accepts the file via its standard document/file content
        block and reads it directly (e.g. PDF via vision, xlsx via OpenAI's
        spreadsheet augmentation).

    NATIVE_VIA_CODE_EXEC
        Provider accepts the file but only via its code-execution
        container path (e.g. Anthropic xlsx via container_upload + the
        code_execution_20250825 tool). The provider's chat client must
        emit a different content block shape, AND the file is uploaded
        as anonymous bytes (filename stripped to 'blob.bin', MIME set to
        'application/octet-stream') because the container_upload path is
        empirically the only verified consumption pattern. Real metadata
        (filename, type) is surfaced to the model via a text instruction
        block at chat time.

    UNSUPPORTED
        Default for any (provider, mime) combination not explicitly listed.
        File is not uploaded to this provider, and any chat call that needs
        the file is skipped with a structured reason.

The matrix is sparse: only NATIVE and NATIVE_VIA_CODE_EXEC rows are listed.
Every other (provider, canonical_mime) pair defaults to UNSUPPORTED.

If a provider adds a new capability tomorrow, the change is one row here.
The router, file_library, dispatcher, and UI don't change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from routing.mime_canonical import SUPPORTED_CANONICAL_NAMES


class Strategy(Enum):
    """How a provider handles a given file type."""
    NATIVE = "native"
    NATIVE_VIA_CODE_EXEC = "native_via_code_exec"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Capability:
    """
    Describes one (provider, canonical_mime) combination.

    fidelity:    1=lossy/text-only fallback, 2=ok, 3=full-fidelity
                 (e.g. PDF with vision is 3; CSV-as-text on Gemini is 2)
    max_size_mb: provider-stated upper limit on file size for this type;
                 0 means "no documented limit" or "unlimited within reason"
    notes:       human-readable caveats — surfaced in UI tooltips and logs
                 so the user knows e.g. that OpenAI caps xlsx parsing at 1k rows
    """
    strategy: Strategy
    fidelity: int = 2
    max_size_mb: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# The matrix.
#
# Sparse: only NATIVE and NATIVE_VIA_CODE_EXEC entries are listed. Any
# (provider, mime) not present is treated as UNSUPPORTED by the router.
# Keeps this file scannable.
#
# When you add a provider, add ONE section. When you add a file type,
# add ONE row per supporting provider. No other file in the codebase
# needs to change.
# ---------------------------------------------------------------------------

CAPABILITY_MATRIX: dict[tuple[str, str], Capability] = {
    # -------------------- OpenAI --------------------
    # Responses API + user_data purpose. Native xlsx with spreadsheet
    # augmentation: parses up to first 1k rows per sheet across all
    # sheets, adds model-generated summaries. Filename extension must
    # be the real one (.xlsx etc.) — we tested anonymous-bytes upload
    # to OpenAI and it was rejected at chat time with an explicit
    # "Expected context stuffing file type to be a supported format"
    # error listing the allowed extensions.
    ("openai", "pdf"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=50,
                                   notes="vision + text extraction"),
    ("openai", "xlsx"): Capability(Strategy.NATIVE, fidelity=2, max_size_mb=50,
                                   notes="parses first 1000 rows per sheet, "
                                         "adds model-generated summaries"),
    ("openai", "csv"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=50),
    ("openai", "png"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),
    ("openai", "jpeg"): Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),

    # -------------------- Azure OpenAI --------------------
    # purpose="assistants" — does NOT accept xlsx or csv even though OpenAI
    # proper does. Lags upstream OpenAI on file types as of 2026-04.
    # Code Interpreter on Azure Responses API documented but xlsx upload
    # to Azure Files API still rejected per Microsoft Q&A as recently as
    # 2025; treat as UNSUPPORTED until empirically verified.
    ("azure_openai", "pdf"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=50,
                                         notes="vision + text extraction"),
    ("azure_openai", "png"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),
    ("azure_openai", "jpeg"): Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),

    # -------------------- Anthropic --------------------
    # PDF/images via 'document' content blocks (files-api beta).
    # xlsx via 'container_upload' content block + code_execution_20250825
    # tool (code-execution + files-api beta both required). Office files
    # are uploaded as anonymous bytes (blob.bin + octet-stream) per
    # empirical verification — the container_upload path doesn't care
    # about filename/MIME at consumption time, and uniform anonymous-bytes
    # keeps one consistent code path if/when docx/pptx are added.
    ("anthropic", "pdf"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=32),
    ("anthropic", "png"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=30),
    ("anthropic", "jpeg"): Capability(Strategy.NATIVE, fidelity=3, max_size_mb=30),
    ("anthropic", "xlsx"): Capability(
        Strategy.NATIVE_VIA_CODE_EXEC,
        fidelity=3,
        max_size_mb=30,
        notes=(
            "Analysed via Anthropic's code execution container "
            "(openpyxl/pandas). Higher fidelity than OpenAI's 1000-row "
            "truncation; ~5min container time billed per call (free under "
            "monthly 1550-hour allowance)."
        ),
    ),

    # -------------------- Gemini --------------------
    # Per Google docs (Jan 2026): "document vision only meaningfully understands
    # PDFs. Other types will be extracted as pure text." So PDF is full-fidelity;
    # CSV is text-fallback (still useful — model reads the raw CSV).
    #
    # xlsx is empirically unsupported (verified Apr 2026):
    #   - Files API rejects xlsx MIME ("Unsupported MIME type")
    #   - inlineData with xlsx MIME rejected at request time
    #   - inlineData with octet-stream rejected at request time
    #   - Vertex docs explicitly state code execution does NOT accept file URIs
    # Conclusion: no path exists for Gemini xlsx today. Marked UNSUPPORTED.
    ("gemini", "pdf"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=50,
                                   notes="vision + text, up to 1000 pages"),
    ("gemini", "csv"):  Capability(Strategy.NATIVE, fidelity=2, max_size_mb=20,
                                   notes="extracted as plain text, "
                                         "no structured parsing"),
    ("gemini", "png"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),
    ("gemini", "jpeg"): Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),

    # -------------------- Grok (xAI) --------------------
    # Files API + input_file blocks (OpenAI-compatible). Per xAI docs,
    # supported types include txt, md, code, csv, json, pdf. xlsx is NOT
    # listed — marked UNSUPPORTED. Vision is jpg/png only (no webp/gif).
    ("grok", "pdf"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=48),
    ("grok", "csv"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=48),
    ("grok", "png"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),
    ("grok", "jpeg"): Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),
}


# All providers that appear in the matrix. Source of truth for "what
# providers does HECTOR know about" — used by the router to detect typos
# and by the UI to enumerate available chips.
KNOWN_PROVIDERS: frozenset[str] = frozenset(p for p, _ in CAPABILITY_MATRIX.keys())


# Strategies that count as "the file gets to this provider somehow."
# Both NATIVE and NATIVE_VIA_CODE_EXEC are upload+chat workable paths;
# the difference is consumption shape, not whether the provider "supports"
# the file. The router uses this set to decide supported vs skipped.
SUPPORTED_STRATEGIES: frozenset[Strategy] = frozenset({
    Strategy.NATIVE,
    Strategy.NATIVE_VIA_CODE_EXEC,
})


def lookup(provider: str, canonical_mime: str) -> Capability:
    """
    Return the Capability for a (provider, canonical_mime) pair.

    Sparse-table semantics: a missing entry means UNSUPPORTED, not an error.
    Returns a Capability with Strategy.UNSUPPORTED for unknown combinations,
    so callers can treat the result uniformly without try/except.
    """
    cap = CAPABILITY_MATRIX.get((provider, canonical_mime))
    if cap is not None:
        return cap
    return Capability(strategy=Strategy.UNSUPPORTED)


# ---------------------------------------------------------------------------
# Self-validation: run at import time so typos in this file fail loudly
# at app startup, not silently when a user tries to upload a file.
# ---------------------------------------------------------------------------

def _validate_matrix() -> None:
    for (provider, mime), cap in CAPABILITY_MATRIX.items():
        if not provider or not provider.islower():
            raise ValueError(
                f"Provider name must be lowercase non-empty: {provider!r}"
            )
        if mime not in SUPPORTED_CANONICAL_NAMES:
            raise ValueError(
                f"Capability matrix references unknown canonical mime "
                f"{mime!r} for provider {provider!r}. "
                f"Known: {sorted(SUPPORTED_CANONICAL_NAMES)}. "
                f"Either fix the typo or add the canonical name to "
                f"mime_canonical.py first."
            )
        if cap.strategy == Strategy.UNSUPPORTED:
            raise ValueError(
                f"Matrix should only list NATIVE or NATIVE_VIA_CODE_EXEC "
                f"entries. UNSUPPORTED is the default for missing entries. "
                f"Remove the explicit UNSUPPORTED row for "
                f"({provider!r}, {mime!r})."
            )
        if not (1 <= cap.fidelity <= 3):
            raise ValueError(
                f"Fidelity must be 1, 2, or 3 for ({provider!r}, {mime!r}); "
                f"got {cap.fidelity}"
            )
        if cap.max_size_mb < 0:
            raise ValueError(
                f"max_size_mb must be >= 0 for ({provider!r}, {mime!r}); "
                f"got {cap.max_size_mb}"
            )


_validate_matrix()