"""
routing/capability_matrix.py

Declarative table of what each provider supports natively, per canonical
file type. The router reads this table and applies policy. No logic lives
here — only data. Adding a provider or a file type is an edit to this file
plus, at most, a new MIME alias in mime_canonical.py.

Phase 1 scope: native-only architecture. Strategy is either NATIVE (provider
accepts and meaningfully interprets the file via its own API) or UNSUPPORTED
(no native path; the file/provider combination is skipped).

If "provider-side code execution" is added later, introduce a new Strategy
value (e.g. NATIVE_WITH_CODE_EXEC) and add corresponding rows. Don't conflate
strategies in existing rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from routing.mime_canonical import SUPPORTED_CANONICAL_NAMES


class Strategy(Enum):
    """How a provider handles a given file type."""
    NATIVE = "native"
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
# Sparse: only NATIVE entries are listed. Any (provider, mime) not present
# is treated as UNSUPPORTED by the router. Keeps this file scannable.
#
# When you add a provider, add ONE section. When you add a file type,
# add ONE row per supporting provider. No other file in the codebase
# needs to change.
# ---------------------------------------------------------------------------

CAPABILITY_MATRIX: dict[tuple[str, str], Capability] = {
    # -------------------- OpenAI --------------------
    # Responses API + user_data purpose. Native xlsx with spreadsheet
    # augmentation added Feb 2026: the API parses up to first 1k rows
    # and adds model-generated summaries.
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
    ("azure_openai", "pdf"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=50,
                                         notes="vision + text extraction"),
    ("azure_openai", "png"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),
    ("azure_openai", "jpeg"): Capability(Strategy.NATIVE, fidelity=3, max_size_mb=20),

    # -------------------- Anthropic --------------------
    # Files API + document/image content blocks. Per Anthropic docs, the
    # document block is PDF only; csv/xlsx/docx must be inlined as text.
    # Inlining-as-text isn't "native" under our Phase 1 definition.
    ("anthropic", "pdf"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=32),
    ("anthropic", "png"):  Capability(Strategy.NATIVE, fidelity=3, max_size_mb=30),
    ("anthropic", "jpeg"): Capability(Strategy.NATIVE, fidelity=3, max_size_mb=30),

    # -------------------- Gemini --------------------
    # Per Google docs (Jan 2026): "document vision only meaningfully understands
    # PDFs. Other types will be extracted as pure text." So PDF is full-fidelity;
    # CSV is text-fallback (still useful — model reads the raw CSV); xlsx is
    # binary, not in the pass-through list, marked UNSUPPORTED.
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
                f"Matrix should only list NATIVE entries. UNSUPPORTED is "
                f"the default for missing entries. Remove the explicit "
                f"UNSUPPORTED row for ({provider!r}, {mime!r})."
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