"""
routing/mime_canonical.py

Maps real-world MIME strings to short canonical names used throughout the
routing layer (capability matrix, router, adapter capability declarations).

Why: the same file type can have multiple MIME strings (e.g. .xlsx is
sometimes "application/vnd.ms-excel" and sometimes the long openxmlformats
one). Routing logic that compares full MIMEs is brittle. We normalize once,
at the edge, and use short canonical names internally.

Canonical names supported in Phase 1: pdf, xlsx, csv, png, jpeg.
Add more as new file types are introduced (docx, pptx, mp4, etc.).
"""
from __future__ import annotations

# Real MIME → canonical name. Add aliases as you encounter them in the wild.
_MIME_TO_CANONICAL: dict[str, str] = {
    # PDF
    "application/pdf": "pdf",

    # Excel — modern (.xlsx) and legacy (.xls) both canonicalize to "xlsx"
    # because for routing purposes they're the same capability question.
    # If we ever need to distinguish, split into "xlsx" and "xls" here.
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "application/x-excel": "xlsx",
    "application/x-msexcel": "xlsx",

    # CSV
    "text/csv": "csv",
    "application/csv": "csv",
    "text/x-csv": "csv",

    # PNG
    "image/png": "png",

    # JPEG — the spec calls it image/jpeg; image/jpg is a common typo we accept
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/pjpeg": "jpeg",
}


def canonical_mime(mime: str) -> str | None:
    """
    Return the canonical name for a MIME string, or None if unknown.

    Lookup is case-insensitive. Whitespace is stripped. Returns None for
    empty input or unrecognized MIMEs — callers decide how to handle that
    (typically: mark the file as unsupported across all providers).
    """
    if not mime:
        return None
    return _MIME_TO_CANONICAL.get(mime.strip().lower())


# Reverse lookup: canonical → "preferred" real MIME. Used when an adapter
# needs to send the file to a provider and we need the official MIME string.
# We pick one canonical real MIME per type (the most widely accepted one).
_CANONICAL_TO_MIME: dict[str, str] = {
    "pdf":  "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv":  "text/csv",
    "png":  "image/png",
    "jpeg": "image/jpeg",
}


def preferred_mime(canonical: str) -> str | None:
    """Return the preferred official MIME string for a canonical name."""
    return _CANONICAL_TO_MIME.get(canonical)


# All canonical names this version of the routing layer knows about.
# Used in tests and by the capability matrix to validate it isn't using
# an unknown canonical name.
SUPPORTED_CANONICAL_NAMES: frozenset[str] = frozenset(_CANONICAL_TO_MIME.keys())