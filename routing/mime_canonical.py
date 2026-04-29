"""
routing/mime_canonical.py

Maps real-world MIME strings to short canonical names used throughout the
routing layer (capability matrix, router, adapter capability declarations).

Why: the same file type can have multiple MIME strings (e.g. .xlsx is
sometimes "application/vnd.ms-excel" and sometimes the long openxmlformats
one). Routing logic that compares full MIMEs is brittle. We normalize once,
at the edge, and use short canonical names internally.

Canonical names supported:
    pdf, xlsx, xls, csv, docx, doc, pptx, ppt, png, jpeg.
Add more as new file types are introduced.

Note on xls vs xlsx:
    xls (legacy binary Excel) and xlsx (Open XML) are kept as separate
    canonical names so the matrix can express different capabilities for
    each. For example, OpenAI's spreadsheet augmentation may treat them
    differently, and Anthropic's container needs xlrd for xls but openpyxl
    for xlsx. If a provider supports both identically, it's two matrix
    rows pointing at the same Capability.
"""
from __future__ import annotations

# Real MIME → canonical name. Add aliases as you encounter them in the wild.
# Lookup is case-insensitive (canonical_mime() lowercases first).
_MIME_TO_CANONICAL: dict[str, str] = {
    # PDF
    "application/pdf": "pdf",

    # Excel — modern (.xlsx) Open XML
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsx",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12": "xlsx",

    # Excel — legacy (.xls) binary
    "application/vnd.ms-excel": "xls",
    "application/x-excel": "xls",
    "application/x-msexcel": "xls",

    # CSV
    "text/csv": "csv",
    "application/csv": "csv",
    "text/x-csv": "csv",

    # Word — modern (.docx)
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",

    # Word — legacy (.doc)
    "application/msword": "doc",

    # PowerPoint — modern (.pptx)
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",

    # PowerPoint — legacy (.ppt)
    "application/vnd.ms-powerpoint": "ppt",

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
    "xls":  "application/vnd.ms-excel",
    "csv":  "text/csv",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc":  "application/msword",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt":  "application/vnd.ms-powerpoint",
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