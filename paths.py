"""
Path helpers for HECTOR-AI.

Two distinct concerns, both centralised here so the rest of the codebase
doesn't need to know how the app is being run.

1. Bundled read-only resources (icons, skill files, anything shipped with
   the app and not meant to change at runtime). These live in different
   places depending on how the app is running:
     - Development:        next to the source tree (Path(__file__).parent)
     - PyInstaller bundle: extracted to sys._MEIPASS at startup
     - macOS .app bundle:  also under sys._MEIPASS when frozen by
                           PyInstaller; pyinstaller handles the
                           Contents/Resources placement transparently.

   Use resource_path("assets/logo.png") and you get the right path back.

2. Per-user writable data (the SQLite registry, any cache files, anything
   that changes at runtime and must persist across sessions). Lives in
   OS-correct locations:
     - Windows: %APPDATA%\\HECTOR-AI\\
                  (typically C:\\Users\\<user>\\AppData\\Roaming\\HECTOR-AI\\)
     - macOS:   ~/Library/Application Support/HECTOR-AI/
     - Linux:   ~/.local/share/HECTOR-AI/

   Use user_data_dir() and you get the right path back. Directory is
   created on demand so callers can immediately write into it.

Why a single module: every file that needs a path goes through here.
Keeps the import line obvious and the intent unambiguous, and gives
us one place to change behaviour if we ever need to (e.g. adding a
'portable mode' that puts everything next to the .exe).
"""
from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Bundled resources — read-only, ship with the app
# ---------------------------------------------------------------------------

def resource_path(rel: str) -> Path:
    """Return the absolute path to a bundled resource.

    Args:
        rel: Path relative to the project root in dev (or to the bundle
             root when frozen). Forward slashes work on all platforms;
             pathlib normalises them.

    Behaviour:
        - Frozen (PyInstaller): looks under sys._MEIPASS, which is the
          temp directory PyInstaller extracts the bundle to at startup
          for one-file builds, or the bundle's data dir for one-folder
          builds. Same attribute either way.
        - Development: looks next to this file (project root).

    The returned path is NOT verified to exist — caller decides whether
    a missing resource is an error or just a soft case (the way main.py
    treats a missing logo as fine).
    """
    if getattr(sys, "frozen", False):
        # PyInstaller sets sys._MEIPASS to the bundle/extract dir.
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # Dev mode: this file lives at the project root, so its parent
        # IS the project root.
        base = Path(__file__).resolve().parent

    return base / rel


# ---------------------------------------------------------------------------
# Per-user writable data — OS-correct application support directory
# ---------------------------------------------------------------------------

# App identifier used for the data directory name on every platform.
# MUST stay in sync with QApplication.setOrganizationName / setApplicationName
# in main.py — Qt uses these to decide where QSettings goes, and we use
# the same name to keep all our user data under a single folder.
_APP_NAME = "HECTOR-AI"


def user_data_dir() -> Path:
    """Return the OS-correct per-user writable directory for HECTOR-AI.

    Creates the directory if it doesn't exist, so callers can immediately
    write into it. Idempotent — calling repeatedly is fine.

    IMPORTANT: this function uses Qt's QStandardPaths to get the OS-correct
    location. QStandardPaths consults QCoreApplication's organisation/app
    name, which means a QApplication MUST exist before calling this. In
    HECTOR's flow, the call sites are inside __init__ methods that run
    after main.py has constructed QApplication, so this is fine — but
    don't call user_data_dir() at module import time.

    Returns:
        Absolute Path. Directory is guaranteed to exist on return.

    Locations by platform:
        Windows: C:\\Users\\<user>\\AppData\\Roaming\\HECTOR-AI\\
        macOS:   /Users/<user>/Library/Application Support/HECTOR-AI/
        Linux:   /home/<user>/.local/share/HECTOR-AI/
    """
    # Lazy import. Importing PySide6 at module top would force every
    # consumer of paths.py to depend on Qt, which we don't want — and
    # this module is otherwise pure-stdlib.
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not base:
        # Fallback for the unlikely case Qt returns empty (no QApplication
        # yet, or unusual platform). Use the home dir + app name. Better
        # than crashing; gives the user a sensible-ish location they can
        # find.
        base = str(Path.home() / f".{_APP_NAME.lower()}")

    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path