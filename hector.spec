# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for HECTOR-AI.

Cross-platform: this same spec file works for both Windows and macOS
builds. PyInstaller's spec format reads sys.platform at build time, so
platform-specific behaviour is encoded inline below.

Usage:
    pyinstaller hector.spec

Output:
    Windows: dist/HECTOR-AI/HECTOR-AI.exe + supporting files
    macOS:   dist/HECTOR-AI.app  (bundle)

Mode:
    One-folder (--onedir) so we get faster startup and easier debugging
    than --onefile. We zip the folder for distribution.

Console:
    Enabled for this first build — Python errors land in a console
    window so we can see what crashes if anything does. Switch to
    console=False after the first successful end-to-end test.

Icon:
    Not configured here. Adding the .ico/.icns later as a polish pass
    once the build pipeline is verified end-to-end.

History of macOS keyboard input issue (READ BEFORE CHANGING):

    v0.1.0 / v0.1.1 — macOS bundle launched, GUI rendered correctly,
    but every keystroke produced a system beep and no text appeared
    in the prompt or Settings inputs. Right-click paste worked because
    that's a clipboard operation, not a keyboard event.

    v0.1.2 — added collect_all('PySide6') on the wrong hypothesis that
    the platforminputcontexts plugins were missing. Bundle size grew
    from ~55MB to ~289MB. Did NOT fix the keyboard issue.

    v0.1.3 — added NSPrincipalClass=NSApplication to the macOS
    Info.plist. Without this key, macOS does not recognise the .app
    as a proper Cocoa GUI application; keyboard events have no first
    responder and the system rings the alert bell. THIS WAS THE FIX.
    Verified working by the client on a real Apple Silicon M4
    MacBook Air running Sequoia 15.7.3.

    v0.1.4 — reverts the v0.1.2 collect_all('PySide6') change. Since
    NSPrincipalClass was the actual fix, the aggressive PySide6
    bundling is no longer needed. Bundle size returns to roughly
    v0.1.1 levels (~55MB Windows, ~80MB macOS). The default PyInstaller
    PySide6 hook is sufficient for HECTOR's Qt usage (QtCore, QtGui,
    QtWidgets, QtNetwork). Also retains the v0.1.3 NSPrincipalClass
    + defensive Info.plist defaults — those stay forever.
"""
from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

# SPEC_DIR is the directory this spec file lives in (project root).
# PyInstaller defines the magic variable `SPECPATH` at runtime.
SPEC_DIR = Path(SPECPATH).resolve()

ENTRY_POINT = str(SPEC_DIR / "main.py")

# ---------------------------------------------------------------------------
# Bundled data files
# ---------------------------------------------------------------------------
# Project-specific data goes here. PySide6's data files are handled by
# PyInstaller's default PySide6 hook (no explicit handling needed in v0.1.4+).

datas = [
    # (src_relative_to_spec, dest_dir_in_bundle)
    ("assets", "assets"),
]

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
# PyInstaller's static analysis follows top-level imports. Things it
# misses are typically dynamic (loaded by string name at runtime). The
# main culprit for HECTOR is `keyring`, which picks a backend at
# runtime based on the OS.
#
# We list backends for ALL platforms here. Bundling unused backends is
# harmless — they're never loaded — and it keeps the same spec working
# on both Windows and Mac builds.

hiddenimports = [
    # Qt SVG support — needed for inline SVG icons in response cards.
    # The default PySide6 hook sometimes misses this, especially on
    # macOS PyInstaller builds.
    "PySide6.QtSvg",
    # cryptography — Fernet encryption for secrets storage.
    "cryptography",
    "cryptography.fernet",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.hashes",
    "cryptography.hazmat.primitives.kdf.pbkdf2",
    "cryptography.hazmat.backends",
    "cryptography.hazmat.backends.openssl",
    "cryptography.hazmat.bindings._rust",
]

# ---------------------------------------------------------------------------
# Excludes
# ---------------------------------------------------------------------------
# Modules to NOT bundle even if PyInstaller thinks they're imported.
# Keeping this list short and only adding things that actually waste
# space; over-excluding causes runtime ImportErrors that are confusing
# to debug.

excludes = [
    # tkinter ships with Python and PyInstaller bundles it by default,
    # but HECTOR uses Qt — tkinter is dead weight (~5MB).
    "tkinter",
    # unittest is dev-only.
    "unittest",
]

# ---------------------------------------------------------------------------
# Analysis — what to bundle
# ---------------------------------------------------------------------------

a = Analysis(
    [ENTRY_POINT],
    pathex=[str(SPEC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# EXE — the launcher
# ---------------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # one-folder mode: don't embed binaries in the EXE
    name="HECTOR-AI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX compression sometimes triggers AV false positives
    console=True,           # First build: keep console for error visibility
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ---------------------------------------------------------------------------
# COLLECT — gather everything into the output folder
# ---------------------------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HECTOR-AI",
)

# ---------------------------------------------------------------------------
# macOS .app bundle wrapper
# ---------------------------------------------------------------------------
# On macOS, after COLLECT produces dist/HECTOR-AI/, we additionally wrap
# it as a .app bundle so it behaves like a native Mac application
# (double-click to launch, shows up in Launchpad, etc.).
#
# Info.plist keys explained:
#
#   NSPrincipalClass = "NSApplication"
#       CRITICAL for keyboard input. Tells macOS this bundle is a
#       proper Cocoa GUI app. Without it, macOS treats the bundle
#       as something more like a CLI tool that opened a window;
#       keyboard events have no "first responder" object and macOS
#       rings the system alert bell on every keystroke. Adding this
#       key was THE fix for the v0.1.0/v0.1.1/v0.1.2 keyboard-beep
#       symptom on macOS. Verified working in v0.1.3 by the client
#       on real M4 hardware.
#
#   LSUIElement = False
#       Defensive: NOT a menu-bar-only / agent app. Default is False;
#       being explicit prevents confusion with PyInstaller's defaults.
#
#   LSBackgroundOnly = False
#       Defensive: NOT a daemon / background-only process. Same
#       defensive reasoning as LSUIElement.

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="HECTOR-AI.app",
        icon=None,            # Add .icns here in a later pass
        bundle_identifier="com.karri.hectorai",
        info_plist={
            # CRITICAL — see comment above. Fixes keyboard beep on macOS.
            "NSPrincipalClass": "NSApplication",
            # Defensive defaults — explicit is better than implicit.
            "LSUIElement": False,
            "LSBackgroundOnly": False,
            # Bundle identity.
            "CFBundleName": "HECTOR-AI",
            "CFBundleDisplayName": "HECTOR-AI",
            "CFBundleVersion": "1.0.0rc2",
            "CFBundleShortVersionString": "1.0.0rc2",
            # Tell macOS this is a regular GUI app, not a tool.
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            # Allow the app to run on Apple Silicon natively.
            "LSMinimumSystemVersion": "11.0",
            # High-DPI support.
            "NSHighResolutionCapable": True,
        },
    )
