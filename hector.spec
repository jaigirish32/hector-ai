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

PySide6 plugin bundling:
    Uses collect_all('PySide6') to bundle EVERY Qt module, plugin, and
    data file. This is heavier than the default PySide6 hook (adds maybe
    20–50 MB to the bundle) but it's the safe choice because the default
    hook misses the platforminputcontexts plugins on macOS, which are
    required for the keyboard event chain to reach widgets. Without
    those plugins, the macOS bundle launches and renders fine but
    keystrokes are dropped (right-click paste still works because
    that's a clipboard operation, not a keyboard event). v0.1.0 and
    v0.1.1 hit this; v0.1.2 fixes it via collect_all.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

# SPEC_DIR is the directory this spec file lives in (project root).
# PyInstaller defines the magic variable `SPECPATH` at runtime.
SPEC_DIR = Path(SPECPATH).resolve()

ENTRY_POINT = str(SPEC_DIR / "main.py")

# ---------------------------------------------------------------------------
# PySide6 — collect everything
# ---------------------------------------------------------------------------
# collect_all returns (datas, binaries, hiddenimports) for the named
# package. We use it for PySide6 to ensure every Qt plugin (including
# platform input contexts on macOS) gets bundled. The default PySide6
# hook tries to be selective but has been observed to miss
# platforminputcontexts in 6.11.x with PyInstaller 6.20.x — exactly
# the failure mode that causes "GUI works, keyboard input dropped" on
# macOS. collect_all is the safe sledgehammer.
pyside_datas, pyside_binaries, pyside_hiddenimports = collect_all("PySide6")

# ---------------------------------------------------------------------------
# Bundled data files
# ---------------------------------------------------------------------------
# Project-specific data goes here. PySide6's data files come in via
# pyside_datas (above) and get merged into Analysis.datas below.

datas = [
    # (src_relative_to_spec, dest_dir_in_bundle)
    ("assets", "assets"),
]
datas.extend(pyside_datas)

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
    # keyring backends — keyring picks one at runtime by OS
    "keyring.backends.Windows",
    "keyring.backends.macOS",
    "keyring.backends.SecretService",
    "keyring.backends.fail",
    "keyring.backends.chainer",
    "keyring.backends.null",
    # win32ctypes — keyring's Windows credential vault dependency
    "win32ctypes.pywin32",
    "win32ctypes.pywin32.pywintypes",
    "win32ctypes.pywin32.win32cred",
]
hiddenimports.extend(pyside_hiddenimports)

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
    binaries=pyside_binaries,
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

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="HECTOR-AI.app",
        icon=None,            # Add .icns here in a later pass
        bundle_identifier="com.karri.hectorai",
        info_plist={
            "CFBundleName": "HECTOR-AI",
            "CFBundleDisplayName": "HECTOR-AI",
            "CFBundleVersion": "0.1.2",
            "CFBundleShortVersionString": "0.1.2",
            # Tell macOS this is a regular GUI app, not a tool.
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            # Allow the app to run on Apple Silicon natively.
            "LSMinimumSystemVersion": "11.0",
            # High-DPI support.
            "NSHighResolutionCapable": True,
        },
    )
