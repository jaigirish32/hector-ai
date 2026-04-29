# HECTOR-AI — Local Build Guide

Step-by-step for producing a runnable HECTOR-AI build on your own machine.
This is for **manual testing of the PyInstaller spec** before we ship the
GitHub Actions workflow that does the same thing in CI.

---

## Prerequisites

- Python 3.12 in your active venv (`.venv` in the project root)
- HECTOR runs normally via `python main.py` (verify before building — if
  it doesn't run from source, it won't run packaged either)

---

## One-time setup

PyInstaller isn't a runtime dependency of HECTOR — only a build-time tool.
Install it into your venv directly:

```powershell
pip install pyinstaller
```

This adds `pyinstaller.exe` to your venv's `Scripts/` folder. Verify:

```powershell
pyinstaller --version
```

Should print a version like `6.x.x`.

---

## Building

From the project root, with venv activated:

```powershell
pyinstaller hector.spec
```

What this does:
1. Reads `hector.spec` to know what to bundle.
2. Walks all imports starting from `main.py`.
3. Bundles them, plus the `assets/` folder, plus the keyring backends, into
   `dist/HECTOR-AI/`.
4. Produces a `HECTOR-AI.exe` launcher in that folder.

Build time: 30 seconds to 2 minutes depending on your machine. First build
is slowest because PyInstaller analyses every dependency from scratch.

If the build succeeds, you'll see:
```
INFO: Building COLLECT COLLECT-00.toc completed successfully.
```

If it fails, the error is usually on the last 5–10 lines. Common ones:
- `ModuleNotFoundError` — a hidden import is missing. Add it to the
  `hiddenimports` list in `hector.spec` and rebuild.
- `Permission denied` — `dist/` or `build/` directory is locked. Close
  any Explorer window or other process holding it open.

---

## Testing the build

**Critical:** test in a directory OTHER than the project root. PyInstaller
artifacts must work without any project files on disk — running from
`dist/HECTOR-AI/HECTOR-AI.exe` directly might silently pick up files
from the parent project that won't be there for the user.

Copy the output folder somewhere clean:

```powershell
Copy-Item -Recurse dist\HECTOR-AI C:\Temp\hector-test
```

Then:

```powershell
cd C:\Temp\hector-test
.\HECTOR-AI.exe
```

A console window will open (this is intentional for the first build —
shows Python errors if anything crashes). Then the HECTOR window
should appear.

**What to verify:**

1. The window opens.
2. The logo appears (top-left card and title bar).
3. Click "Settings" — your existing API keys should be visible (they're
   stored in OS credential vault, not in the bundle, so they survive).
4. Click "+ Add file" and pick a small file. Should upload successfully.
5. Run a comparison. Should work on whichever providers you have keys for.
6. Close the app. Reopen. Files added in step 4 should still be there
   (the SQLite registry is at `%APPDATA%\Karri\HECTOR-AI\hector.db` —
   shared between source and packaged versions of HECTOR).

If all six work, the Windows build is good.

---

## Troubleshooting

### "App opens then disappears immediately"

The console window is closing too fast to read. Run from PowerShell so
the console persists:

```powershell
cd C:\Temp\hector-test
.\HECTOR-AI.exe
```

Read the error in the PowerShell window after the app closes.

### "ImportError: No module named X"

X needs to be added to `hiddenimports` in `hector.spec`. Rebuild:

```powershell
pyinstaller hector.spec
```

(PyInstaller will reuse cached analysis where possible — the rebuild is
faster than the first build.)

### "Logo doesn't appear"

The asset bundling didn't include `assets/logo.png`. Check that
`dist/HECTOR-AI/_internal/assets/logo.png` exists. If not, the `datas`
entry in the spec didn't take. Re-check the spec and rebuild.

### "Windows Defender / SmartScreen warning"

Expected for unsigned PyInstaller builds. Click "More info" → "Run
anyway." This warning will go away when we add code signing (separate
work — needs a code-signing certificate, ~$300/year).

---

## Cleaning up

If you want to discard the build and start fresh:

```powershell
Remove-Item -Recurse -Force build, dist
```

These two folders are listed in `.gitignore` (or should be — verify).
They're regenerated every time you run `pyinstaller hector.spec`.

---

## What this DOESN'T do

This is a manual local build for testing. It does NOT:
- Build a Mac version (needs a Mac — we'll use GitHub Actions for that).
- Sign the executable (needs a certificate).
- Produce an installer (.msi/.exe-installer) — just a folder.
- Bump the version number — that's manual in `pyproject.toml`.

The GitHub Actions workflow (Commit 3) handles automated builds for both
platforms on every release tag. This local build is just to verify the
spec works before committing the workflow.
