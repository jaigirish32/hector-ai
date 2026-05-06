"""
Debug helper for tracing the streaming path during v0.2.0 development.

Flip DEBUG_STREAMING to False to silence all dbg() output without
removing the calls. Once streaming is stable, the dbg() calls can be
removed entirely or left in for future debugging.

Output goes to stderr (not stdout) so it doesn't mix with normal app
output. flush=True ensures the print is visible immediately even when
the app is hanging.
"""
import sys

DEBUG_STREAMING = True


def dbg(tag: str, msg: str) -> None:
    """Emit a tagged debug line to stderr if DEBUG_STREAMING is on."""
    if DEBUG_STREAMING:
        print(f"[{tag}] {msg}", file=sys.stderr, flush=True)