"""
Shared rate-limit retry logic for provider clients.

Each provider's complete() method wraps its API call in `with_rate_limit_retry()`.
The helper:

  - Calls the function. If it succeeds, returns the result.
  - If it raises an SDK rate-limit exception, the helper looks up the
    retry-after wait. The provider supplies its own parser (each SDK
    exposes the header differently, and Gemini doesn't expose it at all).
    If the parser returns None, we fall back to 1s/2s/4s exponential.
  - Up to 3 retries (4 total attempts). After exhausting, raises
    our domain RateLimitError with a "rate limited after 3 retries"
    message so the user knows we did try.
  - Non-rate-limit exceptions propagate immediately — auth errors,
    connection errors, etc. should not retry.

Why each provider passes its own SDK exception class and header parser:
  Each provider knows its own SDK best. Anthropic and OpenAI both
  expose response headers via exc.response.headers. Gemini doesn't.
  The helper stays generic; provider-specific bits stay in the provider.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from providers.base import RateLimitError

from providers._dbg import dbg

T = TypeVar("T")

# Number of RETRIES (not total attempts). 3 retries = 4 total tries.
_MAX_RETRIES = 3

# Exponential fallback wait in seconds, used only when the SDK exception
# does not expose a retry-after header. Index = retry attempt (0-based).
_FALLBACK_WAITS = (1, 2, 4)


def with_rate_limit_retry(
    fn: Callable[[], T],
    *,
    sdk_rate_limit_exception: type[BaseException],
    parse_retry_after_seconds: Callable[[BaseException], int | None],
    provider_label: str,
) -> T:
    """Run fn(), retrying on rate-limit exceptions.

    Args:
        fn: zero-arg callable that performs the actual API call.
        sdk_rate_limit_exception: the SDK's specific rate-limit exception
            type. We catch this; everything else propagates.
        parse_retry_after_seconds: given an instance of the SDK's
            exception, return seconds to wait, or None if no header was
            available. Each provider supplies a parser tailored to its SDK.
        provider_label: human-readable provider name used in the final
            error message after retries are exhausted ("Anthropic", etc.).

    Returns:
        The successful result of fn().

    Raises:
        RateLimitError: if all retries are exhausted on rate-limit errors.
        Anything else: propagated unchanged from fn().
    """
    last_sdk_exc: BaseException | None = None

    for attempt in range(_MAX_RETRIES + 1):
        dbg("RETRY", f"{provider_label} attempt {attempt}/{_MAX_RETRIES} starting")
        try:
            result = fn()
            dbg("RETRY", f"{provider_label} attempt {attempt} SUCCESS")
            return result
        except sdk_rate_limit_exception as exc:
            dbg("RETRY", f"{provider_label} attempt {attempt} got SDK RateLimit")
            last_sdk_exc = exc
            if attempt == _MAX_RETRIES:
                dbg("RETRY", f"{provider_label} retries EXHAUSTED, raising")
                raise RateLimitError(
                    f"{provider_label} rate limited after {_MAX_RETRIES} retries.",
                    raw=str(exc),
                ) from exc

            wait_seconds = parse_retry_after_seconds(exc)
            dbg("RETRY", f"{provider_label} parser returned wait={wait_seconds}")
            if wait_seconds is None or wait_seconds < 0:
                wait_seconds = _FALLBACK_WAITS[attempt]
                dbg("RETRY", f"{provider_label} using fallback wait={wait_seconds}s")

            dbg("RETRY", f"{provider_label} sleeping for {wait_seconds}s")
            time.sleep(wait_seconds)
            dbg("RETRY", f"{provider_label} sleep done, looping to next attempt")

    # Unreachable — the loop either returns or raises. Defensive raise
    # in case Python's flow analysis ever needs it.
    raise RateLimitError(
        f"{provider_label} rate limited after {_MAX_RETRIES} retries.",
        raw=str(last_sdk_exc) if last_sdk_exc else "",
    )