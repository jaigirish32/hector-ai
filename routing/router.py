"""
routing/router.py

Pure routing logic. Reads the capability matrix and returns, for a given
file + set of selected providers, which providers can handle it natively
and which can't (with structured reasons).

No I/O, no SDK imports. Unit-testable in isolation.

HECTOR runs every request against multiple providers in parallel for
side-by-side comparison. The router does not pick "the best provider" —
the user already picked their providers via the UI chips. The router
answers a per-provider capability question for the file at hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from routing.capability_matrix import (
    Capability,
    KNOWN_PROVIDERS,
    Strategy,
    lookup,
)
from routing.mime_canonical import canonical_mime


class SkipReason(Enum):
    """Why a provider was excluded from the supported list."""
    UNKNOWN_PROVIDER = "unknown_provider"     # caller passed a name we don't recognise
    UNKNOWN_MIME = "unknown_mime"             # MIME couldn't be canonicalised
    NOT_NATIVE = "not_native"                 # provider doesn't natively support this type
    FILE_TOO_LARGE = "file_too_large"         # file exceeds provider's documented limit


@dataclass(frozen=True)
class SupportedProvider:
    """A provider that can handle the file natively."""
    name: str
    capability: Capability


@dataclass(frozen=True)
class SkippedProvider:
    """A provider that can't handle the file, with structured reason."""
    name: str
    reason: SkipReason
    detail: str = ""    # human-readable elaboration for UI tooltips


@dataclass(frozen=True)
class RoutingPlan:
    """
    Outcome of routing one file across the selected providers.

    The supported list is what the orchestrator will fan out to. The
    skipped list is what the UI greys out, with reasons surfaced so the
    user understands why each provider is unavailable for this file.
    """
    canonical_mime: str | None     # None if the input MIME was unrecognised
    supported: list[SupportedProvider]
    skipped: list[SkippedProvider]

    @property
    def has_any_support(self) -> bool:
        return len(self.supported) > 0


def route(
    *,
    mime: str,
    file_size_bytes: int,
    selected_providers: list[str],
) -> RoutingPlan:
    """
    Decide which selected providers can natively handle a file.

    Args:
        mime: Real MIME string from the file picker / upload (e.g.
              "application/pdf"). Will be canonicalised internally.
        file_size_bytes: Size of the file in bytes. Used to enforce
              per-provider size limits documented in the matrix.
        selected_providers: The providers the user picked via UI chips.
              Order is preserved in the output's supported/skipped lists.

    Returns:
        RoutingPlan with two lists: providers that natively support
        this file (with their Capability) and providers that don't
        (with structured SkipReasons).

    The function never raises for normal inputs. Unknown providers
    appear in the skipped list with reason UNKNOWN_PROVIDER; unknown
    MIMEs cause every provider to be skipped with reason UNKNOWN_MIME.
    """
    canonical = canonical_mime(mime)
    file_size_mb = max(0, file_size_bytes // (1024 * 1024))

    supported: list[SupportedProvider] = []
    skipped: list[SkippedProvider] = []

    for provider in selected_providers:
        # 1. Validate provider name. Typos in the caller's chip list
        # show up here rather than producing silent zero-result routes.
        if provider not in KNOWN_PROVIDERS:
            skipped.append(SkippedProvider(
                name=provider,
                reason=SkipReason.UNKNOWN_PROVIDER,
                detail=f"{provider!r} is not in the capability matrix.",
            ))
            continue

        # 2. If we couldn't canonicalise the MIME, no provider can be
        # said to support it. Mark them all as UNKNOWN_MIME.
        if canonical is None:
            skipped.append(SkippedProvider(
                name=provider,
                reason=SkipReason.UNKNOWN_MIME,
                detail=f"MIME type {mime!r} is not recognised.",
            ))
            continue

        # 3. Look up the capability. UNSUPPORTED is the default for
        # any (provider, mime) combination not explicitly listed.
        cap = lookup(provider, canonical)
        if cap.strategy != Strategy.NATIVE:
            skipped.append(SkippedProvider(
                name=provider,
                reason=SkipReason.NOT_NATIVE,
                detail=f"{provider} does not natively support .{canonical} files.",
            ))
            continue

        # 4. Native support exists, but check the size limit.
        # max_size_mb == 0 means "no documented limit, don't enforce".
        if cap.max_size_mb > 0 and file_size_mb > cap.max_size_mb:
            skipped.append(SkippedProvider(
                name=provider,
                reason=SkipReason.FILE_TOO_LARGE,
                detail=(
                    f"{provider} accepts .{canonical} up to "
                    f"{cap.max_size_mb} MB; this file is {file_size_mb} MB."
                ),
            ))
            continue

        # 5. Provider is fully cleared.
        supported.append(SupportedProvider(name=provider, capability=cap))

    return RoutingPlan(
        canonical_mime=canonical,
        supported=supported,
        skipped=skipped,
    )