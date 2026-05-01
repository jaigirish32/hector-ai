"""
Settings storage for HECTOR-AI.

API keys go to an encrypted file in the per-user data directory,
managed by the EncryptedStorage class. Encryption uses Fernet (AES)
with a key derived from machine-stable identifiers, so the file is
unreadable if copied to a different machine.

Non-secret preferences go to QSettings (standard OS preferences location).

This module is deliberately UI-free — any view can use it, and we can
swap the storage backend later without touching the UI.

Migration note: prior versions used the OS keyring (Windows Credential
Manager / macOS Keychain). v0.1.6 onward uses the encrypted file.
There is no automatic migration — users on v0.1.5 or earlier will
need to re-enter their API keys once on first launch of v0.1.6+.
Old keychain entries are harmless and can be deleted manually.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings

from encrypted_storage import EncryptedStorage
from models import DEFAULT_MODELS, Provider


# Note: KEYRING_SERVICE is no longer used — kept here briefly for
# reference in case any external code expects it. Will be removed
# in a future version. Storage now uses EncryptedStorage (see import).


# Per-provider secrets we track. Each is stored as a separate entry
# in the OS vault so they're independently rotatable.
class SecretKey:
    OPENAI_API_KEY = "openai_api_key"
    AZURE_OPENAI_API_KEY = "azure_openai_api_key"
    AZURE_OPENAI_ENDPOINT = "azure_openai_endpoint"
    AZURE_OPENAI_DEPLOYMENT_PREFIX = "azure_openai_deployment:"  # prefix + model id
    ANTHROPIC_API_KEY = "anthropic_api_key"
    GOOGLE_API_KEY = "google_api_key"
    XAI_API_KEY = "xai_api_key"


class SettingsManager:
    """Facade for all HECTOR-AI configuration storage."""

    def __init__(self) -> None:
        # QSettings uses app/org names from main.py — same ones you set earlier.
        self._qsettings = QSettings()
        # Encrypted file storage for secrets. Lazy-loads on first read,
        # writes through to disk on every change. No OS prompts, no
        # cross-platform behavior differences.
        self._storage = EncryptedStorage()
    # ------------------------------------------------------------------
    # Secret storage (API keys, endpoints) — OS credential vault
    # ------------------------------------------------------------------

    def get_secret(self, key: str) -> str:
        """Return a stored secret, or '' if never set."""
        try:
            return self._storage.get(key)
        except Exception:
            # On rare I/O or decryption errors, fail soft — return empty
            # rather than crash. The user can re-enter their keys.
            return ""

    def set_secret(self, key: str, value: str) -> None:
        """Store or update a secret. Empty value deletes it."""
        if not value:
            self.delete_secret(key)
            return
        try:
            self._storage.set(key, value)
        except Exception as exc:
            raise SettingsError(f"Could not save secret: {exc}")


    def delete_secret(self, key: str) -> None:
        """Remove a stored secret. Safe to call if the secret isn't set."""
        try:
            self._storage.delete(key)
        except Exception:
            # Best-effort delete. If it fails, the worst case is the
            # value is still in the file — caller should not crash.
            pass

    def has_secret(self, key: str) -> bool:
        """Return True if a non-empty secret is stored for this key."""
        return bool(self.get_secret(key))

    # ------------------------------------------------------------------
    # Non-secret preferences — QSettings (plain file in app config dir)
    # ------------------------------------------------------------------

    def get_preference(self, key: str, default: str = "") -> str:
        value = self._qsettings.value(key, default)
        return str(value) if value is not None else default

    def set_preference(self, key: str, value: str) -> None:
        self._qsettings.setValue(key, value)

    # ------------------------------------------------------------------
    # Convenience — which providers are configured?
    # ------------------------------------------------------------------

    def configured_providers(self) -> set[Provider]:
        """Return the set of providers that currently have API keys set."""
        configured: set[Provider] = set()

        if self.has_secret(SecretKey.OPENAI_API_KEY):
            configured.add(Provider.OPENAI)

        if (
            self.has_secret(SecretKey.AZURE_OPENAI_API_KEY)
            and self.has_secret(SecretKey.AZURE_OPENAI_ENDPOINT)
        ):
            configured.add(Provider.AZURE_OPENAI)

        if self.has_secret(SecretKey.ANTHROPIC_API_KEY):
            configured.add(Provider.ANTHROPIC)

        if self.has_secret(SecretKey.GOOGLE_API_KEY):
            configured.add(Provider.GOOGLE)

        if self.has_secret(SecretKey.XAI_API_KEY):
            configured.add(Provider.XAI)

        # Local models don't need keys — always "configured" if any exist.
        if any(m.provider == Provider.LOCAL for m in DEFAULT_MODELS):
            configured.add(Provider.LOCAL)

        return configured

    def is_model_runnable(self, model_id: str) -> bool:
        """True if the provider for this model has its credentials set."""
        from models import get_model

        model = get_model(model_id)
        if model is None:
            return False
        return model.provider in self.configured_providers()


class SettingsError(Exception):
    """Raised when a settings operation fails for a reason we care about."""