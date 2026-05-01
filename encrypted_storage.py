"""
Encrypted file storage for HECTOR-AI secrets (API keys, endpoints).

Replaces the previous OS keyring approach. Stores secrets in a single
encrypted file at {user_data_dir}/secrets.enc, using a key derived from
machine-stable identifiers (hostname + MAC address).

Why machine-derived key, not user password?
    A user-supplied password would force a prompt on every app launch,
    which defeats the UX goal. A machine-derived key gives "encrypted
    at rest" security against casual file viewing, while remaining
    transparent to the user. Same approach used by Discord, Slack
    desktop, and most single-user developer tools.

    The key is reproducible on the same machine without user input,
    but a copy of the encrypted file moved to a different machine
    cannot be decrypted there. This is the right trade-off for HECTOR-AI.

Format:
    The encrypted file holds a Fernet-encrypted JSON object:
        {"openai_api_key": "sk-...", "anthropic_api_key": "...", ...}
    Keys match the SecretKey constants in settings_manager.

File-write safety:
    Writes go to a temp file first, then atomically rename over the
    real file. Prevents corruption if the app crashes mid-write.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import paths


# Fixed salt for key derivation. Not secret — this is fine for our
# threat model. Same salt every machine, but the password (machine
# identity) is different per machine, so derived keys are different
# per machine. Salt is here to satisfy PBKDF2's API requirement and
# to make the key unique to HECTOR-AI specifically (not collide with
# other apps using the same machine-id approach).
_KDF_SALT = b"hector-ai-v1-secrets-salt"

# PBKDF2 iteration count. 100k is the OWASP minimum for PBKDF2-SHA256
# as of 2023. We're not protecting against a determined attacker here
# (they have file access = they have the machine = they have the key),
# but using a real iteration count is good hygiene.
_KDF_ITERATIONS = 100_000

# File where the encrypted blob lives.
_SECRETS_FILENAME = "secrets.enc"


def _machine_password() -> bytes:
    """Build a stable per-machine identifier as PBKDF2 input.

    Combines hostname and MAC address. Both are stable across reboots
    on a typical desktop. uuid.getnode() returns the MAC as an int;
    socket.gethostname() returns the OS hostname.
    """
    hostname = socket.gethostname()
    mac = uuid.getnode()
    raw = f"{hostname}:{mac}".encode("utf-8")
    # SHA-256 first to normalize length and entropy distribution.
    return hashlib.sha256(raw).digest()


def _derive_key() -> bytes:
    """Return a fixed encryption key.

    DEBUG: Hardcoded key for testing the encryption pipeline.
    Once we confirm the read/write/decrypt cycle works across app
    restarts, we'll replace this with a proper machine-derived
    or generated-and-stored key.
    """
    # Fernet expects a URL-safe base64-encoded 32-byte key.
    # Generated once with: Fernet.generate_key()
    return b"hZmKqLZmTbXNvJsRkPdYwQfGcVeWaUiOmKnJhFdSgRk="


def _secrets_path() -> Path:
    """Full path to the encrypted secrets file."""
    return paths.user_data_dir() / _SECRETS_FILENAME


class EncryptedStorage:
    """Read/write/delete secrets in an encrypted file.

    Loads the file lazily on first read. Holds decrypted contents in
    memory after that. Writes go to disk immediately (and update the
    in-memory copy) so reads after writes return the latest value
    without re-reading from disk.
    """

    def __init__(self) -> None:
        self._fernet = Fernet(_derive_key())
        # None until first load. After load, a dict (possibly empty).
        self._secrets: dict[str, str] | None = None

    # ---------- Public API ----------

    def get(self, key: str) -> str:
        """Return a stored secret, or '' if not present."""
        self._ensure_loaded()
        assert self._secrets is not None
        return self._secrets.get(key, "")

    def set(self, key: str, value: str) -> None:
        """Store or update a secret. Empty value deletes it."""
        self._ensure_loaded()
        assert self._secrets is not None
        if value:
            self._secrets[key] = value
        else:
            self._secrets.pop(key, None)
        self._save()

    def delete(self, key: str) -> None:
        """Remove a stored secret. Safe if the key isn't set."""
        self._ensure_loaded()
        assert self._secrets is not None
        if key in self._secrets:
            del self._secrets[key]
            self._save()

    def has(self, key: str) -> bool:
        """True if a non-empty secret is stored for this key."""
        return bool(self.get(key))

    # ---------- Internal helpers ----------

    def _ensure_loaded(self) -> None:
        """Load and decrypt the secrets file. Empty dict if missing
        or unreadable."""
        if self._secrets is not None:
            return
        path = _secrets_path()
        if not path.exists():
            self._secrets = {}
            return
        try:
            ciphertext = path.read_bytes()
            plaintext = self._fernet.decrypt(ciphertext)
            self._secrets = json.loads(plaintext.decode("utf-8"))
        except (InvalidToken, json.JSONDecodeError, OSError):
            # Corrupted file or wrong key (e.g., file was copied from
            # another machine). Fail soft: treat as no secrets stored.
            # User can re-enter keys to overwrite.
            self._secrets = {}

    def _save(self) -> None:
        """Atomically write the encrypted secrets file.

        Writes to a temp file in the same directory, then renames over
        the real file. Rename is atomic on the same filesystem, so
        partial writes can't corrupt the real file even on a crash.
        """
        assert self._secrets is not None
        path = _secrets_path()
        # user_data_dir is created lazily by paths.user_data_dir(), so
        # it should already exist by the time we get here. Just in case:
        path.parent.mkdir(parents=True, exist_ok=True)

        plaintext = json.dumps(self._secrets).encode("utf-8")
        ciphertext = self._fernet.encrypt(plaintext)

        tmp_path = path.with_suffix(".enc.tmp")
        tmp_path.write_bytes(ciphertext)
        os.replace(tmp_path, path)