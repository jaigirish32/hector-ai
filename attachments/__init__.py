"""
Attachment handling for HECTOR-AI.

Manages local file metadata (registry.py) and per-provider file uploads
(uploaders.py). Files are uploaded once per provider and referenced by
their cached file_id on subsequent requests.
"""
from attachments.registry import FileRegistry

__all__ = ["FileRegistry"]