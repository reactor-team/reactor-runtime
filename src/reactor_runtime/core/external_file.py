"""References to step files stored by an external service."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

EXTERNAL_RESULTS_FILE = "_external_results.json"
"""The private step file that contains external source references."""


@dataclass(frozen=True)
class ExternalFile:
    """Describe an externally stored file without fetching its bytes.

    The step store saves the source in a separate descriptor for the runner.
    The public result lists only the file name, type, and availability deadline.
    Access credentials belong in the runner's configuration, never this value.

    Attributes:
        name: One file name, including ``output.mp4`` for an external video.
        content_type: The file's MIME type.
        provider: The runner's configured source resolver name.
        job_id: The source's job identifier, used to refresh download links.
        url: An HTTPS download address. Signed query parameters are allowed.
        expires_at: The result's known availability deadline, as an aware
            datetime. ``None`` means unknown, not unlimited availability.

    Raises:
        ValueError: A field is empty, malformed, or exceeds its size limit.
    """

    name: str
    content_type: str
    provider: str
    job_id: str = field(repr=False)
    url: str = field(repr=False)
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the source reference."""
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name in {".", "..", "result.json", EXTERNAL_RESULTS_FILE}
            or len(self.name) > 255
            or any(c in self.name for c in "/\\\x00\r\n")
        ):
            raise ValueError("external file name must be one non-reserved path segment")
        if not re.fullmatch(r"[a-zA-Z0-9!#$&^_.+-]+/[a-zA-Z0-9!#$&^_.+-]+", self.content_type):
            raise ValueError("external file content_type must be a MIME type")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.provider):
            raise ValueError("external file provider must be a resolver name")
        if not self.job_id or len(self.job_id) > 512 or any(ord(c) < 32 for c in self.job_id):
            raise ValueError("external file job_id must contain 1 to 512 printable characters")
        parsed = urlsplit(self.url)
        if (
            len(self.url) > 16384
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or any(ord(c) < 32 for c in self.url)
        ):
            raise ValueError("external file url must be an HTTPS address without user information")
        if self.expires_at is not None:
            if self.expires_at.utcoffset() is None:
                raise ValueError("external file expires_at must include a timezone")
            object.__setattr__(self, "expires_at", self.expires_at.astimezone(UTC))

    def public_entry(self) -> dict[str, str | None]:
        """Return the file facts safe to include in the public result."""
        return {
            "name": self.name,
            "content_type": self.content_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    def source_entry(self) -> dict[str, str | None]:
        """Return the private descriptor used to resolve the file."""
        return self.public_entry() | {
            "provider": self.provider,
            "job_id": self.job_id,
            "url": self.url,
        }
