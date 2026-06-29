"""The local upload store.

A standalone runtime accepts a client's files directly over HTTP rather than
through a platform object store. The store is the in-memory, session-scoped home
for those bytes: a slot is reserved when the client announces a file, the bytes
arrive on a later request, and the runtime reads them back when a command or an
out-of-band notification references the slot.

The store is also the upload boundary the model never crosses. A slot keeps the
``upload_id`` the client addresses it by and the size it declared — runtime-only
metadata — while :meth:`UploadStore.fetch` hands back only an
:class:`~reactor_runtime.core.model.UploadedFile`, the name, mime type, and
bytes a handler needs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from reactor_runtime.core import UploadedFile

UploadId = str
"""The identifier a client uses to reference an uploaded file."""


class UploadError(Exception):
    """Base for an upload operation the store cannot satisfy."""


class UnknownUploadError(UploadError):
    """No completed slot exists for the referenced upload id.

    Raised when an id names no slot at all, or names a slot whose bytes have not
    been written yet.
    """


class UploadAlreadyCompleteError(UploadError):
    """The slot already holds bytes, so a second write is refused."""


class UploadSizeMismatchError(UploadError):
    """The bytes written do not match the size the slot was created with."""


class UploadIdTakenError(UploadError):
    """A slot is already reserved under the requested upload id."""


@dataclass
class _Slot:
    """A reserved upload — runtime-only metadata, plus the bytes once written.

    The internal wrapper that retains the ``upload_id`` and declared ``size`` the
    model never needs; :meth:`UploadStore.fetch` strips it to an
    :class:`~reactor_runtime.core.model.UploadedFile`.

    Attributes:
        upload_id: The id the client references this slot by.
        name: Original file name.
        mime_type: Declared content type.
        size: The byte count the slot was created with, enforced on write.
        data: The written bytes, or ``None`` until they arrive.
    """

    upload_id: UploadId
    name: str
    mime_type: str
    size: int
    data: bytes | None = None


class UploadStore:
    """An in-memory, session-scoped store of client-uploaded files.

    Uploads are two-step: :meth:`create_slot` reserves a slot from the announced
    metadata and returns its id, then :meth:`put` writes the bytes for that id.
    :meth:`fetch` reads a completed slot back as the model-facing view. The store
    holds bytes for the life of a session so the same upload can be referenced
    more than once; :meth:`clear` drops everything when the session ends.
    """

    def __init__(self) -> None:
        """Start an empty store with no reserved slots."""
        self._slots: dict[UploadId, _Slot] = {}

    def create_slot(
        self, name: str, mime_type: str, size: int, upload_id: UploadId | None = None
    ) -> UploadId:
        """Reserve a slot for an announced file and return its upload id.

        The id is minted here by default. A caller that must address the slot by
        an id it already holds may pass *upload_id* to reserve that exact id
        instead, so a later reference to it resolves to these bytes.

        Args:
            name: Original file name.
            mime_type: Declared content type.
            size: The exact byte count the later write must match.
            upload_id: The id to reserve the slot under; minted when omitted.

        Returns:
            The upload id the slot is reserved under -- the supplied one, or a
            freshly minted one when none was given.

        Raises:
            UploadIdTakenError: If *upload_id* is already reserved.
        """
        if upload_id is None:
            upload_id = uuid.uuid4().hex
        elif upload_id in self._slots:
            raise UploadIdTakenError(upload_id)
        self._slots[upload_id] = _Slot(
            upload_id=upload_id, name=name, mime_type=mime_type, size=size
        )
        return upload_id

    def expected_size(self, upload_id: UploadId) -> int:
        """Return the exact byte count a reserved slot's write must match.

        Lets an ingress reject a write whose declared length is wrong before it
        reads the body, rather than buffering the bytes only to fail them in
        :meth:`put`.

        Args:
            upload_id: The slot's id, from :meth:`create_slot`.

        Returns:
            The byte count the slot was created with.

        Raises:
            UnknownUploadError: If no slot exists for *upload_id*.
            UploadAlreadyCompleteError: If the slot already holds bytes.
        """
        slot = self._slots.get(upload_id)
        if slot is None:
            raise UnknownUploadError(upload_id)
        if slot.data is not None:
            raise UploadAlreadyCompleteError(upload_id)
        return slot.size

    def put(self, upload_id: UploadId, data: bytes) -> None:
        """Store the bytes for a previously reserved slot.

        Args:
            upload_id: The slot's id, from :meth:`create_slot`.
            data: The file's bytes; their length must equal the slot's size.

        Raises:
            UnknownUploadError: If no slot exists for *upload_id*.
            UploadAlreadyCompleteError: If the slot already holds bytes.
            UploadSizeMismatchError: If *data* is not the slot's declared size.
        """
        slot = self._slots.get(upload_id)
        if slot is None:
            raise UnknownUploadError(upload_id)
        if slot.data is not None:
            raise UploadAlreadyCompleteError(upload_id)
        if len(data) != slot.size:
            raise UploadSizeMismatchError(f"expected {slot.size} bytes, got {len(data)}")
        slot.data = data

    async def fetch(self, upload_id: UploadId) -> UploadedFile:
        """Read a completed slot back as the model-facing upload.

        Async so the in-memory store and a future remote fetch share one shape.

        Args:
            upload_id: The slot's id.

        Returns:
            The upload's name, mime type, and bytes — without the upload id.

        Raises:
            UnknownUploadError: If no slot exists for *upload_id* or its bytes
                have not been written.
        """
        slot = self._slots.get(upload_id)
        if slot is None or slot.data is None:
            raise UnknownUploadError(upload_id)
        return UploadedFile(name=slot.name, mime_type=slot.mime_type, data=slot.data)

    def clear(self) -> None:
        """Drop every slot, releasing the session's uploaded bytes."""
        self._slots.clear()
