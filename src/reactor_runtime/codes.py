"""The error codes the runtime itself puts on a reply.

A client branches on an error's code, so every code the runtime sends is named
here instead of spelled at the call site. A model author's own codes travel
through :class:`CommandError` and are not listed.

This module sits at the top level and imports nothing. The codes are shared by
the authoring layer, the runner, and the wire codecs, so a home inside any one
of them would make two of those layers depend on each other.
"""

from __future__ import annotations

INVALID_COMMAND = "invalid_command"
"""A command's payload failed the model's contract, so no handler ran."""

UNRESOLVED_UPLOAD = "unresolved_upload"
"""A command referenced an upload the runtime could not fetch."""

INTERNAL_ERROR = "internal_error"
"""A command handler raised something other than a ``CommandError``."""

PUBLISH_REFUSED = "publish_refused"
"""The runtime refused a client's request to publish a track."""
