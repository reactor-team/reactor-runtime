"""The HTTP surface — the runtime's single ingress.

The runtime is a standalone HTTP black box. This package holds the thin route
groups over the runner's read-only surface and the server that assembles them,
mounts each transport's routes, and runs the ASGI app. No platform knowledge
lives here: the surface is a documented HTTP contract a standalone caller drives
directly.
"""

from reactor_runtime.http.routes import EgressRoutes, SessionRoutes, UploadRoutes
from reactor_runtime.http.server import HttpServer

__all__ = [
    "EgressRoutes",
    "HttpServer",
    "SessionRoutes",
    "UploadRoutes",
]
