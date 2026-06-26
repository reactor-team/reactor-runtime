
"""
STUN/TURN URI normalization for ICE server configuration.

Converts legacy forms (e.g. ``stun:host:port``, ``turn:host:port?transport=udp``)
to RFC-style URIs (e.g. ``stun://host:port``, ``turn://host:port?transport=udp``).
Useful for any code that needs to consume or produce ICE server URIs.
"""

from typing import List, Tuple
from urllib.parse import quote

from reactor_runtime.transport.webrtc.config import IceServer


def parse_ice_uri(raw: str) -> Tuple[str, str, str] | None:
    """Parse a single STUN/TURN URL string into scheme, authority, and query.

    Accepts:
      - ``stun:host:port`` or ``stuns:host:port``
      - ``turn:host:port?transport=udp`` or ``turns:host:port?transport=tcp``
      - ``stun://host:port`` or ``turn://host:port?transport=udp`` (already RFC-style)

    Returns:
      ``(scheme, authority, query)`` where query does not include the leading ``?``.
      Returns ``None`` if the string has no valid scheme or scheme is not stun/stuns/turn/turns.
    """
    raw = raw.strip()
    scheme, after = _split_scheme(raw)
    if not scheme or scheme not in ("stun", "stuns", "turn", "turns"):
        return None

    if after.startswith("//"):
        authority_and_more = after[2:]
    else:
        authority_and_more = after

    if "?" in authority_and_more:
        authority, query = authority_and_more.split("?", 1)
    else:
        authority, query = authority_and_more, ""

    authority = authority.strip()
    query = query.strip()
    return scheme, authority, query


def build_ice_uri(
    scheme: str,
    authority: str,
    query: str = "",
    *,
    username: str | None = None,
    credential: str | None = None,
) -> str:
    """Build an RFC-style STUN/TURN URI from parsed components.

    For TURN, pass username and credential to embed them as user:pass@authority.
    Credentials are percent-encoded. STUN URIs never include credentials.
    """
    if not authority:
        return f"{scheme}://"
    if scheme in ("turn", "turns") and username and credential:
        u = quote(username, safe="")
        p = quote(credential, safe="")
        uri = f"{scheme}://{u}:{p}@{authority}"
    else:
        uri = f"{scheme}://{authority}"
    if query:
        uri += f"?{query}"
    return uri


def normalize_ice_uri(raw: str) -> str | None:
    """Normalize a single STUN/TURN URL string to RFC-style form (no credentials).

    Example: ``stun:host:port`` -> ``stun://host:port``.
    Returns ``None`` if the URL is not a recognized STUN/TURN URI.
    """
    parsed = parse_ice_uri(raw)
    if not parsed or not parsed[1]:
        return None
    scheme, authority, query = parsed
    return build_ice_uri(scheme, authority, query)


def to_stun_turn_uris(servers: List[IceServer]) -> Tuple[List[str], List[str]]:
    """Convert IceServer configs into RFC-style URI lists.

    - STUN: ``stun:host:port`` -> ``stun://host:port`` (no credentials).
    - TURN: ``turn:host:port?transport=udp`` -> ``turn://host:port?transport=udp``;
      if username and credential are set on the IceServer, embeds them as
      ``turn://user:pass@host:port?transport=udp``.

    Unknown schemes are ignored. Credentials are percent-encoded.
    """
    stun_uris: List[str] = []
    turn_uris: List[str] = []

    for server in servers:
        user = server.username
        cred = server.credential

        for raw in server.urls:
            parsed = parse_ice_uri(raw)
            if not parsed:
                continue
            scheme, authority, query = parsed
            if not authority:
                continue

            if scheme in ("stun", "stuns"):
                stun_uris.append(build_ice_uri(scheme, authority, query))
                continue

            if scheme in ("turn", "turns"):
                turn_uris.append(
                    build_ice_uri(
                        scheme, authority, query, username=user, credential=cred
                    )
                )

    return stun_uris, turn_uris


def _split_scheme(rest: str) -> tuple[str, str]:
    """Split 'stun:foo' into ('stun', 'foo')."""
    i = rest.find(":")
    if i < 0:
        return "", rest
    return rest[:i].lower(), rest[i + 1 :]
