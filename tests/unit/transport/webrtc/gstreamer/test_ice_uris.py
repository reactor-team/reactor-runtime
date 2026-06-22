
"""Unit tests for STUN/TURN URI normalization (ice_uris module)."""

import pytest

from reactor_runtime.transport.webrtc.gstreamer.ice_uris import (
    build_ice_uri,
    normalize_ice_uri,
    parse_ice_uri,
    to_stun_turn_uris,
)
from reactor_runtime.transport.webrtc.config import IceServer


# =============================================================================
# parse_ice_uri
# =============================================================================


class TestParseIceUri:
    """Tests for parse_ice_uri."""

    def test_stun_legacy(self):
        assert parse_ice_uri("stun:host:3478") == ("stun", "host:3478", "")

    def test_stun_legacy_uppercase_scheme(self):
        assert parse_ice_uri("STUN:host:3478") == ("stun", "host:3478", "")

    def test_stuns_legacy(self):
        assert parse_ice_uri("stuns:host:5349") == ("stuns", "host:5349", "")

    def test_stun_rfc_style(self):
        assert parse_ice_uri("stun://host:3478") == ("stun", "host:3478", "")

    def test_turn_legacy(self):
        assert parse_ice_uri("turn:host:3478") == ("turn", "host:3478", "")

    def test_turn_legacy_with_query(self):
        assert parse_ice_uri("turn:host:3478?transport=udp") == (
            "turn",
            "host:3478",
            "transport=udp",
        )

    def test_turns_legacy_with_query(self):
        assert parse_ice_uri("turns:host:5349?transport=tcp") == (
            "turns",
            "host:5349",
            "transport=tcp",
        )

    def test_turn_rfc_style_with_query(self):
        assert parse_ice_uri("turn://host:3478?transport=udp") == (
            "turn",
            "host:3478",
            "transport=udp",
        )

    def test_stun_with_whitespace(self):
        assert parse_ice_uri("  stun:host:3478  ") == ("stun", "host:3478", "")

    def test_unknown_scheme_returns_none(self):
        assert parse_ice_uri("http://example.com") is None
        assert parse_ice_uri("https:host:443") is None
        assert parse_ice_uri("foo:bar") is None

    def test_no_scheme_returns_none(self):
        assert parse_ice_uri("host:3478") is None
        assert parse_ice_uri("") is None

    def test_empty_authority_returns_scheme_and_empty_strings(self):
        # Still valid parse; authority is empty
        assert parse_ice_uri("stun:") == ("stun", "", "")
        assert parse_ice_uri("turn://?transport=udp") == ("turn", "", "transport=udp")


# =============================================================================
# build_ice_uri
# =============================================================================


class TestBuildIceUri:
    """Tests for build_ice_uri."""

    def test_stun_no_query(self):
        assert build_ice_uri("stun", "host:3478") == "stun://host:3478"

    def test_stun_with_query(self):
        assert build_ice_uri("stun", "host:3478", "transport=udp") == (
            "stun://host:3478?transport=udp"
        )

    def test_turn_no_credentials(self):
        assert build_ice_uri("turn", "host:3478", "transport=udp") == (
            "turn://host:3478?transport=udp"
        )

    def test_turn_with_credentials(self):
        assert build_ice_uri(
            "turn", "host:3478", "transport=udp", username="user", credential="pass"
        ) == "turn://user:pass@host:3478?transport=udp"

    def test_turn_credentials_percent_encoded(self):
        uri = build_ice_uri(
            "turn",
            "host:3478",
            username="u@ser",
            credential="p:ass",
        )
        assert "u%40ser" in uri
        assert "p%3Aass" in uri
        assert uri.startswith("turn://")

    def test_stun_ignores_credentials(self):
        assert build_ice_uri(
            "stun", "host:3478", username="user", credential="pass"
        ) == "stun://host:3478"

    def test_empty_authority(self):
        assert build_ice_uri("stun", "") == "stun://"

    def test_turns_with_credentials(self):
        assert build_ice_uri(
            "turns", "host:5349", "transport=tcp", username="u", credential="p"
        ) == "turns://u:p@host:5349?transport=tcp"

    def test_turn_partial_credentials_omitted(self):
        # Only both user+credential embed; one alone does not
        assert build_ice_uri("turn", "host:3478", username="u") == "turn://host:3478"
        assert build_ice_uri("turn", "host:3478", credential="p") == "turn://host:3478"


# =============================================================================
# normalize_ice_uri
# =============================================================================


class TestNormalizeIceUri:
    """Tests for normalize_ice_uri."""

    def test_stun_legacy_to_rfc(self):
        assert normalize_ice_uri("stun:host:3478") == "stun://host:3478"

    def test_stuns_legacy_to_rfc(self):
        assert normalize_ice_uri("stuns:host:5349") == "stuns://host:5349"

    def test_turn_with_query(self):
        assert normalize_ice_uri("turn:host:3478?transport=udp") == (
            "turn://host:3478?transport=udp"
        )

    def test_already_rfc_unchanged(self):
        assert normalize_ice_uri("stun://host:3478") == "stun://host:3478"

    def test_invalid_returns_none(self):
        assert normalize_ice_uri("http://example.com") is None
        assert normalize_ice_uri("") is None
        assert normalize_ice_uri("not-a-uri") is None

    def test_empty_authority_returns_none(self):
        assert normalize_ice_uri("stun:") is None


# =============================================================================
# to_stun_turn_uris
# =============================================================================


class TestToStunTurnUris:
    """Tests for to_stun_turn_uris."""

    def test_empty_list(self):
        assert to_stun_turn_uris([]) == ([], [])

    def test_single_stun(self):
        servers = [IceServer(urls=["stun:stun.example.com:3478"])]
        stun, turn = to_stun_turn_uris(servers)
        assert stun == ["stun://stun.example.com:3478"]
        assert turn == []

    def test_single_turn_no_credentials(self):
        servers = [IceServer(urls=["turn:turn.example.com:3478?transport=udp"])]
        stun, turn = to_stun_turn_uris(servers)
        assert stun == []
        assert turn == ["turn://turn.example.com:3478?transport=udp"]

    def test_single_turn_with_credentials(self):
        servers = [
            IceServer(
                urls=["turn:turn.example.com:3478?transport=udp"],
                username="user",
                credential="secret",
            )
        ]
        stun, turn = to_stun_turn_uris(servers)
        assert stun == []
        assert turn == ["turn://user:secret@turn.example.com:3478?transport=udp"]

    def test_mixed_stun_and_turn(self):
        servers = [
            IceServer(urls=["stun:stun.example.com:3478", "turn:turn.example.com:3478"])
        ]
        stun, turn = to_stun_turn_uris(servers)
        assert stun == ["stun://stun.example.com:3478"]
        assert turn == ["turn://turn.example.com:3478"]

    def test_multiple_servers(self):
        servers = [
            IceServer(urls=["stun:a.example.com:3478"]),
            IceServer(urls=["stun:b.example.com:3478"]),
            IceServer(
                urls=["turn:c.example.com:3478"],
                username="u",
                credential="p",
            ),
        ]
        stun, turn = to_stun_turn_uris(servers)
        assert stun == ["stun://a.example.com:3478", "stun://b.example.com:3478"]
        assert turn == ["turn://u:p@c.example.com:3478"]

    def test_unknown_scheme_skipped(self):
        servers = [IceServer(urls=["stun:good:3478", "http://bad", "turn:also:3478"])]
        stun, turn = to_stun_turn_uris(servers)
        assert stun == ["stun://good:3478"]
        assert turn == ["turn://also:3478"]

    def test_empty_authority_skipped(self):
        servers = [IceServer(urls=["stun:", "stun:valid:3478"])]
        stun, turn = to_stun_turn_uris(servers)
        assert stun == ["stun://valid:3478"]
        assert turn == []

    def test_credentials_percent_encoded(self):
        servers = [
            IceServer(
                urls=["turn:host:3478"],
                username="user@name",
                credential="pass:word",
            )
        ]
        stun, turn = to_stun_turn_uris(servers)
        assert turn == ["turn://user%40name:pass%3Aword@host:3478"]
