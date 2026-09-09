import logging
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from reactor_runtime.core import ConnId, RuntimeConfig
from reactor_runtime.http import HttpServer
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.runner import Runner
from reactor_runtime.serve import (
    _apply_env,
    _assemble,
    _log_level_from_env,
    _port_range_from_env,
    _version,
    _video_codecs_from_env,
    _webrtc_config_from_env,
    main,
)
from reactor_runtime.transport.webrtc.config import CodecEntry, IceTransportPolicy, WebRtcConfig
from reactor_runtime.transport.webrtc.peer import WebRtcPeerFactory
from reactor_runtime.transport.webrtc.signaling import SdpAnswer, SdpOffer, TrackMap


async def _unused_factory(
    conn_id: ConnId,
    offer: SdpOffer,
    tracks: TrackMap,
    config: WebRtcConfig,
    version: ProtocolVersion,
) -> tuple[object, SdpAnswer]:
    """A peer factory that must never be invoked during assembly."""
    raise AssertionError("peer factory must not be invoked during assembly")


_UNUSED: WebRtcPeerFactory = cast(WebRtcPeerFactory, _unused_factory)


_WEBRTC_ENV = (
    "STUN_SERVERS",
    "TURN_SERVERS",
    "WEBRTC_PORT_RANGE",
    "ICE_TRANSPORT_POLICY",
    "WEBRTC_CLIENT_PING_TIMEOUT_SECONDS",
    "WEBRTC_BWE_MIN_KBPS",
    "WEBRTC_BWE_MAX_KBPS",
    "WEBRTC_BWE_INITIAL_KBPS",
    "WEBRTC_SENDER_MAX_KBPS",
    "WEBRTC_SENDER_MIN_KBPS",
    "WEBRTC_VIDEO_CODECS",
)

_RUNTIME_ENV = (
    "HOST",
    "PORT",
    "ORPHAN_TIMEOUT_SECONDS",
    "SIGTERM_GRACE_PERIOD",
    "REACTOR_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _clear_adapter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test against a clean environment for the serve adapter."""
    for name in (*_WEBRTC_ENV, *_RUNTIME_ENV):
        monkeypatch.delenv(name, raising=False)


def test_assemble_uses_libwebrtc_peer_factory_by_default() -> None:
    pytest.importorskip("reactor_webrtc")
    # The peer_factory=None branch does a deferred import; verify it succeeds.
    service = _assemble(RuntimeConfig(model_ref="fake:Model"))
    assert "runner" in service._components


def test_assemble_hooks_on_runner_then_http() -> None:
    service = _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)

    components = service._components
    assert set(components) == {"runner", "http"}
    assert isinstance(components["runner"], Runner)
    assert isinstance(components["http"], HttpServer)
    assert components["http"].depends_on == ("runner",)


def test_assemble_answers_health_with_the_process_aggregate() -> None:
    service = _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)
    http = service._components["http"]
    assert isinstance(http, HttpServer)

    response = TestClient(http._app).get("/health")

    # A freshly assembled process discriminates the two wirings: its runner is
    # healthy and loading, so only the aggregate — which also sees the server
    # that has not started — answers unhealthy.
    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "state": "loading",
        "detail": "http server not started",
    }


def test_assemble_publishes_the_identity_of_the_process() -> None:
    service = _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)
    http = service._components["http"]
    assert isinstance(http, HttpServer)

    body = TestClient(http._app).get("/metrics").text

    # The assembly is the one place the identity gets its real values. Every
    # other test builds a holder by hand, so only this one catches a version or
    # a model reference that was never read off the process.
    assert f'runtime_info{{model="fake:Model",version="{_version()}"}} 1.0' in body


def test_assemble_wires_the_runner_shutdown_to_the_service() -> None:
    service = _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)

    runner = service._components["runner"]
    assert isinstance(runner, Runner)
    assert runner.request_shutdown == service.request_shutdown


def test_version_is_a_non_empty_string() -> None:
    assert isinstance(_version(), str)
    assert _version()


def test_main_refuses_when_no_manifest_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        main()


def test_webrtc_config_falls_back_to_a_public_stun_when_unconfigured() -> None:
    config = _webrtc_config_from_env()

    assert len(config.ice_servers) == 1
    assert config.ice_servers[0].urls == ("stun:stun.l.google.com:19302",)
    assert config.ice_servers[0].username is None
    assert config.transport_policy is IceTransportPolicy.ALL
    assert config.port_range is None
    assert config.ping_timeout == 20.0
    assert config.bwe_min_kbps == WebRtcConfig.bwe_min_kbps
    assert config.bwe_max_kbps == WebRtcConfig.bwe_max_kbps
    assert config.bwe_initial_kbps == WebRtcConfig.bwe_initial_kbps
    assert config.sender_max_kbps == WebRtcConfig.sender_max_kbps
    assert config.sender_min_kbps == WebRtcConfig.sender_min_kbps


def test_webrtc_config_reads_bwe_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBRTC_BWE_MIN_KBPS", "800")
    monkeypatch.setenv("WEBRTC_BWE_MAX_KBPS", "8000")
    monkeypatch.setenv("WEBRTC_BWE_INITIAL_KBPS", "3000")

    config = _webrtc_config_from_env()

    assert config.bwe_min_kbps == 800
    assert config.bwe_max_kbps == 8000
    assert config.bwe_initial_kbps == 3000


def test_the_default_sender_ceiling_is_ten_megabits() -> None:
    """The per-sender ceiling exists to clear libwebrtc's resolution-keyed
    default of 2500 kbps, which is where every frame size above 960x540 lands.
    A default that did not clear it would leave the limit in place and the
    knob looking broken."""
    assert WebRtcConfig.sender_max_kbps == 10000
    assert WebRtcConfig.sender_max_kbps == WebRtcConfig.bwe_max_kbps


def test_webrtc_config_reads_sender_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBRTC_SENDER_MAX_KBPS", "8000")
    monkeypatch.setenv("WEBRTC_SENDER_MIN_KBPS", "2000")

    config = _webrtc_config_from_env()

    assert config.sender_max_kbps == 8000
    assert config.sender_min_kbps == 2000


def test_webrtc_config_rejects_a_sender_floor_above_its_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """libwebrtc refuses the pair, so the alternative to failing at boot is a
    RuntimeError repeated on every negotiation for the process's life."""
    monkeypatch.setenv("WEBRTC_SENDER_MAX_KBPS", "2000")
    monkeypatch.setenv("WEBRTC_SENDER_MIN_KBPS", "8000")

    with pytest.raises(SystemExit):
        _webrtc_config_from_env()


def test_an_unset_sender_bound_does_not_trip_the_ordering_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``0`` means "leave at the libwebrtc default", not "a ceiling of zero" —
    so a floor above it is not an inversion and must not be refused."""
    monkeypatch.setenv("WEBRTC_SENDER_MAX_KBPS", "0")
    monkeypatch.setenv("WEBRTC_SENDER_MIN_KBPS", "2000")

    config = _webrtc_config_from_env()

    assert config.sender_max_kbps == 0
    assert config.sender_min_kbps == 2000


def test_webrtc_config_rejects_a_non_integer_bwe_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBRTC_BWE_MIN_KBPS", "not-a-number")

    with pytest.raises(SystemExit):
        _webrtc_config_from_env()


def test_webrtc_config_rejects_bwe_max_below_the_default_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A partial override: only WEBRTC_BWE_MAX_KBPS is set, below the default initial
    # value (4000). Left unchecked, this passes boot and fails every negotiation instead.
    monkeypatch.setenv("WEBRTC_BWE_MAX_KBPS", "3000")

    with pytest.raises(SystemExit):
        _webrtc_config_from_env()


def test_webrtc_config_rejects_bwe_min_above_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBRTC_BWE_MIN_KBPS", "5000")
    monkeypatch.setenv("WEBRTC_BWE_INITIAL_KBPS", "4000")

    with pytest.raises(SystemExit):
        _webrtc_config_from_env()


def test_webrtc_config_rejects_a_negative_bwe_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBRTC_BWE_MIN_KBPS", "-1")

    with pytest.raises(SystemExit):
        _webrtc_config_from_env()


def test_webrtc_config_accepts_a_consistent_bwe_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBRTC_BWE_MIN_KBPS", "800")
    monkeypatch.setenv("WEBRTC_BWE_INITIAL_KBPS", "3000")
    monkeypatch.setenv("WEBRTC_BWE_MAX_KBPS", "3000")

    config = _webrtc_config_from_env()

    assert (config.bwe_min_kbps, config.bwe_initial_kbps, config.bwe_max_kbps) == (
        800,
        3000,
        3000,
    )


def test_video_codecs_from_env_defaults_to_the_config_default() -> None:
    assert _video_codecs_from_env() == WebRtcConfig.supported_video_codecs


def test_video_codecs_from_env_reads_a_preference_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBRTC_VIDEO_CODECS", "vp8,VP9")

    codecs = _video_codecs_from_env()

    assert codecs == (cast(CodecEntry, {"codec": "VP8"}), cast(CodecEntry, {"codec": "VP9"}))


def test_video_codecs_from_env_rejects_an_unknown_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBRTC_VIDEO_CODECS", "VP8,Theora")

    with pytest.raises(SystemExit):
        _video_codecs_from_env()


def test_webrtc_config_reads_video_codecs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBRTC_VIDEO_CODECS", "H264,VP8")

    config = _webrtc_config_from_env()

    assert config.supported_video_codecs == (
        cast(CodecEntry, {"codec": "H264"}),
        cast(CodecEntry, {"codec": "VP8"}),
    )


def test_webrtc_config_reads_stun_turn_policy_and_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUN_SERVERS", "stun:stun.relay.metered.ca:80")
    monkeypatch.setenv(
        "TURN_SERVERS",
        "user;secret;turn:global.relay.metered.ca:80,"
        "user;secret;turns:global.relay.metered.ca:443?transport=tcp",
    )
    monkeypatch.setenv("ICE_TRANSPORT_POLICY", "relay")
    monkeypatch.setenv("WEBRTC_CLIENT_PING_TIMEOUT_SECONDS", "45")

    config = _webrtc_config_from_env()

    assert [server.urls[0] for server in config.ice_servers] == [
        "stun:stun.relay.metered.ca:80",
        "turn:global.relay.metered.ca:80",
        "turns:global.relay.metered.ca:443?transport=tcp",
    ]
    turn = config.ice_servers[1]
    assert turn.username == "user"
    assert turn.credential == "secret"
    assert config.transport_policy is IceTransportPolicy.RELAY
    assert config.ping_timeout == 45.0


def test_webrtc_config_rejects_a_malformed_turn_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TURN_SERVERS", "user;turn:no-credential:3478")

    with pytest.raises(SystemExit):
        _webrtc_config_from_env()


def test_webrtc_config_rejects_an_unknown_ice_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICE_TRANSPORT_POLICY", "direct")

    with pytest.raises(SystemExit):
        _webrtc_config_from_env()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10000:20000", (10000, 20000)),
        (":20000", (1024, 20000)),
        ("10000:", (10000, 65535)),
    ],
)
def test_port_range_parses_each_form(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: tuple[int, int]
) -> None:
    monkeypatch.setenv("WEBRTC_PORT_RANGE", value)

    assert _port_range_from_env() == expected


def test_port_range_is_none_when_unset() -> None:
    assert _port_range_from_env() is None


@pytest.mark.parametrize("value", ["20000:10000", "100:200", "10000", "abc:def"])
def test_port_range_rejects_invalid_input(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("WEBRTC_PORT_RANGE", value)

    with pytest.raises(SystemExit):
        _port_range_from_env()


def test_apply_env_overlays_bind_and_lifecycle_tunables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8090")
    monkeypatch.setenv("ORPHAN_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("SIGTERM_GRACE_PERIOD", "10")

    cfg = _apply_env(RuntimeConfig(model_ref="fake:Model"))

    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8090
    assert cfg.orphan_timeout == 5.0
    assert cfg.grace_period == 10.0


def test_apply_env_keeps_defaults_when_unset() -> None:
    cfg = _apply_env(RuntimeConfig(model_ref="fake:Model"))

    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8080


def test_apply_env_rejects_a_non_integer_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "eighty-ninety")

    with pytest.raises(SystemExit):
        _apply_env(RuntimeConfig(model_ref="fake:Model"))


def test_log_level_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTOR_LOG_LEVEL", "debug")
    assert _log_level_from_env() == logging.DEBUG


def test_log_level_defaults_to_info() -> None:
    assert _log_level_from_env() == logging.INFO
