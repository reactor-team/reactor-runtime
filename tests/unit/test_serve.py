import logging
import socket
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from reactor_runtime.core import ConnId, LiveKitConfig, RuntimeConfig
from reactor_runtime.http import HttpServer
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.runner import Runner
from reactor_runtime.runner.session_host import SessionHost
from reactor_runtime.serve import (
    _apply_env,
    _assemble,
    _listen,
    _log_level_from_env,
    _port_range_from_env,
    _version,
    _video_codecs_from_env,
    _webrtc_config_from_env,
    main,
    serve,
)
from reactor_runtime.service import Service
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
    "EXIT_TIMEOUT_SECONDS",
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


def _served(service: Service) -> tuple[list[dict[str, str]], set[str]]:
    """The transports an assembled process lists, and the routes it serves."""
    components = service._components
    runner, http = components["runner"], components["http"]
    assert isinstance(runner, Runner)
    assert isinstance(http, HttpServer)
    paths = {getattr(route, "path", "") for route in http._app.routes}
    return runner.descriptor()["transports"], paths


def test_assemble_serves_livekit_beside_webrtc_when_the_extra_is_installed() -> None:
    transports, paths = _served(
        _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)
    )

    assert transports == [
        {"protocol": "webrtc", "version": "1.0"},
        {"protocol": "livekit", "version": "1.0"},
    ]
    assert "/sessions/{sid}/transport/livekit/connections" in paths


def test_assemble_serves_webrtc_alone_without_the_livekit_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reactor_runtime.serve.livekit_available", lambda: False)

    transports, paths = _served(
        _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)
    )

    assert transports == [{"protocol": "webrtc", "version": "1.0"}]
    assert not any("/transport/livekit/" in path for path in paths)


def test_assemble_serves_livekit_beside_a_session_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reactor_runtime.serve.batched_capacity", lambda ref: 3)
    cfg = RuntimeConfig(model_ref="fake:Model", livekit=LiveKitConfig(topics=("avatar",)))

    service = _assemble(cfg, peer_factory=_UNUSED)

    host, http = service._components["runner"], service._components["http"]
    assert isinstance(host, SessionHost)
    assert isinstance(http, HttpServer)
    paths = {getattr(route, "path", "") for route in http._app.routes}
    assert any("/transport/livekit/" in path for path in paths)
    assert all(
        {t["protocol"] for t in slot.descriptor()["transports"]} == {"webrtc", "livekit"}
        for slot in host.slots
    )


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
        "capacity": {"slots": 1, "free": 0},
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


def test_assemble_renders_the_transport_instruments_on_the_endpoint() -> None:
    # Every other metrics test builds a holder by hand and reads the registry
    # back directly, which passes whether or not the group that declared those
    # instruments is the one `/metrics` renders. This goes through the assembly
    # and the endpoint instead, so a component wired to a holder of its own —
    # the failure that leaves a metric live in memory and absent from every
    # scrape — is caught here rather than in production.
    service = _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)
    http = service._components["http"]
    assert isinstance(http, HttpServer)

    body = TestClient(http._app).get("/metrics").text

    # A labelled instrument with no children yet renders its HELP and TYPE lines
    # and no samples, so the name is what a scrape carries before a wire is live.
    for family in (
        "runtime_webrtc_rtt_seconds",
        "runtime_webrtc_media_rtt_seconds",
        "runtime_webrtc_loss_ratio",
        "runtime_webrtc_bandwidth_estimate_bytes_per_second",
        "runtime_webrtc_nacks_total",
        "runtime_webrtc_keyframe_requests_total",
        "runtime_webrtc_packets_lost_total",
        "runtime_webrtc_packets_retransmitted_total",
        "runtime_webrtc_bytes_sent_total",
        "runtime_webrtc_frames_sent_total",
        "runtime_media_dropped_frames_total",
    ):
        assert family in body, f"{family} never reached the endpoint"


def test_assemble_wires_the_runner_shutdown_to_the_service() -> None:
    service = _assemble(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)

    runner = service._components["runner"]
    assert isinstance(runner, Runner)
    assert runner.request_shutdown == service.request_shutdown


def test_assemble_hands_the_exit_budgets_to_the_service() -> None:
    cfg = RuntimeConfig(model_ref="fake:Model", grace_period=7.0, exit_timeout=3.0)

    service = _assemble(cfg, peer_factory=_UNUSED)

    assert service._grace_period == 7.0
    assert service._exit_timeout == 3.0


class _ServiceStub:
    """A service whose run is instant, standing in for the assembled one."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.bounded = False

    async def run(self) -> None: ...

    def bound_process_exit(self) -> None:
        self.bounded = True


async def test_serve_reports_the_service_exit_status_and_bounds_the_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _ServiceStub(exit_code=1)
    monkeypatch.setattr("reactor_runtime.serve._assemble", lambda *a, **k: stub)
    monkeypatch.setattr("reactor_runtime.serve._listen", lambda cfg: [])

    code = await serve(RuntimeConfig(model_ref="fake:Model"), peer_factory=_UNUSED)

    assert code == 1
    # The teardown after the loop is the caller's; serve arms its deadline.
    assert stub.bounded


async def test_serve_listens_before_it_imports_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    listening: list[socket.socket] = []

    def listen(cfg: RuntimeConfig) -> list[socket.socket]:
        order.append("listen")
        listening.extend(_listen(cfg))
        return listening

    def capacity(ref: str) -> int:
        order.append("import")
        return 1

    async def run(self: Service) -> None: ...

    monkeypatch.setattr("reactor_runtime.serve._listen", listen)
    monkeypatch.setattr("reactor_runtime.serve.batched_capacity", capacity)
    monkeypatch.setattr(Service, "run", run)
    monkeypatch.setattr(Service, "bound_process_exit", lambda self: None)

    try:
        await serve(
            RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=0), peer_factory=_UNUSED
        )
    finally:
        for sock in listening:
            sock.close()

    # A probe that connects while the model's class imports is queued on the
    # bound socket and answered once the server runs, rather than refused.
    assert order == ["listen", "import"]


def _free_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as probe:
        return probe.getsockname()[1]


@pytest.mark.parametrize("host", ["localhost", ""])
def test_listen_binds_every_address_the_host_resolves_to(host: str) -> None:
    port = _free_port()
    expected = {
        (family, address[0])
        for family, _, _, _, address in socket.getaddrinfo(
            host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    }

    sockets = _listen(RuntimeConfig(model_ref="fake:Model", host=host, port=port))
    try:
        bound = {(sock.family, sock.getsockname()[0]) for sock in sockets}
        assert bound == expected
        assert socket.AF_INET in {family for family, _ in bound}
        with socket.create_connection(("127.0.0.1", port), timeout=5.0):
            pass
    finally:
        for sock in sockets:
            sock.close()


def test_listen_exits_when_the_address_is_taken() -> None:
    with socket.create_server(("127.0.0.1", 0)) as taken:
        port = taken.getsockname()[1]
        with pytest.raises(SystemExit, match=f"cannot listen on 127.0.0.1:{port}"):
            _listen(RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=port))


def _fake_main_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    (tmp_path / "reactor.yaml").write_text("runtime:\n  import: fake:Model\n")
    monkeypatch.chdir(tmp_path)

    async def fake_serve(*args: object, **kwargs: object) -> int:
        return code

    monkeypatch.setattr("reactor_runtime.serve.serve", fake_serve)


def test_main_exits_with_the_failure_status_after_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_main_environment(tmp_path, monkeypatch, code=1)

    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1


def test_main_returns_normally_after_a_clean_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_main_environment(tmp_path, monkeypatch, code=0)

    main()


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
    monkeypatch.setenv("EXIT_TIMEOUT_SECONDS", "3.5")

    cfg = _apply_env(RuntimeConfig(model_ref="fake:Model"))

    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8090
    assert cfg.orphan_timeout == 5.0
    assert cfg.grace_period == 10.0
    assert cfg.exit_timeout == 3.5


def test_apply_env_keeps_defaults_when_unset() -> None:
    cfg = _apply_env(RuntimeConfig(model_ref="fake:Model"))

    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8080
    assert cfg.exit_timeout == 10.0


def test_apply_env_rejects_a_non_numeric_exit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXIT_TIMEOUT_SECONDS", "soon")

    with pytest.raises(SystemExit):
        _apply_env(RuntimeConfig(model_ref="fake:Model"))


@pytest.mark.parametrize("name", ["EXIT_TIMEOUT_SECONDS", "SIGTERM_GRACE_PERIOD"])
@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "-1"])
def test_apply_env_rejects_a_deadline_window_a_timer_cannot_hold(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    # Both windows feed the forced-exit deadline. `float` parses these, but an
    # infinite timer never fires and a NaN or negative one fires at once, so
    # the misconfiguration is refused at boot instead.
    monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit, match="finite number >= 0"):
        _apply_env(RuntimeConfig(model_ref="fake:Model"))


def test_apply_env_accepts_a_zero_exit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXIT_TIMEOUT_SECONDS", "0")

    assert _apply_env(RuntimeConfig(model_ref="fake:Model")).exit_timeout == 0.0


def test_apply_env_rejects_a_non_integer_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "eighty-ninety")

    with pytest.raises(SystemExit):
        _apply_env(RuntimeConfig(model_ref="fake:Model"))


def test_log_level_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTOR_LOG_LEVEL", "debug")
    assert _log_level_from_env() == logging.DEBUG


def test_log_level_defaults_to_info() -> None:
    assert _log_level_from_env() == logging.INFO
