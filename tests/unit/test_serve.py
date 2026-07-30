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
    _load_config,
    _log_level_from_env,
    _port_range_from_env,
    _version,
    _webrtc_config_from_env,
    main,
)
from reactor_runtime.transport.webrtc.config import IceTransportPolicy, WebRtcConfig
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


_MANIFEST = """\
model:
  name: demo
  version: 0.1.0
runtime:
  import: pipeline:Demo
  config: config.yml
"""


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


def test_load_config_reads_the_model_reference_from_runtime_import(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    cfg = _load_config(manifest)

    assert cfg.model_ref == "pipeline:Demo"


def test_load_config_resolves_runtime_config_against_the_manifest_dir(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    cfg = _load_config(manifest)

    assert cfg.config_path == tmp_path / "config.yml"


def test_load_config_leaves_config_path_none_when_unset(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("runtime:\n  import: pipeline:Demo\n")

    cfg = _load_config(manifest)

    assert cfg.config_path is None


def test_load_config_refuses_a_manifest_without_runtime_import(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("model:\n  name: demo\n")

    with pytest.raises(SystemExit):
        _load_config(manifest)


def test_load_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    # A tab where YAML expects spaces is a syntax error, not a mapping problem.
    manifest.write_text("runtime:\n\timport: pipeline:Demo\n")

    with pytest.raises(SystemExit, match="invalid YAML"):
        _load_config(manifest)


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
