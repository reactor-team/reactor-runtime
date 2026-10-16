"""The runtime internals the launcher hooks still have the shape it expects.

The scenarios run only on demand and after a release, so a refactor that
renames or reshapes one of these would otherwise first break a release's
endurance run. Update ``endurance/launcher.py`` in the same change.
"""

import inspect

from endurance import launcher

import reactor_runtime.http.server as http_server
from reactor_runtime import serve
from reactor_runtime.runner.client_stats import ClientStatsGate
from reactor_runtime.runner.connection_manager import ConnectionManager
from reactor_runtime.runner.runner import Runner
from reactor_runtime.transport.webrtc.connection import WebRTCConnection
from reactor_runtime.transport.webrtc.pacer import MediaPacer
from reactor_runtime.transport.webrtc.peer import WebRTCPeer


def test_build_app_takes_the_parameters_the_wrapper_passes() -> None:
    expected = list(inspect.signature(launcher.build_app).parameters)
    assert list(inspect.signature(http_server.build_app).parameters) == expected


def test_media_pacer_stop_takes_no_arguments() -> None:
    assert list(inspect.signature(MediaPacer.stop).parameters) == ["self"]


def test_the_runner_keeps_its_connections_in_a_manager_with_a_count() -> None:
    assert "self._connections = ConnectionManager(" in inspect.getsource(Runner)
    assert isinstance(inspect.getattr_static(ConnectionManager, "count"), property)


def test_the_runner_keeps_client_stats_per_connection_in_a_gate() -> None:
    assert "self._client_stats = ClientStatsGate(" in inspect.getsource(Runner)
    assert isinstance(ClientStatsGate()._accepted_at, dict)


def test_the_counted_transport_classes_are_classes() -> None:
    for cls in (WebRTCConnection, WebRTCPeer, MediaPacer):
        assert inspect.isclass(cls)


def test_serve_has_the_entry_point_the_launcher_hands_over_to() -> None:
    assert callable(serve.main)
