from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.webrtc.version import protocol_for_transport


def test_known_transport_version_maps_to_its_codec() -> None:
    assert protocol_for_transport("1.0") is ProtocolVersion.V0


def test_absent_transport_version_falls_back_to_v0() -> None:
    assert protocol_for_transport(None) is ProtocolVersion.V0


def test_unknown_transport_version_falls_back_to_v0() -> None:
    assert protocol_for_transport("9.9") is ProtocolVersion.V0
