from reactor_runtime.transport.webrtc.config import (
    IceServer,
    IceTransportPolicy,
    WebRtcConfig,
)


def test_defaults_match_the_old_transport_settings() -> None:
    cfg = WebRtcConfig()

    assert cfg.ice_servers == ()
    assert cfg.transport_policy is IceTransportPolicy.ALL
    assert cfg.ping_timeout == 20.0
    assert cfg.hw_codecs_enabled is False
    assert cfg.bwe_min_kbps == 500
    assert cfg.bwe_max_kbps == 10000
    assert cfg.bwe_target_kbps == 4000
    assert cfg.bwe_target_update_threshold == 0.05
    assert cfg.rtx_max_size_packets == 512
    assert cfg.rtx_max_size_time_ms == 200
    assert cfg.rtp_payload_mtu == 1200
    assert cfg.ice_tcp is False
    assert cfg.upnp is False
    assert cfg.ice_gathering_timeout_ms == 3000


def test_default_codec_preference_order() -> None:
    cfg = WebRtcConfig()

    assert [c["codec"] for c in cfg.supported_video_codecs] == ["VP9", "VP8", "H264", "AV1", "H265"]
    assert cfg.supported_video_codecs[0]["parameters"] == {"profile-id": "0"}
    assert [c["codec"] for c in cfg.supported_audio_codecs] == ["Opus"]
    assert len(cfg.rtp_header_extensions) == 1


def test_config_is_overridable() -> None:
    cfg = WebRtcConfig(
        ice_servers=(IceServer(urls=("stun:stun.example:3478",)),),
        transport_policy=IceTransportPolicy.RELAY,
        hw_codecs_enabled=True,
        bwe_target_kbps=2000,
    )

    assert cfg.transport_policy is IceTransportPolicy.RELAY
    assert cfg.hw_codecs_enabled is True
    assert cfg.bwe_target_kbps == 2000
    assert cfg.ice_servers[0].urls == ("stun:stun.example:3478",)
