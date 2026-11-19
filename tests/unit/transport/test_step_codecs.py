"""The step frame codec table that negotiation and the frame codecs share."""

from __future__ import annotations

import logging

import pytest
import reactor_webrtc as rw

from reactor_runtime.transport import step_codecs
from reactor_runtime.transport.step_codecs import (
    BY_ENCODING,
    BY_SDP_NAME,
    STEP_FRAME_CODECS,
    use_nvdec,
    video_codecs,
)
from reactor_wire.v1 import step_pb2


def test_each_codec_is_named_once_in_every_vocabulary() -> None:
    codecs = [c.codec for c in STEP_FRAME_CODECS]

    assert len(BY_SDP_NAME) == len(BY_ENCODING) == len(STEP_FRAME_CODECS)
    assert all(codecs.count(codec) == 1 for codec in codecs)


def test_every_encoded_wire_encoding_has_a_codec() -> None:
    encoded = set(step_pb2.FrameEncoding.values()) - {
        step_pb2.FRAME_ENCODING_UNSPECIFIED,
        step_pb2.FRAME_ENCODING_RAW,
    }

    assert set(BY_ENCODING) == encoded


def test_each_sdp_name_is_the_bindings_codec_name() -> None:
    # The SDP name is the binding's variant name upper-cased, the same mapping
    # the peer uses for a configured codec preference.
    variants = {
        name.upper(): value
        for name in dir(rw.VideoCodec)
        if isinstance(value := getattr(rw.VideoCodec, name), rw.VideoCodec)
    }

    for codec in STEP_FRAME_CODECS:
        assert variants[codec.sdp_name] == codec.codec


def test_the_video_codec_backends_are_collected_once() -> None:
    assert video_codecs() is video_codecs()


def _fresh_backends(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Reset the process's codecs to unopened, recording the ``nvdec`` each opening asks for."""
    opened: list[bool] = []

    def _codecs(*, nvdec: bool = False) -> object:
        opened.append(nvdec)
        return object()

    monkeypatch.setattr(step_codecs, "_backends", [])
    monkeypatch.setattr(step_codecs, "_nvdec", False)
    monkeypatch.setattr(step_codecs.rw, "VideoCodecs", _codecs)
    return opened


def test_the_codecs_decode_in_software_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _fresh_backends(monkeypatch)
    video_codecs()
    assert opened == [False]


def test_nvdec_set_before_first_use_opens_the_codecs_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _fresh_backends(monkeypatch)
    use_nvdec(True)
    video_codecs()
    assert opened == [True]


def test_nvdec_set_after_first_use_is_ignored_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    opened = _fresh_backends(monkeypatch)
    video_codecs()
    with caplog.at_level(logging.WARNING, logger=step_codecs.__name__):
        use_nvdec(True)
    video_codecs()
    assert opened == [False]
    assert "already open" in caplog.text


def test_repeating_the_setting_after_first_use_says_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fresh_backends(monkeypatch)
    video_codecs()
    with caplog.at_level(logging.WARNING, logger=step_codecs.__name__):
        use_nvdec(False)
    assert caplog.text == ""
