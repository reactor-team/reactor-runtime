import numpy as np
import pytest

from reactor_runtime import Audio, MediaInput, Output, TrackPayload, Video
from reactor_runtime.core.values import TrackDirection, TrackKind


class GameOutput(Output):
    main_video: Video
    narration: Audio


class GameInput(MediaInput):
    camera: Video


def test_output_tracks_are_resolved_as_outbound() -> None:
    tracks = GameOutput.__tracks__
    assert set(tracks) == {"main_video", "narration"}
    assert tracks["main_video"].kind is TrackKind.VIDEO
    assert tracks["main_video"].direction is TrackDirection.OUT
    assert tracks["narration"].kind is TrackKind.AUDIO


def test_input_tracks_are_resolved_as_inbound() -> None:
    tracks = GameInput.__tracks__
    assert set(tracks) == {"camera"}
    assert tracks["camera"].direction is TrackDirection.IN


def test_audio_default_sample_rate_is_carried_as_rate() -> None:
    assert GameOutput.__tracks__["narration"].rate == 48_000.0


def test_audio_subclass_overrides_sample_rate() -> None:
    class Narration(Audio):
        sample_rate = 16_000

    class Out(Output):
        speech: Narration

    assert Out.__tracks__["speech"].rate == 16_000.0


def test_an_audio_track_buffers_frames_unless_it_declares_seconds() -> None:
    assert GameOutput.__tracks__["narration"].buffer_seconds is None
    assert GameInput.__tracks__["camera"].buffer_seconds is None


def test_an_audio_subclass_carries_its_buffer_seconds() -> None:
    class AgentAudio(Audio):
        sample_rate = 24_000
        buffer_seconds = 120

    class In(MediaInput):
        agent_audio: AgentAudio

    track = In.__tracks__["agent_audio"]
    assert (track.rate, track.buffer_seconds) == (24_000.0, 120.0)


@pytest.mark.parametrize("seconds", [0, -1.5, True, "120", float("nan"), float("inf")])
def test_a_buffer_seconds_that_is_not_a_finite_positive_number_is_refused(seconds: object) -> None:
    class AgentAudio(Audio):
        buffer_seconds = seconds

    with pytest.raises(ValueError, match="buffer_seconds must be a finite positive number"):

        class In(MediaInput):
            agent_audio: AgentAudio


@pytest.mark.parametrize("rate", [0, float("nan"), float("inf")])
def test_a_buffer_seconds_needs_a_finite_positive_sample_rate(rate: float) -> None:
    class AgentAudio(Audio):
        sample_rate = rate
        buffer_seconds = 120

    with pytest.raises(ValueError, match="buffer_seconds needs a positive sample_rate"):

        class In(MediaInput):
            agent_audio: AgentAudio


def test_a_class_with_no_track_fields_has_no_tracks() -> None:
    class Empty(Output):
        pass

    assert Empty.__tracks__ == {}


def test_a_bare_payload_binds_the_array_and_no_metadata() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    output = GameOutput(main_video=frame, narration=np.zeros((1, 4), dtype=np.int16))
    assert output.main_video is frame
    assert output.__metadata__ == {}


def test_a_wrapped_payload_keeps_the_array_on_the_track() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    output = GameOutput(
        main_video=TrackPayload(frame, metadata={"seed": 1}),
        narration=np.zeros((1, 4), dtype=np.int16),
    )
    assert output.main_video is frame
    assert output.__metadata__ == {"main_video": {"seed": 1}}


def test_a_payload_without_metadata_records_none() -> None:
    output = GameOutput(
        main_video=TrackPayload(np.zeros((2, 2, 3), dtype=np.uint8)),
        narration=np.zeros((1, 4), dtype=np.int16),
    )
    assert output.__metadata__ == {}


def test_payloads_must_still_cover_every_track() -> None:
    with pytest.raises(TypeError, match="expects payloads"):
        GameOutput(main_video=TrackPayload(np.zeros((2, 2, 3), dtype=np.uint8)))
