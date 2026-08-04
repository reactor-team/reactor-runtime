import numpy as np
import pytest

from reactor_runtime import Audio, Input, Output, TrackPayload, Video
from reactor_runtime.core.values import TrackDirection, TrackKind


class GameOutput(Output):
    main_video: Video
    narration: Audio


class GameInput(Input):
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
