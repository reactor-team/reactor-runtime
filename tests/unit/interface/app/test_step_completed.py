from typing import Any

import pytest

from reactor_runtime import StepCompleted


def test_a_bare_report_has_no_media_files_or_error() -> None:
    step = StepCompleted()
    assert step.output is None
    assert step.files == {}
    assert step.error is None
    assert step.elapsed is None


def test_the_files_are_copied_so_the_caller_cannot_change_them() -> None:
    files = {"last_frame.png": b"png"}
    step = StepCompleted(files=files)
    files["late.txt"] = b"x"
    assert step.files == {"last_frame.png": b"png"}


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "is not a file name"),
        (".", "is not a file name"),
        ("..", "is not a file name"),
        ("frames/last.png", "path separator"),
        ("frames\\last.png", "path separator"),
        ("result.json", "reserved"),
        ("output.mp4", "reserved"),
    ],
)
def test_a_file_name_that_is_not_one_plain_name_is_rejected(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StepCompleted(files={name: b"data"})


def test_file_contents_must_be_bytes() -> None:
    files: dict[str, Any] = {"notes.txt": "text"}
    with pytest.raises(TypeError, match="must be bytes"):
        StepCompleted(files=files)


def test_bytes_data_is_kept_as_it_is() -> None:
    assert StepCompleted(data=b"\x93\x01\x02").data == b"\x93\x01\x02"


def test_a_dict_of_data_reads_back_as_compact_json() -> None:
    step = StepCompleted(data={"actions": [0.5, -1], "done": False})
    assert step.data == b'{"actions":[0.5,-1],"done":false}'


def test_data_that_json_cannot_encode_is_rejected() -> None:
    with pytest.raises(ValueError, match="not JSON-serialisable"):
        StepCompleted(data={"when": object()})


def test_data_must_be_bytes_or_a_dict() -> None:
    data: Any = "text"
    with pytest.raises(TypeError, match="must be bytes or a dict"):
        StepCompleted(data=data)
