"""External file reports retain source references without exposing them publicly."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from reactor_runtime import ExternalFile, StepCompleted


def source(**changes: Any) -> ExternalFile:
    fields: dict[str, Any] = {
        "name": "output.mp4",
        "content_type": "video/mp4",
        "provider": "example",
        "job_id": "job-1",
        "url": "https://media.example/video.mp4?signature=private",
    }
    return ExternalFile(**(fields | changes))


def test_source_details_are_separate_from_public_file_facts() -> None:
    expiry = datetime(2026, 10, 28, tzinfo=timezone(timedelta(hours=2)))
    file = source(expires_at=expiry)
    assert file.public_entry() == {
        "name": "output.mp4",
        "content_type": "video/mp4",
        "expires_at": "2026-10-27T22:00:00+00:00",
    }
    assert file.source_entry()["job_id"] == "job-1"
    assert file.source_entry()["url"] == file.url
    assert "private" not in repr(file)
    assert "job-1" not in repr(file)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "../video.mp4"},
        {"name": "_external_results.json"},
        {"name": "result.json"},
        {"name": "video\n.mp4"},
        {"content_type": "video/mp4\r\nX-Test: injected"},
        {"provider": ""},
        {"job_id": ""},
        {"url": "http://media.example/video.mp4"},
        {"url": "https://user:password@media.example/video.mp4"},
        {"url": "https://media.example/video.mp4#fragment"},
        {"expires_at": datetime(2026, 10, 28)},
    ],
)
def test_invalid_source_fields_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="external file"):
        source(**changes)


def test_unknown_expiry_is_preserved() -> None:
    assert source().public_entry()["expires_at"] is None
    assert source(expires_at=datetime(2026, 10, 28, tzinfo=UTC)).expires_at is not None


def test_external_files_can_accompany_local_files() -> None:
    file = source()
    step = StepCompleted(files={"usage.json": b"{}"}, external_files=(file,))
    assert step.external_files == (file,)
    assert step.files == {"usage.json": b"{}"}


def test_duplicate_and_reserved_file_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        StepCompleted(external_files=(source(), source()))
    with pytest.raises(ValueError, match="unique"):
        StepCompleted(files={"extra.mp4": b""}, external_files=(source(name="extra.mp4"),))
    with pytest.raises(ValueError, match="reserved"):
        StepCompleted(files={"_external_results.json": b"{}"})
    with pytest.raises(ValueError, match="at most 256"):
        StepCompleted(external_files=tuple(source(name=f"{i}.mp4") for i in range(257)))


def test_total_descriptor_size_is_bounded() -> None:
    with pytest.raises(ValueError, match="within 1 MB"):
        StepCompleted(
            external_files=tuple(
                source(name=f"{i}.mp4", url="https://media.example/" + "x" * 16000)
                for i in range(100)
            )
        )
