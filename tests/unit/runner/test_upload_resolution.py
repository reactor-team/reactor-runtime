import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from reactor_runtime.core import TypeSpec, UploadedFile
from reactor_runtime.runner.upload_resolution import declares_upload, resolve_uploads
from reactor_runtime.upload_store import UnknownUploadError

FILES = {
    "up_1": UploadedFile(name="a.png", mime_type="image/png", data=b"a"),
    "up_2": UploadedFile(name="b.png", mime_type="image/png", data=b"b"),
    "up_3": UploadedFile(name="c.png", mime_type="image/png", data=b"c"),
}


async def fetch(upload_id: str) -> UploadedFile:
    try:
        return FILES[upload_id]
    except KeyError:
        raise UnknownUploadError(upload_id) from None


def ref(upload_id: str) -> dict[str, str]:
    return {"upload_id": upload_id}


@dataclass
class Holder:
    file: UploadedFile


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (UploadedFile, True),
        (UploadedFile | None, True),
        (list[UploadedFile], True),
        (list[UploadedFile] | None, True),
        (list[UploadedFile | None], True),
        (list[list[UploadedFile]], True),
        (str, False),
        (list[str], False),
        (dict[str, str], False),
        (dict[str, UploadedFile], False),
        (Holder, False),
        (Any, False),
    ],
)
def test_declares_upload_follows_optional_and_list_wrappers_only(
    annotation: Any, expected: bool
) -> None:
    assert declares_upload(TypeSpec.of(annotation)) is expected


async def test_a_single_reference_resolves_to_its_file() -> None:
    assert await resolve_uploads(TypeSpec.of(UploadedFile), ref("up_1"), fetch) == FILES["up_1"]


async def test_a_list_resolves_every_entry_in_the_client_order() -> None:
    value = [ref("up_3"), ref("up_1"), ref("up_2")]

    resolved = await resolve_uploads(TypeSpec.of(list[UploadedFile]), value, fetch)

    assert resolved == [FILES["up_3"], FILES["up_1"], FILES["up_2"]]


async def test_the_order_holds_when_later_entries_arrive_first() -> None:
    async def staggered(upload_id: str) -> UploadedFile:
        # The first entry is the slowest, so a walk that collected in arrival
        # order would put it last.
        await asyncio.sleep(0.05 if upload_id == "up_1" else 0)
        return FILES[upload_id]

    value = [ref("up_1"), ref("up_2"), ref("up_3")]
    resolved = await resolve_uploads(TypeSpec.of(list[UploadedFile]), value, staggered)

    assert resolved == [FILES["up_1"], FILES["up_2"], FILES["up_3"]]


async def test_an_optional_list_passes_none_through() -> None:
    spec = TypeSpec.of(list[UploadedFile] | None)

    assert await resolve_uploads(spec, None, fetch) is None
    assert await resolve_uploads(spec, [ref("up_1")], fetch) == [FILES["up_1"]]


async def test_a_file_already_in_place_passes_through_untouched() -> None:
    spec = TypeSpec.of(list[UploadedFile])

    resolved = await resolve_uploads(spec, [FILES["up_1"], ref("up_2")], fetch)

    assert resolved == [FILES["up_1"], FILES["up_2"]]


async def test_an_entry_that_is_not_a_reference_is_left_for_validation() -> None:
    # The resolver never judges shape: an element the contract will reject
    # stays exactly as sent so the rejection names what the client sent.
    spec = TypeSpec.of(list[UploadedFile])
    value = [ref("up_1"), "not-a-reference", {"upload_id": 7}, {"uploadId": "up_2"}]

    resolved = await resolve_uploads(spec, value, fetch)

    assert resolved == [FILES["up_1"], "not-a-reference", {"upload_id": 7}, {"uploadId": "up_2"}]


async def test_a_value_the_spec_does_not_type_as_an_upload_is_untouched() -> None:
    value = {"upload_id": "up_1", "note": "a model's own mapping"}

    assert await resolve_uploads(TypeSpec.of(dict[str, str]), value, fetch) is value
    assert await resolve_uploads(TypeSpec.of(list[str]), ["up_1"], fetch) == ["up_1"]


async def test_a_list_that_is_not_a_list_is_untouched() -> None:
    # A scalar where an array was declared is the contract's to refuse.
    assert await resolve_uploads(TypeSpec.of(list[UploadedFile]), "up_1", fetch) == "up_1"


async def test_one_unknown_entry_fails_the_whole_list() -> None:
    value = [ref("up_1"), ref("missing"), ref("up_2")]

    with pytest.raises(UnknownUploadError):
        await resolve_uploads(TypeSpec.of(list[UploadedFile]), value, fetch)
