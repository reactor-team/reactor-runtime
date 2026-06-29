"""The process-global interface registries and their schema consequences.

These exercise the registration contract directly: a declaration registers, an
empty base does not, the schema reads the registry union (so a message a model
only ever broadcasts is still published), and the per-test clear isolates one
test's declarations from the next.
"""

import pytest

from reactor_runtime import (
    EVENT_REGISTRY,
    INPUT_REGISTRY,
    MESSAGE_REGISTRY,
    OUTPUT_REGISTRY,
    Input,
    ModelMessage,
    Output,
    ReactorModel,
    Video,
    all_input_tracks,
    all_output_tracks,
    event,
)
from reactor_runtime.interface.model import ModelContract


def test_declaring_a_subclass_registers_it() -> None:
    class Out(Output):
        main: Video

    class In(Input):
        camera: Video

    class Note(ModelMessage):
        text: str

    assert OUTPUT_REGISTRY["Out"] is Out
    assert INPUT_REGISTRY["In"] is In
    assert MESSAGE_REGISTRY["note"] is Note
    assert set(all_output_tracks()) == {"main"}
    assert set(all_input_tracks()) == {"camera"}


def test_an_event_handler_registers_its_command() -> None:
    class Model(ReactorModel):
        @event(name="go")
        async def go(self) -> None: ...

    assert "go" in EVENT_REGISTRY


def test_a_field_less_base_is_not_registered() -> None:
    class AbstractOut(Output):
        pass

    class AbstractMessage(ModelMessage):
        pass

    assert OUTPUT_REGISTRY == {}
    assert MESSAGE_REGISTRY == {}


def test_a_broadcast_only_message_is_published_in_the_schema() -> None:
    class Out(Output):
        main: Video

    class Alert(ModelMessage):
        """Something the model wants the client to know."""

        level: str

    class Model(ReactorModel):
        output: Out

        @event(name="go")
        async def go(self) -> None:
            # Never returns Alert — the model would self.send() it instead.
            ...

    schema = ModelContract.of(Model).render_schema().to_openapi()
    assert "alert" in schema["webhooks"]
    assert schema["webhooks"]["alert"]["post"]["summary"] == (
        "Something the model wants the client to know."
    )


def test_the_track_union_spans_multiple_output_classes() -> None:
    class First(Output):
        video_a: Video

    class Second(Output):
        video_b: Video

    assert set(all_output_tracks()) == {"video_a", "video_b"}


def test_a_track_name_shared_across_an_inheritance_chain_dedups() -> None:
    class Base(Output):
        main: Video

    class Derived(Base):
        extra: Video

    # Derived re-resolves `main` (inherited) plus `extra`; the union collapses
    # the shared name rather than treating it as a conflict.
    assert set(all_output_tracks()) == {"main", "extra"}


def test_a_track_declared_in_both_directions_is_rejected_on_read() -> None:
    class Out(Output):
        shared: Video

    class In(Input):
        shared: Video

    class Model(ReactorModel):
        output: Out
        input: In

    with pytest.raises(ValueError, match="both input and output"):
        _ = ModelContract.of(Model).tracks


def test_isolation_first_model_sees_only_its_own_surface() -> None:
    class OnlyFirst(ModelMessage):
        a: str

    class Model(ReactorModel):
        @event(name="only_first")
        async def first(self) -> None: ...

    assert set(MESSAGE_REGISTRY) == {"only_first"}
    assert set(EVENT_REGISTRY) == {"only_first"}


def test_isolation_second_model_does_not_see_the_first() -> None:
    class OnlySecond(ModelMessage):
        b: str

    class Model(ReactorModel):
        @event(name="only_second")
        async def second(self) -> None: ...

    assert set(MESSAGE_REGISTRY) == {"only_second"}
    assert set(EVENT_REGISTRY) == {"only_second"}
