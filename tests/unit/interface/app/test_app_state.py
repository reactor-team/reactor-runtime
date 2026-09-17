"""The typed state on :class:`ReactorApp`: generated setters, session scope, schema pin."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime import (
    Audio,
    CommandError,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    ReactorApp,
    UploadedFile,
    Video,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.interface.pipeline import Idle, ReactorPipeline

_GOLDEN = Path(__file__).parent / "golden" / "brightness_openapi.json"


class Frame(Output):
    main_video: Video


class State(InputState):
    speed: float = InputField(default=1.0, ge=0.0, le=10.0)
    seed: int = InputField(default=0)
    _started: bool = False


class App(ReactorApp):
    state: State

    async def run(self) -> None: ...


class Bare(ReactorApp):
    async def run(self) -> None: ...


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(App)


def _ready(app: ReactorApp) -> None:
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda chunk: None
    )


# -- contract wiring ----------------------------------------------------------


def test_public_state_fields_become_commands_on_a_reactor_app() -> None:
    commands = ModelContract.of(App).commands
    assert "set_speed" in commands
    assert "set_seed" in commands
    assert "set__started" not in commands


def test_generated_setter_carries_the_field_constraints() -> None:
    info = ModelContract.of(App).commands["set_speed"].command.__command_fields__["speed"].info
    assert info.ge == 0.0
    assert info.le == 10.0


def test_a_hand_written_event_wins_over_the_generated_setter() -> None:
    class Custom(ReactorApp):
        state: State

        @event(name="set_speed", description="hand written")
        def set_speed(self, speed: float = InputField(default=2.0)) -> None:
            self.state.speed = speed

        async def run(self) -> None: ...

    spec = ModelContract.of(Custom).commands["set_speed"]
    assert spec.description == "hand written"
    assert "set_seed" in ModelContract.of(Custom).commands


def test_an_app_without_state_declares_no_setters_and_constructs() -> None:
    assert Bare.__app_state__ is None
    assert not any(name.startswith("set_") for name in ModelContract.of(Bare).commands)
    assert Bare().state is None


# -- session scope ------------------------------------------------------------


async def test_state_is_built_at_session_start_and_cleared_at_session_end() -> None:
    app = App()
    _ready(app)
    assert app.state is None
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert isinstance(app.state, State)
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert app.state is None


async def test_state_survives_a_client_leaving_and_rejoining_mid_session() -> None:
    app = App()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    app.state.speed = 7.0
    await app._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1002), 1))
    assert app.state.speed == 7.0


async def test_hooks_see_the_state_at_both_ends_of_the_session() -> None:
    seen: list[Any] = []

    class Hooked(ReactorApp):
        state: State

        @session_started
        def start(self) -> None:
            self.state._started = True

        @session_ended
        def end(self) -> None:
            seen.append(self.state._started)

        async def run(self) -> None: ...

    app = Hooked()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert app.state._started is True
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert seen == [True]
    assert app.state is None


async def test_a_bare_app_keeps_no_state_across_a_session() -> None:
    app = Bare()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert app.state is None


# -- command dispatch ---------------------------------------------------------


async def test_set_field_command_updates_the_live_state() -> None:
    app = App()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    command = ModelContract.of(App).validate("set_speed", {"speed": 3.5})
    await app._dispatch_command(CommandEnvelope(command, ConnId(1001), None))
    assert app.state.speed == 3.5


async def test_a_setter_with_no_live_state_is_a_no_op() -> None:
    app = App()
    _ready(app)
    command = ModelContract.of(App).validate("set_seed", {"seed": 5})
    await app._dispatch_command(CommandEnvelope(command, ConnId(1001), None))
    assert app.state is None


# -- schema pin ---------------------------------------------------------------
#
# The golden file is the OpenAPI document this pipeline rendered on 3.3.2,
# where the state feature lived on ReactorPipeline. The class is kept here,
# verbatim, so the pin keeps its meaning: the same declarations render the
# same bytes from ReactorPipeline and from ReactorApp.


class BrightnessOutput(Output):
    """The generated video and the matching audio tone."""

    main_video: Video
    main_audio: Audio


class BrightnessSnapshot(ModelMessage):
    """The live generation parameters, returned when a client asks for them."""

    brightness: float = MessageField(description="Active brightness multiplier.")
    paused: bool = MessageField(description="Whether frame generation is paused.")
    resolution: str = MessageField(description="Active output resolution.")
    text: str = MessageField(description="Caption currently drawn over each frame.")


class BrightnessSet(ModelMessage):
    """Confirmation that the brightness was applied, echoing the value in effect."""

    brightness: float = MessageField(description="Brightness multiplier now in effect.")


class ImageSet(ModelMessage):
    """Acknowledgement that an uploaded reference image was accepted."""

    filename: str = MessageField(description="Name of the image now in effect.")


class BrightnessState(InputState):
    """The generation parameters a client can change live.

    Each public field becomes a ``set_<field>`` command automatically, so a
    client drives the look and pitch without the model declaring any handler.
    """

    brightness: float = InputField(
        default=1.0, ge=0.0, le=2.0, description="Brightness multiplier (0=black, 1=half, 2=white)."
    )
    paused: bool = InputField(default=False, description="Pause frame generation.")
    resolution: str = InputField(
        default="480p",
        choices=["480p", "720p", "1080p", "2160p"],
        description="Output resolution (2160p is 4K UHD).",
    )
    text: str = InputField(
        default="",
        max_length=200,
        description="Caption drawn over every frame; empty draws nothing.",
        moderate=True,
    )


class Brightness(ReactorPipeline):
    """Generate an animated gradient and tone whose look and pitch track the state."""

    state: BrightnessState
    fps = 30
    _reference: UploadedFile | None = None

    def load(self, config_path: Path | None) -> None: ...

    @event(name="get_state", description="Return the current generation parameters.")
    def get_state(self) -> BrightnessSnapshot:
        return BrightnessSnapshot(
            brightness=self.state.brightness,
            paused=self.state.paused,
            resolution=self.state.resolution,
            text=self.state.text,
        )

    @event(name="set_brightness", description="Set the brightness and confirm the value in effect.")
    def set_brightness(
        self,
        brightness: float = InputField(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Brightness multiplier (0=black, 1=half, 2=white).",
        ),
    ) -> BrightnessSet:
        self.state.brightness = brightness
        return BrightnessSet(brightness=brightness)

    @event(name="set_image", description="Set the reference image and acknowledge it.")
    def set_image(self, image: UploadedFile = InputField(moderate=True)) -> ImageSet:
        if not image.mime_type.startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} is not an image.")
        self._reference = image
        return ImageSet(filename=image.name)

    def inference(self) -> Iterator[Any]:
        while True:
            yield Idle


def _openapi(model_cls: type) -> str:
    """Render the document the way the golden file was written: bytes, not a dict."""
    doc = (
        ModelContract.of(model_cls).render_schema(version="v1.0.0", name="brightness").to_openapi()
    )
    return json.dumps(doc, indent=2) + "\n"


def test_the_brightness_pipeline_renders_the_pinned_document(
    register_model: Callable[[type], None],
) -> None:
    # Byte identity, not dict equality, so an ordering change in the render is
    # caught too.
    register_model(Brightness)
    assert _openapi(Brightness) == _GOLDEN.read_text()


def test_the_same_model_on_reactor_app_renders_the_same_document(
    register_model: Callable[[type], None],
) -> None:
    register_model(Brightness)

    class BrightnessApp(ReactorApp):
        """Generate an animated gradient and tone whose look and pitch track the state."""

        state: BrightnessState

        get_state = Brightness.get_state
        set_brightness = Brightness.set_brightness
        set_image = Brightness.set_image

        async def run(self) -> None: ...

    assert BrightnessApp.__doc__ == Brightness.__doc__
    assert _openapi(BrightnessApp) == _GOLDEN.read_text()


def test_frame_fixture_is_an_output() -> None:
    assert isinstance(Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8)), Output)


# -- the step lock and the live gate ------------------------------------------


async def test_a_command_handler_waits_for_the_step_lock() -> None:
    app = App()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    command = ModelContract.of(App).validate("set_speed", {"speed": 9.0})
    async with app._step_lock:
        task = asyncio.create_task(
            app._dispatch_command(CommandEnvelope(command, ConnId(1001), None))
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Parked on the lock a run() loop would hold around one unit of work.
        assert app.state.speed == 1.0
    await task
    assert app.state.speed == 9.0


async def test_a_lifecycle_hook_waits_for_the_step_lock() -> None:
    entered = asyncio.Event()

    class Hooked(ReactorApp):
        @session_started
        async def start(self) -> None:
            entered.set()

        async def run(self) -> None: ...

    app = Hooked()
    _ready(app)
    async with app._step_lock:
        task = asyncio.create_task(app._dispatch_reactor_event(SessionStarted("s")))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not entered.is_set()
    await task
    assert entered.is_set()


async def test_a_session_end_waits_for_the_step_lock_before_clearing_the_state() -> None:
    # No @session_ended hook: the clear must still queue behind the lock a
    # run() loop holds around a step, or a process_input() suspended on an
    # await resumes to a None state.
    app = App()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    ended = SessionEnded("s", EndReason.STOPPED)
    async with app._step_lock:
        task = asyncio.create_task(app._dispatch_reactor_event(ended))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not app._live.is_set()  # the gate drops at once, so the loop stops next
        assert isinstance(app.state, State)  # but the state outlives the step
    await task
    assert app.state is None


async def test_a_session_start_waits_for_the_step_lock_before_building_the_state() -> None:
    # No @session_started hook: the build must still queue behind the lock.
    app = App()
    _ready(app)
    async with app._step_lock:
        task = asyncio.create_task(app._dispatch_reactor_event(SessionStarted("s")))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert app.state is None
    await task
    assert isinstance(app.state, State)


async def test_the_live_gate_needs_a_started_session_and_a_client() -> None:
    app = Bare()
    _ready(app)
    assert not app._live.is_set()
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    assert not app._live.is_set()  # a client, but no session yet
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert app._live.is_set()
    await app._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    assert not app._live.is_set()
    await app._dispatch_reactor_event(ClientConnected(ConnId(1002), 1))
    assert app._live.is_set()
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert not app._live.is_set()


async def test_a_session_end_drops_the_gate_before_its_hook_runs() -> None:
    seen: list[bool] = []

    class Hooked(ReactorApp):
        @session_ended
        def end(self) -> None:
            seen.append(self._live.is_set())

        async def run(self) -> None: ...

    app = Hooked()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    assert app._live.is_set()
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert seen == [False]


# -- an app that declares no state owns the attribute -------------------------


async def test_an_app_without_state_keeps_its_own_state_attribute_across_a_session() -> None:
    # A 3.3.2 model that uses `self.state` for its own purposes declares no
    # `state:` annotation, so the base never builds, clears, or reads it.
    own = object()

    class Legacy(ReactorApp):
        def load(self, config_path: Any) -> None:
            self.state = own

        async def run(self) -> None: ...

    app = Legacy()
    app.load(None)
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert app.state is own
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert app.state is own


def test_an_unresolvable_state_annotation_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):

        class Dangling(ReactorApp):
            state: "NoSuchStateClass"  # type: ignore[ty:unresolved-reference]  # noqa: F821, UP037  # the reference under test

            async def run(self) -> None: ...

    assert Dangling.__app_state__ is None
    assert not any(name.startswith("set_") for name in ModelContract.of(Dangling).commands)
    messages = [record.getMessage() for record in caplog.records]
    assert any("state annotation could not be resolved" in message for message in messages)
