"""Define OpenDreamer's public Reactor schema: its track, its state, its message.

Everything a client can see, and nothing else. The configuration and
conditioning types live with the model, because they describe the checkpoint
rather than the wire.

The controls are the state. Every public field becomes a ``set_<field>`` command
the runtime generates, validates, and documents, and they are level-triggered: a
value a client sets stays set and is read on every generated frame, which is what
makes holding a key or a look direction expressible without an event for the
press and another for the release.

What the world was started from is private, because starting one is not setting
a value: an upload has to be carried as a file, and a demo has to be named from
a fixed list. Those arrive through the ``load_scene`` and ``set_image``
commands, which reset the session and hand the model its conditioning.
"""

from __future__ import annotations

from opendreamer_model import KEYS, MOUSE_BUTTONS

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

_CAMERA_DELTA_MIN = -200.0
_CAMERA_DELTA_MAX = 200.0


class OpenDreamerOutput(Output):
    """Stream the next generated Minecraft frame on `main_video`."""

    main_video: Video


class OpenDreamerState(InputState):
    """The controls of one playable OpenDreamer world.

    Each public field is a `set_<field>` command. Values persist until changed,
    and every generated frame is built from whatever they hold at the time. The
    two private fields hold the scene the world starts from; ``set_image`` and
    ``load_scene`` are what write them.
    """

    paused: bool = InputField(
        default=False,
        description=(
            "Whether frame generation is paused. Pausing preserves the current world and "
            "stops producing `main_video` frames; resuming continues from where it stopped."
        ),
    )
    seed: int = InputField(
        default=0,
        ge=0,
        le=2_147_483_647,
        description=(
            "Seed the world is generated from. Changing it restarts the world, so it is also "
            "how to get a different outcome from the same scene."
        ),
    )
    keys: str = InputField(
        default="",
        max_length=200,
        description=(
            "Minecraft keyboard keys held down, separated by spaces — for example `w space`. "
            f"Accepted keys are: {', '.join(KEYS)}. Anything else is ignored. Every held key "
            "applies to each generated frame until this value changes, so releasing a key "
            "means sending the set without it."
        ),
    )
    buttons: str = InputField(
        default="",
        max_length=40,
        description=(
            "Minecraft mouse buttons held down, separated by spaces — for example `left`. "
            f"Accepted buttons are: {', '.join(MOUSE_BUTTONS)}. Anything else is ignored. "
            "Each held button applies to every generated frame until this value changes."
        ),
    )
    look_x: float = InputField(
        default=0.0,
        ge=_CAMERA_DELTA_MIN,
        le=_CAMERA_DELTA_MAX,
        description=(
            "Horizontal camera movement applied to each generated frame, in [-200, 200]. It "
            "keeps turning the view while set, so send 0 to stop."
        ),
    )
    look_y: float = InputField(
        default=0.0,
        ge=_CAMERA_DELTA_MIN,
        le=_CAMERA_DELTA_MAX,
        description=(
            "Vertical camera movement applied to each generated frame, in [-200, 200]. It "
            "keeps turning the view while set, so send 0 to stop."
        ),
    )
    hotbar: int = InputField(
        default=0,
        ge=-1,
        le=1,
        description=(
            "Hotbar scroll applied to each generated frame: -1 scrolls down, 1 scrolls up, 0 "
            "leaves the selection alone. It keeps scrolling while set, so send 0 to stop."
        ),
    )
    # What the world was started from, for the snapshot clients receive. Empty
    # means no world has been started and nothing is being generated. The image
    # itself is not kept here: `set_image` hands the bytes straight to the model.
    _scene: str = ""


class StateUpdate(ModelMessage):
    """What the world is currently set to.

    Sent whenever any of it changes, and once more to a viewer as it connects,
    so a client never has to track what it believes the world is set to —
    including the values it did not set itself, which matters when several
    viewers share one world.
    """

    paused: bool = MessageField(description="Whether frame generation is paused.")
    scene: str = MessageField(
        description=(
            "What the world was started from: a demo name, `uploaded`, or empty when no world "
            "has been started and nothing is being generated."
        )
    )
    seed: int = MessageField(description="Seed the current world is generated from.")
    keys: str = MessageField(description="Keyboard keys held down for this frame.")
    buttons: str = MessageField(description="Mouse buttons held down for this frame.")
    look_x: float = MessageField(description="Horizontal camera movement applied to this frame.")
    look_y: float = MessageField(description="Vertical camera movement applied to this frame.")
    hotbar: int = MessageField(description="Hotbar scroll applied to this frame.")

    @classmethod
    def from_state(cls, state: OpenDreamerState) -> StateUpdate:
        """Build a snapshot of the controls the world is running under."""
        return cls(
            paused=state.paused,
            scene=state._scene,
            seed=state.seed,
            keys=state.keys,
            buttons=state.buttons,
            look_x=state.look_x,
            look_y=state.look_y,
            hotbar=state.hotbar,
        )
