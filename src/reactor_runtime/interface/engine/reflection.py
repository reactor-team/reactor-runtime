"""Reading an engine's declarations into a serving surface.

The engine declares classes; this module turns them into the things the runtime
already knows how to serve. A :class:`UserInput` becomes a command whose payload
is the input's fields, a :class:`MediaInput` becomes an input track, and the
:class:`Init` becomes the command that opens a rollout. Nothing here talks to a
client — it produces the same :class:`Command` classes and track holders a
hand-written model produces, so the contract, schema, and typed SDKs downstream
need no notion that an engine was involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reactor_runtime.core.fields import NO_DEFAULT, FieldInfo
from reactor_runtime.core.model import Command
from reactor_runtime.core.naming import pascal_to_snake
from reactor_runtime.engine_contract.fields import NO_DEFAULT as CONTRACT_NO_DEFAULT
from reactor_runtime.engine_contract.fields import FieldSpec
from reactor_runtime.engine_contract.inputs import (
    USER_INPUT_REGISTRY,
    Init,
    MediaInput,
    UserInput,
)
from reactor_runtime.interface.engine.store import MediaSpec
from reactor_runtime.interface.events.decorators import make_command
from reactor_runtime.interface.tracks.descriptors import Audio, Video
from reactor_runtime.interface.tracks.input import Input
from reactor_runtime.interface.tracks.output import Output

INIT_COMMAND = "init"
"""The wire name every engine's :class:`Init` is served under, whatever it is called."""

VIDEO_TRACK = "main_video"
"""The outbound track an engine's frames are served on."""


@dataclass(frozen=True)
class EngineInputs:
    """An engine's declared input surface, resolved into serving terms.

    Attributes:
        events: Wire name to the input class it carries, for every plain event.
        media: Track name to how frames on it are materialized.
        init: The engine's initialization class, or ``None`` when it declares
            none and a rollout opens with no arguments.
    """

    events: dict[str, type[UserInput]]
    media: dict[str, MediaSpec]
    init: type[Init] | None


def discover_inputs(engine_cls: type) -> EngineInputs:
    """Resolve the contract classes an engine declares.

    An engine that names them itself — ``declared_inputs = (Move, Look, Camera)``
    — is taken at its word. Otherwise the declarations are the ones that share
    the engine's package, which is what an engine distributed as a package looks
    like from the outside.

    Args:
        engine_cls: The engine pipeline class.

    Returns:
        The declared events, media tracks, and initialization class.

    Raises:
        TypeError: If the engine declares more than one :class:`Init`, or two
            inputs resolve to the same wire name.
    """
    events: dict[str, type[UserInput]] = {}
    media: dict[str, MediaSpec] = {}
    init: type[Init] | None = None

    for input_cls in _declared_by(engine_cls):
        name = wire_name(input_cls)
        if issubclass(input_cls, MediaInput):
            _claim(media, name, input_cls, [events])
            media[name] = MediaSpec(
                track=name, input_cls=input_cls, chunk_size=input_cls.chunk_size
            )
        elif issubclass(input_cls, Init):
            if init is not None and init is not input_cls:
                raise TypeError(
                    f"{engine_cls.__name__} declares two Init classes "
                    f"({init.__name__} and {input_cls.__name__}); an engine has one."
                )
            init = input_cls
        else:
            _claim(events, name, input_cls, [media])
            events[name] = input_cls

    return EngineInputs(events=events, media=media, init=init)


def wire_name(input_cls: type[UserInput]) -> str:
    """Return the name a client addresses an input by.

    The class name in ``snake_case``, except for the initialization class: an
    engine names that after itself, but a client always sends ``init``.
    """
    if issubclass(input_cls, Init):
        return INIT_COMMAND
    return pascal_to_snake(input_cls.__name__)


def command_for(name: str, input_cls: type[UserInput]) -> type[Command]:
    """Build the command whose payload is *input_cls*'s declared fields.

    Args:
        name: The wire name the command is served under.
        input_cls: The declared input the payload mirrors.

    Returns:
        A command class the ordinary contract machinery validates against.
    """
    fields: list[tuple[str, Any] | tuple[str, Any, Any]] = [
        (field, input_cls.__input_annotations__[field], field_info(spec))
        for field, spec in input_cls.__input_fields__.items()
    ]
    return make_command(name, fields)


def field_info(spec: FieldSpec) -> FieldInfo:
    """Translate a contract field's constraints into the runtime's own vocabulary."""
    return FieldInfo(
        default=NO_DEFAULT if spec.default is CONTRACT_NO_DEFAULT else spec.default,
        description=spec.description,
        ge=spec.ge,
        le=spec.le,
        min_length=spec.min_length,
        max_length=spec.max_length,
        choices=spec.choices,
    )


def track_holder(name: str, media: dict[str, MediaSpec]) -> type[Input] | None:
    """Build the :class:`Input` holder declaring an engine's media tracks.

    Declaring the class registers the tracks, so a transport negotiates them and
    the schema publishes them exactly as it does for a hand-written model.

    Args:
        name: Class name for the holder, for readable diagnostics.
        media: The media specs to declare, by track name.

    Returns:
        The holder class, or ``None`` when the engine declares no media.
    """
    if not media:
        return None
    annotations = {
        track: (Video if spec.input_cls.media_kind == "video" else Audio)
        for track, spec in media.items()
    }
    return type(name, (Input,), {"__annotations__": annotations})


def output_holder(name: str) -> type[Output]:
    """Build the :class:`Output` carrying an engine's frames.

    An engine returns decoded video and says nothing about where it goes, so the
    runtime declares the one outbound track it lands on. Declaring the class
    registers the track, so negotiation and the schema treat it as any other.

    Args:
        name: Class name for the holder, for readable diagnostics.

    Returns:
        The holder class, declaring a single video track.
    """
    return type(name, (Output,), {"__annotations__": {VIDEO_TRACK: Video}})


def missing_init_fields(init_cls: type[Init] | None) -> list[str]:
    """Return the initialization fields a client must supply, having no default."""
    if init_cls is None:
        return []
    return [name for name, spec in init_cls.__input_fields__.items() if spec.required]


def default_init(init_cls: type[Init] | None) -> Init | None:
    """Build the initialization the runtime fabricates when a client sends none.

    Returns ``None`` when the engine declares a field with no default, since
    there is nothing to fabricate from and no rollout can exist until a client
    sends one.
    """
    if init_cls is None:
        return None
    if missing_init_fields(init_cls):
        return None
    return init_cls()


def init_values(init: Init) -> dict[str, Any]:
    """Return an initialization's fields as the keyword arguments ``initialize_cache`` takes."""
    return {name: getattr(init, name) for name in type(init).__input_fields__}


def _declared_by(engine_cls: type) -> list[type[UserInput]]:
    """Return the input classes an engine declares, in declaration order."""
    explicit = getattr(engine_cls, "declared_inputs", None)
    if explicit is not None:
        return list(explicit)
    package = _package_of(engine_cls)
    return [
        input_cls for input_cls in USER_INPUT_REGISTRY.values() if _package_of(input_cls) == package
    ]


def _package_of(cls: type) -> str:
    """Return the top-level package a class was declared in."""
    return cls.__module__.partition(".")[0]


def _claim(
    target: dict[str, Any], name: str, input_cls: type[UserInput], others: list[dict[str, Any]]
) -> None:
    """Reject a wire name two declarations both want."""
    taken = name in target or any(name in other for other in others)
    if taken:
        raise TypeError(f"two declared inputs both resolve to the wire name '{name}'")
