"""A director app's client contract, read from the signature of its ``generate()``.

Each parameter of ``generate()`` is a field of the ``enqueue`` command. A
parameter typed :class:`Shot` takes a shot id on the wire. The messages that
describe a shot carry its inputs typed the same way, so a client generated from
the schema reads a typed shot. All of it is built once, when the class is
defined.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import operator
from collections.abc import Callable
from dataclasses import dataclass
from types import GenericAlias, UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from reactor_runtime.apps.director.shots import Shot, ShotEntry
from reactor_runtime.core.fields import NO_DEFAULT, FieldInfo, InputField
from reactor_runtime.core.model import Command, UploadedFile
from reactor_runtime.interface.events.decorators import RESERVED_PARAMS, make_command
from reactor_runtime.interface.events.messages import MessageField, ModelMessage

CONTINUE_PARAMETER = "continue_from"
"""The ``Shot`` parameter whose shot, in autoplay, plays straight on into this one."""

METADATA_MAX_LENGTH = 2000
"""The longest ``metadata`` string ``enqueue`` takes."""

_ADDED_FIELDS = frozenset({"metadata", "position"})

ShotStatusName = Literal["queued", "building", "ready", "playing", "built", "failed"]
"""The values of :class:`ShotStatus`, as a shot's description types its status."""


@dataclass(frozen=True)
class DirectorMessages:
    """The message types a director app sends, built for its own ``generate()``."""

    state: type[ModelMessage]
    queued: type[ModelMessage]
    generated: type[ModelMessage]
    failed: type[ModelMessage]
    started: type[ModelMessage]
    finished: type[ModelMessage]
    stopped: type[ModelMessage]


@dataclass(frozen=True)
class DirectorSpec:
    """What a director app's ``generate()`` declares.

    Attributes:
        parameters: The parameter names, in order.
        shot_parameters: The parameters typed :class:`Shot`.
        continues: :data:`CONTINUE_PARAMETER` when it is a ``Shot`` parameter.
        is_async: Whether ``generate()`` is a coroutine function.
        enqueue: The ``enqueue`` command.
        messages: The message types.
    """

    parameters: tuple[str, ...]
    shot_parameters: frozenset[str]
    continues: str | None
    is_async: bool
    enqueue: type[Command]
    messages: DirectorMessages


def build_spec(cls: type, generate: Callable[..., Any]) -> DirectorSpec:
    """Read *generate*'s signature into the class's contract.

    Raises:
        TypeError: A parameter is positional-only or variadic, takes a name
            the class adds (``metadata``, ``position``) or the runtime
            injects (``client``), or is typed ``Shot`` with a default other
            than ``None``.
    """
    owner = f"{cls.__qualname__}.generate"
    try:
        hints = get_type_hints(generate)
    except Exception as exc:
        raise TypeError(f"{owner}: cannot resolve its parameter types: {exc}") from exc

    fields: list[tuple[str, Any] | tuple[str, Any, Any]] = []
    info_fields: list[tuple[str, Any]] = []
    shot_parameters: set[str] = set()
    names: list[str] = []
    for name, param in inspect.signature(generate).parameters.items():
        if name == "self":
            continue
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise TypeError(
                f"{owner}: parameter '{name}' must be a named parameter a client can set"
            )
        if name in _ADDED_FIELDS or name in RESERVED_PARAMS:
            raise TypeError(f"{owner}: parameter '{name}' is a name the director app uses itself")
        annotation = hints.get(name, Any)
        default = param.default
        if _is_shot(annotation):
            shot_parameters.add(name)
            fallback = default.default if isinstance(default, FieldInfo) else default
            if (
                fallback is not inspect.Parameter.empty
                and fallback is not NO_DEFAULT
                and fallback is not None
            ):
                raise TypeError(f"{owner}: Shot parameter '{name}' may only default to None")
            annotation = _on_the_wire(annotation)
        names.append(name)
        info_fields.append((name, _described(annotation)))
        if default is inspect.Parameter.empty:
            fields.append((name, annotation))
        else:
            fields.append((name, annotation, default))

    fields.append(
        (
            "metadata",
            str,
            InputField(
                default="",
                max_length=METADATA_MAX_LENGTH,
                description=(
                    "A string of the client's own, echoed back on every message about the shot."
                ),
            ),
        )
    )
    fields.append(
        (
            "position",
            int | None,
            InputField(
                default=None,
                ge=0,
                description=(
                    "Where the shot enters the generation queue: 0 is the front. Omitted, the back."
                ),
            ),
        )
    )
    continues = CONTINUE_PARAMETER if CONTINUE_PARAMETER in shot_parameters else None
    return DirectorSpec(
        parameters=tuple(names),
        shot_parameters=frozenset(shot_parameters),
        continues=continues,
        is_async=inspect.iscoroutinefunction(generate),
        enqueue=make_command("enqueue", fields),
        messages=_build_messages(cls, info_fields),
    )


def describe(entry: ShotEntry, fps: float) -> dict[str, Any]:
    """Return *entry* as the plain mapping every message carries for a shot, its length at *fps*."""
    return {
        "shot_id": entry.id,
        "status": entry.status.value,
        "metadata": entry.metadata,
        "inputs": {name: _plain(value) for name, value in entry.inputs.items()},
        "frames": entry.frames,
        "seconds": None if entry.frames is None else round(entry.frames / fps, 3),
        "error": entry.error,
    }


def _is_shot(annotation: Any) -> bool:
    if annotation is Shot:
        return True
    return _optional_inner(annotation) is Shot


def _optional_inner(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1 and len(get_args(annotation)) == 2:
            return args[0]
    return None


def _on_the_wire(annotation: Any) -> Any:
    """The command field type of a ``Shot`` parameter: its id."""
    return str if annotation is Shot else str | None


@dataclass(frozen=True)
class _UploadInfo:
    name: str
    mime_type: str


def _optional(annotation: Any) -> Any:
    """``annotation | None``, built from a type known only at run time."""
    return operator.or_(annotation, type(None))


def _list_of(annotation: Any) -> Any:
    """``list[annotation]``, built from a type known only at run time."""
    return GenericAlias(list, (annotation,))


def _described(annotation: Any) -> Any:
    """The type a shot's description gives an input of type *annotation*."""
    if annotation is UploadedFile:
        return _UploadInfo
    inner = _optional_inner(annotation)
    if inner is not None:
        return _optional(_described(inner))
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return _list_of(_described(args[0])) if args else list
    return annotation


def _plain(value: Any) -> Any:
    """A JSON-ready form of one input, as a shot's description carries it."""
    if isinstance(value, UploadedFile):
        return {"name": value.name, "mime_type": value.mime_type}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _build_messages(cls: type, info_fields: list[tuple[str, Any]]) -> DirectorMessages:
    """Build the message types whose shots carry inputs of *info_fields*' types."""
    module = cls.__module__
    inputs = dataclasses.make_dataclass("ShotInputs", info_fields, frozen=True, module=module)
    info = dataclasses.make_dataclass(
        "ShotInfo",
        [
            ("shot_id", str),
            ("status", ShotStatusName),
            ("metadata", str),
            ("inputs", inputs),
            ("frames", int | None),
            ("seconds", float | None),
            ("error", str | None),
        ],
        frozen=True,
        module=module,
    )
    capacity = dataclasses.make_dataclass(
        "ShotCapacity",
        [("generation", int), ("playout", int), ("history", int)],
        frozen=True,
        module=module,
    )

    def message(name: str, doc: str, fields: dict[str, tuple[Any, str]]) -> type[ModelMessage]:
        namespace: dict[str, Any] = {
            "__doc__": doc,
            "__module__": module,
            "__annotations__": {field: annotation for field, (annotation, _) in fields.items()},
        }
        for field, (_, description) in fields.items():
            namespace[field] = MessageField(description=description)
        return type(name, (ModelMessage,), namespace)

    shot = {"shot": (info, "The shot, whole.")}
    return DirectorMessages(
        state=message(
            "DirectorState",
            "Everything the session holds, sent to a client when it joins and to every client "
            "after each change. A frontend can render from this message alone.",
            {
                "autoplay": (bool, "Whether built shots play in order without a play per shot."),
                "hold_last_frame": (
                    bool,
                    "Whether the last frame holds between shots instead of black.",
                ),
                "build_paused": (bool, "Whether new builds wait; one already running finishes."),
                "playing": (_optional(info), "The shot on the output tracks, or null."),
                "generation": (_list_of(info), "Shots waiting to build or building, front first."),
                "playout": (_list_of(info), "Built shots waiting to play, front first."),
                "history": (
                    _list_of(info),
                    "Built shots later shots can still continue, oldest first.",
                ),
                "capacity": (capacity, "The most shots each place holds."),
                "valid_commands": (list[str], "The commands the session accepts right now."),
            },
        ),
        queued=message("ShotQueued", "The reply to enqueue: the shot it queued.", shot),
        generated=message("ShotGenerated", "A shot finished building.", shot),
        failed=message(
            "ShotFailed",
            "A shot did not build, or depends on one that did not. Its error says why.",
            shot,
        ),
        started=message("ShotStarted", "A shot's first frames reached the output tracks.", shot),
        finished=message("ShotFinished", "A shot played to its end.", shot),
        stopped=message("ShotStopped", "stop or reset cut the shot that was playing.", shot),
    )
