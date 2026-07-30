"""The input vocabulary an engine declares its surface with.

The four bases an engine declares its input surface with. A model author writes
these next to the model code and imports nothing else; a runtime reads the
declarations back and derives the surface it serves — a command per
:class:`UserInput`, an input track per :class:`MediaInput`, an initialization
payload from the :class:`Init`.

A subclass is a declaration in three senses at once: the class name is the wire
name, the annotated fields are the payload, and the :func:`InputField` defaults
on those fields are the validation. Nothing else has to be registered.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal, dataclass_transform, get_origin, get_type_hints

from reactor_runtime.engine_contract.fields import NO_DEFAULT, FieldSpec, InputField

USER_INPUT_REGISTRY: dict[str, type[UserInput]] = {}
"""Every declared :class:`UserInput` subclass, by fully qualified name.

Auto-populated when a subclass outside this module is created, so a runtime can
find an engine's inputs without the engine registering them. The contract's own
bases are not members.
"""

_RUNTIME_FILLED = frozenset({"timestamp_ms", "data", "pts"})
"""Attributes a runtime fills in on arrival, never read off the wire."""

_MISSING = object()


def _build_input(cls: type[UserInput]) -> None:
    """Resolve one subclass's wire fields and cache them on it.

    Inherited fields come first, then the class's own, so an instance's
    constructor order follows the declaration order down the hierarchy. The
    runtime-filled attributes are skipped: they are not part of any payload.

    Raises:
        TypeError: If a field's ``InputField`` default is a mutable container.
    """
    fields: dict[str, FieldSpec] = {}
    annotations: dict[str, Any] = {}
    for base in reversed(cls.__mro__[1:]):
        fields.update(getattr(base, "__input_fields__", {}))
        annotations.update(getattr(base, "__input_annotations__", {}))

    own = cls.__dict__.get("__annotations__", {})
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    for name in own:
        annotation = hints.get(name, own[name])
        if name in _RUNTIME_FILLED or _is_classvar(annotation):
            continue
        raw = cls.__dict__.get(name, _MISSING)
        if isinstance(raw, FieldSpec):
            spec = raw
        elif raw is _MISSING:
            spec = FieldSpec()
        else:
            spec = FieldSpec(default=raw)
        _raise_if_default_not_static(cls.__qualname__, name, spec.default)
        fields[name] = spec
        annotations[name] = annotation
        # A declaration reads as `speed: float = InputField(...)`, which leaves
        # the spec object bound as a class attribute. Clear it so an instance
        # that never received a value cannot read the spec back as its value.
        if name in cls.__dict__:
            delattr(cls, name)

    cls.__input_fields__ = _required_first(fields)
    cls.__input_annotations__ = annotations


def _is_classvar(annotation: Any) -> bool:
    """Return whether *annotation* declares a class variable rather than a field."""
    if isinstance(annotation, str):
        return annotation.startswith(("ClassVar", "typing.ClassVar"))
    return annotation is ClassVar or get_origin(annotation) is ClassVar


def _required_first(fields: dict[str, FieldSpec]) -> dict[str, FieldSpec]:
    """Order fields so the required ones come first, as a signature reads."""
    required = {name: spec for name, spec in fields.items() if spec.required}
    optional = {name: spec for name, spec in fields.items() if not spec.required}
    return required | optional


def _raise_if_default_not_static(owner: str, name: str, default: Any) -> None:
    """Reject a mutable default, which would be shared by every instance."""
    if isinstance(default, (list, dict, set)):
        raise TypeError(
            f"{owner}: default for '{name}' is a mutable {type(default).__name__}. "
            "Use an immutable default (tuple, frozenset) or None."
        )


@dataclass_transform(kw_only_default=True, field_specifiers=(InputField, FieldSpec))
class UserInput:
    """One typed event a client can send.

    Subclass with annotated fields; each becomes part of the wire payload, and
    an :func:`InputField` default attaches its validation constraints::

        class Move(UserInput):
            direction: Literal["forward", "back", "left", "right"]
            speed: float = InputField(default=1.0, ge=0.0, le=4.0)

    An instance is constructed by keyword. A field that declares no default is
    required, both of the constructor and of the client.

    Attributes:
        timestamp_ms: When the input arrived, on a monotonic millisecond clock.
            Stamped by the runtime; a client never sends it, and it is what the
            single ordered window is sorted by.
    """

    timestamp_ms: int = 0

    __input_fields__: ClassVar[dict[str, FieldSpec]] = {}
    """The wire fields this class declares, in constructor order."""

    __input_annotations__: ClassVar[dict[str, Any]] = {}
    """The resolved annotation of each wire field."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _build_input(cls)
        if cls.__module__ != __name__:
            USER_INPUT_REGISTRY[f"{cls.__module__}.{cls.__qualname__}"] = cls

    def __init__(self, **values: Any) -> None:
        """Populate the declared fields from keyword arguments.

        Args:
            values: One value per declared field. A field with a default may be
                omitted.

        Raises:
            TypeError: If a value names a field the class does not declare, or a
                field without a default is missing.
        """
        fields = type(self).__input_fields__
        unknown = set(values) - set(fields)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} has no field(s) {sorted(unknown)}; "
                f"declared fields are {sorted(fields)}"
            )
        missing: list[str] = []
        for name, spec in fields.items():
            if name in values:
                setattr(self, name, values[name])
            elif spec.default is not NO_DEFAULT:
                setattr(self, name, spec.default)
            else:
                missing.append(name)
        if missing:
            raise TypeError(f"{type(self).__name__} is missing required field(s) {missing}")
        self.timestamp_ms = 0

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name, None)!r}" for name in type(self).__input_fields__
        )
        return f"{type(self).__name__}({fields})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            getattr(self, name, None) == getattr(other, name, None)
            for name in type(self).__input_fields__
        )


class MediaInput(UserInput):
    """A :class:`UserInput` that carries media rather than a wire payload.

    Declaring a subclass declares an input track: the class name is the track
    name and the base is the modality. Frames arrive on the track, never as an
    event, so a subclass declares no fields of its own — the runtime fills
    :attr:`data` and :attr:`pts` and hands instances to the mapping in the
    window like any other input.

    Set :attr:`chunk_size` to consume a fixed batch per instance. The runtime
    accumulates that many frames, in capture order, and delivers one instance
    carrying the whole batch, stamped at the last frame's arrival; an incomplete
    batch waits across windows rather than arriving short.

    Attributes:
        data: The frames this instance carries, filled by the runtime.
        pts: Capture time of the frames in seconds, filled by the runtime, or
            ``None`` when the transport gave none.
        chunk_size: How many frames one instance carries.
    """

    media_kind: ClassVar[Literal["video", "audio"]]
    chunk_size: ClassVar[int] = 1

    data: Any = None
    pts: float | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__input_fields__:
            raise TypeError(
                f"{cls.__name__} is a MediaInput and declares wire field(s) "
                f"{sorted(cls.__input_fields__)}. Media arrives on a track, so a "
                "MediaInput carries only the frames the runtime fills in."
            )
        if cls.chunk_size < 1:
            raise TypeError(f"{cls.__name__}.chunk_size must be at least 1, got {cls.chunk_size}")


class VideoInput(MediaInput):
    """An inbound video track. Subclass to declare one; the class name is the track name."""

    media_kind: ClassVar[Literal["video", "audio"]] = "video"


class AudioInput(MediaInput):
    """An inbound audio track. Subclass to declare one; the class name is the track name."""

    media_kind: ClassVar[Literal["video", "audio"]] = "audio"


class Init(UserInput):
    """The model's initialization state — one subclass per engine.

    The fields are the state a rollout starts from and their defaults are the
    default initialization, so a client that sends nothing still gets a model.
    A field declared without a default has to come from the client before a
    rollout can exist at all.

    It rides the window like any other input. Its one extra role is at the
    boundary: the runtime consumes a leading ``Init`` to create the rollout,
    and after that a fresh one reaching the mapping means the client asked for
    a new sequence. An ``Init`` is a whole initialization, not a patch — a
    field the client omits takes the declared default.
    """


@dataclass_transform()
class ModelInput:
    """The full conditioning of one inference step.

    Engine-facing and never on the wire: the mapping folds a window into one of
    these and hands it to ``generate``. Subclass with annotated fields; the
    subclass is turned into a dataclass, so ordinary defaults apply::

        class LingBotStepInput(ModelInput):
            camera_trajectory: Tensor
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not dataclasses.is_dataclass(cls):
            dataclasses.dataclass(cls)


__all__ = [
    "USER_INPUT_REGISTRY",
    "AudioInput",
    "Init",
    "InputField",
    "MediaInput",
    "ModelInput",
    "UserInput",
    "VideoInput",
]
