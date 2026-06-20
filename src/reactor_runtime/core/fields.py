"""Field declaration and default-value validation.

The author-facing way to attach a default and validation constraints to a
command or state field: :func:`InputField` builds a :class:`FieldInfo`, and the
pure helpers here check, at definition time, that a declared default is static
and satisfies its own constraints. Request-time validation of incoming values
reuses :func:`validate_field`, so a default and a client payload are judged by
exactly the same rules.

Depends on nothing else in the package, so it sits at the root of the import
graph alongside the neutral value vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

NO_DEFAULT: Final = object()
"""Sentinel marking a field that declares no default and is therefore required.

A distinct object rather than ``None`` because ``None`` is itself a valid
default for an optional field.
"""

_MUTABLE_DEFAULT_TYPES = (list, dict, set)


@dataclass(frozen=True)
class FieldInfo:
    """Validation constraints and metadata for a single input field.

    Build instances with :func:`InputField` rather than constructing directly.
    When the author declares no default, :attr:`default` holds :data:`NO_DEFAULT`
    and the field is required.

    Attributes:
        default: The field's default value, or :data:`NO_DEFAULT` when required.
        description: Human-readable description, surfaced in the rendered schema.
        ge: Minimum allowed value, inclusive.
        le: Maximum allowed value, inclusive.
        min_length: Minimum length for a string or sequence value.
        max_length: Maximum length for a string or sequence value.
        choices: Exhaustive set of allowed values.
        moderate: Whether the field's value is eligible for content moderation.
    """

    default: Any = NO_DEFAULT
    description: str | None = None
    ge: int | float | None = None
    le: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    choices: list[Any] | None = None
    moderate: bool = True


def InputField(  # noqa: N802 — a capitalised factory reads as a type in field declarations
    default: Any = NO_DEFAULT,
    *,
    default_factory: Any = None,
    description: str | None = None,
    ge: int | float | None = None,
    le: int | float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    choices: list[Any] | None = None,
    moderate: bool = True,
) -> Any:
    """Declare a default value and validation constraints for a field.

    Use as the default for an ``@event`` handler parameter or a state field.
    Values that violate the declared constraints are rejected before a handler
    runs. The return type is ``Any`` so the call can stand in as the default of
    a field of any annotated type.

    Args:
        default: Default value for the field.
        default_factory: Unsupported; passing one raises ``TypeError``. Defaults
            must be statically representable — use a literal ``default=...``.
        description: Human-readable description, surfaced in the rendered schema.
        ge: Minimum allowed value, inclusive.
        le: Maximum allowed value, inclusive.
        min_length: Minimum length for a string or sequence value.
        max_length: Maximum length for a string or sequence value.
        choices: Exhaustive set of allowed values.
        moderate: Whether the field's value is eligible for content moderation
            when moderation is enabled. Only free-text strings and uploaded
            files are ever moderated; typed, enum, and bounded numeric fields
            are rejected before a handler sees them, so there is nothing left to
            moderate.

    Returns:
        A :class:`FieldInfo` carrying the supplied default and constraints.

    Raises:
        TypeError: If ``default_factory`` is supplied.
    """
    if default_factory is not None:
        raise TypeError(
            "InputField(default_factory=...) is not supported. Defaults must be "
            "statically representable — use a literal `default=...`."
        )
    return FieldInfo(
        default=default,
        description=description,
        ge=ge,
        le=le,
        min_length=min_length,
        max_length=max_length,
        choices=choices,
        moderate=moderate,
    )


def raise_if_default_not_static(owner: str, field_name: str, default: Any) -> None:
    """Raise ``TypeError`` when *default* is a mutable container.

    A mutable default (``list`` / ``dict`` / ``set``) would be shared across
    every instance of the class and leak state between sessions — the same
    reason ``dataclasses`` forbids ``field(default=[])``. ``None`` is the
    canonical "unset" value for an optional field and is always allowed.

    Args:
        owner: Qualified name of the declaring class, for the error message.
        field_name: The field being checked.
        default: The declared default value.

    Raises:
        TypeError: If *default* is a mutable container.
    """
    if default is None:
        return
    if isinstance(default, _MUTABLE_DEFAULT_TYPES):
        raise TypeError(
            f"{owner}: default for '{field_name}' is a mutable "
            f"{type(default).__name__}, which cannot be a static default. Use "
            "`default=None` and build the container inside the handler, or an "
            "immutable alternative (tuple, frozenset)."
        )


def validate_field(name: str, value: Any, info: FieldInfo) -> tuple[bool, str]:
    """Check *value* against a field's :class:`FieldInfo` constraints.

    Args:
        name: The field name, for the failure reason.
        value: The value to check.
        info: The constraints to check against.

    Returns:
        ``(True, "")`` when the value satisfies every constraint, otherwise
        ``(False, reason)`` describing the first violation.
    """
    if info.choices is not None and value not in info.choices:
        return False, f"{name}: {value!r} not in choices {info.choices}"

    if info.ge is not None:
        try:
            if value < info.ge:
                return False, f"{name}: {value} < ge({info.ge})"
        except TypeError:
            return False, f"{name}: {value!r} is not comparable to ge({info.ge})"

    if info.le is not None:
        try:
            if value > info.le:
                return False, f"{name}: {value} > le({info.le})"
        except TypeError:
            return False, f"{name}: {value!r} is not comparable to le({info.le})"

    if info.min_length is not None:
        try:
            if len(value) < info.min_length:
                return False, f"{name}: length {len(value)} < min_length({info.min_length})"
        except TypeError:
            return False, f"{name}: {value!r} has no length for min_length({info.min_length})"

    if info.max_length is not None:
        try:
            if len(value) > info.max_length:
                return False, f"{name}: length {len(value)} > max_length({info.max_length})"
        except TypeError:
            return False, f"{name}: {value!r} has no length for max_length({info.max_length})"

    return True, ""


def raise_if_default_invalid(owner: str, field_name: str, default: Any, info: FieldInfo) -> None:
    """Raise ``TypeError`` when *default* cannot stand for *field_name*.

    The single definition-time gate on an ``InputField`` default: the default
    must be statically representable and must satisfy its own constraints, so a
    class whose default is already out of range fails at import rather than on
    the first request. The :data:`NO_DEFAULT` sentinel and an explicit ``None``
    bypass both checks.

    Args:
        owner: Qualified name of the declaring class, for the error message.
        field_name: The field being checked.
        default: The declared default value.
        info: The constraints the default must satisfy.

    Raises:
        TypeError: If *default* is mutable or violates its own constraints.
    """
    if default is NO_DEFAULT or default is None:
        return
    raise_if_default_not_static(owner, field_name, default)
    ok, reason = validate_field(field_name, default, info)
    if not ok:
        raise TypeError(
            f"{owner}: default for '{field_name}' violates its own InputField "
            f"constraints ({reason})."
        )
