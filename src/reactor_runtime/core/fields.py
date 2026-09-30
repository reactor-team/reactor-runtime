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

import dataclasses
import inspect
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
        moderate: Whether the field is marked for content moderation in the
            rendered schema, which states either polarity. Defaults to
            ``False``; see :func:`InputField` for what the mark does and which
            fields it applies to.
    """

    default: Any = NO_DEFAULT
    description: str | None = None
    ge: int | float | None = None
    le: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    choices: list[Any] | None = None
    moderate: bool = False


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
    moderate: bool = False,
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
        moderate: Whether to mark the field for content moderation. ``False``
            by default, so a field is marked only when you ask for it. Either
            way the rendered schema states the answer: the field carries
            ``x-reactor-moderate: true`` or ``x-reactor-moderate: false``. The
            mark is a preference and nothing more — it starts no check, and the
            runtime moderates nothing itself. Whether a check runs against a
            marked field is a deployment decision taken from that schema. Only
            free-text strings and uploaded files are ever eligible — typed,
            enum, and bounded numeric fields carry no free text, so the mark
            does nothing for them.

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


def own_annotations(cls: type) -> dict[str, Any]:
    """Return the annotations *cls* declares itself, none from its bases.

    Goes through :func:`inspect.get_annotations` rather than the class
    ``__dict__``. From Python 3.14 (PEP 649) a class stores its annotations
    lazily, behind ``__annotate__``, and a direct dict read finds nothing; the
    accessor materialises them on every version. Inherited annotations are
    read from the bases' cached field records, never re-resolved.
    """
    return inspect.get_annotations(cls)


def inherited_record(cls: type, attribute: str) -> dict[str, Any]:
    """Merge the field records the bases of *cls* store on themselves.

    A declaration base (``InputState``, ``ModelMessage``, ``Command``) caches
    the fields each subclass declares in a class-level dict. A subclass starts
    from this merge, so the fields it inherits are part of its own record. Bases
    are read from the most distant to the closest, so a nearer base wins a name
    both declare, and a re-declared name keeps the position the first base gave
    it, which is the order a dataclass lays inherited fields out in.

    Args:
        cls: The class being declared.
        attribute: The name of the record attribute on each base.

    Returns:
        A new dict; the bases' records are not modified.
    """
    merged: dict[str, Any] = {}
    for base in reversed(cls.__mro__[1:]):
        merged.update(base.__dict__.get(attribute, {}))
    return merged


def apply_dataclass(cls: type, *, required: list[str], inherits: bool) -> None:
    """Turn a freshly declared subclass into a dataclass, once.

    A class whose parent is already a dataclass is still converted, so the
    fields it declares become fields of its own. Its fields are keyword-only
    when it inherits any, because a dataclass refuses a required field placed
    after an inherited field with a default, and keyword-only fields have no
    such order.

    A required field hides any class attribute a parent holds under the same
    name for the duration of the conversion. The dataclass reads a field's
    default with ``getattr``, so without this a field declared required again
    would silently inherit the parent's default. Once the placeholder is
    removed the parent's attribute is reachable through the subclass again, so
    ``Child.x`` reads the parent's value while ``dataclasses.fields(Child)``
    and the field record both say required. The runtime reads defaults from
    the record, never from the class attribute.

    Args:
        cls: The class to convert. A class that already carries its own
            ``__dataclass_fields__`` is left alone.
        required: The names of the fields *cls* declares without a default.
        inherits: Whether *cls* inherits fields from a declared base.
    """
    if "__dataclass_fields__" in cls.__dict__:
        return
    for name in required:
        setattr(cls, name, dataclasses.MISSING)
    dataclasses.dataclass(cls, kw_only=inherits)
    for name in required:
        delattr(cls, name)


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
