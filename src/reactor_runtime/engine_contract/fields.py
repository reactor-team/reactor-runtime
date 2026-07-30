"""Field declaration for contract inputs — :func:`InputField` and :class:`FieldSpec`.

The constraint vocabulary a :class:`~reactor_runtime.engine_contract.inputs.UserInput`
field is declared with. It is deliberately self-contained: an engine imports the
contract and nothing else, so the contract cannot reach into a runtime's own
field types. A runtime translates a :class:`FieldSpec` into whatever its
validation and schema layers use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

NO_DEFAULT: Final = object()
"""Sentinel marking a field that declares no default and is therefore required."""


@dataclass(frozen=True)
class FieldSpec:
    """The default and validation constraints of one input field.

    Build instances with :func:`InputField` rather than constructing directly.

    Attributes:
        default: The field's default value, or :data:`NO_DEFAULT` when required.
        description: Human-readable description, surfaced in a serving
            runtime's published schema.
        ge: Minimum allowed value, inclusive.
        le: Maximum allowed value, inclusive.
        min_length: Minimum length for a string or sequence value.
        max_length: Maximum length for a string or sequence value.
        choices: Exhaustive set of allowed values.
    """

    default: Any = NO_DEFAULT
    description: str | None = None
    ge: int | float | None = None
    le: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    choices: list[Any] | None = None

    @property
    def required(self) -> bool:
        """Whether the field must be supplied because it declares no default."""
        return self.default is NO_DEFAULT


def InputField(  # noqa: N802 — a capitalised factory reads as a type in field declarations
    default: Any = NO_DEFAULT,
    *,
    description: str | None = None,
    ge: int | float | None = None,
    le: int | float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    choices: list[Any] | None = None,
) -> Any:
    """Declare a default value and validation constraints for an input field.

    Use as the default of an annotated field on a
    :class:`~reactor_runtime.engine_contract.inputs.UserInput` subclass. The
    return type is ``Any`` so the call can stand in as the default of a field of
    any annotated type.

    Args:
        default: Default value for the field; omit to make the field required.
        description: Human-readable description, surfaced in a serving
            runtime's published schema.
        ge: Minimum allowed value, inclusive.
        le: Maximum allowed value, inclusive.
        min_length: Minimum length for a string or sequence value.
        max_length: Maximum length for a string or sequence value.
        choices: Exhaustive set of allowed values.

    Returns:
        A :class:`FieldSpec` carrying the supplied default and constraints.
    """
    return FieldSpec(
        default=default,
        description=description,
        ge=ge,
        le=le,
        min_length=min_length,
        max_length=max_length,
        choices=choices,
    )
