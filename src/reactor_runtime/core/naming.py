"""Conversion between the author's PascalCase class names and wire names.

A command or message is declared as a PascalCase class but travels on the wire
under its ``snake_case`` name (``SetBrightness`` becomes ``set_brightness``).
These two functions are the single place that mapping lives, so a class name and
the string a client sends stay in lockstep.
"""

from __future__ import annotations

import re

_PASCAL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def pascal_to_snake(name: str) -> str:
    """Return the ``snake_case`` wire name for a PascalCase class name."""
    return _PASCAL_BOUNDARY.sub("_", name).lower()


def snake_to_pascal(name: str) -> str:
    """Return the PascalCase class name for a ``snake_case`` wire name."""
    return "".join(word.capitalize() for word in name.split("_"))
