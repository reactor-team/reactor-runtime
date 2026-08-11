# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

"""Typed message tags for the controller ↔ worker queues.

Messages are tuples whose first element is a tag below; enums pickle by
reference across the spawn boundary and are compared by identity, so a
mistyped tag is an ``AttributeError`` at the definition site instead of
a silently ignored verb at runtime.
"""

from __future__ import annotations

import enum


class Verb(enum.Enum):
    """Controller → worker commands, broadcast to every rank."""

    SEED = enum.auto()
    INIT_SESSION = enum.auto()
    CHUNK = enum.auto()
    DROP_SESSION = enum.auto()
    EXIT = enum.auto()


class Reply(enum.Enum):
    """Worker → controller results. Every rank replies to every verb;
    the controller collects the full reply set before proceeding."""

    READY = enum.auto()
    OK = enum.auto()
    FRAMES = enum.auto()
    ERROR = enum.auto()
