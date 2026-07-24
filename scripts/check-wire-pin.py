#!/usr/bin/env python3
"""Gate the pinned wire release against the proto sources in this repository.

The runtime never commits its generated bindings. The build vendors them from
the ``wire/v*`` release pinned in ``[tool.reactor-wire]``, so a published wheel
carries bytes that were released and breaking-checked rather than a local
regeneration. The cost of that indirection is that the pin can fall behind
``proto/``: every check in this repo runs against bindings generated from the
in-repo sources, while the wheel a release builds carries whatever schema the
pinned artifact froze. Code reaching for a message added since the pin then
raises ``AttributeError`` in the published package and nowhere else.

The comparison is on the descriptors embedded in each ``_pb2.py``, reduced to
the set of names, field numbers, and types they declare. Two generations of the
same schema compare equal even when the codegen plugin in ``buf.gen.yaml``
moves under them, so bumping that plugin cannot wedge a release behind a
cosmetic diff, while a schema the pin has not caught up with still fails.

Two modes, mirroring the HTTP contract gate:

``report`` (per PR, informational)
    Describe the drift and exit ``0``. A proto change is expected to drift
    until it merges, its ``wire/v*`` release is cut, and the pin follows.

``enforce`` (at release, gating)
    Exit ``1`` on any drift, so a release cannot ship bindings that disagree
    with the sources it was built from.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from pathlib import Path

from google.protobuf import descriptor_pb2

REPO_ROOT = Path(__file__).resolve().parent.parent


def pinned_version() -> str:
    """Return the wire-protocol version pinned in pyproject.toml."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return str(tomllib.load(f)["tool"]["reactor-wire"]["version"])


def _serialized_descriptor(module: Path) -> bytes:
    """Return the serialized FileDescriptorProto a generated ``_pb2.py`` embeds."""
    for node in ast.walk(ast.parse(module.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "AddSerializedFile"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, bytes)
        ):
            return node.args[0].value
    raise SystemExit(f"error: {module} embeds no serialized descriptor")


def descriptors(root: Path) -> dict[str, descriptor_pb2.FileDescriptorProto]:
    """Return every descriptor generated under *root*, keyed by its .proto path."""
    found: dict[str, descriptor_pb2.FileDescriptorProto] = {}
    for module in sorted(root.rglob("*_pb2.py")):
        proto = descriptor_pb2.FileDescriptorProto()
        proto.ParseFromString(_serialized_descriptor(module))
        found[proto.name] = proto
    return found


def _field_signature(field: descriptor_pb2.FieldDescriptorProto) -> str:
    """Render a field as the parts a client can observe: name, number, and type."""
    label = descriptor_pb2.FieldDescriptorProto.Label.Name(field.label)
    # type_name carries the fully-qualified name for message and enum fields and
    # is empty for scalars, where the type enum is the only description there is.
    kind = field.type_name or (
        descriptor_pb2.FieldDescriptorProto.Type.Name(field.type).removeprefix("TYPE_").lower()
    )
    return f"{field.name} #{field.number} {label.removeprefix('LABEL_').lower()} {kind}"


def declarations(proto: descriptor_pb2.FileDescriptorProto) -> set[str]:
    """Return every message, field, enum, and enum value *proto* declares.

    A flat set of dotted names is enough to gate the pin: both sides describe
    the same schema at two points in time, so anything added, removed, renamed,
    renumbered, or retyped surfaces as a set difference the reader can act on.
    """
    names: set[str] = set()

    def walk_enum(prefix: str, enum: descriptor_pb2.EnumDescriptorProto) -> None:
        name = prefix + enum.name
        names.add(name)
        names.update(f"{name}.{value.name} = {value.number}" for value in enum.value)

    def walk_message(prefix: str, message: descriptor_pb2.DescriptorProto) -> None:
        name = prefix + message.name
        names.add(name)
        names.update(f"{name}.{_field_signature(field)}" for field in message.field)
        for nested in message.nested_type:
            walk_message(f"{name}.", nested)
        for nested_enum in message.enum_type:
            walk_enum(f"{name}.", nested_enum)

    for message in proto.message_type:
        walk_message("", message)
    for enum in proto.enum_type:
        walk_enum("", enum)
    return names


def drift(
    pinned: dict[str, descriptor_pb2.FileDescriptorProto],
    generated: dict[str, descriptor_pb2.FileDescriptorProto],
) -> list[str]:
    """Describe every schema difference between the pinned and generated bindings."""
    lines = [f"{name}: not in the pinned release" for name in sorted(generated.keys() - pinned)]
    lines += [f"{name}: not in proto/" for name in sorted(pinned.keys() - generated)]
    for name in sorted(pinned.keys() & generated.keys()):
        was, now = declarations(pinned[name]), declarations(generated[name])
        detail = [f"+{added}" for added in sorted(now - was)]
        detail += [f"-{removed}" for removed in sorted(was - now)]
        if detail:
            lines.append(f"{name}: " + ", ".join(detail))
    return lines


def main() -> int:
    """Compare the two binding trees and report or enforce the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["report", "enforce"])
    parser.add_argument(
        "--pinned", required=True, type=Path, help="bindings vendored from the pinned release"
    )
    parser.add_argument(
        "--generated", required=True, type=Path, help="bindings generated from the in-repo proto"
    )
    args = parser.parse_args()

    version = pinned_version()
    differences = drift(descriptors(args.pinned), descriptors(args.generated))
    if not differences:
        print(f"wire pin ok: {version} matches proto/")
        return 0

    print(f"the pinned wire release {version} disagrees with proto/:")
    for line in differences:
        print(f"  {line}")

    if args.mode == "report":
        print(
            "::notice::merging a proto change cuts a wire/v* release; bump "
            "[tool.reactor-wire] version to it before the next runtime release"
        )
        return 0

    print(
        f"::error::this release would vendor wire {version}, whose schema is not the one "
        "proto/ describes. Bump [tool.reactor-wire] version to the wire/v* release cut "
        "from these sources, then re-tag.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
