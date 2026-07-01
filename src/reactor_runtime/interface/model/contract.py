"""The per-model contract — :class:`ModelContract`.

The messages a model sends back and its media tracks are read from the
process-global registries every declaration populates, so a message or track
reaches the schema by being declared, never by being wired onto the model class.
Commands are the exception: they are handler-bound, and a command can be
*synthesised* (a pipeline's ``set_<field>`` setters stamp their handlers directly
rather than through the ``@event`` decorator that fills the registry), so the
registry is not a complete command set. The complete set is the one a single
class-level traversal resolves — binding each command to its handler method and
its response type, and collecting the lifecycle hooks — cached on the class. That
resolved set is what validation, dispatch, and the rendered schema all read, so
they can never disagree.

The handler-bound parts are built once, when the model class is created;
:meth:`ModelContract.of` is the accessor for the cached result. The message and
track registries are read lazily, so a message class declared after the model
class still reaches the schema.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from reactor_runtime.core.fields import NO_DEFAULT, validate_field
from reactor_runtime.core.model import Command
from reactor_runtime.core.naming import pascal_to_snake
from reactor_runtime.core.values import TrackInfo
from reactor_runtime.interface.events.decorators import (
    CONNECTED_ATTR,
    DISCONNECTED_ATTR,
    EVENT_ATTR,
    FILE_UPLOADED_ATTR,
    SESSION_ENDED_ATTR,
    SESSION_STARTED_ATTR,
    EventHandler,
)
from reactor_runtime.interface.events.messages import MESSAGE_REGISTRY, ModelMessage
from reactor_runtime.interface.model.schema import (
    CommandSchema,
    MessageSchema,
    ModelSchema,
    command_field_schema,
    message_field_schema,
    track_schema,
)
from reactor_runtime.interface.tracks.input import all_input_tracks
from reactor_runtime.interface.tracks.output import all_output_tracks


class ContractError(Exception):
    """A client payload that fails the contract at request time.

    Attributes:
        field: The offending field, command, or argument name.
        reason: Why it was rejected.
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


@dataclass(frozen=True)
class CommandSpec:
    """A resolved command and everything needed to dispatch and document it.

    Attributes:
        name: The command's wire name.
        command: The :class:`Command` subclass a payload validates into.
        handler: The unbound handler method the command dispatches to.
        description: Human-readable description, surfaced in the schema.
        response: The message type the handler returns, or ``None``.
        is_async: Whether the handler is a coroutine function.
        reserved: Reserved parameters the runtime injects, in injection order.
    """

    name: str
    command: type[Command]
    handler: Callable[..., Any]
    description: str
    response: type[ModelMessage] | None
    is_async: bool
    reserved: tuple[str, ...]


@dataclass(frozen=True)
class LifecycleHooks:
    """The lifecycle handler methods a model declares, by scope.

    Attributes:
        session_started: Runs once when the session begins.
        session_ended: Runs once when the session ends.
        connected: Runs each time a client connects.
        disconnected: Runs each time a client disconnects.
        file_uploaded: Runs when a client uploads a file.
    """

    session_started: Callable[..., Any] | None = None
    session_ended: Callable[..., Any] | None = None
    connected: Callable[..., Any] | None = None
    disconnected: Callable[..., Any] | None = None
    file_uploaded: Callable[..., Any] | None = None


@dataclass(frozen=True)
class ModelContract:
    """The assembled, cached, client-facing contract of one model.

    The handler-bound parts (:attr:`commands`, :attr:`lifecycle`) are resolved
    once from the class. The media :attr:`tracks` and outbound :attr:`messages`
    are read lazily from the process-global registries every declaration
    populates.

    Attributes:
        model: The model identifier, lowercase.
        description: Human-readable model description.
        commands: Command name to its resolved spec.
        lifecycle: The model's lifecycle hook methods.
    """

    model: str
    description: str
    commands: dict[str, CommandSpec]
    lifecycle: LifecycleHooks

    @property
    def tracks(self) -> dict[str, TrackInfo]:
        """The model's media tracks, inbound first, from the track registries.

        Reads the union of every declared :class:`Output` and :class:`Input`.
        Inbound tracks come first so their m-line indices precede the outbound
        ones during transport negotiation.

        Raises:
            ValueError: If a track name is declared in both directions.
        """
        merged: dict[str, TrackInfo] = dict(all_input_tracks())
        for name, info in all_output_tracks().items():
            if name in merged:
                raise ValueError(f"track name '{name}' is declared as both input and output")
            merged[name] = info
        return merged

    @property
    def messages(self) -> dict[str, type[ModelMessage]]:
        """The outbound messages the model can send, from the message registry.

        Every declared :class:`ModelMessage` — whether a command's reply or one
        the model only broadcasts — rather than only those a handler returns.
        """
        return dict(MESSAGE_REGISTRY)

    @classmethod
    def of(cls, model_cls: type) -> ModelContract:
        """Return the contract cached on *model_cls*.

        Args:
            model_cls: A model class whose contract was built at creation.

        Returns:
            The cached contract.

        Raises:
            TypeError: If *model_cls* carries no contract.
        """
        contract = getattr(model_cls, "__reactor_contract__", None)
        if not isinstance(contract, ModelContract):
            raise TypeError(f"{model_cls.__qualname__} has no model contract")
        return contract

    @classmethod
    def build(cls, model_cls: type) -> ModelContract:
        """Assemble the contract from a single traversal of *model_cls*.

        Walks the MRO binding each command and lifecycle hook to its handler
        method, resolving a command's response type from its handler's return
        annotation. Inheritance is ordinary: a subclass inherits every command
        its bases declare, and overrides one by re-applying ``@event`` (the
        most-derived ``@event`` definition wins, binding that class's method as
        the handler). A plain, undecorated method of the same name is not an
        override — it neither un-declares the inherited command nor rebinds it, so
        the command keeps the base's ``@event`` definition and runs the base
        method. The model's tracks and outbound messages are not snapshotted
        here — they are read from the registries through :attr:`tracks` /
        :attr:`messages` when the schema renders.

        Args:
            model_cls: The model class to assemble the contract for.

        Returns:
            The assembled contract.

        Raises:
            ValueError: If two distinct handlers claim the same command name.
        """
        commands: dict[str, CommandSpec] = {}
        claimed_by: dict[str, str] = {}
        hooks: dict[str, Callable[..., Any]] = {}

        for klass in model_cls.__mro__:
            for attr_name, attr in vars(klass).items():
                handler = getattr(attr, EVENT_ATTR, None)
                if isinstance(handler, EventHandler):
                    if handler.name in commands:
                        # A less-derived copy of the same method is the inherited
                        # original; a different method is a genuine name clash.
                        if claimed_by[handler.name] != attr_name:
                            raise ValueError(f"duplicate command name '{handler.name}'")
                        continue
                    commands[handler.name] = _command_spec(handler, attr)
                    claimed_by[handler.name] = attr_name
                    continue
                for lifecycle_attr, key in _LIFECYCLE_ATTRS.items():
                    if getattr(attr, lifecycle_attr, False):
                        hooks.setdefault(key, attr)

        return cls(
            model=pascal_to_snake(model_cls.__name__),
            description=_normalize(model_cls.__doc__),
            commands=commands,
            lifecycle=LifecycleHooks(**hooks),
        )

    def validate(self, name: str, raw_args: dict[str, Any]) -> Command:
        """Validate a client payload into a typed command.

        Each accepted value is coerced into its Python form, so an enum field
        holds a member rather than its wire value. An upload field keeps its
        reference: the bytes are fetched by the runtime, not here.

        Args:
            name: The command name the client sent.
            raw_args: The raw argument mapping from the wire.

        Returns:
            The constructed, validated command.

        Raises:
            ContractError: If the command is unknown, an argument is unexpected
                or missing, or a value fails its field's type or constraints.
        """
        spec = self.commands.get(name)
        if spec is None:
            raise ContractError(name, "unknown command")

        fields = spec.command.__command_fields__
        for arg in raw_args:
            if arg not in fields:
                raise ContractError(arg, "unexpected argument")

        kwargs: dict[str, Any] = {}
        for field_name, command_field in fields.items():
            if field_name in raw_args:
                value = raw_args[field_name]
                type_reason = command_field.spec.check(value)
                if type_reason is not None:
                    raise ContractError(field_name, type_reason)
                ok, constraint_reason = validate_field(field_name, value, command_field.info)
                if not ok:
                    raise ContractError(field_name, constraint_reason)
                kwargs[field_name] = command_field.spec.coerce(value)
            elif command_field.info.default is NO_DEFAULT:
                raise ContractError(field_name, "missing required argument")

        return spec.command(**kwargs)

    def render_schema(self, version: str = "v0.0.0") -> ModelSchema:
        """Render the contract as a versioned :class:`ModelSchema`.

        Args:
            version: The release tag to stamp, carrying a leading ``v``.

        Returns:
            The schema, ready to emit as an OpenAPI document.
        """
        commands = {
            name: CommandSchema(
                description=spec.description,
                schema={
                    field_name: command_field_schema(command_field)
                    for field_name, command_field in spec.command.__command_fields__.items()
                },
            )
            for name, spec in self.commands.items()
        }
        messages = {
            name: MessageSchema(
                description=_normalize(message.user_doc),
                schema={
                    field_name: message_field_schema(message_field)
                    for field_name, message_field in message.__message_fields__.items()
                },
            )
            for name, message in self.messages.items()
        }
        tracks = {name: track_schema(track) for name, track in self.tracks.items()}
        return ModelSchema(
            version=version,
            name=self.model,
            description=self.description,
            tracks=tracks,
            commands=commands,
            messages=messages,
        )


_LIFECYCLE_ATTRS = {
    SESSION_STARTED_ATTR: "session_started",
    SESSION_ENDED_ATTR: "session_ended",
    CONNECTED_ATTR: "connected",
    DISCONNECTED_ATTR: "disconnected",
    FILE_UPLOADED_ATTR: "file_uploaded",
}


def _command_spec(handler: EventHandler, method: Callable[..., Any]) -> CommandSpec:
    """Build a :class:`CommandSpec`, resolving the handler's response type."""
    return CommandSpec(
        name=handler.name,
        command=handler.command,
        handler=method,
        description=handler.description,
        response=_response_type(method),
        is_async=handler.is_async,
        reserved=handler.reserved,
    )


def _response_type(handler: Callable[..., Any]) -> type[ModelMessage] | None:
    """Resolve a handler's return annotation to a message type, or ``None``.

    A handler that returns a :class:`ModelMessage` subclass registers it as the
    command's response; any other return (including none) means no response.
    """
    try:
        hints = get_type_hints(handler)
    except Exception:
        return None
    returned = hints.get("return")
    if isinstance(returned, type) and issubclass(returned, ModelMessage):
        return returned
    return None


def _normalize(doc: str | None) -> str:
    """Dedent and trim a docstring for use as a schema description."""
    if not doc:
        return ""
    return inspect.cleandoc(doc)
