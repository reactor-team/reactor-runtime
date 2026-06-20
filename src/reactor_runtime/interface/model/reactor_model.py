"""The model authoring base — :class:`ReactorModel`.

What a model author subclasses. Declaring a subclass assembles its
:class:`ModelContract` once, from a single traversal of the class, and caches it
on the class — the commands its ``@event`` handlers expose, the messages they
return, its tracks, and its lifecycle hooks.

This is the contract-bearing shell. The engine that gives a model its run loop,
queues, and dispatchers is built separately on top of this base; subclassing here
resolves the contract and nothing more.
"""

from __future__ import annotations

from typing import ClassVar

from reactor_runtime.interface.model.contract import ModelContract


class ReactorModel:
    """Base class an author subclasses to define a model.

    Decorate methods with ``@event`` to expose commands, and with the lifecycle
    decorators to hook session and connection events. Declaring the subclass
    resolves the contract and caches it on the class, reachable through
    :meth:`ModelContract.of`.
    """

    __reactor_contract__: ClassVar[ModelContract]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.__reactor_contract__ = ModelContract.build(cls)
