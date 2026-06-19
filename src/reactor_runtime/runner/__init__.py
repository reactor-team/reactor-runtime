"""Runner core — the session brain pieces.

The components the runner composes to drive one session: the pure session state
machine and the connection multiplexer. They carry the session lifecycle and the
live-connection registry without knowing anything about transports, the model, or
asyncio — the runner wires them to the rest of the process.
"""

from reactor_runtime.runner.state_machine import SessionStateMachine

__all__ = [
    "SessionStateMachine",
]
