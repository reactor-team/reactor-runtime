from dataclasses import dataclass
from typing import Any, Callable, List
from reactor_runtime.transport.webrtc.gstreamer.gst import GObject
import logging

logger = logging.getLogger(__name__)


@dataclass
class _SignalConnection:
    """
    Represents a single GObject signal connection.

    We keep:
      - obj: the GObject instance the signal is connected to
      - handler_id: the integer ID returned by obj.connect(...)
      - signal_name: for debugging/logging purposes
    """

    obj: Any
    handler_id: int
    signal_name: str


class SignalManager:
    """
    Small utility to manage a collection of GObject signal connections.

    Motivation:
      - In GStreamer / GLib code, it's common to connect many signals
        (bus messages, webrtcbin callbacks, pad-added, etc.).
      - During teardown, leaving signal handlers connected can cause:
          * callbacks firing after shutdown
          * references preventing objects from being collected
          * difficult-to-debug crashes if callbacks run on freed resources

    This helper centralizes all connect() calls and provides a single
    disconnect_all() method for deterministic cleanup.
    """

    def __init__(self) -> None:
        # List of tracked signal connections in the order they were created.
        self._connections: List[_SignalConnection] = []

    def connect(
        self,
        obj: Any,
        signal_name: str,
        handler: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """
        Connect a signal handler to a GObject and track the connection.

        Args:
            obj:
                GObject (or GI proxy) exposing .connect() and .disconnect().
            signal_name:
                The signal name (e.g., "pad-added", "on-negotiation-needed").
            handler:
                Python callable invoked when the signal is emitted.
            *args / **kwargs:
                Additional user data passed to GObject.connect().

        Returns:
            The handler_id returned by obj.connect(...). This ID is required
            to disconnect the handler later.

        Notes:
            Tracking handler IDs is essential because disconnecting is done by ID,
            and we often cannot rely on object lifetime alone during teardown.
        """
        handler_id = obj.connect(signal_name, handler, *args, **kwargs)

        self._connections.append(
            _SignalConnection(
                obj=obj,
                handler_id=handler_id,
                signal_name=signal_name,
            )
        )

        return handler_id

    def block_all(self) -> None:
        """
        Block all tracked signal handlers.

        This prevents callbacks from being invoked while the pipeline
        is shutting down, reducing the risk of races between signal
        emission (possibly from streaming threads) and teardown.

        Notes:
            - Does NOT remove the connections from the internal list.
            - Safe to call multiple times.
            - Tolerates partially disposed objects.
            - This is the teardown mechanism the GStreamer client relies on:
              block the handlers, then let the pipeline's ``set_state(NULL)``
              disposal disconnect them. Do NOT call :meth:`disconnect_all`
              after disposal — see its docstring.
        """
        # Block in reverse order for symmetry with disconnect_all
        for conn in reversed(self._connections):
            try:
                # Extra safety: ensure object still looks like a valid GObject
                gp = getattr(conn.obj, "__gpointer__", None)
                if not gp:
                    continue

                # Only block if still connected (when available)
                if hasattr(GObject, "signal_handler_is_connected"):
                    if not GObject.signal_handler_is_connected(
                        conn.obj, conn.handler_id
                    ):
                        continue

                GObject.signal_handler_block(conn.obj, conn.handler_id)

            except Exception as exc:
                logger.debug(
                    "Failed to block signal '%s' (handler_id=%s): %s",
                    conn.signal_name,
                    conn.handler_id,
                    exc,
                )

    def disconnect_all(self) -> None:
        """
        Disconnect all tracked signal handlers.

        Only safe to call while every tracked object is still alive. Once the
        owning pipeline has been disposed (``set_state(NULL)``), GStreamer has
        already dropped these handlers and finalized the elements; calling
        ``.disconnect()`` then double-unrefs a finalized GObject, which is a
        native GStreamer-CRITICAL ("ref_count > 0" assertion) / heap
        corruption — NOT a Python exception, so the ``try`` below cannot catch
        it, and it surfaces as a non-deterministic segfault. The client teardown
        therefore relies on :meth:`block_all` plus pipeline disposal instead of
        calling this method.

        Safety:
            - Disconnects in reverse order (LIFO), which is generally safer when
              later connections depend on earlier ones (or when callbacks chain).
            - Tolerates partially destroyed objects/handlers: GObject may raise if
              the object is already finalized or the handler ID is invalid.
        """
        # Disconnect in reverse order to be safer with chained callbacks.
        while self._connections:
            conn = self._connections.pop()
            try:
                conn.obj.disconnect(conn.handler_id)
            except Exception as exc:
                # Common cases:
                #   - object already disposed
                #   - handler already disconnected
                #   - GI proxy raising due to invalid underlying pointer
                logger.warning(
                    "Failed to disconnect signal '%s' (handler_id=%s): %s",
                    conn.signal_name,
                    conn.handler_id,
                    exc,
                )
