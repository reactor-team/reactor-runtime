"""Confirm on the runner why the loopback stops inside ``create_peer_connection``.

The loopback stalls there and never returns: the last log line before the
faulthandler dump is the one immediately preceding the call, and the only other
thread in the dump has a Python thread state with no Python frame — a native
thread waiting to enter Python.

That is a deadlock between the GIL and libwebrtc's signalling thread. The
installed ``reactor_webrtc`` holds the GIL while it creates a peer connection,
and creating one is a proxy call: it is posted to the signalling thread and the
caller blocks until it finishes. The loopback's stand-in client is gathering ICE
candidates at that exact moment, and each candidate is delivered by the
signalling thread into a Python callback. So the creating thread waits for the
signalling thread while the signalling thread waits for the GIL, and neither can
be interrupted, because the creating thread is in native code and never yields.

This test drives that collision directly, with no tracks, negotiation, media or
frame metadata involved — a candidate callback parked in Python, then one call.
It runs in a subprocess, because the deadlock cannot be observed from inside the
process that is stuck: a watchdog thread would need the GIL the stuck caller
holds. A timeout turns the hang into a failure.

Failing here means the deadlock is in the binding rather than in anything this
package does with it. It is a probe, not a regression test — the fix and its
permanent test belong to ``reactor-webrtc``, and this file goes away with them.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("reactor_webrtc")

_TIMEOUT_S = 45

_PROBE = textwrap.dedent("""
    import asyncio, threading, time
    import reactor_webrtc as rw

    factory = rw.PeerConnectionFactory()

    # Fires on the signalling thread. The sleep releases the GIL, so the main
    # thread is free to take it, and the callback then needs it back to return
    # into native code — the state a caller holding the GIL traps it in.
    entered = threading.Event()

    def hold_the_signalling_thread(_candidate):
        entered.set()
        time.sleep(2.0)

    async def main():
        observer = rw.PeerConnectionObserver()
        observer.on_ice_candidate = hold_the_signalling_thread
        first = factory.create_peer_connection(rw.RtcConfiguration(), observer)
        first.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.RecvOnly)
        offer = await first.create_offer()
        await first.set_local_description(offer)
        assert entered.wait(30), "no ICE candidate was delivered"
        factory.create_peer_connection(rw.RtcConfiguration(), rw.PeerConnectionObserver())

    asyncio.run(main())
    print("returned")
""")


def test_creating_a_peer_connection_returns_while_a_candidate_callback_is_in_flight() -> None:
    """Create a peer connection with the signalling thread parked in Python."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"create_peer_connection did not return in {_TIMEOUT_S}s: the binding holds "
            f"the GIL across the dispatch to the signalling thread, which is waiting "
            f"for the GIL. This is what stalls the loopback."
        )
    assert done.returncode == 0, f"probe exited {done.returncode}:\n{done.stderr}"
    assert "returned" in done.stdout
