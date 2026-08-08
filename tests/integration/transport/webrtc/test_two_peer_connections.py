"""Isolate creating more than one peer connection on the shared media engine.

The loopback stalls in ``create_peer_connection`` on the CI runner, and the thread
dump puts it on the *second* one: the client's peer connection is created and even
produces an offer before the runtime's own call never returns. Locally the same
sequence runs two hundred times over without slowing down.

Everything else the loopback does — tracks, negotiation, media, frame metadata —
sits after that point, so none of it can be the cause. This file removes all of
it. If the second call stalls here, the fault is in the media engine or the
platform rather than in anything this package does with it, which is the
difference between a bug to report upstream and a bug to fix here.

No ICE servers are configured (``WebRtcConfig.ice_servers`` is empty by default),
so candidate gathering has nothing external to reach.
"""

from __future__ import annotations

import logging

import reactor_webrtc as rw

from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.peer import _build_rtc_config, _get_factory

logger = logging.getLogger(__name__)


def test_two_peer_connections_on_one_factory() -> None:
    """Create two peer connections on the shared factory, and nothing else.

    Logs before and after each call so a stall names which one it was, since a
    hang leaves only what was already printed.
    """
    factory = _get_factory()
    config = _build_rtc_config(WebRtcConfig())

    logger.info("creating peer connection 1")
    first = factory.create_peer_connection(config, rw.PeerConnectionObserver())
    logger.info("peer connection 1 created")

    logger.info("creating peer connection 2")
    second = factory.create_peer_connection(config, rw.PeerConnectionObserver())
    logger.info("peer connection 2 created")

    assert first is not None
    assert second is not None


def test_a_third_peer_connection_after_dropping_the_first() -> None:
    """Create, drop, and create again, to separate concurrency from reuse.

    If two live peer connections are the problem, this passes; if creating a
    second one at all is, it stalls the same way. The two say different things
    about where to look.
    """
    factory = _get_factory()
    config = _build_rtc_config(WebRtcConfig())

    logger.info("creating peer connection A")
    first = factory.create_peer_connection(config, rw.PeerConnectionObserver())
    logger.info("peer connection A created; dropping it")
    del first

    logger.info("creating peer connection B")
    second = factory.create_peer_connection(config, rw.PeerConnectionObserver())
    logger.info("peer connection B created")

    assert second is not None


async def test_a_second_peer_connection_after_the_first_is_set_up() -> None:
    """Build up the first peer connection the way a client does, then create a second.

    Bare peer connections coexist fine, so what stalls the loopback is the state
    the first one is carrying by the time the second is created, not the count.
    This adds that state one piece at a time — transceivers, a track, a data
    channel, an offer applied locally — and creates the second peer connection at
    the end.

    Every step logs before and after, so a stall names the piece that caused it:
    the last line printed is the step that never finished.
    """
    factory = _get_factory()
    config = _build_rtc_config(WebRtcConfig())

    logger.info("step 1: create the first peer connection")
    first = factory.create_peer_connection(config, rw.PeerConnectionObserver())
    logger.info("step 1 done")

    logger.info("step 2: add a recvonly video transceiver")
    first.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.RecvOnly)
    logger.info("step 2 done")

    logger.info("step 3: add a recvonly audio transceiver")
    first.add_transceiver(rw.MediaKind.Audio, rw.TransceiverDirection.RecvOnly)
    logger.info("step 3 done")

    logger.info("step 4: create a video track and send it on a new transceiver")
    track = factory.create_video_track("probe-cam")
    sender = first.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.SendOnly)
    sender.set_track(track)
    logger.info("step 4 done")

    logger.info("step 5: create a data channel")
    first.create_data_channel("probe-data")
    logger.info("step 5 done")

    logger.info("step 6: create the offer")
    offer = await first.create_offer()
    logger.info("step 6 done")

    logger.info("step 7: apply the offer locally (starts ICE gathering)")
    await first.set_local_description(offer)
    logger.info("step 7 done")

    logger.info("step 8: create the second peer connection")
    second = factory.create_peer_connection(config, rw.PeerConnectionObserver())
    logger.info("step 8 done")

    assert second is not None


async def test_a_second_peer_connection_whose_observer_carries_callbacks() -> None:
    """Repeat the build-up, but give the second peer connection real callbacks.

    The one thing the passing variants leave out is the observer the runtime
    actually uses: five bound Python callables rather than an empty observer. With
    callbacks registered the native side can call into Python while the creating
    thread still holds the GIL, which is the shape of a deadlock and would explain
    a stall with no Python frame on the other thread.

    Same eight steps, same logs. Step 8 is the only line that differs from the
    variant that passes, so a stall here names the observer as the cause and
    nothing else.
    """
    factory = _get_factory()
    config = _build_rtc_config(WebRtcConfig())

    logger.info("step 1: create the first peer connection")
    first = factory.create_peer_connection(config, rw.PeerConnectionObserver())
    logger.info("step 1 done")

    logger.info("step 2-3: add recvonly video and audio transceivers")
    first.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.RecvOnly)
    first.add_transceiver(rw.MediaKind.Audio, rw.TransceiverDirection.RecvOnly)
    logger.info("step 2-3 done")

    logger.info("step 4: create a video track and send it on a new transceiver")
    track = factory.create_video_track("probe-cam-cb")
    sender = first.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.SendOnly)
    sender.set_track(track)
    logger.info("step 4 done")

    logger.info("step 5: create a data channel")
    first.create_data_channel("probe-data-cb")
    logger.info("step 5 done")

    logger.info("step 6-7: create the offer and apply it locally")
    offer = await first.create_offer()
    await first.set_local_description(offer)
    logger.info("step 6-7 done")

    seen: list[str] = []
    observer = rw.PeerConnectionObserver()
    observer.on_connection_state_change = lambda state: seen.append(f"state:{state}")
    observer.on_ice_gathering_change = lambda state: seen.append(f"gathering:{state}")
    observer.on_ice_candidate = lambda candidate: seen.append("candidate")
    observer.on_track = lambda kind, remote: seen.append(f"track:{kind}")
    observer.on_data_channel = lambda channel: seen.append("channel")

    logger.info("step 8: create the second peer connection, observer with callbacks")
    second = factory.create_peer_connection(config, observer)
    logger.info("step 8 done")

    assert second is not None
