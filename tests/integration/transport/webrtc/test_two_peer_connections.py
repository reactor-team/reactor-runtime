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
