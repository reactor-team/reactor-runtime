from reactor_runtime.core import ConnId
from reactor_runtime.runner.offer_epochs import OfferEpochs


def test_an_unstamped_offer_is_never_stale() -> None:
    epochs = OfferEpochs()
    epochs.session_started()
    assert not epochs.consume(ConnId(1))


def test_an_offer_stamped_in_the_live_session_is_not_stale() -> None:
    epochs = OfferEpochs()
    epochs.session_started()
    epochs.stamp(ConnId(1))
    assert not epochs.consume(ConnId(1))


def test_an_offer_stamped_in_an_earlier_session_is_stale() -> None:
    epochs = OfferEpochs()
    epochs.session_started()
    epochs.stamp(ConnId(1))
    epochs.session_started()
    assert epochs.consume(ConnId(1))


def test_consume_takes_the_stamp_with_it() -> None:
    epochs = OfferEpochs()
    epochs.session_started()
    epochs.stamp(ConnId(1))
    epochs.session_started()
    assert epochs.consume(ConnId(1))
    # The stamp is gone, so a second look falls back to "unstamped".
    assert not epochs.consume(ConnId(1))


def test_a_reoffer_restamps_into_the_live_session() -> None:
    epochs = OfferEpochs()
    epochs.session_started()
    epochs.stamp(ConnId(1))
    epochs.session_started()
    epochs.stamp(ConnId(1))
    assert not epochs.consume(ConnId(1))
