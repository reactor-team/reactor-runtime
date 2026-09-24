"""Failures a :class:`~reactor_runtime.distributed.DistributedRunner` raises to its caller.

A worker's own exception is never wrapped: when every rank raises the same
error, the caller receives rank 0's instance unchanged. The types here cover
what the runner itself detects.
"""

from __future__ import annotations


class WorkerCrashed(RuntimeError):  # noqa: N818 (named for the event a caller catches)
    """A worker process died without answering.

    The message names the rank and its exit code. The group is unusable after
    this; ``shutdown()`` ends it.
    """


class WorkerTimeout(TimeoutError):  # noqa: N818 (named for the event a caller catches)
    """A call did not get an answer from every rank within its timeout.

    The processes are alive but one of them is stuck, usually inside a
    collective a peer abandoned. The group is unusable after this.
    """


class RankDesync(RuntimeError):  # noqa: N818 (named for the state a caller catches)
    """Some ranks raised and others did not, or ranks raised different types of error.

    The message names the first failing rank and carries its exception as
    ``__cause__``. Rank 0's result, if any, came out of a collective a failing
    rank was meant to contribute to, so it is not returned. The group refuses
    further calls.
    """


class SharedSlotAllocationFailed(MemoryError):  # noqa: N818 (named for the event a caller catches)
    """A shared-memory block could not be created or grown.

    The message states the size that was asked for and, where the host exposes
    it, the size of ``/dev/shm``. Docker's default is 64 MB; ``--shm-size`` on
    ``docker run`` and a ``medium: Memory`` ``emptyDir`` on a pod raise it.
    """
