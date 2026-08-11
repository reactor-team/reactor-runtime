# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

"""Typed failures surfaced by :class:`reactor_runtime.distributed.WorkerGroup`."""


class WorkerError(RuntimeError):
    """A worker rank reported an error.

    Any rank may report (a non-leader crasher must not be masked by a
    leader stuck in a collective). Errors raised from ``start_session``
    are recoverable — the workers stay alive and the controller may
    retry; errors from ``generate_chunk`` tear the group down.
    """


class WorkerCrashed(RuntimeError):
    """A worker process died without reporting (segfault, OOM-kill).

    The group is unusable; the pipeline should let the exception
    propagate so the runtime fails fast and the pod restarts.
    """
