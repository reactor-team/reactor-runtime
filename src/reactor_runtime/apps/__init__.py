"""Base classes built on :class:`~reactor_runtime.ReactorApp` for one kind of model each.

A class here owns the loop and the commands for its kind of model, and leaves
the model to write ``load()`` and ``generate()``. Each class keeps serving
streamed and stepped sessions alike, so switching a session between them needs
no change in the model.

* :mod:`reactor_runtime.apps.director`: models that build a whole shot of
  video in one call and play it when a client asks.
"""
