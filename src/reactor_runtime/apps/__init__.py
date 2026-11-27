"""Base classes built on :class:`~reactor_runtime.ReactorApp` for one kind of model each.

A class here owns the loop and the commands for its kind of model, and leaves
the model to write ``load()`` and ``generate()``. Each class keeps serving
streamed and stepped sessions alike, so switching a session between them needs
no change in the model.

* :mod:`reactor_runtime.apps.robot`: robot policies that turn stamped camera
  frames and robot state into chunks of actions.
"""
