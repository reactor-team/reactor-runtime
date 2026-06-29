"""The passthrough example: a minimal model that emits a solid colour frame.

Run it from this directory with ``python -m reactor_runtime.serve`` — the
``reactor.yaml`` here points the runtime at the model.
"""

from examples.passthrough.passthrough import Passthrough

__all__ = ["Passthrough"]
