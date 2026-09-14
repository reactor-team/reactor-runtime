"""The application authoring base, :class:`ReactorApp`, and the step types.

What an author subclasses. Declaring the subclass assembles its
:class:`ModelContract` from one traversal of the class. :class:`StepOutcome` is
what one call to ``generate()`` did, built by the runtime and read by the
application.
"""

from reactor_runtime.interface.app.outcome import StepOutcome
from reactor_runtime.interface.app.reactor_app import ReactorApp

__all__ = ["ReactorApp", "StepOutcome"]
