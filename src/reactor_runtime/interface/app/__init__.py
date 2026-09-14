"""The application authoring base, :class:`ReactorApp`, and its typed state.

What an author subclasses. Declaring the subclass assembles its
:class:`ModelContract` from one traversal of the class; declaring
``state: MyState`` on it turns every public :class:`InputState` field into a
``set_<field>`` command.
"""

from reactor_runtime.interface.app.input_state import InputState
from reactor_runtime.interface.app.outcome import StepOutcome
from reactor_runtime.interface.app.reactor_app import ReactorApp

__all__ = ["InputState", "ReactorApp", "StepOutcome"]
