"""The application authoring base, :class:`ReactorApp`.

What an author subclasses. Declaring the subclass assembles its
:class:`ModelContract` from one traversal of the class.
"""

from reactor_runtime.interface.app.reactor_app import ReactorApp

__all__ = ["ReactorApp"]
