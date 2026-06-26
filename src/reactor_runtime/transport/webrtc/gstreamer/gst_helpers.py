from typing import Any

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst, GObject
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger

logger = get_logger(__name__)


def structure_field_uint(structure: Gst.Structure, field_name: str, value: int) -> None:
    """
    Set a structure field as ``G_TYPE_UINT``. ``rtprtxsend`` maps use
    ``g_value_get_uint`` / ``gst_structure_get_uint`` on these structures;
    plain Python strings or wrongly-typed values break the maps.
    """
    v = GObject.Value()
    v.init(GObject.TYPE_UINT)
    v.set_uint(int(value) & 0xFFFFFFFF)
    structure.set_value(field_name, v)


def try_set_property(element: Gst.Element, prop: str, value: Any) -> bool:
    """
    Best-effort property setter for any Gst.Element.

    Behavior:
        - Only attempts to set property if it exists.
        - Silently ignores invalid values.

    Why this matters:
        Different elements (encoders, decoders, payloaders) expose different
        properties. Some properties exist only in specific GStreamer versions
        or plugin builds. This avoids hard failures when running on slightly
        different runtime environments.
    """
    if element.find_property(prop) is None:
        return False

    try:
        element.set_property(prop, value)
        return True
    except (TypeError, ValueError) as e:
        logger.warning(
            "Failed to set GStreamer element property",
            prop=prop,
            value=value,
            element=element.get_name(),
            error=str(e),
        )
        return False


def add_many(
    bin: Gst.Bin, *elements: Gst.Element, sync_with_parent: bool = False
) -> None:
    """
    Add multiple elements to a Gst.Bin.

    Parameters
    ----------
    bin:
        Target Gst.Bin where elements will be added.
    *elements:
        One or more Gst.Element instances to be added.
    sync_with_parent:
        If True, calls `sync_state_with_parent()` after adding each element.
        This is useful when the parent bin/pipeline is already in a
        non-NULL state (e.g., PLAYING), ensuring the new element follows
        the parent's current state immediately.

    Notes
    -----
    - This helper does not link elements; it only adds them to the bin.
    - Elements must not already belong to another bin.
    - If the parent bin is already running and `sync_with_parent` is False,
      the element will remain in NULL state until explicitly set.
    """
    for e in elements:
        # Gst.Bin.add() returns the C gboolean on most PyGObject builds, but some
        # return None even on success — so verify the element actually landed in
        # this bin rather than trusting the binding's return value. Checking
        # `is not bin` (not just `is None`) also catches a failed add of an element
        # that already belongs to another bin, where get_parent() returns that
        # other bin.
        bin.add(e)
        if e.get_parent() is not bin:
            raise RuntimeError(f"Failed to add {e.get_name()} to {bin.get_name()}")
        if sync_with_parent:
            e.sync_state_with_parent()


def link_many(*elements: Gst.Element) -> None:
    """
    Link multiple Gst.Element objects sequentially.

    Elements are linked in order:
        elements[0] -> elements[1] -> elements[2] -> ...

    Parameters
    ----------
    *elements:
        Two or more Gst.Element instances to link in sequence.

    Raises
    ------
    RuntimeError:
        If any link operation fails.

    Notes
    -----
    - Uses `Gst.Element.link()`, which links compatible src/sink pads.
    - Fails fast on the first unsuccessful link.
    - Does not perform caps negotiation or pad selection explicitly.
    - For dynamic/request pads, use explicit pad linking instead.
    """
    for a, b in zip(elements, elements[1:]):
        if not a.link(b):
            raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")


def link_pads(src_pad: Gst.Pad, sink_pad: Gst.Pad) -> None:
    """
    Link two explicit Gst.Pad objects.

    Parameters
    ----------
    src_pad:
        Source pad (must be of type SRC).
    sink_pad:
        Sink pad (must be of type SINK).

    Raises
    ------
    RuntimeError:
        If the pad link operation does not return Gst.PadLinkReturn.OK.

    Notes
    -----
    - Intended for dynamic pads (e.g., request pads, sometimes pads).
    - Caller is responsible for ensuring caps compatibility.
    - Does not attempt to unlink existing connections.
    - Should be called from the GLib/GStreamer main context thread.
    """
    if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
        raise RuntimeError(
            f"Failed to link {src_pad.get_name()} -> {sink_pad.get_name()}"
        )


def make_element(factory: str, name: str | None) -> Gst.Element:
    """
    Make a GStreamer element from a factory name with proper error handling.

    Args:
        factory:
            The GStreamer element factory name
            (e.g. "x264enc", "rtph264pay", "queue", "videoconvert").

        name:
            Optional instance name inside the pipeline.
            Naming elements is useful for:
                - Debugging (GST_DEBUG logs)
                - Retrieving elements via get_by_name()
                - Pipeline graph inspection (dot files)

    Returns:
        A successfully instantiated Gst.Element.

    Raises:
        RuntimeError if the factory does not exist or the element
        could not be created.

    Notes:
        Gst.ElementFactory.make() returns None when:
            - The plugin is not installed
            - The factory name is incorrect
            - The plugin failed to load
        Failing fast here avoids obscure link or state-change errors later.
    """

    # Attempt to create element from factory.
    element = Gst.ElementFactory.make(factory, name)

    # Fail fast if creation failed.
    if not element:
        raise RuntimeError(f"Failed to create {factory} [name={name}]")

    return element
