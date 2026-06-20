"""Engine-facing internals — buffers, the core, and the bridge.

These are the moving parts the runtime wires around a model: the rate-controlled
output buffer, the readable input buffers, the :class:`ReactorCore` run loop, and
the :class:`ModelBridge` door. They are exposed for runtime integration, not as
part of the curated authoring API, so a model author rarely imports them.
"""
