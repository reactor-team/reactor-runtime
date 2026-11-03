# Pointer

## `set_pointer(x, y, down)`

A touch or a drag on the video: `x, y` in 0…1 over the output frame, origin at the top left; `down`
true while pressed. A pointer acts on what is playing, so it is changed during play (live). Each call is the pointer's state now; a drag is a sequence of calls. The model
reads the states that arrived during a chunk at that chunk's boundary.

- **One shape per chunk.** The states of one chunk are brought to the fixed point count the model was
  traced at (resampled along the path, or padded with a mask), so no drag length adds a compile shape.
- **Converted to the model's frame** the way the source normalises coordinates; the description states
  the source's convention and the conversion.
- A model that takes several fingers takes `set_pointer(points)`, a list of `(x, y, down)`, padded to a
  fixed count the same way.
