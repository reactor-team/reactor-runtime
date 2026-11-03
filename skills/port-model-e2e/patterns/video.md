# Video in

## `video_in`, the input track of a video-to-video model

Declared as a `Video` field named `video_in` on a `MediaInput` subclass, the way
`examples/echo/echo.py` declares its `webcam`. The client publishes frames to it over the connection
that carries the output, and each output chunk is generated from the input chunk at the same moment.

- **Each chunk reads exactly the input frames it rewrites**, chosen by timestamp at the model's frame
  rate, and fitted to its resolution as the source preprocesses its input.
- **`start()` is refused until the frames of the first chunk have arrived** (`waiting_for_video`),
  and `state_update` says how many have.
- **When the stream stalls mid-run, the model waits** for the next chunk's frames. It does not repeat
  or invent frames, and `state_update` reports the stall.
- **Latency is measured and reported**: from an input frame's arrival to its output frame, on the
  served model.

What conditions the rewrite (the prompt, the references, a pointer) is its own command on its own
page.

## A video as a reference

A clip that says what to look or move like is a reference: it goes through `set_references` with its
images ([image.md](image.md)). Brought to the frame count and resolution the source takes; a clip
longer than the model can take is refused with `video_too_long`, giving the limit in frames and
seconds, never cut silently.
