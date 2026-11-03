# Video in

A video reaches a model in one of two ways, chosen from what the source consumes.

| The source | Transport | Role name, for example |
|---|---|---|
| rewrites each input chunk into the matching output chunk, in time | a **live input track** | `source` |
| needs the whole clip before it starts, or reads ahead of the output | an **uploaded clip**, set before play | `source_clip` |

A model may take both (a live stream plus a reference clip). Each is its own input row.

## A live input track

Declare it as a `Video` field of a `MediaInput` subclass, named for its role, the way
`examples/echo/echo.py` declares `webcam`. The client publishes frames to it over the same
connection that carries the output.

- **Per chunk, read exactly the frames that chunk rewrites**: the number of input frames the source
  takes per output chunk, at the model's frame rate. Frames that arrived in between are resampled to
  that count by timestamp, not taken first-come.
- **Fit every frame to the model's resolution on arrival**, as the source preprocesses its input.
- **Refuse a step until enough frames have arrived** for the first chunk (`waiting_for_video`), and
  say in the state how many have arrived. When the stream stalls mid-run, the model waits for the
  frames of the next chunk; it does not repeat or invent frames, and the state reports the stall.
- **Latency is reported**, not tuned away: the time from an input frame's arrival to its output
  frame, measured on the served model.

## An uploaded clip

`set_<role>(video: UploadedFile)`, set before play. Decode off the event loop, refuse what is not a
video (`unsupported_media`) or cannot be decoded (`undecodable_video`), and bring it to the model's
resolution and frame rate the way the source does. A clip longer than the model can take is
refused with `video_too_long`, giving the limit in frames and seconds; never cut silently. A shorter
clip follows the source (stop at its end, or loop), and the description says which.

## Timing and other inputs

The live track is changed during play by nature: every chunk reads new frames. Everything that
conditions how the frames are rewritten (prompt, references) is its own input on its own page. When
the input video and a first frame both define the start, the [README](README.md) precedence table
says which wins.
