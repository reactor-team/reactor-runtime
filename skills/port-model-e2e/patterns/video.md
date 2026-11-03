# Video in

A video reaches a model in one of two ways, chosen by how it arrives, not by its role.

| Transport | Use it when | Declared as |
|---|---|---|
| **A live input track** | the client sends frames as they happen (a camera, a screen) | a `Video` field of a `MediaInput` subclass, as `examples/echo/echo.py` declares `webcam`; named `video_in` for the video to rewrite |
| **An uploaded file** | the client picks a file | `set_source_video(video: UploadedFile \| None)` for the video to rewrite; a clip that says what to look or move like goes through `set_references` ([image.md](image.md)) |

## On arrival

For both, bring the frames to what the model takes, the way the source preprocesses its input: its
resolution (resize and crop, or pad, as the source does) and its frame rate (resampled by
timestamp). State both in the description.

**A live track** is read per chunk:

- each chunk reads exactly the input frames it rewrites, chosen by timestamp, not first-come;
- `start()` is refused until the frames of the first chunk have arrived (`waiting_for_video`), and
  `state_update` says how many have;
- when the stream stalls mid-run, the model waits for the next chunk's frames; it does not repeat or
  invent frames, and `state_update` reports the stall;
- the latency from an input frame's arrival to its output frame is measured on the served model and
  reported.

**An uploaded file** is decoded off the event loop. Refuse what is not a video (`unsupported_media`)
or cannot be decoded (`undecodable_video`). A video longer than the model can take is refused with
`video_too_long`, giving the limit in frames and seconds; it is never cut silently. A shorter one
follows the source (stop at its end, or loop), and the description says which.

## Kind

Read from the source, per input. A video the model encodes once before the generation loop is session setup. One it reads again at every chunk is a live control: a new file
replaces the old one at the next chunk boundary, and a live track changes with every frame by
nature.
