# Video in

This page fixes how a video reaches a served model. What the model does with it (its role) and
when it may be set (its timing) are declared per input in the model's input table: a model may
take a video to rewrite chunk by chunk, a reference video it keeps to, a clip whose motion it
follows, or several of these at once.

| Role, for example | What the model does with it |
|---|---|
| `source` | rewrites it: each output chunk is the matching input chunk, transformed |
| `reference_video` | keeps to its appearance or style |
| `motion_reference` | follows its motion |

## Two transports

| Transport | Use it when | Declared as |
|---|---|---|
| **A live input track** | the client sends frames as they happen (a camera, a screen) and the model consumes them in time | a `Video` field of a `MediaInput` subclass, named for its role, as `examples/echo/echo.py` declares `webcam` |
| **An uploaded clip** | the client picks a file | `set_<role>(video: UploadedFile \| None)`; `None` removes it |

Choose by how the input arrives, not by the role: a reference can be a file or a stream, and a
source to rewrite can be a recorded file played into the model.

## On arrival

For both transports, bring the frames to what the model takes, the way the source preprocesses its
input: its resolution (resize and crop, or pad, as the source does) and its frame rate (resampled by
timestamp). State both in the description.

**An uploaded clip** is decoded off the event loop. Refuse what is not a video (`unsupported_media`)
or cannot be decoded (`undecodable_video`). A clip longer than the model can take is refused with
`video_too_long`, giving the limit in frames and seconds; it is never cut silently. Bring it to the
fixed frame count the model was traced at, the way the source does (pad, loop or stop at its end),
and say which in the description.

**A live track** is read per chunk:

- each chunk reads exactly the input frames it consumes, by timestamp, not first-come;
- a step is refused until the frames of the first chunk have arrived (`waiting_for_video`), and the
  state says how many have;
- when the stream stalls mid-run, the model waits for the next chunk's frames; it does not repeat or
  invent frames, and the state reports the stall;
- the latency from an input frame's arrival to its output frame is measured on the served model and
  reported.

## Timing

Read it from the source, per input. A video the model encodes once before the generation loop is
set before play. A video it reads again at every chunk (a reference re-injected into each chunk's
attention, or a source consumed chunk by chunk) can change during play, applying at the next chunk
boundary: a new uploaded clip replaces the old one there, and a live track changes with every frame
by nature. When an input video and a first frame both define the start, the
[README](README.md) precedence table says which wins.
