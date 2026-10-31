# Echo example

Loop the client's webcam and microphone back, with a real-time video effect.
Two inbound tracks, two outbound tracks, four settings, no weights. It runs on
a CPU and needs nothing installed but the CLI and Docker.

This is the reference for a model that consumes the client's media, in a
streamed session or a stepped one, written as two halves:

- `echo_model.py` is the model. It applies one of seven effects to one frame
  and draws a caption over it, behind `load`, `generate`, and `reset`. It holds
  no state between frames, imports nothing from `reactor_runtime`, and runs
  in a notebook.
- `echo.py` is the application. It declares the tracks and the settings,
  refuses a step in `process_input` until a webcam frame has arrived and
  drains the microphone alongside it, forwards the frame in `generate`, and
  pairs the result with its audio and metadata in `process_output`.

The two meet on two dataclasses, `EchoInput` and `EchoResult`. Audio never
enters the model: it is application plumbing, read next to the frame and
emitted next to the result.

`process_output` returns `None` until `burst` frames have gathered, then emits
them as one batch. That is how a model that produces in bursts drives the same
loop, and it makes the transport's pacing observable in a live session. A
resolution change mid-burst ends the batch early, because a batch is one
array.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model and
defines the image in its `build:` block, and `requirements.txt` lists the
model's own dependencies.

```sh
cd examples/echo
reactor build --no-dockerfile
reactor run
```

`--no-dockerfile` renders the image from `reactor.yaml`'s `build:` block.
`build.runtime_version` must match the `reactor-runtime==<version>` pin in
`requirements.txt`; bump both together to upgrade.

`reactor run` serves WebRTC signaling on `http://localhost:8080`. Connect from
the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) (pick **Local
(Direct)**), allow the camera and the microphone, and the processed stream
comes back.

## Commands

Every public field on `EchoState` is a generated `set_<field>` command. The
app writes no handler by hand.

- `set_effect` one of `none`, `grayscale`, `sepia`, `edges`, `invert`, `blur`,
  `pixelate`.
- `set_intensity` `0` leaves the frame as is, `1` applies the effect in full;
  in between is a mix.
- `set_caption` text drawn over every frame, up to 200 characters; empty
  clears it. It is free text, so the schema marks it for moderation.
- `set_burst` frames batched into one emit, `1` to `120`. `1` sends every
  frame as it is produced; `120` is four seconds of media at 30 fps.

Every setting is held: the model reads the current value of each field on
every step. Whatever a client attaches to a webcam frame as metadata comes
back on the frame it produced; in a batch, a frame that carried nothing comes
back with an empty entry.

## Stepped session

The application writes the three step hooks and leaves `run()` to the
runtime, so the same code serves a stepped session, where each client request
runs one step instead of streaming. Start the session with
`"generation": "stepped"`. A request carries its webcam frames, and any
microphone audio, in `media_input`, and sets any of the four settings through
`state`. Its result carries the newest of those frames on `main_video`, with
the effect applied and the metadata it arrived with, and the audio on
`main_audio`. The settings hold from one step to the next.

A step that carries no webcam frame is refused: `process_input` answers
`REFUSED` with `waiting for a webcam frame`, and the step takes no number.
With `burst` above `1`, a step that only gathers its frame carries no media,
and the step that fills the batch carries every frame of it.

With the Python SDK, from the release that adds stepped sessions
(`Reactor.step()`), against the `reactor run` above:

```python
from reactor_sdk import Reactor

reactor = Reactor("echo", local=True, generation="stepped")
await reactor.connect()

webcam = await reactor.track("webcam").publish()
webcam.push_frame(frame, user_data=b'{"seq": 1}')   # buffered for the next step

result = await reactor.step(state={"effect": "sepia", "caption": "hello"})
[out] = result.track("main_video").frames
out.array                  # the frame pushed above, in sepia, captioned
out.metadata               # b'{"seq": 1}'
```
