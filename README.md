<div align="center">

<img src="assets/banner.png" alt="Reactor Runtime" width="100%" />

**Build real-time AI models in Python.**

[📖 Documentation](https://docs.reactor.inc/deploy/overview) · [🚀 Quickstart](https://docs.reactor.inc/deploy/development/quickstart) · [🌐 Reactor](https://reactor.inc)

</div>

---

Reactor Runtime turns an inference pipeline into a real-time, interactive media and data stream. You write `load()` and `generate()`; the runtime drives them one step at a time and handles the session lifecycle, the WebRTC media transport, and the wire protocol that connects clients to your model. Viewers watch frames as they are generated and change what the model is doing mid-stream, with no restart and no re-queue.

## Highlights

- 📡 **Real-time streaming.** Frames reach clients over WebRTC as your model produces them, not after a whole video is done. The runtime paces playout from the time each step took, so a model needs no rate limiter of its own.
- 🎮 **Live interaction.** Clients send commands mid-generation: change a prompt, move a camera, adjust a parameter. The next frame reflects it.
- 🔌 **No transport code.** You never import a WebRTC library, manage a WebSocket, or encode video. The runtime ships its own media engine as a wheel, so a plain Python container is all a model needs.
- ✅ **Typed, validated commands.** Declare the commands your model accepts with standard Python types and constraints. The runtime validates every payload before your handler runs and compiles the surface into an OpenAPI schema that drives typed client SDKs.
- 🔎 **Traceable logs.** `get_logger()` writes structured records — readable `key=value` in a terminal, JSON for a log pipeline. Every record a session writes carries that session's id automatically, so one filter recovers everything a single run logged.
- 📦 **One container, anywhere.** The `reactor` CLI scaffolds a workspace, builds a small image, and runs it locally. The same image deploys to [Reactor](https://reactor.inc)'s GPU cloud unchanged.

## How it works

You ship one `ReactorApp` subclass: the application the runtime drives and the client talks to. Declare the media it sends and the state a client can set, load your weights once, and write what one step of generation does:

```python
from pathlib import Path

from reactor_runtime import InputField, InputState, Output, ReactorApp, Video


class MyState(InputState):
    prompt: str = InputField(default="a sunny meadow", moderate=True, description="Scene to render.")
    paused: bool = InputField(default=False, description="Hold generation on the last frame.")


class MyOutput(Output):
    main_video: Video


class MyModel(ReactorApp):
    state: MyState

    def load(self, config_path: Path | None) -> None:
        self.pipe = load_my_pipeline()

    def generate(self, input: MyState) -> MyOutput:
        return MyOutput(main_video=self.pipe.forward(prompt=input.prompt))
```

That is a complete application. The runtime calls `generate()` in a loop for as long as someone is watching and emits what it returns. Every public field on `MyState` is a command the client can send: here `set_prompt` and `set_paused`, validated from the fields, and the next step reads the new values.

State and message classes can be subclassed. A subclass keeps every field of its parents and adds its own, so `class SeededState(MyState): seed: int = InputField(default=0)` exposes `set_prompt`, `set_paused`, and `set_seed`. A field declared again in the subclass replaces the parent's, default and constraints included. The fields a subclass adds are keyword-only in its constructor, which lets a required field follow an inherited one that has a default. A private field (leading underscore) always needs a default, because the state is built from defaults when a session starts.

A step is three calls, and `generate()` is the one you must write. `process_input()` runs before it, reading `self.state` and the media tracks, and decides whether a step can happen: return the input `generate()` gets, or raise `ApplicationError("reason")` to skip the step without touching the model. `process_output(outcome)` runs after it, with the result or the error, and returns the media to emit; send a message from there with `await self.send()` and it reaches the client before the step's frames. Both have defaults, so the model above writes neither.

```python
from reactor_runtime import ApplicationError, MessageField, ModelMessage, StepOutcome


class FrameReady(ModelMessage):
    prompt: str = MessageField(description="The prompt this frame was rendered from.")


class MyModel(ReactorApp):
    state: MyState

    async def process_input(self) -> MyState:
        if self.state.paused:
            raise ApplicationError("paused")
        return self.state

    def generate(self, input: MyState) -> MyOutput: ...

    async def process_output(self, outcome: StepOutcome) -> Output | None:
        if outcome.error is not None:
            raise outcome.error
        await self.send(FrameReady(prompt=self.state.prompt))
        return outcome.to_output()
```

`generate()` fails by raising, and the runtime hands the exception to `process_output()` as `outcome.error` rather than letting it escape. That is where you decide. Recover an error you expect from your model: reset it, send a message, return `None`, and the loop goes on to the next step. Re-raise anything else, as the example does: a raise out of `process_output()` is a crash of the model, not of the step. The runtime logs the traceback, stops dispatching commands, ends the session with an error the client sees, and does not restart the loop, which is what an uncaught exception in a hand-written `run()` does too. The default `process_output()` re-raises, so a failing model ends loudly instead of serving nothing in silence. A refusal from `process_input()` is not a failure and never reaches `process_output()`.

```python
    async def process_output(self, outcome: StepOutcome) -> Output | None:
        if isinstance(outcome.error, RolloutExhausted):   # an error the model is known to raise
            self.engine.reset()
            self.output.flush()
            await self.send(WorldRestarted(reason="rollout window reached"))
            return None                                   # nothing to show; the next step starts over
        if outcome.error is not None:
            raise outcome.error                           # anything else is a bug: end the loop
        return outcome.to_output()
```

`self.output.flush()` drops everything queued for the clients, the audio the transport still holds included, and cuts the picture to black. A model that stops what it was saying and carries on rendering keeps the picture instead:

```python
self.output.flush(hold=True)   # drop the queued media, keep the last frame on screen
```

Command handlers and lifecycle hooks run between steps, never during one. A model has one method per lifecycle hook kind (`@session_started`, `@session_ended`, `@connected`, `@disconnected`, `@file_uploaded`). A subclass extends an inherited hook by overriding that method and calling `super()`; the override runs whether or not it repeats the decorator. Marking a second method with the same kind, in one class or in a subclass, fails when the class is defined, so a hook is never dropped in silence. The default `run()` is the loop that drives the three hooks, and it reports each step that runs with `complete_step()`, which is what the runtime counts a session's steps by. To write that report yourself, for example to keep extra files with the step, return a `StepCompleted(output=..., files=...)` from `process_output()` instead of the `Output`. Override `run()` to write your own loop against `emit()`, `send()`, `complete_step()`, `@event`, `self.connected`, and the tracks; `process_input()`, `generate()`, and `process_output()` are then not called, and your loop calls `complete_step(StepCompleted(output=...))` each time it finishes a unit of work. Do that for a loop that is not one step per emit, such as a renderer that emits several times per step or a model that must block on an input.

A command handler refuses by raising `CommandError(code, message)`: `code` is a short, stable token the client branches on, and `message` reaches the client unchanged, so write it for them and name the codes in the command's description. The runtime answers that command's request with the failure, so the caller learns on its own call, no other client in the session hears about a command it did not send, and the runtime records that the command failed. Do not refuse by sending an error message and returning: the runtime counts that as success and acknowledges the command. Any other exception a handler raises reaches the client as an internal error with a fixed message, so catch what a check raises and raise a `CommandError` with a code of its own. The waypoint example's `set_image` does both.

An input track buffers the newest 128 frames and drops the oldest when it is full. How much audio that holds depends on the frame size the transport delivers, so an audio input track can bound its buffer in seconds instead. Its audio arrives at the declared `sample_rate` on every transport:

```python
class AgentAudio(Audio):
    sample_rate = 24_000
    buffer_seconds = 120      # the newest 120 s of audio, whatever the frame size


class MyInput(MediaInput):
    agent_audio: AgentAudio
```

Frames a full buffer drops are counted per track in the session's `runtime_stats`. A `buffer_seconds` bound also holds at most one frame per millisecond of it, and a `read()` of more frames than that fails with `ValueError`. A read of frames whose samples do not fit waits for smaller frames, so pass `timeout` to bound it.

A session can start with its setup already in hand: `POST /start_session` takes a `starting_input` of state values and commands, which the runtime applies on the normal command path before any client connects. A system client sends them, a connection the runtime opens itself, so a handler that takes `client` gets a `ClientInfo` whose `system` is true. The step loop takes its first step only once every starting command has landed, uploads included, so no step runs on the defaults the starting input replaces. In a live session the system client then leaves and the loop waits for a viewer, as it always does; a model with its own `run()` sees the system client connect like any client. A live session stays `waiting` while its starting input applies, because the system client does not occupy it, and `/events` journals the system client's `connection_opened` and `connection_closed` as `waiting -> waiting`. A session with `steps` is occupied by the system client until it ends, so it streams with no other client. `steps` is a whole number from 1 to 10,000. The runtime ends it itself once the model has reported that many steps, with a normal stop whose close reason is "Session ended: the requested steps are complete." A session with `steps` has no client to correct its starting input, so a starting command the runtime cannot apply (an upload whose bytes never arrive, or arguments the contract refuses) ends it at once with the reason `start_refused` and the refusal as its close reason. If the handler of a starting command fails instead, by raising `CommandError` or any other exception, every step the model has not reported yet is recorded with the handler's message, so a session at step 3 of 4 saves steps 3 and 4 with that error, and the session ends with the reason `error`. The model keeps running and takes the next session.

A model can keep each step it reports as a folder of files, turned on in `reactor.yaml`:

```yaml
runtime:
  step_results:
    enabled: true
    video: {codec: h264, preset: veryfast, crf: 23}   # or codec: h265
    audio: {codec: aac, bitrate_kbps: 128}
    queue: 8
```

Every track of the step's output becomes its own stream in one `output.mp4`, beside the extra `files` the step carried and a `result.json` that lists them, with the measured length, frame count, frame rate, and size of each video track in its `usage`. `queue` bounds how many steps the model reports may wait to be saved; a step that finds it full is not saved, so saving never holds the model up. The steps a failed starting command records are saved past that bound, so each of them gets its folder. Each step is its own encode, so this suits a model that finishes a step every few seconds rather than every frame.

When step results are on, `GET /sessions/{id}/steps` lists a session's saved folders by the session's own id; each stays readable for five minutes after it is written, after the session ends too.

A session can also join a LiveKit room, where every other participant is a client. Participants that speak the Reactor protocol need nothing more. For the room's own signals, the model says in `reactor.yaml` which RPC methods, data topics, and byte streams reach which of its commands, messages, and input tracks:

```yaml
runtime:
  livekit:
    rpc:
      - method: lk.clear_buffer        # a call to this method runs clear_buffer
        command: clear_buffer
      - method: lk.playback_finished   # sending playback_finished calls this method
        message: playback_finished
    topics: [avatar]                   # a data packet on avatar.<command> runs <command>
    media_inputs:
      lk.audio_stream: agent_audio     # byte streams on this topic feed agent_audio
```

The runtime checks the block against the model when it loads: every command, message, and input track it names must exist, and a method can run only one command. A mistake stops the model from loading, so it shows at deploy time rather than in a session.

The LiveKit SDK is an optional part of the runtime. A model that connects to rooms lists it in its workspace's `requirements.txt`, and `build.runtime_version` still picks the release:

```text
reactor-runtime[livekit]
```

A running session connects to a room through a connection of its own, the way a WebRTC client connects. The token is the credential of the session's participant: the room, the identity, and the grants come from it, and it never reaches the journal.

```text
POST /sessions/{session_id}/transport/livekit/connections
{"url": "wss://example.livekit.cloud", "token": "eyJhbGciOi..."}

202 {"connection_id": 4711}
```

The join runs in the background. A `GET` on the connection answers `202` while it runs, then reports `open`, `failed`, or `closed` with a reason, and a `DELETE` leaves the room. The room connection is a system connection: once the join succeeds it publishes one LiveKit track per output track, named after it, and occupies the session, so the session streams and the step loop runs whether or not anyone in the room is watching. A session has one room connection at a time. Losing the room closes the room connection and leaves the session running, so the session ends like any session left without a connection, after the orphan timeout.

Every other participant in the room is a client of the session, with a connection of its own: the model gets `@connected` when it joins and `@disconnected` when it leaves, and the journal names its identity in the room. A participant speaks the Reactor protocol as data messages on two topics, `reactor.data` and `reactor.control`, the two families a WebRTC client sends on its two data channels; a message larger than 15,000 bytes travels as a byte stream on the same topic. A command's reply reaches the participant that sent it, and a broadcast reaches every participant. When the room connection closes, its participants' connections close with it.

Scaffold, build, and run it with the CLI:

```sh
reactor init my-model
cd my-model
reactor run
```

`reactor run` builds a container with the runtime inside and serves WebRTC signaling on port 8080. Point a browser at it with the [JS SDK](https://docs.reactor.inc/sdk-reference/using-the-sdk), or connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) and watch frames stream immediately.

Log from the same import, passing context as keyword arguments:

```python
from reactor_runtime import get_logger

logger = get_logger(__name__)

logger.info("scene changed", prompt=self.prompt)
```

Records render as `key=value` text by default, or as one JSON object per line under `REACTOR_LOG_FORMAT=json`. While a session is live, its id is stamped on every record, so tracing one run's logs never requires threading an id through your call sites. Every record also carries the lifecycle phase it was written in, at both granularities: `state`, the session state machine's word, and `runtime_state`, the coarse word the health endpoint serves — so the logs of one phase — loading weights, a live session, teardown — are filterable by whichever vocabulary you are reading off another surface. The stamp is applied where records are written rather than where they are made, so a plain `logging.getLogger(__name__)` and the libraries your model imports are covered too.

## Install

Everything runs through the [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation). There is nothing to install on your host but the CLI and Docker; the runtime ships inside the image the CLI builds for your workspace.

```sh
brew install reactor-team/tools/reactor-cli
```

Not on macOS, or pinning a release in CI? See [Install the CLI](https://docs.reactor.inc/deploy/platform/installation).

## Learn more

- [Quickstart](https://docs.reactor.inc/deploy/development/quickstart): from zero to a model deployed on Reactor's GPUs
- [Runtime overview](https://docs.reactor.inc/deploy/development/overview): what the runtime handles, and the outline of a model
- [Model anatomy](https://docs.reactor.inc/deploy/development/reactor-app/model-anatomy): every member of a `ReactorApp`, line by line
- [The step loop](https://docs.reactor.inc/deploy/development/reactor-app/step-loop): `process_input()`, `generate()`, `process_output()`, and the frame rate
- [Starter example](./examples/starter/README.md): the model `reactor init` scaffolds: one class, `generate()` alone, the smallest complete `ReactorApp`
- [Echo example](./examples/echo/README.md): the client's webcam and microphone in, an effect applied, both sent back, in batches
- [Waypoint example](./examples/waypoint/README.md): a world model on a GPU, seeded from an upload and steered live

## Development

This repository holds the runtime package itself: the authoring interface, the session runner, the media transport, and the wire protocol. To work on it, use [mise](https://mise.jdx.dev/), which pins the toolchain and forwards every task through a thin `make` shim:

```sh
mise run install      # install deps, generate wire bindings, and git hooks
mise run lint         # ruff check, ruff format --check, and mise.lock drift
mise run format       # apply ruff formatting
mise run typecheck    # ty (strict)
mise run test         # unit tests on the floor Python
mise run test-matrix  # unit tests on every supported Python
```

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).
