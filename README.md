<div align="center">

<img src="assets/banner.png" alt="Reactor Runtime" width="100%" />

**Build real-time AI models in Python.**

[📖 Documentation](https://docs.reactor.inc/deploy/overview) · [🚀 Quickstart](https://docs.reactor.inc/deploy/development/quickstart) · [🌐 Reactor](https://reactor.inc)

</div>

---

Reactor Runtime turns an inference pipeline into a real-time, interactive media and data stream. You write `load()` and `generate()`; the runtime drives them one step at a time and handles the session lifecycle, the media transport (WebRTC, or a LiveKit room), and the wire protocol that connects clients to your model. Viewers watch frames as they are generated and change what the model is doing mid-stream, with no restart and no re-queue.

## Highlights

- 📡 **Real-time streaming.** Frames reach clients over WebRTC as your model produces them, not after a whole video is done. The runtime paces playout from the time each step took, so a model needs no rate limiter of its own.
- 🎮 **Live interaction.** Clients send commands mid-generation: change a prompt, move a camera, adjust a parameter. The next frame reflects it.
- 🔌 **No transport code.** You never import a WebRTC library, manage a WebSocket, or encode video. The runtime ships its own media engine as a wheel, so a plain Python container is all a model needs.
- 🏠 **LiveKit rooms.** A session can join a LiveKit room: the model's tracks are published there, and every other participant in the room is a client.
- ✅ **Typed, validated commands.** Declare the commands your model accepts with standard Python types and constraints. The runtime validates every payload before your handler runs and compiles the surface into an OpenAPI schema that drives typed client SDKs.
- 🗂️ **Step results.** Start a session with its input and a number of steps, and keep each step's output as a folder of files.
- 🧩 **Many sessions, one copy of the weights.** Split a model into a worker half that `reactor-runtime serve-worker` loads once and a session half each session runtime runs, and every session shares the worker's weights and GPU. See `examples/shared_worker`.
- 🔎 **Traceable logs.** `get_logger()` writes structured records, and every record a session writes carries its id, so one filter recovers a whole run.
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

A step is three calls, and `generate()` is the one you must write. `process_input()` runs before it and decides whether a step can happen: return the input `generate()` gets, or raise `ApplicationError("reason")` to skip the step. `process_output(outcome)` runs after it, with the result or the error, and returns the media to emit; a message sent there with `await self.send()` reaches the client before the step's frames. Both have defaults, so the model above writes neither.

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

`generate()` fails by raising, and the exception reaches `process_output()` as `outcome.error`. Recover an error you expect there and return `None` to go on to the next step. Re-raise anything else to end the session with an error, as the default `process_output()` does.

Commands and lifecycle hooks such as `@session_started` and `@connected` run between steps, never during one. A command handler refuses by raising `CommandError(code, message)`, which fails the caller's own request. For a loop that is not one step per emit, override `run()` and write it against `emit()`, `send()`, and `complete_step()`.

A session can also start with its setup in hand and run a set number of steps: `POST /start_session` takes a `starting_input` of state values and commands, and `steps`, and the session ends once the model has reported that many. Each step can be kept as a folder of files, turned on in `reactor.yaml`:

```yaml
runtime:
  step_results:
    enabled: true
    video: {codec: h264, preset: veryfast, crf: 23}   # or codec: h265
    audio: {codec: aac, bitrate_kbps: 128}
    queue: 8
```

A step's tracks are saved as one `output.mp4`, with a `result.json` that lists the step's files, and `GET /sessions/{id}/steps` serves the folders.

A session can also be stepped: `POST /start_session` with `"generation": "stepped"` runs one step per `RequestStep` a client sends on the data channel instead of streaming, and answers each request with a `StepResult` that carries the step's media, so a stepped client must negotiate data-channel chunking. The model code is the same in both modes; a model that overrides `run()` or `submit_command()` cannot be stepped. The [stepped echo example](./examples/stepped_echo/README.md) answers each step with what it saw, so a client can check every part of its request.

A session can join a LiveKit room. The room connection publishes the model's output tracks, and every other participant in the room is a client of the session. The model's `reactor.yaml` says which RPC methods, data topics, and byte streams reach which of its commands, messages, and input tracks:

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

The LiveKit SDK is an optional part of the runtime, so a model that joins rooms lists `reactor-runtime[livekit]` in its `requirements.txt`. A running session joins a room through a connection of its own:

```text
POST /sessions/{session_id}/transport/livekit/connections
{"url": "wss://example.livekit.cloud", "token": "eyJhbGciOi..."}
```

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

Records render as `key=value` text, or as one JSON object per line under `REACTOR_LOG_FORMAT=json`. Every record written during a session carries its id and the runtime's lifecycle state, including the records of `logging.getLogger()` and of the libraries your model imports.

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
- [Events and messages](https://docs.reactor.inc/deploy/development/reactor-app/events-and-messages): commands, refusals with `CommandError`, and messages to clients
- [Video and audio tracks](https://docs.reactor.inc/deploy/development/reactor-app/video-and-audio-tracks): output and input tracks, and how input is buffered
- [Starter example](./examples/starter/README.md): the model `reactor init` scaffolds: one class, `generate()` alone, the smallest complete `ReactorApp`
- [Echo example](./examples/echo/README.md): the client's webcam and microphone in, an effect applied, both sent back, in batches
- [Waypoint example](./examples/waypoint/README.md): a world model on a GPU, seeded from an upload and steered live
- [Stepped echo example](./examples/stepped_echo/README.md): one frame and a JSON report back for every step, the reference for a stepped session

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
