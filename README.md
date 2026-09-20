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

Command handlers and lifecycle hooks run between steps, never during one. The default `run()` is the loop that drives the three hooks. Override it to write your own loop against `emit()`, `send()`, `@event`, `self.connected`, and the tracks; `process_input()`, `generate()`, and `process_output()` are then not called. Do that for a loop that is not one step per emit, such as a renderer that emits several times per step or a model that must block on an input.

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
