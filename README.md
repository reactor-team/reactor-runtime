<div align="center">

<img src="assets/banner.png" alt="Reactor Runtime" width="100%" />

**Build real-time AI models in Python.**

[📖 Documentation](https://deploy-docs.reactor.inc) · [🚀 Quickstart](https://deploy-docs.reactor.inc/development/quickstart) · [🌐 Reactor](https://reactor.inc)

</div>

---

Reactor Runtime turns an inference pipeline into a real-time, interactive media and data stream. You write `load()` and either own the loop in `run()` or let the runtime step your model for you; the runtime handles the session lifecycle, the WebRTC media transport, and the wire protocol that connects clients to your model. Viewers watch frames as they are generated and change what the model is doing mid-stream, with no restart and no re-queue.

## Highlights

- 📡 **Real-time streaming.** Frames reach clients over WebRTC as your model produces them, not after a whole video is done. `emit()` paces your loop to the rate clients play at, so a model needs no rate limiter of its own.
- 🎮 **Live interaction.** Clients send commands mid-generation: change a prompt, move a camera, adjust a parameter. The next frame reflects it.
- 🧩 **Your model stays yours.** Bind a plain Python class with `model:` and the runtime drives it one step at a time — weights and caches on one side of the line, the wire contract on the other, with neither importing the other.
- 🔌 **No transport code.** You never import a WebRTC library, manage a WebSocket, or encode video. The runtime ships its own media engine as a wheel, so a plain Python container is all a model needs.
- ✅ **Typed, validated commands.** Declare the commands your model accepts with standard Python types and constraints. The runtime validates every payload before your handler runs and compiles the surface into an OpenAPI schema that drives typed client SDKs.
- 📦 **One container, anywhere.** The `reactor` CLI scaffolds a workspace, builds a small image, and runs it locally. The same image deploys to [Reactor](https://reactor.inc)'s GPU cloud unchanged.

## How it works

A model is a `ReactorModel` subclass in `model.py`. Declare the media it sends, load your weights once, and write the loop that receives or produces frames, data, and more:

```python
from pathlib import Path

from reactor_runtime import InputField, Output, ReactorModel, Video, event


class MyOutput(Output):
    main_video: Video


class MyModel(ReactorModel):
    fps = 24

    def load(self, config_path: Path | None) -> None:
        self.pipe = load_my_pipeline()
        self.prompt = "a sunny meadow"

    @event(name="set_prompt", description="Scene the model renders")
    async def set_prompt(
        self, prompt: str = InputField(default="a sunny meadow", moderate=True)
    ) -> None:
        self.prompt = prompt

    async def run(self) -> None:
        while True:
            await self.connected.wait()
            while self.connected.is_set():
                frame = self.pipe.forward(prompt=self.prompt)
                await self.emit(MyOutput(main_video=frame))
```

That is a complete model. `run()` produces frames for as long as someone is watching, and any client can send `set_prompt` at any time to change what the next frame renders.

### Let the runtime drive the loop

A model that produces one step at a time does not have to write `run()` at all. Keep the model itself a plain class — weights, caches, and a `generate()` that takes and returns plain values — and bind it with `model:`. The runtime steps it while a client is connected and paces playout from how long each step took:

```python
from reactor_runtime import (
    InputField, InputState, NotReady, Output, ReactorModel, Video,
)


class MyState(InputState):
    prompt: str = InputField(default="a sunny meadow", moderate=True)


class MyOutput(Output):
    main_video: Video


class MyModel(ReactorModel):
    state: MyState          # every public field becomes a set_<field> command
    model: MyGenerator      # a plain class: no reactor_runtime import

    def load(self, config_path):
        self.model = MyGenerator()
        self.model.load(config_path)

    def map_step(self, state, input) -> dict:
        if not state.prompt:
            raise NotReady("no prompt set")
        return {"prompt": state.prompt}
```

The step is `map_step` → `generate` → `to_output`. `map_step` builds the arguments and declines with `NotReady` when there is nothing to step on; `to_output` puts the products on tracks — it defaults to the single declared one, so this model writes none. A step can also return a `ModelMessage` instead of, or alongside, its media, which is what a model whose product is data rather than video does.

Overriding `run()` stays available and costs nothing else: the state and its generated commands, the `@event` handlers, the tracks, and the model binding all keep working. See [`examples/stepped`](./examples/stepped) for both halves written out.

Scaffold, build, and run it with the CLI:

```sh
reactor init my-model
cd my-model
reactor run
```

`reactor run` builds a container with the runtime inside and serves WebRTC signaling on port 8080. Point a browser at it with the [JS SDK](https://docs.reactor.inc), or connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) and watch frames stream immediately.

## Install

Everything runs through the [`reactor` CLI](https://deploy-docs.reactor.inc/platform/installation). There is nothing to install on your host but the CLI and Docker; the runtime ships inside the image the CLI builds for your workspace.

```sh
brew install reactor-team/tools/reactor-cli
```

Not on macOS, or pinning a release in CI? See [Install the CLI](https://deploy-docs.reactor.inc/platform/installation).

## Learn more

- [Quickstart](https://deploy-docs.reactor.inc/development/quickstart): from zero to a streaming model in 2 minutes
- [Model anatomy](https://deploy-docs.reactor.inc/development/reactor-model/model-anatomy): every piece of a Reactor model, line by line
- [The run loop](https://deploy-docs.reactor.inc/development/reactor-model/run-loop): emitting frames, batches, and frame rates

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
