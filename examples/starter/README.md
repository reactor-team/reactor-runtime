# Starter example

The smallest complete model: the Reactor logo turning over a field of coarse
static. The image is the model's weights, read from the workspace's `weights/`
directory. A client sets the spin speed, pauses the spin, and tunes how often
the static re-rolls. It runs on a CPU and needs nothing installed but the CLI
and Docker.

This is the model `reactor init` scaffolds, written as one class in
`starter.py`. It declares what the client can set (`StarterState`) and what it
receives (`StarterOutput`), reads the config and the logo in `load`, and
renders one frame in `generate`. That is the whole step: `generate` receives
the live state, which is what the runtime's default `process_input` hands it,
and returns the output, which the default `process_output` emits as it is.
Neither hook is written here.

## One class, on purpose

The class holds the model's own state (the angle, the static, the frame
count) beside the code that answers the client. At this size that is the
right shape, and it is the shape a first model should take: the smallest
thing that streams.

It is also the shape that stops scaling. Once a model has weights worth
loading in a notebook, or inputs worth deciding on before the model runs, the
two concerns belong in two places: an application half that owns
`process_input` and `process_output` and answers to the client, and a model
half that owns the weights and imports nothing from `reactor_runtime`. The
split is optional, and nothing in the runtime enforces it. `examples/echo`
and `examples/waypoint` are written that way, and the
[`application-model-isolation`](../../skills/application-model-isolation/SKILL.md)
skill says when and how to make the move.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model,
the config it reads, and the weights it mounts, and defines the image in its
`build:` block; `requirements.txt` lists the model's own dependencies.

```sh
cd examples/starter
reactor build --no-dockerfile
reactor run
```

`--no-dockerfile` renders the image from `reactor.yaml`'s `build:` block.
`build.runtime_version` must match the `reactor-runtime==<version>` pin in
`requirements.txt`; bump both together to upgrade.

`reactor run` serves WebRTC signaling on `http://localhost:8080`. Connect from
the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) (pick **Local
(Direct)**); frames start as soon as a client connects.

`config.yaml` holds the frame size and the logo's file name. `weights/logo.png`
is the image the model spins; swap it for any PNG to see the weights change
what streams. With no image in `weights/`, the model draws a placeholder ring
and says so in its log.

## Commands

Every public field on `StarterState` is a generated `set_<field>` command. The
app writes no handler by hand.

- `set_spin_speed` turns per second, `0.05` to `5`.
- `set_paused` `true` holds the logo at its current angle; `false` resumes.
  The static keeps moving either way, so the stream is visibly live.
- `set_static_interval` frames the static holds before it re-rolls, `1` to
  `300`.

Every setting is held: the model reads the current value of each field on
every step. A session end puts the animation back to its first frame.
