# Waypoint example

An explorable world. A client uploads a seed frame, and the model generates
four frames per step from it, steered live by keyboard and mouse controls. The
model is Overworld's Waypoint 1.5, driven through the upstream
[`world_engine`](https://github.com/Overworldai/world_engine) package.

This example is the reference for splitting an application into two halves:

- `waypoint_model.py` is the model. It holds the engine, its world, and its
  step index behind three methods, `load`, `generate`, and `reset`. It imports
  nothing from `reactor_runtime` and runs in a notebook.
- `waypoint.py` is the application. It declares what the client can set and
  what it receives, decides when a step can happen in `prepare_step`, forwards
  the step to the model in `generate`, and turns the result into frames and
  messages in `collect_step`.

The two meet on two dataclasses, `WaypointStepInput` and `WaypointStepResult`.
The application never reads the model's engine; what it needs to know about a
step rides in the result.

When `generate` raises, the exception reaches `collect_step` as
`outcome.error`, and the application decides. This example re-raises: the
model's only own error, `NotSeeded`, cannot happen because `prepare_step`
refuses a step before it has a seed, so anything that does arrive is a bug or
a GPU failure, and ending the session with an error beats serving a frozen
world. A model with an error it expects recovers there instead: reset the
engine, send a message, return `None`, and the loop continues.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model and
defines the image in its `build:` block, and `requirements.txt` lists the
model's own dependencies. The `build:` block declares `cuda_version`, because
the engine compiles kernels at load, `git` as a system package, because
`world_engine` installs from its git tag, and `UV_INDEX_STRATEGY` in
`build_env`, because `world_engine`'s build needs a `setuptools` newer than
the one the PyTorch index carries. There is nothing to install on your
host but the CLI and Docker. The weights come from Hugging Face on first load;
set `HF_TOKEN` if the repository asks for one.

```sh
cd examples/waypoint
reactor build --no-dockerfile
reactor run --gpus all
```

`--no-dockerfile` renders the image from `reactor.yaml`'s `build:` block.
`build.runtime_version` must match the `reactor-runtime==<version>` pin in
`requirements.txt`; bump both together to upgrade.

`reactor run` serves WebRTC signaling on `http://localhost:8080`. Connect from
the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) (pick **Local
(Direct)**) and upload an image with `set_image`; frames start on the next
step.

`config.yml` holds the model's own settings: the weights to load, the
quantization, and how many idle steps to run at load so the compiled kernels
are ready before the first client waits on them.

## Commands

Every public field on `WaypointState` is a generated `set_<field>` command.
`set_image` and `reset` are written by hand: the upload is decoded and fitted
into the seed frame rather than stored, and a reset needs no field.

- `set_image` upload the seed frame (PNG or JPEG, fitted to 1280x720). The
  next step starts a new world from it.
- `set_paused` `true` holds generation on the last frame; `false` resumes.
- `set_action` movement shorthand: `idle`, `forward`, `back`, `left`, `right`,
  `forward_left`, `forward_right`, `back_left`, `back_right`.
- `set_buttons` extra pressed keycodes as a comma-separated list, for example
  `32,16`.
- `set_mouse_x`, `set_mouse_y` mouse velocity consumed on the next step.
- `set_scroll_wheel` `-1`, `0`, or `1`.
- `reset` restart the world from the current seed frame.

The model sends one message, `waypoint_status`, on connect, as the reply to
`set_image` and `reset`, and every 50 steps: whether a seed is set, whether
generation is paused, and the index of the last completed step. Each frame on
`main_video` carries `{"step": n}` as metadata.
