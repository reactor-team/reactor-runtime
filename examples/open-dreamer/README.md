# OpenDreamer example

A real GPU model on the step surface: the public
[OpenDreamer world model](https://github.com/next-state/open-dreamer) served as a
playable Minecraft world. A client picks what the world starts from, holds keys
and moves the mouse, and watches frames stream back.

Three files, and the split between them is the point:

- **`opendreamer_model.py`** — the world model. It imports no `reactor_runtime`,
  takes plain values (held keys as strings, camera deltas as floats, a seed) and
  hands back a plain RGB frame. Everything that persists between steps — the JAX
  KV caches, the RNG, how much of the conditioning window it has observed — is
  private to it, so it runs in a script with the runtime uninstalled.
- **`opendreamer_types.py`** — the wire. The controls a client may set, the video
  track, and the one message the model sends. Nothing else is visible to a
  client.
- **`opendreamer.py`** — the application. It imports no JAX and touches no cache.
  It declares the state, maps it onto one model step, and owns the three commands
  that are not values.

There is no `run()`. The runtime steps the model while a client is connected and
paces playout from how long each step took, which for this model is its real
frame rate.

## The step

```
map_step(state, input)  ->  model.generate(**inputs)  ->  to_output(frame)  ->  wire
```

`map_step` only reads. The controls are level-triggered, so a step is a function
of whatever they hold right now, and asking what to generate changes nothing.
Note what is *missing* from it: no check for whether a world has been started.
The model returns `None` until it has one, the runtime reads that as a declined
step, and the condition lives in one place instead of two.

`on_state_changed` is the other half. The runtime calls it when the state
actually changed, so one viewer learns about a control another viewer moved, and
a reset that stops generation is still announced — which no per-frame message
could do, because after a reset there are no frames.

## Prerequisites

- The `reactor` CLI and Docker.
- An NVIDIA GPU, driver, and Container Toolkit. CPU inference is unsupported.
- About 8 GB in the weights cache for the checkpoint, plus image space for the
  CUDA dependencies and the 184 MB public VPT sample.

## Run

This directory is a `reactor` workspace. Build the image, then give the container
one GPU:

```sh
cd examples/open-dreamer
reactor build
reactor run --gpus device=0
```

`reactor build` installs the runtime and the model's dependencies, then fetches
the pinned OpenDreamer source and the public demo clip. `reactor run` downloads
the checkpoint into the mounted weights cache on first use and reuses it
afterwards, then compiles the JAX generation and conditioning paths before
reporting ready on `http://localhost:8080`.

```sh
curl -s localhost:8080/health
```

## What a client can do

Every control is a field on `OpenDreamerState`, so the runtime generates,
validates, and documents a `set_<field>` command for each one — no handler is
written for any of them:

| Command | Effect |
| --- | --- |
| `set_keys` | Keys held down, space-separated: `w space`. Every held key applies to each generated frame until the value changes, so releasing means sending the set without it. |
| `set_buttons` | Mouse buttons held down: `left`, `right`, `middle`. |
| `set_look_x` / `set_look_y` | Camera movement applied to every frame while set, in [-200, 200]. Send 0 to stop. |
| `set_hotbar` | Hotbar scroll: -1, 0, or 1. |
| `set_seed` | The seed the world runs under. Changing it starts the world again from the same scene. |
| `set_paused` | Stops producing frames and preserves the world. |

Three commands are hand-written, because none of them is setting a value:

| Command | Effect |
| --- | --- |
| `load_scene(scene)` | Start a fresh world from a demo clip, or `random` to be given one. |
| `set_image(image)` | Start a fresh world from an uploaded Minecraft screenshot. |
| `reset` | Return everything to how a session starts: controls at their defaults, no scene, world discarded. |

Until a scene has been chosen nothing is generated. `state_update` carries a
snapshot of the whole session, sent when anything changes and once more to a
viewer as it connects.

## Starting from an image

```js
const image = await uploadFile(file);
await sendCommand("set_image", { image });
```

The model centre-crops the upload, pads it to its tokenizer shape, repeats it
across the conditioning window, and pairs each frame with a neutral action. A
still image has no motion history, so its rollout is less stable than a demo clip
backed by consecutive frames and aligned actions. Any image is accepted;
Minecraft frames from the model's own distribution behave best.

An upload that cannot be decoded fails the command with `invalid_image` while the
client is still waiting for its answer, rather than being discovered inside a
step with nobody to tell. That is the whole reason the application hands the
model its conditioning explicitly instead of letting it notice a change.

## Notes

- `opendreamer.yaml` pins the upstream revision, the checkpoint, and the
  conditioning windows. The Dockerfile installs that same revision under
  `/opt/open-dreamer` and sets `OPENDREAMER_PATH` in the image.
- The scenes belong to the application, not the model: how many starting points
  exist, what they are called, and which one `random` resolves to are product
  decisions. The model builds a conditioning window from a clip or an image and
  never learns their names.
- The same scene and seed reproduce the same rollout.
- Stop `reactor run` to remove the container and release its GPU memory.
