# FlashDreams Waypoint example

Waypoint 1.5 served from [NVIDIA FlashDreams](https://github.com/NVIDIA/flashdreams)
by naming it. An explorable world starts from an image and is steered live by
keyboard and mouse; the model generates four frames per step at 1024x512 and
plays them at 60 fps. The same model `examples/waypoint` serves through its
own two Python files is served here with none: FlashDreams' adapter for
Waypoint holds the pipeline, the frame size and rate, the seed-image loader,
and the key mapping, and the runtime's `action2v` family class turns that into
a `ReactorApp`.

The workspace is `reactor.yaml`, `config.yml`, and `requirements.txt`.

```yaml
# reactor.yaml
runtime:
  import: reactor_runtime.flashdreams.action2v:Action2V   # the family class, shipped by the runtime
  config: config.yml

# config.yml
application: action2v-waypoint-1-5-1b   # the FlashDreams application slug
example_image: true                     # start each session from the adapter's example image
```

`action2v` is the FlashDreams family driven by keyboard and mouse. Any model
registered in that family is served the same way: change the slug. For more
control, subclass `Action2V` in a Python file beside the manifest and point
`runtime.import` at it; the repository README shows a subclass that adds a
field and scales the pointer motion in `process_input()`.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the class and
defines the image in its `build:` block, and `requirements.txt` lists the
model's dependencies. There is nothing to install on your host but the CLI and
Docker.

```sh
cd examples/flashdreams-waypoint
reactor build
reactor run --gpus all --weights /path/to/weights -e HF_HUB_OFFLINE=0
```

`reactor run` serves WebRTC signaling on `http://localhost:8080`. Connect from
the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) (pick **Local
(Direct)**); frames start as soon as a client connects, from the adapter's
example image. Upload your own with `set_image`.

`build.runtime_version` names the reactor-runtime release the image installs,
and it must carry `reactor_runtime.flashdreams`. The class imports FlashDreams
when the model loads, so the runtime itself has no FlashDreams dependency.

### The image

Three things in the `build:` block are FlashDreams' requirements, and the
comments in `reactor.yaml` say why each is there:

- `cuda_version` 13.x: FlashDreams builds against CUDA 13 with PyTorch 2.9 or
  later. The declared version selects the CUDA base image and the matching
  PyTorch index.
- `system_packages: [git]`: FlashDreams' v2 packages are on GitHub only, so
  they install from a pinned commit.
- `build.run`: the family package (`flashdreams-action2v`) and the model
  package (`flashdreams-waypoint`) install without their declared
  dependencies. `flashdreams-action2v` asks for `flashdreams[serving]`, whose
  `aiortc` pins `av<18` while the runtime needs `av>=18`, and the family app
  never imports that extra. Everything the two do import comes from
  `requirements.txt`: `flashdreams` core with its real dependencies, and
  Pillow. Pin the same commit in both files.

### The weights

A FlashDreams pipeline downloads its checkpoints from Hugging Face when it is
built, into the FlashDreams cache and the Hugging Face cache. The model half
points both at the weights root the runtime provides, `REACTOR_WEIGHTS_PATH`,
and sets `HF_HUB_OFFLINE=1`, so a deployed model reads only from its bundle
and a missing file fails at load rather than reaching for the network.

The bundle is that directory. To fill it, run once with an empty weights
directory and downloads allowed, as the `reactor run` line above does with
`-e HF_HUB_OFFLINE=0`; the run that follows needs neither the flag nor the
network. For Waypoint the directory holds 3.5 GB:

```
<weights>/
  default_inputs/waypoint/crystal_desert_blade.jpg      # the adapter's example image
  huggingface/                                          # HF_HUB_CACHE
    models--Overworld--Waypoint-1.5-1B/snapshots/<sha>/model.safetensors
    models--Overworld-Models--taehv1_5/snapshots/<sha>/taehv1_5.pth
    blobs/  refs/  .locks/
```

The Hugging Face cache stores each file once under `blobs/` and reaches it
through a symlink in `snapshots/`, so copy the directory with symlinks
preserved or dereferenced, never dropped.

## Commands

Every public field on the family's state is a generated `set_<field>` command.
`set_image`, `move`, and `reset` are written by hand: an upload is decoded
rather than stored as a field, pointer motion adds up between steps, and a
reset needs no field.

- `set_image` upload the image the world starts from (PNG or JPEG; the
  adapter sizes it). The next step starts a new world from it.
- `set_keys` the keys held down, comma-separated, for example `w,shift`:
  letters, digits, `space`, `shift`, `ctrl`, `enter`, `tab`, and the arrow
  keys. Held until changed; send `""` to release them all.
- `move` pointer and wheel motion since the last `move`, as a fraction of the
  frame: `dx` of its width, `dy` of its height, positive right and down, and
  `wheel`. The next step spends it.
- `set_seed` the noise seed for the next world.
- `set_paused` `true` holds generation; `false` resumes.
- `reset` start over from the same image.

Frames arrive on `main_video`, each tagged with `{"rollout": r, "index": i}`:
the world it belongs to and the model's own step count within it. A new
rollout's first frames flush whatever is still queued of the old one. When a
world reaches the model's limit of 10,000 steps, the model sends
`rollout_restarted` with the step count and starts a new world from the same
image.
