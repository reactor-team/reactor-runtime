# FlashDreams SANA-WM example

SANA-WM streaming served from [NVIDIA FlashDreams](https://github.com/NVIDIA/flashdreams)
by naming it. A world described by a prompt and started from an image, flown
with the keyboard; the model generates twenty-four frames per step at
1280x704 and plays them at 16 fps. The adapter FlashDreams ships for SANA-WM
holds the pipeline, the frame size and rate, the first-frame loader, and the
camera calibration, and the runtime's `cam2v` family class turns that into a
`ReactorApp`. The workspace is `reactor.yaml`, `config.yml`, and
`requirements.txt`.

```yaml
# reactor.yaml
runtime:
  import: reactor_runtime.flashdreams.cam2v:Cam2V   # the family class, shipped by the runtime
  config: config.yml

# config.yml
application: cam2v-sana-wm-streaming   # the FlashDreams application slug
example_data: true                     # start from the adapter's example: frame and prompt
example_idx: 0
```

This is the same family class the [Lingbot example](../flashdreams-lingbot/README.md)
serves, with a different slug and a different model package in the image:
that is what serving a FlashDreams family comes down to. The two differ
underneath in a way the family absorbs: SANA-WM's adapter declares a
`generate_step` hook that turns the camera poses into the model's own
conditioning request and keeps the camera's history across the rollout,
where Lingbot's hands the poses to the pipeline as they are. The family's
model half runs every step through that hook, so both are served the way
FlashDreams itself serves them.

## Run

```sh
cd examples/flashdreams-sana-wm
reactor build
reactor run --gpus all --weights /path/to/weights -e HF_HUB_OFFLINE=0
```

`reactor run` serves WebRTC signaling on `http://localhost:8080`. Connect from
the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) (pick **Local
(Direct)**); frames start as soon as a client connects, from the example data.
Hold `w` with `set_keys` to fly forward, `a` or `d` to turn. The first
session after a start waits for the weights (see below).

`build.runtime_version` names the reactor-runtime release the image installs,
and it must carry `reactor_runtime.flashdreams`. The pinned `3.6.0` does not:
it predates the module, so the image it builds cannot import the class, and
the steps above work once the pin names the first release that ships it. Until
then the example runs with a runtime checkout mounted over the image's
installed package, which is how it was verified.

### The image

The `build:` block carries FlashDreams' requirements, each with its reason as
a comment in `reactor.yaml`: `cuda_version` 13.x, `git` as a system package,
`UV_INDEX_STRATEGY` in `build_env`, and a `build.run` step that installs the
family package (`flashdreams-cam2v`) and the model package
(`flashdreams-sana-wm`) without their declared dependencies. `flashdreams-cam2v`
asks for the `local-window` and `serving` extras, for a desktop window the
family never opens and a WebRTC stack whose `aiortc` pins `av<18` while the
runtime needs `av>=18`. `flashdreams-sana-wm`'s dependencies are ordinary
libraries, and `requirements.txt` lists them by name (Diffusers, Accelerate,
SentencePiece, torchvision, Pillow) beside `flashdreams[runners]`, so one pip
pass resolves all of them against the runtime's own. Pin the same commit in
both files.

### The weights

A FlashDreams pipeline downloads its checkpoints from Hugging Face when it is
built, into the FlashDreams cache and the Hugging Face cache. The model half
points both at the weights root the runtime provides, `REACTOR_WEIGHTS_PATH`,
and sets `HF_HUB_OFFLINE=1`, so the checkpoints are read from the bundle and
a missing checkpoint fails at load rather than reaching for the network.

The adapter's example data is outside that rule. FlashDreams fetches it the
first time `example_data: true` asks for it and keeps it in the FlashDreams
cache under the weights root, and that fetch does not read `HF_HUB_OFFLINE`:
with the files in the bundle no request is made; without them, the adapter
tries the network and `load()` fails with the reason when it cannot. So fill
the bundle with the same `example_data` and `example_idx` the deployment uses.

The bundle is that directory. To fill it, run once with an empty weights
directory and downloads allowed, as the `reactor run` line above does with
`-e HF_HUB_OFFLINE=0`; the run that follows needs neither the flag nor the
network. For SANA-WM the directory holds 105 GB:

```
<weights>/
  example_data/sana_wm/00/                              # demo_0.png, demo_0.txt
  huggingface/                                          # HF_HUB_CACHE
    models--Efficient-Large-Model--SANA-WM_streaming/snapshots/<sha>/
      sana_dit/            # the streaming transformer, 32 GB
      refiner_diffusers/   # the LTX-2 latent refiner, 41 GB
      gemma3_12b/          # the refiner's text encoder, 24 GB
      ltx2_causal_vae/     # the causal VAE, 5 GB
    models--Efficient-Large-Model--gemma-2-2b-it/snapshots/<sha>/   # the transformer's text encoder, 10 GB
    blobs/  refs/  .locks/
```

SANA-WM's pipeline reads its checkpoints on the first step rather than when
it is built, so `load()` returns in seconds and the first session's first
step carries the load: about a hundred seconds from a filled bundle on a
B200. The pipeline then keeps the weights, and later sessions start in about
ten seconds. The model does not compile kernels, so the bundle holds no
`torchinductor/` directory.

The Hugging Face cache stores each file once under `blobs/` and reaches it
through a symlink in `snapshots/`, so copy the directory with symlinks
preserved or dereferenced, never dropped.

## Commands

Every public field on the family's state is a generated `set_<field>` command.
`set_image` and `reset` are written by hand: an upload is decoded rather than
stored as a field, and a reset needs no field.

- `set_prompt` the text the world follows. SANA-WM cannot change its prompt
  mid-way, so a new prompt starts a new world from the current image.
- `set_image` upload the image the world starts from (PNG or JPEG; the
  adapter sizes it to 1280x704). The next step starts a new world from it,
  with the current prompt. The camera intrinsics stay those the adapter
  derived from the example image, and the world scale stays 1.
- `set_keys` the camera keys held down, comma-separated: `w`/`s` forward and
  back, `q`/`e` strafe, `a`/`d` or `j`/`l` turn, `i`/`k` look up and down.
  Held until changed; send `""` to stop.
- `set_seed` the noise seed for the next world.
- `set_paused` `true` holds generation; `false` resumes.
- `reset` start over from the same image and prompt.

Frames arrive on `main_video`, each tagged with `{"rollout": r, "index": i}`:
the world it belongs to and the model's own step count within it. A new
rollout's first frames flush whatever is still queued of the old one.
SANA-WM's adapter ends a rollout after 10 steps, fifteen seconds of video;
the model then sends `rollout_restarted` with the step count and starts a new
world from the same image and prompt. A subclass that wants the world to hold
at its end overrides `process_output()`, as the repository README shows.
