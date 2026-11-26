# reactor.yaml: the fields that apply, and the settings that moved to the environment

The full reference is https://docs.reactor.inc/deploy/platform/reactor-yaml. With the project + environment CLI,
`reactor.yaml` describes **the image only**: the project name, the runtime entry point and the build. The GPU,
instance count, regions, variables and weights belong to an environment and are set with `reactor deploy`,
`reactor environment update` and `reactor scale` flags.

## A commented example

```yaml
$schema: reactor/v2

model:
  name: acme/video-world-1.3b      # <org>/<name>: the project. <org> is the Account Slug from reactor whoami.

runtime:
  import: app:App                  # module:Class of the ReactorApp subclass. reactor validate does not
                                   # check it; a wrong value fails only when the container starts.
  config: config.yaml              # handed to load(); the runtime parses nothing in it.
  weights_path: hf://acme/video-world-1.3b@<commit-sha>   # for `reactor run` only, and the runtime's
                                   # fallback when $REACTOR_WEIGHTS_PATH is unset. A deploy does not read it.

build:
  runtime_version: "3.11.9"        # the reactor-runtime release; the only place it is pinned.
  cuda_version: "12.9.1"           # required for a GPU image; picks the CUDA base image and torch index
  python_requirements: requirements.txt
  system_packages: [git, ffmpeg]   # Debian package names only, no versions
  runtime_env:                     # baked into the image; the model process sees them
    HF_HUB_OFFLINE: "1"            # a missed hub download fails loudly instead of downloading
    TRANSFORMERS_OFFLINE: "1"
```

## Settings that moved to the environment

| Setting | Old place (not applied now) | Set it with |
|---|---|---|
| GPU type and count per instance | `model.resources.gpu` | `--gpu NVIDIA_B200x1` on `deploy` / `environment update`. A new project does not take it from the file: pass it on the first deploy |
| Instance count | `deployment.instances[].count` | `--desired N` on `deploy` / `scale` / `environment update`; `0` stops every instance |
| Regions | `deployment.instances[].region` | `--require-regions us-east` (only these) or `--prefer-regions` (these first) |
| Weights | `runtime.weights_path` at publish | `--weights` with a tag, a digest, a local directory or `hf://…@<sha>`; `none` clears them |
| Variables | `build.runtime_env` (still baked in) | `--var KEY=VALUE` / `--unset KEY`, no build; `--secret NAME` for credentials |
| Release | `model.version` | none: an image is named by its digest; `--build-tag` / `reactor image tag` names one |
| Host CPU and memory | `model.resources.cpu` / `memory` | not settable: managed by Reactor. Ask Reactor if the model needs a specific host-memory allocation |

`model.description`, `model.public` and `model.display-name` are legacy and not applied. The CLI does not read a
`deployment.yaml`. The parser keeps the old keys for compatibility; parsing a key does not mean a deploy applies it.

## Scaffold defaults to change after `reactor init`

| The scaffold writes | What it does if left | Set it to |
|---|---|---|
| `runtime.weights_path: ./weights` and a `weights/logo.png` | `reactor run` mounts the logo as the weights, and `--weights ./weights` would push it. A deploy without `--weights` uploads nothing | delete `weights/`; point `weights_path` at the real weights (`hf://<org>/<repo>@<sha>` or the directory) for local runs |
| `build.runtime_version` an older release (3.8.2 in CLI v1.20261124.15) | the image installs that older runtime | the current release from https://docs.reactor.inc/deploy/development/changelog |
| `cuda_version` commented out | a CPU image: torch without CUDA | uncomment and set the version the source's torch build needs |
| no region, no GPU | a first deploy without `--gpu` falls back to the model's existing accelerator, or CPU; do not rely on it | `--gpu` and `--require-regions` from Phase 0, on the first deploy |

Pin the runtime only in `build.runtime_version`. The scaffold's `requirements.txt` says to leave `reactor-runtime`
out, and its `.dockerignore` already excludes `weights/`.

## Notes from the runs

- The parser drops unknown keys without an error. Run `reactor validate` and compare spellings: `model:` uses
  hyphens (`extra-args`), every other block uses underscores (`runtime_version`, `runtime_env`).
- `build.runtime_env` and `--var` both reach the model process. Bake in what never changes (`HF_HUB_OFFLINE`);
  read a switch you may flip (a debug flag, a stall bound) from a variable and set it with `--var`.
- A `Dockerfile` in the workspace takes the build over: `base_image`, `cuda_version`, `python_version`,
  `system_packages` stop applying. A port built from `build:` should have none.
- Take `runtime_version` from the changelog, `cuda_version` from what the source's torch build needs, and the GPU
  and region from the account's allocation and the model's size.
