# Local checks without a GPU

A remote fix loop costs about 13-14 minutes (SKILL.md Phase 6). Each check below costs seconds to minutes on the
porting machine and removes a class of failure from that loop. Run them in this order; each catches what the
earlier ones cannot.

| Check | Catches | Does not catch |
|---|---|---|
| `reactor validate` | reactor.yaml schema and structure, offline | anything in Python; a wrong `runtime.import` (it fails only when the container starts) |
| `python -m py_compile` | syntax errors | everything else |
| `ruff check --select F` | undefined names (F821), unused or missing imports, a name used before assignment | a module that is missing from the image |
| Import test in the built image | a dependency missing from `requirements.txt`, a version conflict, an import-time error in the vendored source, a wrong `runtime.import` | shape and dtype errors |
| GPU-free tests of the application half ([`port-on-reactor-app.md`](../../port-model-e2e/references/port-on-reactor-app.md) §12), in the built image | the contract assembles, commands validate and refuse with their codes, `reset` and session end clear the inputs | anything the model half computes |
| CPU shape smoke run | shape mismatches at module boundaries, wrong reshapes, a cache that grows wrongly across chunks | GPU memory, numerics of CUDA kernels |
| Build context count | the remote build's size limits | |
| `reactor deploy <env> --dry-run` | project resolution, the GPU, region and weights flags as the CLI reads them | whether capacity is free |

## Lint

```sh
ruff check --select F .            # or: python -m pyflakes .
```

Exclude the vendored repo's unused files if they are noisy, but keep every file the model imports. A misspelled
name in a branch the warm-up does not reach would otherwise surface only in a user's session.

## Import test in the built image (needs Docker)

`reactor build` builds the image from `reactor.yaml` with the local Docker daemon (local is its default;
`--remote` builds on the platform) and tags it `reactor-local/<workspace-dir-name>:dev`, the tag `reactor run`
uses. The image holds exactly the dependencies the platform will install.

```sh
reactor build
docker run --rm --entrypoint python reactor-local/<dir>:dev -c "import app, model"
docker run --rm -it --entrypoint bash reactor-local/<dir>:dev   # inspect: pip list, the installed torch, files
```

- Import both halves. The application half must import with no GPU; the model half imports `torch` and the
  vendored package, which also works on a CPU.
- An extension that loads CUDA libraries at import time (flash-attn) can fail on a machine with no NVIDIA driver
  (`libcuda.so.1` not found). That error here is expected; any other import error is a real one.
- `--platform` defaults to `linux/amd64`, which the platform needs. On an ARM machine Docker emulates it, which
  is slow; the image is still the right one to test.
- `reactor deploy <env> --build-local` builds the same way and pushes the image instead of building remotely.

Without Docker, the same test in any Python 3.12 environment with `pip install -r requirements.txt` catches most
of it, but not a conflict with the runtime's own pins (`build.runtime_version`).

## CPU shape smoke run

Build the model's modules with a tiny config (few layers, small width, small spatial size) and random weights,
on the CPU in float32, and run two or three chunks through the same per-chunk code the port uses. Assert the
output shape of each chunk and that the caches carried between chunks have the shapes the next chunk expects.
This exercises the shape asserts at module boundaries without the weights or a GPU. Keep the script in the
workspace but out of the image (`.dockerignore`).

What stays untested: GPU memory, CUDA-only code paths (a fused kernel, flash-attn), and numerics. Warm-up in
`load()` covers those at deploy.

## Build context

Limits of a remote build: 4096 entries, 32 MiB per file, 256 MiB in total, 64 MiB compressed. `--dry-run` does not
check them; count the workspace yourself (`find . -type f | wc -l`, `du -sh .`). Put large non-code files of a
vendored repo (papers, demo videos, notebooks) and the weights in `.dockerignore`.

## Dependency pins

Dependency pins in the source that conflict with the runtime's own requirements fail the build (for example a
source pin `numpy<2` against a runtime that needs `numpy>=2.1`). Resolve them now: drop or relax the source pin,
and record the change as unproven. `requirements.txt` never lists `reactor-runtime`.

## CUDA extensions (flash-attn and similar)

Compiling an extension from source in the remote build is slow and adds to every fix loop, since builds share no
layer cache. Prefer a prebuilt wheel that matches torch, CUDA and Python exactly, pinned by URL in
`requirements.txt`. For torch 2.8, CUDA 12 and Python 3.12:

```
flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

The tags in the file name must match the image: `cu12` the CUDA major version, `torch2.8` the installed torch,
`cp312` the image's Python (`build.python_version`, default 3.12), `cxx11abiTRUE` the ABI of the torch wheel.
The remote CUDA base image is `nvidia/cuda:<cuda_version>-devel-ubuntu24.04` (it has `nvcc`), so a source build
works, only slowly.

A prebuilt wheel installed in a remote build without compiling (the build log shows
`+ flash-attn==2.8.3+cu12torch2.8cxx11abitrue`) and loaded at run time: the serve log printed
`attention backend: flash_attn_2=True flash_attn_3=False`. Log that line at start-up, because a source that falls
back to SDPA when the import fails does so silently. On a 14B video DiT on one B200, warm-up chunks took 3.1–4.1 s
with flash-attn and 3.1–4.1 s with PyTorch SDPA, so the extension cost a dependency and bought no measurable speed
there. Measure before keeping one.

If the extension is optional in the source, keep a fallback (for example PyTorch SDPA) and log which path the
model took at load, so the serve log says which kernel ran.
