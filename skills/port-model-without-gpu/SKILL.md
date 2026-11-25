---
name: port-model-without-gpu
description: "Port a research model to Reactor and get it serving on the platform when you have NO GPU anywhere: the platform is the only place the code runs. The porting itself (the ReactorApp, the application/model split, commands, weights, chunk streaming, warm-up) is taught by port-model-e2e and read from there; this skill is the loop around it - an account and GPU-allocation check, local checks without a GPU (lint, an import test in the built image, a CPU shape smoke run), one `reactor deploy <environment>` with the project + environment CLI (v1.20261124 or later) that builds remotely and rolls out with the weights (a local folder or a pinned public `hf://` repo), every error read from `reactor environment instances` and `reactor logs`, and a fix. Parity with the source cannot be proven, only written down as unproven. Every port logs a timing line per chunk, a stall watchdog and a resource line, as a stopgap until the platform reports these itself. Covers multi-GPU ports. Includes scripts for the account check, the session token and a session-check client. Use when someone says \"port this model but I have no GPU\", \"build and serve it remotely\", \"port without a local GPU\", or when the porting machine has no CUDA device. NOT for porting on a GPU node where the source can run (port-model-e2e), not for speeding up an already-served model, and not for the older model + release CLI (`reactor model publish`). Status: experimental."
---

# Port a model without a local GPU

## This skill or port-model-e2e

| | [`port-model-e2e`](../port-model-e2e/SKILL.md) | `port-model-without-gpu` (this skill) |
|---|---|---|
| You have | a GPU node you can run containers on | no GPU anywhere on your side |
| The loop | run the source's example as a baseline, port, replay the baseline on the port, serve on the node | local static checks → `reactor deploy` → read `reactor environment instances` / `reactor logs` → fix |
| Parity with the source | proven, step by step, against the recorded baseline | cannot be proven; every unchecked choice is written down as unproven |
| Where it ends | served on the node, with a local play page | an environment of a project on the platform, `ready`, played from a frontend |

Use `port-model-e2e` whenever a GPU node is available, even if the port will later be deployed: a baseline and a
replay prove what this skill can only record. Use this skill when none is. Either way the **porting content comes
from `port-model-e2e`** and [`application-model-isolation`](../application-model-isolation/SKILL.md); this skill
adds only what the remote loop needs ([`references/port-additions.md`](references/port-additions.md)).

> **Status: experimental.** Written for the project + environment CLI (v1.20261124+). The measured facts
> (timings, failure messages, resource limits) come from ports run with the previous CLI on one endpoint with
> B200 GPUs (multi-GPU on RTX 6000 Pro): a 1.3B video model and a 14B variant, each served on its first deploy,
> the 14B also on 2 GPUs. They depend on the platform, not on the command set. Where this skill says
> "unverified", no run has confirmed the behaviour with the new CLI yet.

**In:** a model's source repository, pinned to a commit, and **one weights directory** holding every file
the model loads: on the machine you work from, or as one public Hugging Face repo. The machine needs a network
connection and the `reactor` CLI (v1.20261124+), but no GPU. Docker is optional (Phase 3). You also need a Reactor
account with Serve access and a GPU allocation, signed in with `reactor auth login` (or `$REACTOR_API_KEY`).

**Out:** the model running in an environment of its project, `reactor status <env>` `ready`, the serve log
showing the model loaded and a timing line per warm-up chunk, a client session that receives frames, and a person
playing it from a frontend.

Everything that needs a GPU happens on the platform. After the local checks, the only feedback is what the
platform sends back: the build output, the instance phases, and the serve logs. Keep each one whole, and note when
each action started and when its answer arrived.

## The model in one paragraph

A **project** (`<org>/<name>`, from `model.name` in `reactor.yaml`) holds images, weight sets and environments.
An **environment** (`production` exists from the start; `reactor environment create staging` adds one) owns what
runs: image, weights, GPU type and count, instance count, regions, variables. `reactor deploy <env>` builds the
workspace remotely and writes the environment's next **revision**; `reactor environment update <env>` changes the
image, weights or a setting with no build. There is no `model.version` to bump: an image is named by its digest.
`reactor.yaml` describes only the image (`model.name`, `runtime:`, `build:`); its `model.resources` and
`deployment:` blocks are not applied.

In CLI v1.20261124.15, `--help` on a two-word command (`reactor environment update --help`) fails with
`unknown command`; read https://docs.reactor.inc/deploy/cli-reference/ for those flags.

## What is in this folder

| File | Used in | What it is |
|---|---|---|
| `references/port-additions.md` | Phase 2 | Where each part of the port is taught in `port-model-e2e`, and what the port adds without a GPU: debuggable failures, the unproven list, multi-GPU facts from remote runs, required observability |
| `references/reactor-yaml.md` | Phases 2, 5 | The `reactor.yaml` fields that still apply, the scaffold defaults to change, which settings moved to the environment |
| `references/local-checks.md` | Phase 3 | What can be checked without a GPU, and what each check catches; CUDA extensions |
| `references/errors.md` | Phases 5-6 | Every failure seen so far, resource failures and hangs included: where its message appears, how long it takes, how to fix it; changing an environment without a build |
| `scripts/check_account.sh` | Phase 0 | Sign-in, account slug, GPU allocation, one PASS/FAIL line each |
| `scripts/mint_token.sh` | Phase 7 | Mints a short-lived session token scoped to one model (and environment); prints only the JWT |
| `scripts/session_check.py` | Phase 7 | A Python SDK client: connects, sends your commands, records the video, writes `summary.json`. Untested end to end |

## Phases

| Phase | What | Exit criterion |
|---|---|---|
| 0 | Account readiness, GPU count, placement | `check_account.sh` passes for the GPU type and count per instance |
| 1 | Read the source against the weights directory | every file the code loads is found in the given weights directory |
| 2 | Write the port | a workspace written to `port-model-e2e` Phase 4 plus `port-additions.md` (observability and the unproven list included), whose model reads only from the weights path; scaffold defaults fixed |
| 3 | Local checks, no GPU | `references/local-checks.md`: validate, lint, import test, CPU shape smoke, build context, dependency pins |
| 4 | Choose the weights source | a pinned `hf://…@<sha>` reference, or a pushed weight set with a tag |
| 5 | Deploy | `reactor status <env>` `ready`, and the serve log shows the model loaded and a timing line per warm-up chunk |
| 6 | Read errors and fix | (only when 5 fails) the same exit as Phase 5 |
| 7 | Session check, then a person plays it | the session check receives frames; a person drives it from a frontend |

## Phase 0: account readiness, GPU count, placement

```sh
reactor auth login                       # or export REACTOR_API_KEY; REACTOR_SERVER picks the endpoint
reactor whoami                           # "Account Slug": the <org> of every project name
reactor doctor                           # key, server reachability, Docker, workspace files
scripts/check_account.sh NVIDIA_B200 1   # the GPU type and count per instance the model needs
```

- **No org slug** (`Account Slug: not set`). Every project is `<org>/<name>`, and a project create needs the slug.
  Finish the account's Serve setup, or ask Reactor support to set it.
- **No allocation of that GPU type in a region.** Only `GET /accounts/{id}/accelerators` (the script) shows the
  account's grants. `reactor gpu list` and `reactor region list` list what the platform supports, not what the
  account holds. `held` counts GPUs your running environments hold, including one you are about to replace.

**Choose the GPU count from memory first.** Add up, in the serving dtype, the main network, the text encoder,
the VAE, and the caches a rollout grows (KV cache, feature caches). Start with the fewest GPUs that hold that with
room to spare. A 14B video model whose source shows only an 8-GPU command served on one B200 (bf16 network 37 GB,
text encoder 11 GB, cache ~23 GB; 77.6 GiB allocated after warm-up).

**Add GPUs for speed when the source has a sequence-parallel path.** The same 14B port on 2 RTX 6000 Pro, reusing
the source's sequence-parallel attention with one rank per GPU, ran a 16-frame chunk in 4.8-5.5 s against
7.4-8.7 s on one: 1.55x. Real time is otherwise a later optimisation; the per-chunk timing line shows where you are.

**Placement.** An account's N GPUs may sit on different machines. A 1xN instance then waits in `pending` with
`waiting_for_capacity` and "No node has N free <GPU> in <region>", indefinitely, while the quota stays held. No
command shows which GPUs share a machine: ask the account owner, or try it and watch for that phase.

## Phase 1: read the source against the weights directory

Inventory the load surface from the code as
[`source-and-weights.md`](../port-model-e2e/references/source-and-weights.md) §2 does, and check that every file is
in the given weights directory, noting its path relative to the directory root: that is the layout the port reads.
The source is not run (no GPU), so that section's example run and §3's staging do not apply. A public Hugging Face
repo is pinned by its commit sha in Phase 4, so the deploy names exact bytes.

## Phase 2: write the port

```sh
reactor init <org>/<name>     # scaffolds ./<name>/: reactor.yaml, model.py, config.yaml, requirements.txt,
                              # .dockerignore, README.md, weights/logo.png
```

Write the code as `port-model-e2e` Phase 4 teaches it
([`port-on-reactor-app.md`](../port-model-e2e/references/port-on-reactor-app.md), with the streaming refactor first
when the source needs one), following [`references/port-additions.md`](references/port-additions.md) §1 for what
differs without a GPU. Write the manifest to `references/reactor-yaml.md` and fix the scaffold defaults there.

Four points decide how short Phases 5 and 6 are:

- The model reads weights only from the root `reactor_runtime.get_weights_path()` returns. No hub download at
  startup.
- `load()` warms up at the largest input size, so a per-chunk bug or a GPU memory shortfall fails at deploy, not in
  a user's session.
- Shape asserts at module boundaries, with the shapes in the message (`port-additions.md` §2).
- **Observability, required until the platform reports it** (`port-additions.md` §5): a timing line per chunk, a
  stall watchdog around each chunk (inside every rank on multi-GPU), and a resource line.

Without a GPU nothing proves the port matches the source: keep the unproven list (`port-additions.md` §3).

## Phase 3: local checks, no GPU

Follow `references/local-checks.md`. In short: `reactor validate` (reactor.yaml only), `ruff check --select F`
(undefined names), an import test inside the image `reactor build` makes when Docker is available, a CPU smoke run
of the shape code with small random tensors, the build-context limits, and the dependency pins. Prefer a prebuilt
wheel for CUDA extensions such as flash-attn. `reactor deploy <env> --dry-run` prints the plan and sends no write.

## Phase 4: choose the weights source

| Weights | Use | Notes |
|---|---|---|
| A public Hugging Face repo | `--weights hf://<org>/<repo>@<sha>` on `reactor deploy` or `reactor environment update` | Uploads nothing. The platform downloads the pinned files at start, in phase `syncing_weights`: 86 GB took about 10 min of a 13.5-min start. `reactor weights push` does not take `hf://` |
| A private or gated repo, or a local folder | `reactor weights push <dir> --tag v1.0.0`, then `--weights v1.0.0` | Content-addressed: files already stored are skipped. Symlinks are refused; resolve them first. `reactor weights list` / `show <tag>` |

An environment keeps its weights across code deploys: pass `--weights` on the first deploy and again only when the
weights change. Weights are cached per machine, so an instance that lands on a machine which already holds the set
skips most of the sync. In the container the runtime reads the set from `$REACTOR_WEIGHTS_PATH` either way.

Weight sets belong to the project, so there is nothing to copy between releases: another environment selects the
same set by tag or digest (`reactor environment show <env> -o json` prints the digests). For an `hf://` set, pass the
same `hf://…@<sha>` reference again; whether its digest also works with `--weights` is unverified.

## Phase 5: deploy

```sh
reactor deploy production \
  --gpu NVIDIA_B200x1 --require-regions us-east \
  --weights hf://<org>/<repo>@<sha> \
  --timeout 45m                         # above weights sync + load + warm-up
reactor logs production --follow        # in a second terminal: "model loaded", one timing line per warm-up chunk
```

The first deploy creates the project and runs one instance. `--gpu TYPExN` and `--require-regions` come from
Phase 0; a new project does not take the GPU from `reactor.yaml`. Use `staging` (`reactor environment create
staging` first) when `production` already serves people. Later deploys keep the GPU, count, regions, variables and
weights unless a flag names them.

`reactor deploy` waits for the rollout. Whether it returns promptly on a load failure is unverified, so watch the
instances as well (`references/errors.md`, Watching instances). A failed start showed about 2-3 min after the
image was ready, as phase `failed` with REASON `load_failed`. Two states never end on their own: `loading` past
your warm-up time (no startup timeout exists; read the watchdog's stack dumps) and `pending` with
`waiting_for_capacity`.

## Phase 6: read errors and fix

Find the row in `references/errors.md`, read the message where it says, fix, and `reactor deploy <env>` again.

**One fix loop takes about 12 min:** about 9 min of remote build, then 2-3 min from the image to the error. Remote
builds share no layer cache, so a one-line fix reinstalls every dependency. Read the whole traceback and fix
everything it implies before rebuilding. Phase 3's import test and smoke run catch a whole class of these for free.

- A configuration switch, a variable, the GPU type or count, the instance count: `reactor environment update` or
  `reactor scale`, no build (`references/errors.md`, the end). Read such switches from the environment.
- While a fix is in flight, `reactor scale <env> --desired 0` or `reactor rollback <env> --to <revision>`: a
  failing instance restarts in a loop and keeps holding its GPU.

For every log you act on, record the command, when the action started and when the log arrived, whether the
log alone said what was wrong and where, and what you had to guess.

## Phase 7: session check, then a person plays it

```sh
export REACTOR_JWT="$(scripts/mint_token.sh <org>/<name>)"          # <org>/<name>:staging for staging
python scripts/session_check.py --model <org>/<name> --commands commands.json \
  --first-frame first.jpg --seconds 30 --out out/session-1
```

`commands.json` holds the model's own example inputs as a list of commands; the script's header shows the
format. It needs `reactor-sdk`, `numpy`, `pillow`, `imageio`, `imageio-ffmpeg`, and no GPU. Read `summary.json`:
connect time, start to first frame, frame count and size, frames per second, every message the model sent. A
working 1.3B video model on one B200 connected in about 2 s and sent its first frame about 1 s after `start`.

Read the serve log's timing lines for the session too: frames per second at the client is the playout rate; the
generate time per chunk says whether the model keeps up.

Then a person plays it from a frontend built on the JS SDK, its controls generated from the served schema as
[`play-page.md`](../port-model-e2e/references/play-page.md) §1 describes.
Before handing a link out, run `reactor status <env>` again: a deployed model was once found at 0 instances two
days later with no reason recorded.
