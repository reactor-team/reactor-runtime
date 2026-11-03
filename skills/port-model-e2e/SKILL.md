---
name: port-model-e2e
description: "Port a research video model that has never run on Reactor, end to end, on one GPU node; written for chunk-by-chunk models (prompt-driven streams, action-driven world models, video-to-video). Given a code repository and a checkpoint (a Hugging Face repo or a local directory), run the repository's own example in a container you built, record it as a baseline, refactor it into a chunk stream when the source plans a whole video at once, port the model to a ReactorApp on the latest reactor-runtime, replay the baseline on the port and prove the two agree, serve the port on the node, hold it to the served-model checklist, and build a local play page for it with the baseline's examples. Asks a few questions once at the start, runs to the end without stopping, then offers to contribute the port to the Reactor Cookbook. Use when asked to port, bring up, or make playable a model from a research repo or a Hugging Face link."
---

# Porting a research model, end to end

**The goal.** You are given a code repository and a checkpoint. You hand back:

1. **A baseline**: the repository's own example, run in a container you built, with its outputs
   and its per-step cost recorded.
2. **A served port**: the model as a `ReactorApp`, served on the node, with a local play page that
   loads the baseline's examples, and instructions for opening it from where the person sits. The
   [Reactor Sandbox](https://reactor-sandbox.vercel.app/) reaches the same server as a fallback.
3. **Proof that the two agree**: the port runs the baseline's exact inputs and produces the same
   output, bit for bit, at every step.

## What this skill covers well

Video models that generate chunk by chunk, of three kinds:

- **Prompt-driven streams**: text, optionally a first frame; the prompt can change mid-stream.
- **Action-driven world models**: text and a first frame, then actions applied chunk by chunk.
  An action is whatever the model was trained to take: held keys for movement, camera or mouse
  look, a click on a point of the frame, or a drag with one or several points.
- **Video-to-video**: a video comes in, with a prompt or references, and each output chunk rewrites
  the matching input chunk.

A model that is bidirectional over a whole clip (every frame attends to every other, so it cannot
become a chunk stream) is outside this skill: Phase 0 recognises it, and the run stops and reports.
Other kinds of model are not covered by its guidance: the phases still apply, but the baseline schedule, the replay comparison and the
checklist are written for video, and the report says so.

## Inputs

| Input | Required | Notes |
|---|---|---|
| Code repository | yes | URL or path, and a commit. Its README's example is the baseline |
| Checkpoint | yes | `hf://org/repo[@revision]`, a Hugging Face URL, or a local directory |
| GPU node | yes | You run on it. Use only the GPU numbers you were given, and name them in every command (`--gpus '"device=2"'`, `CUDA_VISIBLE_DEVICES=2`), so nothing falls back to GPU 0 or to all GPUs on a node others share |

## Questions, asked once, at the start

Almost everything about a port is read from the code. Ask only what the code cannot decide, in
**one** message before Phase 1, each with a proposed answer so the person can reply "all defaults":

| Question | Proposed default |
|---|---|
| Which of the source's examples is the baseline? (only when the README has several) | the README's canonical invocation |
| Each input, as **server config**, **set before play** (setup) or **changed during play** (live) (table below), with its modality and role | your proposal per input, with its reason |
| Model slug | the repository name, lowercased |
| Stop after the source example, so you can judge the model before it is ported? | no. When yes, the run stops once the untouched source example has run and rendered, before any edit to the source |

**These are the only questions until the report.** Ask them once, at the start, and then work to the
end without stopping to ask anything else; a port takes hours, and a person should not have to sit
beside it. When you have no way to ask at all (you were started as a headless or scheduled job, with
no one reading your output), do not ask: take the proposed defaults, and put them at the top of the
report so the person can see what was decided for them and rerun with other answers.

**Server config, set before play (setup), changed during play (live).** Every input the source takes
gets exactly one kind. Name each kind for the person first and give its formal term in parentheses,
in the question and in the report, so the person can understand it now and recognise it later in
the commands. This table is the one definition every file of this skill uses:

| Kind | What it means for the person playing | Where it lives | Typical inputs |
|---|---|---|---|
| **server config** | set by the server, not by the person; the same for every session | the config file `load()` reads | checkpoint subpaths, step count, dtype, sampler; the resolution when the checkpoint supports only one |
| **set before play** (setup) | the person sets it, then presses start; locked while it plays (a value sent while running is refused) and editable again after `stop` or `reset` | a command valid while idle | first frame, seed; the resolution when the model offers several |
| **changed during play** (live) | the person changes it while the video plays; it takes effect at the next chunk and holds until changed | a public state field or a command applied at the next step | prompt, actions; reference images when the architecture can take them at a chunk boundary (otherwise set before play) |

The kind is one of three things declared for each input, beside its **modality** (how it travels:
text, image, video, movement) and its **role** (what the model does with it: first frame, reference,
prompt). None follows from another: an image can be a first frame set before play or a reference
changed during play. [patterns/](patterns/README.md) is the command standard: the lifecycle, the
events, and one command name per concept, so a client that drives one ported model drives the next
by changing arguments only.

Some arguments exist only because the research script renders one video to a file and exits: an
output path, a limit on how many frames to make, the code that saves the mp4. A served model streams
frames to a client and runs until it is stopped, so these are removed rather than mapped to
anything. And no client input gets a default value in the model or its config: every value a client
can set comes from the client.

**Asking it.** Give each kind its name and its one-line meaning from the table, then list the inputs
under it, with a reason wherever the proposal departs from what the source does. For example:

> **Server config** (set by the server, not by you): 4 denoising steps, 480×832, bf16.
> **Set before play** (setup): you set it, then press start; locked while it plays. The first frame,
> the seed.
> **Changed during play** (live): change it while the video plays; takes effect at the next ~0.5 s chunk. The
> prompt, the camera keys. The source keeps one prompt for a whole video, but the model re-reads the
> text at every chunk, so a new prompt steers what comes next.

**Proposing what is set before play and what changes during play.** Start from the code: an input
the source consumes inside its per-chunk loop changes during play; one it encodes once before the
loop is set before play. Then correct for what an
interactive model needs. A research script often fixes the prompt or the actions for the whole run
only because it renders one offline video, while a person playing the model changes them between
turns. When the architecture can take a new value at a chunk boundary (the text or action
conditioning is applied per chunk, or can be re-encoded there), propose the prompt and the actions
as changed during play even though the source never changes them; say so in the proposal, and say what the
architecture does with the change. A value that changes mid-run where the source never did has no
baseline to compare against: the replay proves the schedule the source ran, and the person judges
the mid-run change by playing it.

These are read from the code, never asked:

- **Starting.** The model waits until the client has set what it needs and starts generation on the
  client's command. The model half carries no default prompt, image or action; values the source
  hard-codes in its example script go into the baseline's conditions file, not into the model, and a
  person playing the port enters values on the play page.
- **An empty prompt is a value.** A prompt the client sets to empty (or to only spaces, which strip
  to empty) is accepted and encoded as the source encodes the empty string, the way a text-to-video
  model's unconditional branch already does. What the model refuses is starting before the client
  has sent a prompt at all. Refuse the empty prompt only when the source's own code fails on it, and
  say so in the report.
- **Clients per session.** One client per session. Several clients sharing one session (spectators,
  multiplayer) is a product feature added later, not part of a port.
- **Playback fps.** Nothing to decide up front. Read the frame rate the model was trained at from
  the source's config or its video writer, port the model, and measure how fast it generates. The
  report sets the two side by side. Slower means it does not keep up, and the gap is reported.
  Faster leaves the person a choice afterwards: keep the faster playback (motion a little quicker
  than trained, often fine), or pace output down to the trained rate.
- **Output tracks and weight loading.** Read from the code. Only a case the code leaves genuinely
  ambiguous becomes a question.

## Phases

| Phase | What | Detail | Exit criterion |
|---|---|---|---|
| 0 | Orient | below | the node runs a GPU container; latest runtime release looked up; step shape read; workspaces created |
| 1 | The source, in your container | [source-and-weights.md](references/source-and-weights.md) | the repository's own example runs in your image with the network off, loading only from the weights root |
| 2 | The baseline | [baseline-and-replay.md](references/baseline-and-replay.md) | the source reproduces itself bit for bit across two processes, or its remaining run-to-run difference is recorded as a floor with the operation that causes it; conditions file and trajectory recorded |
| 3 | **Streaming refactor**, only when needed | below, and [streaming-refactor.md](references/streaming-refactor.md) | "not needed" recorded with its reason, or a per-chunk function whose output over the baseline's schedule equals the baseline bit for bit (or within the floor) |
| 4 | The port | [port-on-reactor-app.md](references/port-on-reactor-app.md) | a `ReactorApp` workspace that builds, renders its schema and passes its GPU-free tests |
| 5 | The replay | [baseline-and-replay.md](references/baseline-and-replay.md); the port is built and served as in [serving.md](references/serving.md) §1–2 | the port under the baseline's conditions matches it bit for bit at every step (or within the floor), or within a tolerance declared before the run for a named, unavoidable change ([baseline-and-replay.md](references/baseline-and-replay.md) §5.2) |
| 6 | Serve | [serving.md](references/serving.md) | the port is healthy on the node; every command round-trips; a second session starts clean; the log shows no handler error |
| 7 | Checklist | [checklist.md](references/checklist.md) | every check under the four principles passes or is explained. A new finding joins the principle it belongs to; only a finding no principle covers adds a principle |
| 8 | Play page | [play-page.md](references/play-page.md) | a local page served on the node drives every command, loads the baseline's examples, and passed a headless browser run on the node |
| 9 | Report, then the Cookbook offer | below | one report with the three deliverables and every decision made; the person answered whether to contribute the port |

Do not start a phase before the previous one's exit criterion holds. A late failure in a port is
almost always an early check that was skipped: weights that only load from the hub, an example that
never ran in a container, a "porting bug" that was in the source all along because nobody saved the
source's output.

Each phase hands the next one values, not reasoning: Phase 1 an image, a passing command and a
weights root; Phase 2 a conditions file, a trajectory and the stage map; Phase 3 a source
that generates one chunk per call (or "not needed"); Phase 4 a workspace; Phase 5 a step-by-step
comparison; Phase 6 a served model; Phase 7 a table; Phase 8 a page and its address.

## Phase 0: Orient

- **The node can run GPU containers.** Every later phase builds and runs Docker images on this
  node, so check it before anything else:

  ```sh
  docker version                    # client and daemon both answer; no sudo needed
  docker buildx version             # BuildKit, which `reactor build` uses
  nvidia-smi                        # the driver sees the GPUs
  docker run --rm --gpus '"device=N"' nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
                                    # a container sees device N: the NVIDIA container toolkit works
  reactor doctor                    # the CLI's own checks, Docker included
  ```

  Use a device you were given (Phase 1 explains how to tell it is free). A failure in
  `reactor doctor` about an API key or server does not block a local port; a failure about Docker
  does. Installing Docker or the NVIDIA container toolkit needs root on the node, so a missing
  piece is a stop-and-report, not something to install around.
- **The runtime release.** Look up the latest release every time:

  ```sh
  gh release view -R reactor-team/reactor-runtime --json tagName -q .tagName   # latest non-pre-release
  ```

  Every merge to the runtime's main branch publishes a `.dev` pre-release, so a plain
  `gh release list` shows those first; `gh release view` returns the latest full release. That
  version is the one exact pin, in `reactor.yaml`'s `build.runtime_version`; the image installs it
  in the same pass as `requirements.txt`. `requirements.txt` may also list `reactor-runtime`, as
  this repository's examples do; if it does, the version is identical. `reactor init` writes a
  default that can be older than the latest release, so replace it. Upgrade the Reactor CLI
  (`reactor upgrade`) and record `reactor version`. Every other file of this skill refers here for
  the pin.
- **The step shape.** Read the source's inference entry point and name what one step of the port
  will be: a chunk of a continuous stream, or a chunk of output per chunk of an input video. The
  step shape decides the baseline's
  input schedule ([baseline-and-replay.md](references/baseline-and-replay.md)) and whether Phase 3
  has work to do: a source that plans a whole video in advance but is causal (later frames see only
  earlier ones) becomes a chunk stream there. A model
  that is bidirectional over a whole clip (every frame attends to every other, so it cannot become
  a chunk stream) is outside this skill: stop and report it ("When a phase cannot finish").
- **The template.** `examples/waypoint` in this repository: a GPU model with Hugging Face weights, a
  `build:` block, and the application/model split. `examples/echo` is the reference for a stepped
  session.
- **The workspaces.** `reactor build` tags the image `reactor-local/<directory name>:dev` and has no
  tag flag, so on a node other people build on, the directory name is the tag namespace. Use two
  directories, both under one run directory on the node, named so nobody else's can collide:

  | Directory | Phases | Image tag |
  |---|---|---|
  | `<slug>-source-<you>-<YYYYMMDD>` | 1–2: runs the source and records the baseline | `reactor-local/<slug>-source-<you>-<YYYYMMDD>:dev` |
  | `<slug>-<you>-<YYYYMMDD>` | 4–7: the port, built, replayed, served and checked | `reactor-local/<slug>-<you>-<YYYYMMDD>:dev` |

  The port directory starts as a copy of the source directory's `reactor.yaml` `build:` block and
  `requirements.txt`, so both images come from one declared environment. Everything this run
  produces (logs, `baseline/`, the report) goes in the run directory.

## Phase 3: Streaming refactor, only when needed

A served model is called once per step and must hand back one chunk each time, carrying what it
needs from earlier chunks (a KV cache, a rolling latent buffer, a position index) to the next call.
Many research repositories do not work that way: their example renders a whole video in one call, or
runs a multi-stage scheduler over the full length, because it only ever writes one file. Wrapping
such code in a `ReactorApp` gives a model that cannot take a new prompt or action until the whole
video is done. So the source is reshaped **before** any `ReactorApp` code exists, against the
baseline, in its own phase.

Decide from the stage map Phase 2 drew
([baseline-and-replay.md](references/baseline-and-replay.md) §2.0):

- **Not needed.** The source already produces one chunk per iteration of a loop, and everything that
  carries across chunks is explicit state. Record "not needed" with that reason (the loop and its
  state, by file and line) and go to Phase 4.
- **Needed.** The source cannot produce one chunk at a time from only the input known when that
  chunk starts (work sized to the whole video, an input consumed early, a stage that needs the full
  sequence; [streaming-refactor.md](references/streaming-refactor.md) gives examples, not a closed list).
  Refactor the source, in its own git history on top of the baseline commit, as
  [streaming-refactor.md](references/streaming-refactor.md)
  orders it: one commit per step, each proven bit-identical to the Phase 2 baseline before the next
  begins. The result is a per-chunk function that, driven through the baseline's schedule, produces
  the baseline exactly. If the refactor changes how random numbers are drawn, the baseline is
  re-recorded under the noise rule
  ([baseline-and-replay.md](references/baseline-and-replay.md) §2.2), never to make a diff pass.

The refactor is done on the source, offline, with nothing of the runtime involved: a bug found here
is a refactor bug, and Phase 4 then only wraps a model that already streams.

## Phase 9: Report, then the Cookbook offer

One report, in the run directory, opening with the three deliverables:

- **Baseline**: the command that produced it, the source commit, the image, the conditions file,
  the rendered output, the per-step cost.
- **Port**: the workspace path, the command that serves it, the play page's address, and how to
  open it from where the person sits: the case that applies, with its commands
  ([play-page.md](references/play-page.md) §6). The Reactor Sandbox is named as the fallback.
- **Agreement**: the per-step comparison table, the side-by-side render, and the verdict. Any
  tolerance names the unavoidable change that needed it and was declared before the run.

Then the model's shape and speed, measured on the served port over one full session, with the GPU
model and device count:

| Number | How it is stated |
|---|---|
| Step size | video frames per chunk, the latent frames behind them, and the resolution |
| Frames | frames per session in the baseline, and frames the served session produced |
| Trained fps | the rate the model was trained at, and where it was read |
| Generation fps | steady state and lower bound side by side, the lower bound with the chunk where it occurred; never an average alone. Then faster or slower than trained, by how much, and, when faster, the question whether to keep it or pace down |
| Action latency | an estimate from the measured step time, not a separate measurement: an input waits for the chunk being generated to finish and shows in the next one, so the latency lies between one step time (it arrived just before a chunk boundary) and two (just after). State it as that range in milliseconds. |

Then the checklist table, the environment deviation record (every version that differs from the
source's own environment, and why), every edit made to the source (the streaming refactor's steps with their bit-identical checks, or its "not needed" reason; any change to how it draws random numbers, with its side-by-side against the untouched source), and every decision you
took in place of a person: slug, the split into server config, set before play and changed
during play, fps, the example used as the
baseline.

### The Cookbook offer

The [Reactor Cookbook](https://github.com/reactor-team/reactor-cookbook) collects model workspaces
other people build and run. After the report, and only then, ask the person once whether to
contribute this port there. It is the one question after the start: the person decides after
seeing the report and playing the model. First check, and put in the question, the three things a
contribution needs:

- **The licence.** The source repository's licence allows redistribution, and its notice files go
  along.
- **The weights.** A person outside can download them from a public source the workspace fetches
  itself (a public Hugging Face repository, for example). A file shared by hand is not one.
- **The shape.** The workspace follows the Cookbook's
  [GUIDELINES.md](https://github.com/reactor-team/reactor-cookbook/blob/main/GUIDELINES.md): file
  names, the application/model split, its tests.

Say which hold and what would have to change for the rest. Only on an explicit yes, prepare the
workspace under `models/<slug>/` on a branch of a fork or clone and open the pull request as a draft;
the play page stays with the person and is not part of it. Without an answer, or on a no, stop at
the report.

## When a phase cannot finish

Stop and report; do not guess around it. The usual causes, each a question for a person rather
than something to work around:

- **The repository has no runnable example.** Ask for one. An invented input proves that the
  environment boots, not that the model behaves.
- **A gated checkpoint refuses the token.** Report the repository link and the account the token
  belongs to; the terms are accepted per repository.
- **The source does not reproduce itself, and the difference cannot be named.** After every source
  of nondeterminism is frozen, a remaining run-to-run difference that can be traced to a named
  operation is recorded as the floor and the run continues
  ([baseline-and-replay.md](references/baseline-and-replay.md) §2.1). A difference no operation
  accounts for leaves the baseline without an anchor: report what varies and stop.
- **The model is bidirectional over a whole clip.** Every frame of a clip attends to every other,
  so it cannot become a chunk stream; each request would have to generate a whole clip. This skill
  ports chunk-by-chunk models only: report the model, the evidence in its code, and stop.
- **The node cannot run GPU containers.** Docker, BuildKit, the GPU driver or the NVIDIA container
  toolkit is missing, or the current user cannot reach the Docker daemon. Installing them needs
  root: report which check in Phase 0 failed, with its output, and stop.
- **No free device on the node.** Report what holds the devices; never share one.

Everything else is yours to decide and to write down.
