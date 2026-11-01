---
name: port-model-e2e
description: "Port a research model that has never run on Reactor, end to end, on one GPU node. Given a code repository and a checkpoint (a Hugging Face repo or a local directory), run the repository's own example in a container you built, record it as a baseline, port the model to a ReactorApp on the latest reactor-runtime, replay the baseline on the port and prove the two agree, serve the port on the node so a person can play it from the Reactor Sandbox, and hold it to the served-model checklist. Runs unattended by default. Use when asked to port, bring up, or make playable a model from a research repo or a Hugging Face link. A model already on reactor-runtime 3.0 to 3.4 goes through porting-to-reactor-app; a 2.x model through porting-models-to-standalone-runtime."
---

# Porting a research model, end to end

**The goal.** You are given a code repository and a checkpoint. You hand back:

1. **A baseline**: the repository's own example, run in a container you built, with its outputs
   and its per-step cost recorded.
2. **A served port**: the model as a `ReactorApp`, served on the node, which a person plays from
   the [Reactor Sandbox](https://reactor-sandbox.vercel.app/).
3. **Proof that the two agree**: the port replays the baseline's exact conditions and produces the
   same output, bit for bit, step for step.

Every phase below serves those three. A port that serves but has not been replayed against the
baseline is not finished, and neither is one that replays but behaves badly when served: a port is
done when the model behaves well, not when it runs.

## Inputs

| Input | Required | Notes |
|---|---|---|
| Code repository | yes | URL or path, and a commit. Its README's example is the baseline |
| Checkpoint | yes | `hf://org/repo[@revision]`, a Hugging Face URL, or a local directory |
| GPU node | yes | You run on it. Use only the devices you were given, pinned explicitly |

## Questions, asked once, at the start

Almost everything about a port is read from the code. Ask only what the code cannot decide, in
**one** message before Phase 1, each with a proposed answer so the person can reply "all defaults":

| Question | Proposed default |
|---|---|
| Which of the source's examples is the baseline? (only when the README has several) | the README's canonical invocation |
| Each input the inference entry point takes, as **config**, **setup** or **live** (table below) | your proposal per input, with its reason |
| Model slug | the repository name, lowercased |
| Stop after the source example, so you can judge the model before it is ported? | no. When yes, the run stops once the untouched source example has run and rendered, before any edit to the source |

A run is either **interactive** (a person is there: send the questions and wait for the reply) or
**unattended** (no one is: proceed at once on the proposed defaults and list each one in the report).
After this message nothing in the run waits for a person.

**Config, setup and live.** Every input the source takes gets exactly one kind. This table is the
one definition every file of this skill uses:

| Kind | What it is | Where it lives | Typical inputs |
|---|---|---|---|
| **config** | fixed for the life of the server | the config file `load()` reads | checkpoint subpaths, step count, dtype, sampler; the resolution when the checkpoint supports only one |
| **setup** | set before generation starts, fixed until a reset; a value sent while running is refused | a command valid while idle | first frame, seed; the resolution when the model offers several |
| **live** | may change while generating; lands at the next chunk boundary and holds until changed | a public state field or a command applied at the next step | prompt, actions; reference images when the architecture can take them at a chunk boundary (otherwise setup) |

Inputs that exist only for the offline script (output paths, frame-count caps, video writers) are
deleted. None of the three kinds carries a default for a client input in the model or its config.

**Proposing setup or live.** Start from the code: an input the source consumes inside its per-chunk
loop is live; one it encodes once before the loop is setup. Then correct for what an
interactive model needs. A research script often fixes the prompt or the actions for the whole run
only because it renders one offline video, while a person playing the model changes them between
turns. When the architecture can take a new value at a chunk boundary (the text or action
conditioning is applied per chunk, or can be re-encoded there), propose the prompt and the actions
as live even though the source never changes them; say so in the proposal, and say what the
architecture does with the change. A value that changes mid-run where the source never did has no
baseline to compare against: the replay proves the schedule the source ran, and the person judges
the mid-run change by playing it.

These are read from the code, never asked:

- **Starting.** The model waits until the client has set what it needs and starts generation on the
  client's command. The model half carries no default prompt, image or action; values the source
  hard-codes in its example script go into the baseline's conditions file, not into the model, and a
  person playing the port enters values in the Sandbox.
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
| 0 | Orient | below | latest runtime release looked up; step shape read; workspaces created |
| 1 | The source, in your container | [source-and-weights.md](references/source-and-weights.md) | the repository's own example runs in your image with the network off, loading only from the weights root |
| 2 | The baseline | [baseline-and-replay.md](references/baseline-and-replay.md) | the source reproduces itself bit for bit across two processes, or its residual run-to-run difference is recorded as a floor with the operation that causes it; conditions file and trajectory recorded |
| 3 | The port | [port-on-reactor-app.md](references/port-on-reactor-app.md), [streaming-refactor.md](references/streaming-refactor.md) when the source plans a whole video in advance | a `ReactorApp` workspace that builds, renders its schema and passes its GPU-free tests |
| 4 | The replay | [baseline-and-replay.md](references/baseline-and-replay.md); the port is built and served as in [serving.md](references/serving.md) §1–2 | the port under the baseline's conditions matches it bit for bit at every step (or within the floor), or within a tolerance declared before the run for a named, unavoidable change |
| 5 | Serve | [serving.md](references/serving.md) | the port is healthy on the node; every command round-trips; a second session starts clean; the log shows no handler error; the Sandbox connects to it |
| 6 | Checklist | [checklist.md](references/checklist.md) | every item passes or is explained |
| 7 | Report | below | one report with the three deliverables and every decision made |

Do not start a phase before the previous one's exit criterion holds. A late failure in a port is
almost always an early check that was skipped: weights that only load from the hub, an example that
never ran in a container, a "porting bug" that was in the source all along because nobody saved the
source's output.

Each phase hands the next one values, not reasoning: Phase 1 an image, a passing command and a
weights root; Phase 2 a conditions file and a trajectory; Phase 3 a workspace; Phase 4 a step-by-step
comparison; Phase 5 a served model; Phase 6 a table.

## Phase 0: Orient

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
  will be: a chunk of a continuous stream, a chunk of output per chunk of an input video, or one
  answer (data, no frames) per input. The step shape decides the baseline's
  input schedule ([baseline-and-replay.md](references/baseline-and-replay.md)) and the Phase 3 route:
  a source that plans a whole video in advance but is causal (later frames see only earlier ones)
  becomes a chunk stream through [streaming-refactor.md](references/streaming-refactor.md). A model
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
  | `<slug>-<you>-<YYYYMMDD>` | 3–6: the port, built, replayed, served and checked | `reactor-local/<slug>-<you>-<YYYYMMDD>:dev` |

  The port directory starts as a copy of the source directory's `reactor.yaml` `build:` block and
  `requirements.txt`, so both images come from one declared environment. Everything this run
  produces (logs, `baseline/`, the report) goes in the run directory.

## Phase 7: Report

One report, in the run directory, opening with the three deliverables:

- **Baseline**: the command that produced it, the source commit, the image, the conditions file,
  the rendered output, the per-step cost.
- **Port**: the workspace path, the command that serves it, and how to play it from the Reactor
  Sandbox (the SSH tunnel command and the address to enter, [serving.md](references/serving.md) §4).
- **Agreement**: the per-step comparison table, the side-by-side render, and the verdict. Any
  tolerance names the change that needed it and was declared before the run.

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
source's own environment, and why), every edit made to the source (including any change to how it
draws random numbers, with its side-by-side against the untouched source), and every decision you
took in place of a person: slug, the config, setup and live split, fps, the example used as the
baseline.

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
- **No free device on the node.** Report what holds the devices; never share one.
- **The person can reach the node only through SSH, and no TURN server is available.** Signalling
  crosses an SSH tunnel; WebRTC media does not, so the Sandbox on a laptop connects but shows no video
  ([serving.md](references/serving.md) §4). Report it, with the two ways to fix it: a network route
  to the node, or a TURN server.

Everything else is yours to decide and to write down.
