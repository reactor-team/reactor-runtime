# Phase 1: the source runs in your image, from one weights root, offline

The goal of this phase is to run the repo's **own** example inside a container you
built on the GPU node, with every weight it loads staged into one directory (the
weights root), and to prove the run still works with the network unplugged. Every
later phase stands on this run:

- the baseline (`baseline-and-replay.md`) is recorded in this image;
- the port (`port-on-reactor-app.md`) ships with this `build:` block and this
  `requirements.txt`;
- the served model mounts this weights root.

So when the baseline and the port are compared, the only difference between them
is the port itself, not the environment.

**This phase hands on four things:**
- a workspace directory holding `reactor.yaml` (`model:` and `build:`) and
  `requirements.txt`;
- the image tag it builds;
- the absolute path of the weights root;
- the exact command that runs the example, plus its output.

It also hands on the environment deviation record (section 5).

## 1. Take the devices, and know where the bytes go

**Decide free GPUs by process, never by memory.** A model that is still loading
can show a few hundred MiB for several seconds while it already holds the device.

```sh
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

Join the two lists by **UUID**: compute-apps reports a UUID, not an index. A
device is free when no process lists its UUID.

Make the probe fail closed. If either query returns no parseable lines, stop:
an empty list of busy devices must never read as "all free". Before relying on
the probe, test it against a device you know is busy.

Pin the devices you take, and pass them explicitly on every container you start
(`--gpus '"device=N"'`, or `--gpus device=N` for `reactor run`). Never use
`--gpus all`, and never rely on "the first free device".

**Put the weights root on the node's large volume** (`df -h`).

Choose one absolute path for the root, for example `/data/weights/<slug>`, and
keep using it.

## 2. Read the repo before building anything

**Find the example.** This is the README's canonical invocation, with the inputs
the repo ships (a sample image, a prompt file, a config). It becomes the baseline.
If the repo ships nothing runnable, **stop and ask** for an example. An input the
agent makes up proves only that the environment boots, not that the model
behaves.

**Inventory the load surface from the code, not from the README.** A README lists
what a person needs in order to train. The inference path loads its own subset,
and often one extra file buried in a helper.

```sh
grep -rnE "from_pretrained|hf_hub_download|snapshot_download|huggingface_hub|torch\.hub|load_state_dict|torch\.load|safetensors|load_file" --include='*.py' <repo>
```

Write the inventory as a table with these columns:

| artifact | source (HF repo id + revision, URL, or local path) | bytes | subdirectory under the weights root |
|---|---|---|---|

Name each subdirectory the way the config will reference it. There is usually
more than one artifact: a text encoder, a VAE, an auxiliary estimator.

You can get the bytes without a credential. A gated repo's file list and sizes
are public even when its files are not:

```sh
curl -s "https://huggingface.co/api/models/<org>/<repo>/tree/<revision>?recursive=true" \
  | jq -r '.[] | select(.type=="file") | "\(.size)\t\(.path)"'
```

**Grep for the hazards that decide the environment:**
- **Attention backend.** Native `F.scaled_dot_product_attention` needs nothing
  extra. `flash_attn`, `xformers` and `sage` need a compiled wheel that matches
  the GPU architecture.
- **Pickled checkpoints.** Recent torch defaults `torch.load(..., weights_only=True)`,
  so a pickled checkpoint may need `weights_only=False`.
- **Removed modules.** `distutils` and `imp` imports are gone in Python 3.12.
- **Dev builds.** A module that imports a dev or unreleased class of a library at
  top level is the single best import to probe later.

## 3. Stage every weight into the weights root

The goal state, in one sentence: **every weight the model loads is a real file
under the weights root, and the code finds it there by path.** The HF cache is
not a weights home. Files there are keyed by repo id and resolved over the
network unless you force the code offline, and the gate in section 7 would then
pass or fail depending on what that cache happened to hold.

**From a Hugging Face link:** use `hf download` with `--local-dir` and a pinned
revision, one call per artifact.

```sh
export HF_TOKEN=...          # from your secret store; never on the command line
hf download <org>/<repo> --revision <sha> --local-dir /data/weights/<slug>/<subdir>
```

**Pass one pattern per `--include`.** Run one `hf download` per pattern:

```sh
for p in "transformer/**" "vae/**" "text_encoder/**"; do
  hf download <org>/<repo> --revision <sha> --include "$p" --local-dir /data/weights/<slug>/<subdir>
done
```

**Verify the download against the Hub, with a check that can fail.** For every
file the tree listing (section 2) names under the patterns you fetched, the local
file must exist at exactly that size; on any mismatch the check exits non-zero. A
download or a check that reports success without comparing against the Hub's own
list proves nothing, so test the check itself by making it fail once.

**Rehearse on a public repo first.** Run any download or verify script end to end
against a small public repo of the same shape before pointing it at the real
one.

**A 403 on a gated repo.** Report one link per repo
(`https://huggingface.co/<org>/<repo>`) together with the account the token
belongs to (`hf auth whoami`), and stop. Two facts matter here:
- Accepting the terms while logged in as a different account gives a
  byte-identical 403.
- Gating is per repository, so a model repo and its companion dataset are two
  separate acceptances.

**From a local directory:** use it as the weights root if it already holds real
files in the layout the code expects. Otherwise, copy it into the root.

**The weights root holds real files only, checked from inside the container.** A
bind mount carries a link, not its target, so a link that resolves on the host is
dead inside the container, and the loader then reports a missing file.

Audit the root; this should print nothing:

```sh
find /data/weights/<slug> -type l
```

Replace any hit with real bytes:

```sh
cp -rL <link> <link>.real && rm <link> && mv <link>.real <link>
```

## 4. Declare the environment the port will ship in

Build the source's environment **in the form the ported model will use**: a
Reactor workspace whose `build:` block generates the image. The runtime sits
unused in the image while the source runs. The port's workspace later copies
this `build:` block and `requirements.txt` unchanged, so the baseline and the
port share one declared environment.

**Create the source workspace** under the name SKILL.md Phase 0 gives it,
`<slug>-source-<you>-<YYYYMMDD>`; the directory name becomes the image tag.

```sh
reactor init <org>/<slug>-source-<you>-<YYYYMMDD>   # <org>/ skips the account lookup
cd <slug>-source-<you>-<YYYYMMDD>
rm -r weights/
```

Leave the scaffold's `model.py`, `config.yaml` and `runtime.import` as they
are: they keep the workspace valid for `reactor build`, and the source runs
with Python as the entrypoint (section 6), so the scaffold model never loads.
Then edit `reactor.yaml`:

```yaml
$schema: reactor/v2
model:
  name: <org>/<slug>
  version: v0.1.0
  resources:
    gpu:
      count: <what the repo's example needs>
runtime:
  import: model:Starter        # the scaffold's; never served from this workspace
  config: config.yaml
  weights_path: /data/weights/<slug>
build:
  runtime_version: "<latest release, SKILL.md Phase 0>"   # the one runtime pin
  python_version: "3.12"
  cuda_version: "12.9.1"                                  # see below
  system_packages: [git, ffmpeg]                          # what the source needs
  python_requirements: requirements.txt
```

`runtime.weights_path` is read by the Reactor CLI, not by the runtime: for a
local `reactor run`, the CLI mounts the weights from `--weights`, else
`$REACTOR_WEIGHTS_PATH` in your shell, else this key, and exports the mount
path as `REACTOR_WEIGHTS_PATH` inside the container.

The pin itself, and how to look it up, are in SKILL.md Phase 0. The generated
image installs `reactor-runtime==<runtime_version>` and `-r requirements.txt` in
**one uv pass**, so a conflict fails the build instead of the model load.
`requirements.txt` may list `reactor-runtime`; if it does, the version is
identical. `reactor validate` does not compare the two.

**Python is 3.12 or newer** (the runtime requires it). Most code written for 3.10
runs unchanged. What breaks is the source's **pinned dependencies**: an old
`numpy`, `torch` or `scipy` with no cp312 wheel, or an import of `distutils` or
`imp`. Take the smallest upgrade that installs on 3.12, because that is what the
port will run.

**CUDA and torch follow the GPU, not the repo's pins.**
- A Blackwell GPU (sm_100) needs a CUDA 12.8+ torch wheel. Set `cuda_version`
  accordingly.
- If the host driver is newer than the image's CUDA, nothing changes.
- If the driver is older, list every package that would have to move down, and
  decide before building rather than discovering it at import.
- A CUDA extension compiled during the build sees no GPU, so set
  `TORCH_CUDA_ARCH_LIST` (for example `10.0` for B200) through `build.build_env`.

**One `requirements.txt`, resolved in one pass.** Git-source dependencies are
pinned to a commit or tag, and corrected pins go in the file. Never add a
side-install in a separate layer to dodge a conflict: that hides the conflict
instead of resolving it. Use `opencv-python-headless`, not `opencv-python`, and
check that nothing in the dependency closure pulls the GUI build back in. A
package that builds from source needs `python3-dev`, `ninja` or `git` in
`system_packages`.

If uv cannot find a build backend that lives only on PyPI while `cuda_version`
is set, add `build_env: {UV_INDEX_STRATEGY: unsafe-best-match}`.

**Build on the node.** It is x86 and has Docker, and building there avoids moving
a multi-GB image:

```sh
reactor build          # tags reactor-local/<slug>-source-<you>-<YYYYMMDD>:dev
```

**Fall back to a Dockerfile only for a recorded reason.** Use one when the source
cannot run on 3.12 with any reasonable pin set, or needs a base image the `build:`
block cannot express. `reactor init --force --dockerfile` writes the generated
Dockerfile into the workspace as a starting point, and an on-disk `Dockerfile`
then takes the build over.

Write the reason into the deviation record. The baseline then runs in a different
environment from the port, so a replay divergence is first checked against that
difference (`baseline-and-replay.md`).

## 5. The environment deviation record

Keep `ENVIRONMENT.md` in the workspace. List every way this image differs from
the environment the source declares (its `requirements.txt`, `environment.yml`,
`setup.py` or README):

| package / setting | source declares | this image | why |
|---|---|---|---|
| python | 3.10 | 3.12 | runtime floor |
| torch | 2.4.0+cu121 | 2.8.0+cu128 | sm_100 needs cu128 |
| numpy | 1.24.4 | 1.26.4 | no cp312 wheel for 1.24 |

Collect the installed side from inside the image (`pip freeze` or
`uv pip list`). Note the attention backend actually in use, and the driver
version.

When the replay disagrees with the baseline, this table is the first place to
look. When the baseline disagrees with numbers the repo reports, it is the only
place to look.

## 6. Run the source: a cheap probe first, then the example

The image's entrypoint serves a model, so run the source with Python as the
entrypoint. `reactor run` cannot do this: it forwards its arguments to the
serving entrypoint.

Mount the source and the weights root. Mount the weights root **read-write**:
loaders such as the Hugging Face libraries write lock files beside the weights
they read, and on a `:ro` mount that write fails with
`OSError: [Errno 30] Read-only file system`. Mount the weights root itself, not a
parent, at the same absolute path on both sides.

```sh
IMG=reactor-local/<slug>-source-<you>-<YYYYMMDD>:dev
W=/data/weights/<slug>
RUN="docker run --rm --gpus '\"device=N\"' --shm-size 8g --entrypoint python \
  -v <repo>:/src -w /src -v $W:$W -e REACTOR_WEIGHTS_PATH=$W $IMG"
```

**Cheap probe**, in seconds, with no weights loaded. Iterate the environment
here, never on the full run:

```sh
eval $RUN -c "'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'"
eval $RUN -c "'import <the module with the most fragile top-level import>'"
eval $RUN <the repo's own offline or meta-device check, if it has one>
```

**Then the example**, exactly as the README runs it, pointed at paths under the
weights root:

```sh
eval $RUN <the README's script and arguments> --output /src/out/example
```

Change only what the environment forces: a path, a `weights_only=False`, a
removed import. Note every source edit in `ENVIRONMENT.md`.

The run produces something a person can judge: a clip, or its frames.

**Record what the run cost:** its wall clock, and per-step time if the script
prints any. When a field looks like per-step time, check whether it is
cumulative before dividing by it. This is the only point where the source's own
speed can be measured cheaply.

## 7. Strip Hugging Face out, and prove it with the network off

**Rewrite every load to a path under the weights root:**
- `from_pretrained("org/name")` becomes `from_pretrained(f"{root}/<subdir>")`.
  A local path never touches the Hub, so the library call itself can stay.
- `hf_hub_download`, `snapshot_download` and URL fetches are deleted. The file is
  staged, so reference it directly. A hub id hard-coded in a helper needs a code
  change to accept a path.
- The root comes from the environment (`REACTOR_WEIGHTS_PATH`), not from a
  constant. In the port it becomes `reactor_runtime.get_weights_path()`.

**Force offline in the image.** Add `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
to `build.build_env` and to the run's environment. When the port is written,
carry them into its `runtime_env` too; that is where they get lost. A component
that insists on hub-cache semantics gets an `hf/` cache directory **inside** the
weights root, with `HF_HOME` pointing at it, so the cache travels with the
weights.

**The gate.** Re-run the example with `--network none`, the weights root mounted,
and no HF cache mounted. The output should match section 6's run; for a
generative model that means the same seed, the same output.

```sh
eval ${RUN/docker run/docker run --network none} <the README's script and arguments>
```

**Check the root from inside the container**, through the mount the model will
get. A host-side `test -e` follows a link that is dead in there:

```sh
docker run --rm --network none --entrypoint bash -v $W:$W $IMG -c \
  "find $W -xtype l -printf 'DEAD LINK %p -> %l\n' | grep -q . && exit 1 || echo 'weights root resolves'"
```

Proceed only when the gate passes. A model that "probably only downloads once"
fails the first time it is served on a machine with no route to the Hub.

## Exit criteria

- The repo's own example ran in your image on your pinned devices, and its output
  is in the run record.
- `find <root> -type l` prints nothing, and the in-container check prints
  `weights root resolves`.
- The example reproduces with `--network none`, loading only from the weights
  root.
- `reactor.yaml` pins `build.runtime_version` to the latest release, looked up
  as SKILL.md Phase 0 says.
- `ENVIRONMENT.md` lists every deviation from the source's declared environment,
  with a reason for each.
