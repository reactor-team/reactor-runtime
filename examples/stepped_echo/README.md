# Stepped echo example

Answer each step with one frame and a report of what the step saw. One inbound
video track, one outbound video track, a label, an upload slot, and two
commands; no weights. It runs on a CPU and needs nothing installed but the CLI
and Docker.

This is the reference for a model built to be stepped: a client sends a step
with frames, state, and commands, and gets back exactly what the model saw, so
a client or a test can check that every part of its request arrived. It writes
`process_input()`, `generate()`, and `process_output()` and leaves `run()` to
the runtime, which is what lets a session be stepped at all. The same code
serves a streamed session too.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model, keeps
every step it reports under `runtime.step_results`, and defines the image in
its `build:` block.

```sh
cd examples/stepped_echo
reactor build --no-dockerfile
reactor run
```

`--no-dockerfile` renders the image from `reactor.yaml`'s `build:` block.
`build.runtime_version` must match the `reactor-runtime==<version>` pin in
`requirements.txt`; bump both together to upgrade. The pin is 3.9.0, the first
release with stepped sessions, so `reactor build` resolves it once that
release is published. `reactor run` serves on
`http://localhost:8080`. Start a stepped session with
`POST /start_session {"generation": "stepped"}` and connect a client that asks
for one.

## The contract

Tracks: `webcam` (video) in, `main_video` (video) out.

State, each a generated `set_<field>` command:

- `label` (`str`, default `""`): free text, reported on every step until it
  changes.
- `overlay` (an upload, default none): a file, reported by name, media type,
  and size on every step until it changes. It goes into a step as an upload
  reference, as the value of `overlay` in the step's state.

Commands:

- `tag` with `value: str`: adds `value` to the tags of the step it is sent
  with. Tags belong to one step and are cleared after it; a `tag` sent between
  two steps lands on the next.
- `fail` with `reason: str`: the step it is sent with fails with `reason` as
  its error. The step still runs, counts, and sends its frame, and the client
  receives a result whose `failure_code` is `FAILED` and whose `error` is
  `reason`.

Each step sends exactly one frame on `main_video`. When the step carried
`webcam` frames, it is the last of them, pixels unchanged; when it carried
none, it is the last webcam frame an earlier step of the session carried, and
a 16x16 black uint8 RGB frame only before the first. The frame's metadata is the step's
report, as UTF-8 JSON:

```json
{
  "label": "the label in force",
  "tags": ["this step's tags, in the order sent"],
  "media_input": 2,
  "input_metadata": "the last webcam frame's metadata as text, or null",
  "input_metadata_base64": "the same metadata's exact bytes, base64, or null",
  "overlay": {"name": "cat.png", "mime_type": "image/png", "size": 1024},
  "error": null
}
```

`media_input` counts the `webcam` frames the step's input buffer held: at most
128, the oldest dropped when a step carries more. Metadata is opaque bytes:
`input_metadata` is it read as UTF-8, with any byte that is not UTF-8 replaced
by U+FFFD, and
`input_metadata_base64` is the bytes themselves, to compare exactly. `overlay` is `null` when no
file is set, and `error` is the reason `fail` gave, or `null`. Each step that works also keeps the same bytes as `report.json`
(`application/json`) with its saved result. A failed step keeps no files, as
the runtime keeps none for a failed step, so its report travels on its frame
alone. A step with no state, commands, or frames runs on the current state.

In a streamed session each turn of the loop is one step: it sends the newest
webcam frame that arrived since the last turn, or the one before when none
did, at up to 30 fps, so the picture holds steady while the loop outpaces the
camera. Before the first webcam frame arrives, it sends the black frame.

## From a client

With the Python SDK, from the release that adds stepped sessions
(`Reactor.step()`), against the `reactor run` above:

```python
from reactor_sdk import Reactor, StepCommand

reactor = Reactor("stepped-echo", local=True, generation="stepped")
await reactor.connect()

webcam = await reactor.track("webcam").publish()
webcam.push_frame(frame, user_data=b'{"seq": 1}')   # buffered for the next step

overlay = await reactor.upload_file("cat.png")
result = await reactor.step(
    state={"label": "first", "overlay": overlay},
    commands=[StepCommand("tag", {"value": "a"})],
)
[out] = result.track("main_video").frames
out.array                  # the frame pushed above, unchanged
out.metadata               # b'{"label": "first", "tags": ["a"], "media_input": 1, ...}'
[report] = result.files    # report.json, application/json

await reactor.step()       # label and overlay hold, tags are empty, the same frame again
failed = await reactor.step(commands=[StepCommand("fail", {"reason": "on purpose"})])
failed.error               # "on purpose"
```
