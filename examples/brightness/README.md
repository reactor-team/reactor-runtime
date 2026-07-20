# Brightness example

A generated, client-controllable gradient with a matching tone, built on
`ReactorPipeline`. No model weights and no client media: it implements an
`inference()` generator over a typed `InputState`, so it is a compact end-to-end
exercise of the pipeline authoring surface — auto-generated `set_<field>`
commands, typed command replies, `Idle` skips, fixed FPS, and multi-track
(video + audio) output. The resolution goes up to 4K (2160p) and a client-set
caption is drawn over every frame.

## Run

The caption is drawn with Pillow, which is not a runtime dependency, so add it
for the run. Run it as a layered dependency with `--with` rather than installing
into the project venv — `uv run` re-syncs the project on every invocation, so a
manually installed package gets dropped:

```sh
cd examples/brightness
uv run --with 'pillow>=10.1' python -m reactor_runtime.serve
```

Serves on `0.0.0.0:8080`. Point a client (e.g. the test-frontend) at it, or:

```sh
curl -s localhost:8080/health
curl -s -X POST localhost:8080/start_session -H 'content-type: application/json' -d '{}'
```

## Commands

`set_paused`, `set_resolution`, and `set_text` are generated automatically from
`BrightnessState`'s public fields; `set_brightness` overrides the auto-generated
setter to reply with a typed confirmation; `get_state` and `set_image` are
hand-written `@event` handlers:

- `set_brightness` — `0.0`–`2.0` (0 = black, 1 = half, 2 = white); replies with a
  `BrightnessSet` message carrying the value now in effect.
- `set_paused` — `true` pauses generation (the generator yields `Idle`); `false` resumes.
- `set_resolution` — `480p | 720p | 1080p | 2160p` (2160p is 4K UHD).
- `set_text` — caption drawn over every frame (up to 200 chars); empty clears it.
- `set_image` — accept an uploaded reference image; replies with an `ImageSet` ack.
- `get_state` — replies with a `BrightnessSnapshot` of the live parameters.

## Recording

`reactor.yaml` enables recording, so the runtime continuously encodes the output
to local fMP4 segments and serves them under `/clips` (needs `ffmpeg` on `PATH`).
`requestClip(durationSeconds)` grabs the last N seconds and `requestRecording()`
grabs the whole session.
