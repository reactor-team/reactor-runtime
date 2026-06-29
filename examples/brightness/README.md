# Brightness example

A generated, client-controllable gradient with a matching tone, built on
`ReactorPipeline`. No model weights and no client media: it implements an
`inference()` generator over a typed `InputState`, so it is the smallest end-to-end
exercise of the pipeline authoring surface — auto-generated `set_<field>`
commands, `Idle` skips, fixed FPS, and multi-track (video + audio) output, all in
pure NumPy.

## Run

NumPy is already a runtime dependency, so there is nothing extra to install:

```sh
cd examples/brightness
uv run python -m reactor_runtime.serve
```

Serves on `0.0.0.0:8080`. Point a client (e.g. the test-frontend) at it, or:

```sh
curl -s localhost:8080/health
curl -s -X POST localhost:8080/start_session -H 'content-type: application/json' -d '{}'
```

## Commands

All three are generated automatically from `BrightnessState`'s public fields:

- `set_brightness` — `0.0`–`2.0` (0 = black, 1 = half, 2 = white).
- `set_paused` — `true` pauses generation (the generator yields `Idle`); `false` resumes.
- `set_resolution` — `480p | 720p | 1080p`.

## Recording

`reactor.yaml` enables recording, so the runtime continuously encodes the output
to local fMP4 segments and serves them under `/clips` (needs `ffmpeg` on `PATH`).
`requestClip(durationSeconds)` grabs the last N seconds and `requestRecording()`
grabs the whole session.
