# Brightness example

A generated, client-controllable gradient with a matching tone, built on
`ReactorPipeline`. No model weights and no client media: it implements an
`inference()` generator over a typed `InputState`, so it is a compact end-to-end
exercise of the pipeline authoring surface — auto-generated `set_<field>`
commands, typed command replies, `Idle` skips, fixed FPS, and multi-track
(video + audio) output. The resolution goes up to 4K (2160p) and a client-set
caption is drawn over every frame.

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model and
defines the image in its `build:` block, and `requirements.txt` lists the
model's own dependencies (Pillow, which draws the caption). There is nothing
to install on your host but the CLI and Docker.

```sh
cd examples/brightness
reactor build
reactor run
```

The build is automatic: `reactor build` renders the image from `reactor.yaml`'s
`build:` block — no Dockerfile to write or maintain. `build.runtime_version`
pins the `reactor-runtime` release the image installs; bump it there to
upgrade.

`reactor run` reuses the image `reactor build` produced (it builds one on first
run if none exists), then serves WebRTC signaling on `http://localhost:8080`.
Rebuild after editing anything baked into the image:

```sh
reactor build && reactor run
```

Connect a client from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/)
(pick **Local (Direct)**), or point the [JS SDK](https://docs.reactor.inc) at it
with `local: true`. A quick liveness check:

```sh
curl -s localhost:8080/health
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
to fMP4 segments in process and serves them under `/clips`.
`requestClip(durationSeconds)` grabs the last N seconds and `requestRecording()`
grabs the whole session.
