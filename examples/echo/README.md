# Echo example

Receives the client's webcam and microphone, applies a real-time video effect,
and echoes both back over WebRTC. Exercises inbound media, commands, the session
and connection lifecycle hooks, bidirectional A/V, and file uploads (an uploaded
image blended over the output video).

## Run

This directory is a `reactor` workspace: `reactor.yaml` names the model, the
`Dockerfile` builds the image, and `requirements.txt` pins the runtime alongside
OpenCV (which powers the effects). The CLI builds the image with the runtime
inside and runs it — nothing to install on your host but the CLI and Docker.

```sh
cd examples/echo
reactor build
reactor run
```

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

- `set_effect` — `none | grayscale | sepia | edges | invert | blur | pixelate`
- `set_intensity` — `0.0`–`1.0`
- `set_caption` — draw a text caption over the output video (up to 200 chars).
- `set_burst` — `1`–`30` frames per emit. `1` emits on every tick; higher holds
  frames back and sends them together, which is how a batching model produces.
  Useful for watching the transport smooth an uneven producer in a live session.
- `set_overlay_image` — blend an uploaded image over the output video.
  `overlay_image` is a file reference (`UploadedFile`); `overlay_strength` is
  `0.0`–`1.0`. From the JS SDK: `const ref = await uploadFile(file); await
  sendCommand("set_overlay_image", { overlay_image: ref, overlay_strength: 0.5 })`.

## Recording

`reactor.yaml` enables recording, so the runtime continuously encodes the echoed
output to fMP4 segments in process and serves them under `/clips`. From the JS
SDK, `requestClip(durationSeconds)` grabs the last N seconds and
`requestRecording()` grabs the whole session; both resolve to a clip the client
downloads from `/clips`.
