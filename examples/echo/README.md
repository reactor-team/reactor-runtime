# Echo example

Receives the client's webcam and microphone, applies a real-time video effect,
and echoes both back over WebRTC. Exercises inbound media, commands, lifecycle
hooks, bidirectional A/V, and file uploads (an uploaded image blended over the
output video).

## Run

The effects use OpenCV, which is not a runtime dependency, so add it for the run.
Run it as a layered dependency with `--with` rather than installing into the
project venv — `uv run` re-syncs the project on every invocation, so a manually
`uv pip install`ed package gets dropped (and pulling in an old OpenCV would
downgrade the runtime's numpy 2.x and break the ABI):

```sh
cd examples/echo
uv run --with 'opencv-python-headless>=4.10' python -m reactor_runtime.serve
```

Serves on `0.0.0.0:8080`. Point a client (e.g. the test-frontend) at it, or:

```sh
curl -s localhost:8080/health
curl -s -X POST localhost:8080/start_session -H 'content-type: application/json' -d '{}'
```

## Commands

- `set_effect` — `none | grayscale | sepia | edges | invert | blur | pixelate`
- `set_intensity` — `0.0`–`1.0`
- `set_overlay_image` — blend an uploaded image over the output video.
  `overlay_image` is a file reference (`UploadedFile`); `overlay_strength` is
  `0.0`–`1.0`. From the JS SDK: `const ref = await uploadFile(file); await
  sendCommand("set_overlay_image", { overlay_image: ref, overlay_strength: 0.5 })`.

## Recording

`reactor.yaml` enables recording, so the runtime continuously encodes the echoed
output to local fMP4 segments and serves them under `/clips`. This needs
`ffmpeg` on `PATH`. From the JS SDK, `requestClip(durationSeconds)` grabs the
last N seconds and `requestRecording()` grabs the whole session; both resolve to
a clip the client downloads from `/clips`.
