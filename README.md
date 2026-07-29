# Reactor Runtime

A Python framework for building real-time, interactive video models.

You write the model. Reactor Runtime handles the session lifecycle, the media
transport, and the wire protocol that connects clients to your model.

## Status

Early development. The public API and wire protocol are not yet stable.

## Requirements

- Python 3.12 or newer.
- The **`reactor-webrtc`** wheel, declared as a dependency, which ships the
  prebuilt libwebrtc media engine. No system-level WebRTC or GStreamer install
  is required; the wheel carries everything the transport needs.
- **`ffmpeg`** on `PATH`, when recording is enabled. The recorder shells out to
  it to encode the model's output into the fMP4 segments `/clips` serves; a
  runtime with recording turned off does not need it. Install it from Homebrew
  (`ffmpeg`) or the `ffmpeg` package on Debian/Ubuntu.

## Development

The project uses [mise](https://mise.jdx.dev/) as its task runner; it pins the
toolchain and installs [uv](https://docs.astral.sh/uv/), which owns Python. A
thin `make` shim forwards the same names, so `make lint` runs `mise run lint`.

```sh
mise run install      # install deps, generate wire bindings, and git hooks
mise run lint         # ruff check, ruff format --check, and mise.lock drift
mise run format       # apply ruff formatting
mise run typecheck    # ty (strict)
mise run test         # unit tests on the floor Python
mise run test-matrix  # unit tests on every supported Python
```

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).
