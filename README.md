# Reactor Runtime

A Python framework for building real-time, interactive video models.

You write the model. Reactor Runtime handles the session lifecycle, the media
transport, and the wire protocol that connects clients to your model.

## Status

Early development. The public API and wire protocol are not yet stable.

## Requirements

- Python 3.12 or newer.
- A system **GStreamer** install with the plugins the WebRTC media engine
  drives. The Python bindings are **PyGObject**, declared as a dependency, but
  they bind to GStreamer's native libraries, which must be present on the
  machine:
  - `gstreamer` 1.x with the base, good, bad, and ugly plugin sets,
  - `libnice` / the `webrtcbin` element for WebRTC,
  - `gobject-introspection` (so PyGObject can load the typelibs).

  On macOS these come from Homebrew (`gstreamer` and the `gst-plugins-*`
  formulae); on Debian/Ubuntu, the `gstreamer1.0-*`, `libnice`, and
  `gir1.2-gstreamer-1.0` packages. Without them, importing the transport or
  running its tests fails.
- **`ffmpeg`** on `PATH`, when recording is enabled. The recorder shells out to
  it to encode the model's output into the fMP4 segments `/clips` serves; a
  runtime with recording turned off does not need it. Install it from Homebrew
  (`ffmpeg`) or the `ffmpeg` package on Debian/Ubuntu.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                     # install dependencies (creates .venv)
uv run lefthook install     # install git hooks (once per clone)
uv run ruff check           # lint
uv run ruff format --check  # format
uv run mypy                 # type check (strict)
uv run pytest               # tests
```

The GStreamer media engine under `transport/webrtc/gstreamer/` is a faithful
port of dynamic GStreamer code. Like the vendored protocol bindings, it is held
to behaviour rather than house style and is excluded from the strict type
checker and the linter; its behavioural coverage lives in its own tests.

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).
