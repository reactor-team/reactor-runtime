# Wire protocol (`reactor_wire.v1`)

The `.proto` files here are the source of truth for the data- and control-channel
contract a client speaks to a model. They are the only checked-in artifact: the
generated language bindings are **not** committed. The runtime vendors a released
bindings package at build time instead.

```
proto/
├── reactor_wire/v1/       ← .proto sources (package reactor_wire.v1)
│   ├── common.proto       ← framing shared by every channel message
│   ├── data.proto         ← data channel: model Command / ModelMessage
│   ├── control.proto      ← control channel: platform + track traffic
│   ├── model.proto, platform.proto, track.proto
└── packaging/
    └── pyproject.toml      ← template for the published reactor-wire wheel
```

## Why bindings are not committed

Generated code in git drifts, churns diffs, and bakes one toolchain's choices
into the tree. Instead, every schema change cuts a versioned release of the
bindings, and the runtime pins and vendors that release. Consumers — this
runtime and any external SDK — get the exact artifact that was built and
breaking-checked, not a local regeneration.

## Releases are CalVer

A push to `main` that touches `proto/` runs [`wire-release.yml`](../.github/workflows/wire-release.yml):
it builds the Python bindings wheel and publishes a GitHub release.

- **Version:** `1.YYYYMMDD.<run_number>` (e.g. `1.20260618.42`). CalVer carries
  no compatibility promise on its own — `buf breaking` does (see below).
- **Tag:** `wire/v<version>`. The `wire/` prefix keeps these off the runtime's
  own `vX.Y.Z` package tags.
- **Artifacts:** `reactor_wire-<version>-py3-none-any.whl` (vendored by the
  runtime) and `reactor-wire-<version>-protos.tar.gz` (the `.proto` sources, for
  building bindings in other languages).

## Compatibility is enforced by `buf breaking`, not the version

Because the version is CalVer and the bindings are vendored, `buf breaking` is
the only thing guaranteeing a release stays compatible:

- **Per PR:** `make proto-breaking` checks against `main` (also a CI job and a
  lefthook hook) for fast feedback.
- **At release:** `make proto-breaking-release` checks the sources about to be
  published against the previous `wire/v*` tag, and is the first step of the
  release workflow. A breaking diff fails the release.

An intentional break is never a mutation of `v1` — it is a new namespace
(`reactor_wire.v2`, a parallel `proto/reactor_wire/v2/` package).

## Consuming the bindings

**This runtime** pins a version in [`pyproject.toml`](../pyproject.toml):

```toml
[tool.reactor-wire]
version = "1.20260618.42"
```

`make install` and `make build` run [`scripts/fetch-wire.py`](../scripts/fetch-wire.py),
which downloads the pinned wheel and vendors `reactor_wire/` into `src/`
(gitignored). To adopt a newer protocol, bump the pin and re-run `make install`.

For local iteration on the `.proto` files *before* a release exists, generate
straight from the local sources with `make proto-gen` (`buf generate`). The
published wheel always vendors the pinned release, never a local generation.
