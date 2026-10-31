---
name: maintaining-examples
description: "Keep the runtime's example models few and current. Use when adding or changing a feature a model author uses, when about to add a workspace under examples/, when bumping the package version for a release, or when reviewing a PR that touches examples/. Covers which example shows which shape, where a new feature goes, and how every example follows every release."
---

# Maintaining the examples

`examples/` holds three workspaces: `starter`, `echo`, and `waypoint`. An
outside developer copies them, so each one must build, pass its tests, and pin
the current release after every change. Each example adds that cost to every
change after it, so the set is fixed.

| Example | The shape it shows |
| --- | --- |
| `starter` | One class that writes `generate()` alone, with the default hooks. The model `reactor init` scaffolds. |
| `echo` | Client media in, the application/model split, a refusal in `process_input()`, batched output, per-frame metadata, and the reference for a stepped session. |
| `waypoint` | A world model on a GPU: weights, a config file, an upload, hand-written commands, and messages to the client. |

All three leave `run()` to the runtime, so each serves a streamed session and
a stepped one with the same code. Keep it that way.

## A new feature joins an existing example

When a change adds a feature a model author uses, such as a hook, a type, a
manifest block, or a session mode, show it in one example if one fits:

1. Pick the example whose shape the feature belongs to. Client media, step
   behaviour, and stepped sessions go to `echo`. Weights, config, uploads,
   and messages go to `waypoint`. `starter` takes only a feature that needs
   nothing beyond `generate()`: a change that adds `process_input()` or
   `process_output()` to it, or splits it in two files, removes its reason to
   exist.
2. Make the smallest change that uses the feature for real. Add to the
   example's contract; do not rename, remove, or change an existing command,
   track, or default. Clients, the endurance suite, and test models built on
   the examples depend on them.
3. Document the feature in the example's README and test it in
   `tests/unit/examples/test_<example>.py`, in the same PR.
4. If no example fits without bending its shape, leave the examples alone.
   The docstring of the symbol that owns the feature is the reference, and the
   tests exercise it. A model that only a test needs is a test fixture, not an
   example: it lives with its tests, as `endurance-tests/model*/` and
   `tests/integration/pod/split_echo/` do.

Do not add a workspace under `examples/` for a feature.
`tests/unit/examples/test_runtime_pins.py` asserts the set, so a new workspace
fails CI until the maintainers decide to change it.

## Every release moves every example

The PR that bumps the package version (`uv version X.Y.Z`) bumps every
example in the same commit:

- `requirements.txt`: `reactor-runtime==X.Y.Z`
- `reactor.yaml`: `build.runtime_version: "X.Y.Z"`

```sh
rg -n 'reactor-runtime==|runtime_version' examples
```

`tests/unit/examples/test_runtime_pins.py` fails until every pin matches the
version in `pyproject.toml`. A pin names the release the merge publishes, so
`reactor build` of an example resolves it once the release workflow has
uploaded it.

If the release changes or deprecates a part of the authoring surface, move
the examples off the old form in the same PR, so no example teaches an API
the release has left.

## Review checklist

- A new feature is shown in one example, or the PR says why none fits.
- No new directory under `examples/`.
- No existing command, track, or default of an example changed.
- A version bump moves every example's pins.
