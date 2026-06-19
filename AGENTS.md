# Agent instructions for reactor-runtime

This repository is the open-source Reactor Runtime, published under Apache 2.0.
Everything in it — code, comments, docs, commit messages, PR descriptions — is
public. These rules are non-negotiable.

## This repo is public: no private-system references

Treat Reactor's private monorepo and production services as if they do not
exist. Do not name them, link to them, depend on them, or write code that
assumes them. The audience is an outside developer with no insider context.

In scope: this package's own APIs, its protocol, and the public-facing product
surface. If a piece of functionality depends on something private, it does not
belong in this repo.

Bad: "The Redis runner connects to the Coordinator over gRPC."
Good: "Production runners are distributed separately as packages built on top
of this runtime."

## Toolchain

`uv` manages everything. The commands below are also exposed as `make` targets.

```sh
uv sync                     # install deps (creates .venv)
uv run lefthook install     # install git hooks (once per clone)
uv run ruff check           # lint
uv run ruff format          # format
uv run mypy                 # type check (strict)
uv run pytest               # tests
```

- Python floor is 3.12. Write modern syntax: `X | None`, builtin generics,
  `type` statements where they help. Never `Optional[...]`/`Dict[...]`.
- mypy runs strict. Do not weaken it with blanket ignores; a targeted
  `# type: ignore[code]` needs a reason the reader can verify.
- Releases are tag-driven: `uv version X.Y.Z` commit + matching `vX.Y.Z` tag.
  Never edit the version field by hand in an unrelated change.

## Tests

Every PR that adds or changes behaviour ships tests in the same PR. Unit tests
live in `tests/unit/`, integration tests in `tests/integration/`, mirroring
the `src/reactor_runtime/` module layout. A failing test must surface its full
traceback — never wrap test execution in machinery that swallows exceptions
from worker threads or event loops.

## Lints

`ruff check`, `ruff format --check`, and `mypy` must pass before a PR is
opened. Do not disable rules file-wide to silence a finding; fix the code or
narrowly suppress with a justification.

## Docstrings

Google style, enforced by ruff's pydocstyle rules. Docstrings document the
contract — what the caller can rely on — not the implementation.

- Every public module, class, and function has a docstring. Private helpers
  need one only when the name cannot carry the intent.
- One-line summary in the imperative ("Return the...", not "Returns the...").
  Add `Args:` / `Returns:` / `Raises:` sections when they say something the
  signature does not.
- This is the public surface model authors read. Write for them, not for
  maintainers of this repo.

## Comments

Comments explain non-obvious intent, trade-offs, or constraints — things the
code cannot say. Before writing one, ask: does a reader who was not part of
this change's iteration need this? If not, delete it.

- Write the end state, as if the code has always been this way. No iteration
  narration: no "previously...", "no longer...", "switched from X to Y",
  "this used to be...". The reader has no context of how the code got here
  and does not care.
- No negations of former behaviour. Describe what the code does, not what it
  stopped doing.
- Never narrate what the code visibly does ("# loop over frames").
- No double negations ("not unsupported", "doesn't fail to...") — state the
  positive fact.

## PR descriptions

Two sections, both prose. The diff already lists what moved; the description
gives a reviewer the context the diff cannot.

```
## Why
One or two paragraphs: the problem, and why this is the right shape of fix.
Reference the user-visible behaviour, constraint, or design decision that
motivates the change.

## What Changed
Narrative that walks a reviewer through the change. Surface non-obvious
decisions, trade-offs, and anything that looks surprising in the diff.
```

Never enumerate files or paste a changelog ("Added X. Updated Y."). If the
public API changes, show idiomatic usage of the new surface — how it feels to
call, not just the signature.

## Commits

Every commit is signed off (`git commit -s`) — DCO is enforced in CI. Title
is an imperative sentence; body explains the why in 1–3 short paragraphs,
following the same end-state writing rules as comments.
