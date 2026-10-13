# Batch example

A small recurrent network that paints a video for every session from the
session's own seed, serving up to four sessions from one process and running
one forward pass per step for all of them. It is the example of
`BatchedReactorApp`.

`batch.py` is one class. `max_sessions = 4` is the whole opt-in: the runtime
hosts that many sessions in the process and admits a fifth only once one
ends. In place of `generate()` the class writes `generate_batch()`, which gets
the input of every session that takes the step, keyed by session id, and
returns each one's output. The latents are stacked into one batch and the
network runs once.

Everything per session lives where a single-session model keeps it.
`self.state` is the calling session's `BatchState` in a handler, a hook, and
`process_input()`; `self.session_id` names the session, and the class keeps
each session's latent by it, made in `@session_started` and dropped in
`@session_ended`. `process_output()` keeps a fingerprint of each frame,
`frame.sha256`, beside the step's `output.mp4`, so a client can check that a
session's frames are the ones the same seed gives when it runs alone.

The network's weights are drawn from a fixed seed, so a session's frames
depend only on its own `seed` and `tint`. It runs on PyTorch when it is
installed, on the GPU when there is one, and on NumPy otherwise.

## Run

```sh
cd examples/batch
reactor build --no-dockerfile
reactor run
```

`config.yaml` sets the frame size and the network's size. Raise `latent`,
`channels`, and `layers` to give a GPU enough work per step for batching to
show.

Start sessions with their own ids, and follow each one:

```sh
curl -s localhost:8080/start_session -d '{"steps": 100, "starting_input": {"state": {"seed": 7}}}' \
  -H 'content-type: application/json' | jq .session_id
curl -s localhost:8080/health            # capacity: {"slots": 4, "free": 3}
curl -sN localhost:8080/sessions/<id>/events
```

`bench.py` starts several step sessions at once against a running runtime and
reports each one's steps per second, read from the `step_completed` facts on
its event stream:

```sh
python bench.py --url http://localhost:8080 --sessions 4 --steps 300
```

## Commands

| Command    | Field  | Description                                        |
|------------|--------|----------------------------------------------------|
| `set_seed` | `seed` | Seeds the session's latent; restarts the picture.  |
| `set_tint` | `tint` | Pulls the colours from cold (0) to warm (1).       |
