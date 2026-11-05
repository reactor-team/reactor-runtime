# Movement

Every model that moves through a scene takes the same walk-and-turn commands, so a person can play
any of them from a keyboard and a client drives all of them with the same calls. A model that can be
conditioned on camera poses also takes a trajectory.

```
 set_move(forward, right)   set_look(yaw, pitch)        every movement model
          │                         │
          └────────────┬────────────┘
                       │ the port turns them into the model's own action
                       ▼
     keys, as trained               poses, integrated over the chunk
     (key-trained model)            (pose-conditioned model)
                                           ▲
                         set_camera_pose(poses)          pose-conditioned models only;
                                                          overrides move and look while set
```

Which one a model is comes from the source: what its conditioning tensor is built from in training.

## `set_move(forward, right)` and `set_look(yaw, pitch)`

| Command | Arguments | Meaning |
|---|---|---|
| `set_move(forward, right)` | each −1…1 | forward/back and right/left; 0 stands still |
| `set_look(yaw, pitch)` | each −1…1 | positive yaw turns right, positive pitch looks up; 0 stops turning |

Values hold until changed. Their kind is read from the source like any other input; for a world
model that is almost always a live control, applying at the next chunk boundary.

**−1…1 is a fraction of the model's own full action, not a physical unit.** Models are trained at
different scales, many with no unit at all (a key is pressed or not), so 1 means "as far or as fast
as this model moves at full input", and the same value moves different models differently. Where the
source defines what full input means (a step in its own world units, a turn in degrees per frame),
`capabilities` states it with where it was read; where it does not, `capabilities` says the scale is
the model's own. Never fill in a number the source does not give.

**A discrete axis takes −1, 0 and 1 only.** A key-trained model declares its axes discrete in
`capabilities`, and a fractional value is refused with `unsupported_value`, listing the accepted
values; it is never rounded. An axis the model lacks (no strafing) is declared absent, and a
non-zero value on it is refused the same way.

**The port does the conversion**, once per chunk: to the one-hot or direction vector a key-trained
source builds, or, for a pose-conditioned model, to poses integrated from the held move and look over
the chunk's duration, starting where the previous chunk ended. Take the step size and the turn per
frame from the source's own controls (its demo's key mapping, its path builder). When the source has
none, choose them and say in the report that they were chosen, not read.

## `set_camera_pose(poses)`, pose-conditioned models only

- `poses`: 4×4 camera-to-world matrices in the OpenCV convention, in the model's own world units, one
  per frame or one per chunk as `capabilities` states.
- Its kind is read from the source: session setup when the model plans the whole trajectory
  at the start; a live control when it reads the next chunk's poses at each boundary,
  in which case each call gives the poses from the next boundary on.
- While poses are set, `set_move` and `set_look` are ignored and `state_update` says the camera pose
  is in control. `set_camera_pose([])` hands control back. When the poses run out, the model holds
  the last one and `state_update` says so.
- A pose count the session cannot use is refused with `invalid_camera_pose`, giving the expected
  count.

## `set_input(keys, mouse)`

For a model trained on raw keyboard and mouse state whose buttons are not movement (jump, attack,
use): `keys` is the full set of keys held now, `mouse` the held mouse state, both in the model's own
vocabulary listed in `capabilities`. Walking and turning on such a model still go through `set_move`
and `set_look`.
