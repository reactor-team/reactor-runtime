# Movement

Movement has two layers. Every model that moves takes the simple layer; a model that can take camera
poses also takes the layer below it.

```
 set_move(forward, right)   set_look(yaw, pitch)        simple layer: every movement model
          │                         │
          └────────────┬────────────┘
                       │ the model, or the port, turns them into its own action
                       ▼
               set_camera_pose(poses)                    pose layer: models that take poses
                       │ overrides move and look while set; [] hands control back
                       ▼
                 the next chunk
```

## The simple layer: `set_move` and `set_look`, changed during play (live)

| Command | Arguments | Meaning |
|---|---|---|
| `set_move(forward, right)` | each −1…1 | forward/back and right/left; 0 stands still |
| `set_look(yaw, pitch)` | each −1…1 | positive yaw turns right, positive pitch looks up; 0 stops turning |

Values hold until changed and apply at the next chunk boundary.

**−1…1 is a fraction of the model's own full action, not a physical unit.** Models are trained at
different scales, many with no unit at all (a key is pressed or not), so 1 means "as far or as fast
as this model moves at full input", and the same value moves different models differently. Where the
source does define what full input means (a step size in its own world units, a turn in degrees per
frame), `capabilities` states it with where it was read; where it does not, `capabilities` says the
scale is the model's own. Never fill in a number the source does not give.

**A discrete axis takes −1, 0 and 1 only.** A model trained on keys declares its axes discrete in
`capabilities`, and a fractional value is refused with `unsupported_value`, listing the accepted
values. It is never rounded. An axis the model lacks (no strafing) is declared absent; a non-zero
value on it is refused the same way.

**Turning the simple layer into the model's action** happens in the port, once per chunk: keys as the
one-hot or direction vector the source builds, or, for a model conditioned on poses, poses
integrated from the held move and look over the chunk's duration, starting where the previous chunk
ended. Take the step size and turn per frame from the source's own controls (its demo's key mapping,
its path builder). When the source has none, choose them, say in the report that they were chosen and
not read, and let `set_speed` adjust them while playing.

## `set_speed(move, turn)`, changed during play (live)

Scales what full input means: `move` and `turn` are multipliers on the model's full-input step and
turn, 1 being the model's own. A model with discrete or fixed-speed actions does not expose it.

## The pose layer: `set_camera_pose(poses)`, changed during play (live)

For models that can be conditioned on camera poses, directly or through a conversion in the port.

- `poses`: 4×4 camera-to-world matrices in the OpenCV convention, in the model's own world units,
  one per frame or one per chunk as `capabilities` states. Each call gives the poses for the chunks
  from the next boundary on; the model consumes them in order.
- While poses are set, `set_move` and `set_look` are ignored, and `state_update` says the camera pose
  is in control. `set_camera_pose([])` hands control back to move and look. When the given poses run
  out, the model holds the last pose and says so in `state_update`.
- A model that cannot take poses does not expose `set_camera_pose`; a client reads that from
  `capabilities`.

## `set_input(keys, mouse)`, changed during play (live)

For a model trained on raw keyboard and mouse state whose buttons are not movement (jump, attack,
use): `keys` is the full set of keys held now, `mouse` the held mouse state, both in the model's own
vocabulary listed in `capabilities`. Movement on such a model still goes through `set_move` and
`set_look` when its keys move the camera.
