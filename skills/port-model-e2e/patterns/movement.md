# Movement

Every model that moves through a scene accepts the same held keys and mouse look, so a person can
play any of them from a keyboard and a client drives all of them with the same calls. A model trained
on camera poses also accepts a camera path, for someone who designs the trajectory in advance.

```
                      held keys + mouse look  (every movement model)
                                   │
          ┌────────────────────────┴────────────────────────┐
   trained on camera poses                           trained on keys
          │                                                  │
   the port converts each chunk's                     the keys and look go
   keys and look into that chunk's poses              to the model as trained
          │                                                  │
   also accepts set_camera_path;                        no set_camera_path
   an active path overrides keys and look             in its schema
```

The model's native form is read from the source: what its conditioning tensor is built from during
training (a pose per frame or per latent frame, or a key and mouse encoding).

## Commands every movement model has

| Command | Argument | Meaning |
|---|---|---|
| `set_keys` | `held: list[str]` | the complete set of keys held now; replaces the previous set; `[]` releases all |
| `set_look` | `yaw: float`, `pitch: float` | look speed in degrees per second, held until changed; `0, 0` stops turning |

- **Keys are named by what they do**, from this vocabulary: `forward`, `back`, `left`, `right`,
  `up`, `down`, `run`, `jump`. A model declares the subset it supports in the `set_keys`
  description; a key outside it is refused with `unsupported_key`, listing the supported ones. A
  key specific to one model (an action with no equivalent here) keeps a descriptive name and is
  listed the same way.
- **Look is a speed, not a delta**, so it holds across chunk boundaries the way a held key does.
  Positive yaw turns right, positive pitch looks up. The description states the range.
- **A model whose look is discrete** (trained on a few look directions or speeds) accepts exactly
  those values and refuses any other with `unsupported_value`, listing them. A value is never snapped
  to the nearest one.
- **Both apply at the next chunk boundary and hold until changed.** The snapshot reports the keys
  and look sent and those applied at the last step.

## Converting keys and look to poses (pose-trained models)

The conversion lives in the port, in the application half, and runs once per chunk: integrate the
held keys and the look speed over the chunk's duration into that chunk's camera poses, starting from
the pose the previous chunk ended at, in the cadence the model takes (one pose per frame or per
latent frame).

Two numbers fix it, and both come from the model, never from feel:

- **translation speed**: the model's world units per second at full forward speed, read from the
  camera paths it was trained on or from the source's own path builder;
- **turn rate scale**: degrees per second that a look speed of 1 maps to, from the same source.

State both, and where each was read, in the `set_keys` and `set_look` descriptions and in the report.

## The camera path (pose-trained models only)

| Command | Argument | Meaning |
|---|---|---|
| `set_camera_path` | `path: UploadedFile \| None` (JSON) | a whole trajectory; `None` returns control to keys and look |

The file holds `poses`: a list of 4×4 camera-to-world matrices in the OpenCV convention, in the
model's world units, one per frame or per latent frame as the description states; and, when the
model takes them, `intrinsics` as the source expects them. A path whose length does not match what
the session will generate is refused with `invalid_camera_path`, giving the expected count.

Its timing is declared in the input table like any other input: set before play when the model
plans the whole path at the start, changed during play when it reads the next chunk's poses at each
boundary. While a path is active, `set_keys` and `set_look` are refused with `camera_path_active`. A
key-trained model has no `set_camera_path`; a client reads its absence from the schema.
