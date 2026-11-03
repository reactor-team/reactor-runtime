# The command standard for chunk-by-chunk models

A client that drives one ported model should drive the next by changing arguments only. This
directory fixes the calls that make that possible: one lifecycle, one set of events, and one command
per concept. It fixes names and meanings, not features: a model exposes only the commands it has,
and may add its own.

## Three axes per input

Every input a model takes is described on three axes, none derived from another.

| Axis | The question | Where it is fixed |
|---|---|---|
| **Modality** | How does the data travel? Text, image, video, movement, a pointer | the modality's page |
| **Role** | What does the model do with it? First frame, reference, scene description, camera | the command's name |
| **Kind** | When may it be set? | the model's input table, read from its source, and published in `capabilities` |

No command has a fixed kind. A prompt is a live control in most chunk-by-chunk models and session setup in one whose history must restart for new text; a reference is re-read every chunk in
one model and encoded once in another. Read each input's kind from the source.

The kinds, each with the plain explanation to give a person the first time it is named:

| Kind | Meaning |
|---|---|
| **Serving config** | decided when the model starts, the same for every session; not a command |
| **Session setup** | accepted while idle; holds for the whole run; sent while running, refused with `locked_while_running` |
| **Live control** | accepted at any time: sent before `start` it is the starting value, sent while running it applies at the next chunk boundary; held until changed |

## Lifecycle: the same on every model

| Command | Behaviour |
|---|---|
| `start()` | idle → running. Refused when a required input is missing or when already running. A model never starts on its own when an input arrives |
| `pause()` / `resume()` | hold on the last frame; resume continues from there. Live inputs sent while paused apply on resume |
| `reset()` | end generation and **clear** every input, on every rank; allowed from any state. To run the same conditions again, set them again (on a play page: reset, then pick the example) |

## Events

| Message | Sent | Carries |
|---|---|---|
| `state_update` | from `@connected` and as the answer to every command | `state` (idle, running, paused); `settings`: each input's value last sent and the value applied at the last chunk; `capabilities` |
| `chunk_complete` | after every chunk | `index` |
| `generation_started`, `generation_complete` | when a run starts and ends | |

`capabilities` lists every command the model exposes with its kind, its accepted values or range,
and what the values mean where the source defines it.

A refused command raises `CommandError(code, message)`; the runtime delivers it to the client, so no
model message is needed. A change meant for chunk N is sent by the client after `chunk_complete` of
chunk N−1, so no model needs a scheduling command.

## One command per concept

| Concept | Command | Page |
|---|---|---|
| text the model reads | `set_prompt(prompt)` | [text.md](text.md) |
| the frame the video starts on | `set_first_frame(image)` | [image.md](image.md) |
| what to look or sound like | `set_references(items)` | [image.md](image.md) |
| a video to rewrite | `video_in` input track, or `set_source_video(video)` for a file | [video.md](video.md) |
| walk and turn | `set_move(forward, right)`, `set_look(yaw, pitch)` | [movement.md](movement.md) |
| a camera trajectory | `set_camera_pose(poses)` | [movement.md](movement.md) |
| raw keys and mouse that are not movement | `set_input(keys, mouse)` | [movement.md](movement.md) |
| a touch or a drag | `set_pointer(x, y, down)` | [pointer.md](pointer.md) |
| randomness | `set_seed(seed)`, −1 for random | |
| output size, when the model offers several | `set_resolution(name)` | |

Every other input is named `set_<its role>` (`set_style_image`, `set_audio_prompt`), and two roles of
one modality are two commands. An argument a model or an application needs on a standard command is
added as a keyword with a default, so a client that does not send it is unaffected.

**Setting an empty value removes an input** (`""`, `None`, `[]`); there are no clear commands.

**Positions use one coordinate system**: `x, y` in 0…1 over the output frame, origin at the top
left, or a region in the same units.

## The model's input table

Phase 4 writes one row per client input before any command exists; the play page, the checklist and
the report read it, and `capabilities` publishes it.

| Input | Modality | Role | Kind | Command |
|---|---|---|---|---|
| first frame | image | the frame the video starts on | session setup | `set_first_frame` |
| reference | image | an appearance the model keeps to | live control: re-read at every chunk | `set_references` |
| prompt | text | the scene description | live control | `set_prompt` |
| movement | movement | where the camera goes | live control | `set_move`, `set_look` |

## When two inputs control the same thing

| Both set | Wins | The other |
|---|---|---|
| `set_camera_pose`, and `set_move` / `set_look` | the camera pose | ignored until the pose is set to `[]`; `state_update` says which is in control |
| a video to rewrite, and `set_first_frame` | what the source does; read it | the description says which wins |

A conflict found in a port is added here.
