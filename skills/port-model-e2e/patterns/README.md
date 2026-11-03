# The command standard for chunk-by-chunk models

A client that drives one ported model should drive the next with the same calls, changing only
arguments. This directory fixes those calls. Every port follows it: reuse a standard name for a
standard concept, and give a genuinely new feature one short name in the same shape.

The standard fixes names and meanings, not features. A model may lack a standard command (it then
does not expose it), and may have commands the standard does not mention.

## Three axes per input

Every input a model takes is described on three axes, none derived from another.

| Axis | The question | Fixed by |
|---|---|---|
| **Modality** | How does the data travel? Text, image, video, movement, a pointer | the modality's page |
| **Role** | What does the model do with it? First frame, reference, scene description, camera | the model's input table |
| **Kind** | When may it be set? | the standard's default for the command, and the model's `capabilities` |

The three kinds, named for the person first, with the formal term in parentheses:

| Kind | Meaning |
|---|---|
| **Server config** | set by the server, the same for every session; not a command |
| **Set before play** (setup) | accepted while idle only; holds for the whole session; sent while running, it is refused with `locked_while_running` |
| **Changed during play** (live) | accepted at any time; sent before `start`, it is the starting value; applies at the next chunk boundary and holds until changed |

Each standard command has a default kind. A model whose source cannot take a live input mid-run
(it encodes a reference once into its context) declares that command setup in `capabilities`; a
mid-run call is then refused, never answered by a silent restart.

## Lifecycle: five commands, the same on every model

| Command | Behaviour |
|---|---|
| `start()` | idle → running. Refused when a required input is missing or when already running. The model never starts on its own when an input arrives |
| `pause()` / `resume()` | hold on the last frame; resume continues from there. Live inputs sent while paused apply on resume |
| `stop()` | end generation and **keep** every setting; `start()` again replays the same conditions |
| `reset()` | end generation and **clear** every setting; allowed from any state |

## Events: one fixed set

| Message | Sent | Carries |
|---|---|---|
| `state_update` | from `@connected` and as the answer to every command | `state` (idle, running, paused), `settings` (every input's value sent, and the value applied at the last chunk), `capabilities` |
| `chunk_complete` | after every chunk | `index` |
| `generation_started`, `generation_complete` | when a run starts and ends | |

A refused command raises `CommandError(code, message)`; the runtime delivers it to the client, so
there is no model message for it. A change meant for chunk N is sent by the client after
`chunk_complete` of chunk N−1; no model has a scheduling command.

`capabilities` lists every command the model exposes with its kind, its value set or range, and what
its values mean where the source defines it ([movement.md](movement.md)).

## The standard commands

```
# lifecycle
start() · pause() · resume() · stop() · reset()

# set before play (setup)
set_first_frame(image)            image.md
video_in                          input track; video.md
set_seed(seed)                    -1 = random
set_resolution(name)              only when the model offers several

# changed during play (live)
set_prompt(prompt, cut=False)     text.md
set_references(items)             image.md; images, clips, voice
set_move(forward, right)          movement.md; -1..1
set_look(yaw, pitch)              movement.md; -1..1
set_speed(move, turn)             movement.md
set_camera_pose(poses)            movement.md; overrides move and look while set
set_input(keys, mouse)            movement.md; full held state, for button models
set_pointer(x, y, down)           pointer.md; x, y in 0..1 over the output frame
```

A new feature gets `set_<what it is>`, a kind, and a row in the input table: a separate text for
the sound is `set_audio_prompt(prompt)`, live.

**Setting an empty value removes an input**: `set_references([])`, `set_camera_pose([])`. There are no
clear commands.

**Positions use one coordinate system**: `x, y` in 0…1 over the output frame, origin at the top
left, or a region in the same units. No command defines its own.

## The model's input table

Phase 4 writes one row per client input before any command exists; the play page, the checklist and
the report read it.

| Input | Modality | Role | Kind | Command |
|---|---|---|---|---|
| first frame | image | the frame the video starts on | set before play (setup) | `set_first_frame` |
| reference | image | an appearance the model keeps to | changed during play (live), or setup when the source encodes it once | `set_references` |
| prompt | text | the scene description | changed during play (live) | `set_prompt` |
| movement | movement | where the camera goes | changed during play (live) | `set_move`, `set_look`, and `set_camera_pose` when the model takes poses |

## When two inputs control the same thing

| Both set | Wins | The other |
|---|---|---|
| `set_camera_pose`, and `set_move` / `set_look` | the camera pose | ignored until the pose is set to `[]`; `state_update` says which is in control |
| `video_in`, and `set_first_frame` | what the source does; read it | the description says which wins |

A conflict found in a port is added here.
