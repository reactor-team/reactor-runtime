# Input patterns

A client that drives one ported model should be able to drive the next one with the same calls.
These pages make that possible: each fixes how one **modality** reaches a served model, so two
models that take the same kind of input expose the same commands.

## Three axes, declared separately

Every input a model takes is described on three axes. None of them is derived from another.

| Axis | The question it answers | Where it is fixed |
|---|---|---|
| **Modality** | How does the data travel? Text, an image, a video, movement, a touch | the modality's page below |
| **Role** | What does the model do with it? First frame, reference, style, the scene description, a camera path | the model's input table |
| **Timing** | When may it be set? **Server config**, **set before play** or **changed during play** ([SKILL.md](../SKILL.md)) | the model's input table, per input |

A modality fixes neither role nor timing. An image may be the first frame of the video or a
reference the model keeps looking at; a model may take several images at once, each with its own
role; a reference image may be set before play in one model and replaceable mid-generation in
another. A prompt is changed during play in most chunk-by-chunk models and set before play in a
model whose history must restart when the text changes. Read each from the source, per input.

## The model's input table

Phase 4 writes one row per client input, before any command exists. It is the port's interface, and
the play page, the checklist and the report all read it.

| Input | Modality | Role | Timing | Applies at | Command |
|---|---|---|---|---|---|
| `first_frame` | image | the frame the video starts from | set before play | the first step | `set_first_frame` |
| `reference` | image | appearance the model keeps to | set before play and changed during play | the next chunk boundary | `set_reference` |
| `prompt` | text | the scene description | changed during play | the next chunk boundary | `set_prompt` |
| movement | movement | where the camera goes | changed during play | the next chunk boundary | `set_keys`, `set_look`; `set_camera_path` when the model is trained on poses |

**One command per input, named `set_<role>`.** The role is the input's name: `set_first_frame`,
`set_reference`, `set_prompt`, `set_negative_prompt`. Two inputs of one modality are two commands,
because they are two roles. Setting an empty value (`""`, `None`, `[]`) is how an optional input is
removed; there are no separate clear commands. Movement is the exception: its commands are fixed by
[movement.md](movement.md), whatever the model calls them internally.

**One state snapshot answers every command** (see [port-on-reactor-app.md](../references/port-on-reactor-app.md)
§8) and carries, for each input changed during play, both the value last sent and the value the
model applied at its last step.

## Composing a model from pages

A model is the set of modalities it takes. Read each input's page, and nothing else is needed for
the combination: a model that takes text, an input video, movement and a touch follows
[text.md](text.md), [video.md](video.md), [movement.md](movement.md) and the touch rules in
[port-on-reactor-app.md](../references/port-on-reactor-app.md) §5, one input at a time.

Inputs interact only when two of them control the same thing. Those cases are listed here, once:

| When both are set | The one that wins | What the other does |
|---|---|---|
| a camera path, and held keys or mouse look | the camera path | refused with `camera_path_active` until the path is set to empty |
| an input video, and a first frame | whatever the source does; read it | the port follows the source and the description says which wins |

A new conflict found in a port is added as a row here.

## Modalities

| Page | Covers |
|---|---|
| [text.md](text.md) | prompts and any other free text |
| [image.md](image.md) | first frames, references, any still image |
| [video.md](video.md) | a live input stream, or an uploaded clip |
| [movement.md](movement.md) | held keys, mouse look and camera paths, as one interface |

Touch (a click, a drag, several fingers) follows [port-on-reactor-app.md](../references/port-on-reactor-app.md)
§5 until a page for it is written. A modality none of these covers (a force applied to a point, a
sound cue) keeps the source's own form, with a command named for its role, its units and range
stated in the description, its timing declared in the input table, and its shape fixed on arrival.
When a second model takes the same modality, write its page here and move both onto it.
