# Images and other references

An image says nothing about its role. The standard has two roles for still images, each with its own
command and default kind; any other role is a new feature named for what it is.

| Command | Default kind | Role |
|---|---|---|
| `set_first_frame(image: UploadedFile)` | set before play (setup) | frame 1 of the video is this picture |
| `set_references(items: list[UploadedFile])` | changed during play (live) | "look or sound like this": images, clips, voice; the video does not start on them |

**A first frame is setup** because it only means something at frame 1. A model that can continue
from a new picture mid-stream takes it as a new feature (`set_keyframe`), live.

**References are live by default**: a reference says what to look like, and a model that re-reads its
references at every chunk can take new ones at any boundary. A model whose source encodes them once
into the context declares `set_references` setup in `capabilities`.

`set_references` replaces the whole list; `[]` removes every reference. The list is brought to the
fixed count the model was traced at, padded with a mask; more than that count is refused with
`too_many_references`, giving the limit. Each item's type (image, clip, audio) must be one the model
takes, or it is refused with `unsupported_media`.

## On arrival

Decode and fit off the event loop (`asyncio.to_thread`), as `set_image` in
`examples/waypoint/waypoint.py` does:

- refuse what is not the expected media with `unsupported_media`, and what cannot be decoded with
  `undecodable_image` (or `undecodable_video`, `undecodable_audio`);
- fit to the model's resolution exactly as the source does (resize then centre crop, or pad), and
  state the rule and the size in the description.

The handler stores decoded pixels. VAE or image-encoder work runs in the model half, once, at the
step that applies the input, and is cached for the session. `reset` clears every image and
reference on every rank: a stale reference raises nothing and silently conditions the next session.
