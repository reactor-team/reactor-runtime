# Images and references

An image says nothing about its role. Each role is its own command; its kind is read from the source.

| Command | Role |
|---|---|
| `set_first_frame(image: UploadedFile \| None)` | frame 1 of the video is this picture |
| `set_references(items: list[UploadedFile])` | "look or sound like this": images, clips, voice; the video does not start on them |
| `set_<role>(image: UploadedFile \| None)` | any other role (`set_style_image`, `set_mask`) |

A first frame only means something at frame 1, so it is almost always set before play (setup); a
source that can continue from a new picture mid-stream takes that as its own role (`set_keyframe`).
References are changed during play (live) when the source re-reads them at every chunk, and set
before play (setup) when it encodes them once into the context.

`set_references` replaces the whole list; `[]` removes every reference. References that do different
jobs (one sets the style, another the identity) are two roles, two commands; a list holds references
that do the same job. The list is brought to the fixed count the model was traced at, padded with a
mask; more is refused with `too_many_references`, giving the limit. An item of a type the model does
not take is refused with `unsupported_media`.

A role the model needs before it can start is refused at `start()` when missing
(`missing_first_frame`).

## On arrival

Decode and fit off the event loop (`asyncio.to_thread`), as `set_image` in
`examples/waypoint/waypoint.py` does:

- refuse what is not the expected media with `unsupported_media`, and what cannot be decoded with
  `undecodable_image` (`undecodable_video`, `undecodable_audio`);
- fit to the model's resolution exactly as the source does (resize then centre crop, or pad), and
  state the rule and the size in the description.

The handler stores decoded pixels. VAE or image-encoder work runs in the model half, once, at the
step that applies the input, and is cached for the session. `reset` clears every image on every rank:
a stale reference raises nothing and silently conditions the next session.
