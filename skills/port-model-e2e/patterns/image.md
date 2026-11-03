# Image

An image says nothing about its role. The same upload is a first frame in one model and a reference
in another, and one model may take several. Each role is its own input with its own command and its
own timing.

## Commands

| Command | Argument | Answers with |
|---|---|---|
| `set_<role>`: `set_first_frame`, `set_reference`, `set_style` | `image: UploadedFile` | the state snapshot |
| `set_<role>` for a role that takes several images: `set_references` | `images: list[UploadedFile]`, replacing the whole set | the state snapshot |
| `clear_<role>`, for an optional role | none | the state snapshot |

A role the model needs before it can start (a first frame, in an image-to-video model) is refused at
start when missing (`missing_first_frame`). An optional role is absent until set and absent again
after `clear_<role>`.

## On arrival

Decode and fit off the event loop (`asyncio.to_thread`), the way `set_image` in
`examples/waypoint/waypoint.py` does:

- refuse what is not an image with `unsupported_media`, and what cannot be decoded with
  `undecodable_image`;
- fit to the model's resolution exactly as the source does (resize then centre crop, or pad), and
  state the rule and the size in the description;
- a role that takes several images is brought to the fixed count the model was traced at, padded
  with a mask; more than that count is refused with `too_many_images`, giving the limit.

The handler stores the decoded pixels. VAE or image-encoder work runs in the model half, once, at the
step that applies the image, and is cached by image for the session.

## Timing

Declared per role in the input table. Read it from the source: an image encoded once before the
generation loop is set before play; an image the model reads at every chunk (a reference injected
into each chunk's attention) can be changed during play, applying at the next chunk boundary.

- **A first frame** is almost always set before play: it is frame 0, and changing it means a new
  video. When the source supports continuing from a new frame mid-stream, it is a separate role
  (`set_keyframe`) with its own row.
- **A reference** is set before play when the source encodes it once into the context, and changed
  during play when it is re-read per chunk.
- `reset` and session end clear every image input on every rank. A stale reference raises nothing;
  it silently conditions the next session.
