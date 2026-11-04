# Text

## `set_prompt(prompt)`

All the text a model reads goes through `set_prompt`, in the model's own layout: a model whose
prompt has sections (a picture description, a soundscape) takes them in one string or one
structured value, as its source does, and the description shows the layout. A second, independent
text is its own role with its own command (`set_audio_prompt`).

Its kind is read from the source, by the rule in
[`application-model-isolation`](../../application-model-isolation/SKILL.md) rule 4. A live
control in most chunk-by-chunk models, which re-read the text at every chunk: a new prompt applies
at the next chunk boundary and nothing restarts. Session setup when the text prefilled state the run
carries forward, so the model's history must restart for new text to take effect; it never
restarts silently.

## On arrival

- **Marked for moderation.** The prompt is declared with `InputField(..., moderate=True)`
  ([port-on-reactor-app.md](../references/port-on-reactor-app.md) §8).
- **Strip once, in the handler.** The stripped text is the only value stored, encoded and reported.
- **An empty prompt is a value**, encoded as the source encodes the empty string. What is refused is
  `start()` before any prompt was sent (`missing_prompt`).
- **The only limit is the token budget**, counted as the model's encoder counts it, special tokens
  included. Over it: `prompt_too_long`, with the count and the limit. No character limit, anywhere.
- **One shape.** Pad or truncate the token sequence to the traced length, as the source does.
- **Encode at the boundary, not in the handler.** The handler stores the newest prompt; the model
  encodes it once, at the boundary that applies it, and caches encodings by text for the session.
  Prompts superseded before that boundary are never encoded.
