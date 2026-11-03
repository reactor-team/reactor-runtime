# Text

## Commands

| Command | Argument | Answers with |
|---|---|---|
| `set_<role>`, usually `set_prompt` | `text: str` | the state snapshot, with the stored text |

A model with several text inputs has one command per role (`set_prompt`, `set_negative_prompt`,
`set_subject_prompt`). A model whose text is split by time (a prompt per segment) takes it as one
input per segment boundary through the same command: the newest text applies from the next chunk.

## On arrival

- **Strip once, in the handler.** The stripped text is the only value stored, encoded and reported.
- **An empty text is a value.** It is encoded as the source encodes the empty string. What is
  refused is starting before any text was sent (`missing_prompt`).
- **The only limit is the token budget.** Count tokens the way the model's encoder counts them,
  special tokens included, and refuse a text over the budget with `prompt_too_long`, giving the count
  and the limit. No character limit, anywhere.
- **One shape.** Pad or truncate the token sequence to the length the model was traced at, the way
  the source does, so no length adds a compile shape.

## Timing

Declared per input in the input table. For an input changed during play:

- the handler stores the newest text; the model encodes it once, at the chunk boundary that applies
  it, and caches encodings by text for the session. Texts superseded before that boundary are never
  encoded;
- the snapshot carries the text sent and the text applied at the last step;
- when the model's history must restart for a new text to take effect (the source re-encodes the
  whole context, or the cross-attention cache cannot be swapped mid-stream), the input is set before
  play, or the command description says that a change restarts the stream and the state reports it.
  Read which from the source; never cut silently.
