# Model facts, reported at the start

Before anything long runs, the person gets one table of what the model is: read from the source's
code, config, README and checkpoint, each answer with the file and line (or config key) it came from.
It goes out in the same message as the questions, so the person sees the shape of the model in the
first minutes of a port rather than in the report hours later.

This file is the template. A person who wants other questions answered hands you their own template
with the request; use theirs in place of this one, and keep the rules below.

## Rules

- **Every answer names its source.** `configs/inference.yaml: num_frames_per_block` or
  `pipeline.py:212`. A value you could not find is written as **unknown** with where you looked; never
  a guess dressed as a fact.
- **Read, not measured.** Nothing here needs the model to run. Measured numbers (generation fps,
  step time, memory in use) belong to the report; when one later disagrees with what the code said,
  the report says so beside it.
- **Distinguish what the checkpoint fixes from what a session may choose.** A resolution the
  checkpoint was trained at only is a fixed value; a list of supported resolutions is a choice.
- **State limits as the code enforces them.** A maximum length is the number the code checks or the
  size of the buffer that fills (a context window, a KV cache, a positional embedding table), with
  what happens at the limit: an error, a rolling window, quality falling off.

## The questions

| Fact | What to state |
|---|---|
| What it is | one sentence: task (text-to-video, image-to-video, video-to-video, action-driven world model…), architecture family, parameter count |
| Inputs | each input with its **modality** (text, image, video, movement), its **role** (prompt, first frame, reference, action…), and whether the source takes it once or per chunk |
| Outputs | each output track: video (frame size, channels), audio (sample rate), anything else |
| Chunk size | video frames per chunk, and the latent frames behind them |
| Forwards per chunk | denoising steps × network calls per step (classifier-free guidance doubles it; a separate text or action encoder counts once per chunk or once per session — say which) |
| Resolution | the trained resolution, or the supported list; which one the baseline uses |
| Trained fps | the rate the model was trained at, and where it is read |
| Maximum length | frames or seconds per session, what enforces it, and what happens at the limit |
| Conditioning over time | whether prompt or actions can change at a chunk boundary, from how the code applies them |
| Weights | checkpoint size on disk, the files loaded, precision |
| GPU need | memory the source states or a README requirement, and the GPU it was run on; "unknown" when the source is silent |
| Licence | the code's and the weights' licence names |

## How it looks in the message

A table with the fact, the value, and the source, followed by the questions from the skill. Units on
every number: `12 video frames (3 latent frames)`, `480×832`, `4 steps × 2 calls (CFG) = 8 forwards`.
