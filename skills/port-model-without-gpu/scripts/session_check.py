"""Session check: one client session against a deployed Reactor model, recorded.

Connects with the public Python SDK (reactor-sdk) using a short-lived session token,
optionally uploads a first-frame image, sends a list of commands, records the output
video track for N seconds after the first frame, then disconnects. Writes into --out:

  summary.json     connect time, start -> first frame, frame count and size, fps,
                   per-second frame counts, largest gap, stats, any error
  messages.jsonl   every event and message received, with its time since launch
  output.mp4       the received frames
  still_first.png, still_mid.png, still_last.png

Status: untested. This script has not been run end to end against a deployed model yet;
expect to fix SDK call details against the reactor-sdk version you install.

Usage:
  export REACTOR_JWT="$(scripts/mint_token.sh <org>/<name>)"
  python session_check.py --model <org>/<name> --commands commands.json \
      [--first-frame image.jpg] [--seconds 30] [--out out/session-1]

For an environment other than production, mint the token for <org>/<name>:<environment>;
whether the Python SDK takes that form as --model is unverified.

The token is read from $REACTOR_JWT and removed from the environment after use; it is
never taken from argv.

Commands are a JSON list, from a file (--commands) or from $SESSION_COMMANDS. Each item is
{"command": <name>, "args": {...}}. They are sent in order, after the first frame:

  [
    {"command": "set_prompt", "args": {"prompt": "a lighthouse on a cliff at dusk"}},
    {"command": "set_seed",   "args": {"seed": 42}},
    {"command": "start",      "args": {}}
  ]

--first-frame uploads the file and sends --first-frame-command (default set_first_frame)
with {--first-frame-field: <upload ref>} (default field: image) before the list.
The start-to-first-frame clock starts when --start-command (default start) is sent, or
after the last command when the list has no such command.

Needs: pip install reactor-sdk numpy pillow imageio imageio-ffmpeg. No GPU.
Exit status: 0 frames received; 2 SDK error; 3 other error; 4 no frames.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from reactor_sdk import Reactor, ReactorError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="<org>/<name>")
    p.add_argument("--api-url", default=os.environ.get("REACTOR_SERVER", "https://api.reactor.inc"))
    p.add_argument("--commands", type=Path, help="JSON file with the command list (else $SESSION_COMMANDS)")
    p.add_argument("--first-frame", type=Path, help="image to upload before the commands")
    p.add_argument("--first-frame-command", default="set_first_frame")
    p.add_argument("--first-frame-field", default="image")
    p.add_argument("--start-command", default="start")
    p.add_argument("--track", default="main_video", help="output video track to record")
    p.add_argument("--seconds", type=float, default=30.0, help="record this long after the first frame")
    p.add_argument("--first-frame-timeout", type=float, default=180.0)
    p.add_argument("--mp4-fps", type=float, default=16.0, help="frame rate written into output.mp4")
    p.add_argument("--out", type=Path, default=Path("out/session-check"))
    return p.parse_args()


def load_commands(args: argparse.Namespace) -> list[dict]:
    if args.commands is not None:
        raw = args.commands.read_text()
    else:
        raw = os.environ.get("SESSION_COMMANDS", "[]")
    cmds = json.loads(raw)
    if not isinstance(cmds, list) or not all(isinstance(c, dict) and "command" in c for c in cmds):
        raise SystemExit('commands must be a JSON list of {"command": ..., "args": {...}}')
    return cmds


class Recorder:
    """Timestamps every event into messages.jsonl and keeps the received frames."""

    def __init__(self, out: Path) -> None:
        self.t0 = time.monotonic()
        self.frames: list[tuple[float, np.ndarray]] = []
        self.file = open(out / "messages.jsonl", "w")
        self._caps_seen = False

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def log(self, kind: str, **kw) -> None:
        rec = {"t": self.now(), "kind": kind, **kw}
        self.file.write(json.dumps(rec, default=str) + "\n")
        self.file.flush()
        if not (kind == "message" and kw.get("type") == "chunk_complete"):
            print(json.dumps(rec, default=str)[:600], flush=True)

    def shrink(self, msg):
        """Keep a repeated capabilities dict only the first time it arrives."""
        if isinstance(msg, dict) and isinstance(msg.get("data"), dict) and "capabilities" in msg["data"]:
            data = dict(msg["data"])
            if self._caps_seen:
                data["capabilities"] = "<omitted, same as first>"
            self._caps_seen = True
            return {**msg, "data": data}
        return msg


async def run(args: argparse.Namespace, commands: list[dict], rec: Recorder, summary: dict) -> int:
    jwt = os.environ.pop("REACTOR_JWT", None)
    if not jwt:
        raise SystemExit("REACTOR_JWT is not set; mint one with scripts/mint_token.sh")
    reactor = Reactor(model_name=args.model, jwt=jwt, api_url=args.api_url, generation="streamed")
    del jwt

    reactor.on("status_changed", lambda s: rec.log("status", status=str(s)))
    reactor.on("session_id_changed", lambda s: rec.log("session_id", session_id=s))
    reactor.on("error", lambda e: rec.log("error", code=getattr(e, "code", None), message=str(e)))
    reactor.on("track_received", lambda name, mid=None: rec.log("track_received", name=name, mid=mid))
    reactor.on("capabilities_received", lambda c: rec.log("capabilities", caps=c))
    reactor.on("runtime_message", lambda m: rec.log("runtime_message", msg=m))
    reactor.on("message", lambda m: rec.log(
        "message", type=(m.get("type") if isinstance(m, dict) else None), msg=rec.shrink(m)))

    video = reactor.track(args.track)

    @video.on_frame
    def on_frame(frame):
        rec.frames.append((time.monotonic(), np.array(frame, copy=True)))

    rc = 0
    try:
        t = time.monotonic()
        await reactor.connect()
        summary["connect_s"] = round(time.monotonic() - t, 3)
        summary["session_id"] = reactor.session_id
        summary["tracks"] = [(tr.name, str(tr.kind), str(tr.direction)) for tr in reactor.tracks]
        rec.log("connected", seconds=summary["connect_s"], status=str(reactor.status))

        if args.first_frame is not None:
            t = time.monotonic()
            ref = await reactor.upload_file(str(args.first_frame))
            summary["upload_s"] = round(time.monotonic() - t, 3)
            rec.log("uploaded", path=str(args.first_frame), ref=str(ref))
            reply = await reactor.send_command(args.first_frame_command, {args.first_frame_field: ref})
            rec.log("reply", command=args.first_frame_command, reply=rec.shrink(reply))

        names = [c["command"] for c in commands]
        start_idx = names.index(args.start_command) if args.start_command in names else len(commands) - 1
        n_before = len(rec.frames)
        t_start = time.monotonic()
        for i, c in enumerate(commands):
            if i == start_idx:
                n_before = len(rec.frames)
                t_start = time.monotonic()
            reply = await reactor.send_command(c["command"], c.get("args", {}))
            rec.log("reply", command=c["command"], reply=rec.shrink(reply))
        summary["frames_before_start"] = n_before
        summary["_t_start"] = t_start

        deadline = t_start + args.first_frame_timeout
        while len(rec.frames) <= n_before and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if len(rec.frames) <= n_before:
            raise RuntimeError(f"no frame within {args.first_frame_timeout:.0f} s of start")
        t_first = rec.frames[n_before][0]
        summary["start_to_first_frame_s"] = round(t_first - t_start, 3)
        rec.log("first_frame", after_start_s=summary["start_to_first_frame_s"])
        while time.monotonic() < t_first + args.seconds:
            await asyncio.sleep(0.25)

        try:
            st = await reactor.get_stats()
            summary["stats"] = {k: getattr(st, k, None) for k in (
                "rtt_ms", "frames_per_second", "incoming_bitrate_bps", "packet_loss_ratio",
                "candidate_type", "bytes_received")}
            summary["stats"]["inbound"] = [
                {k: getattr(s, k, None) for k in ("kind", "frames_decoded", "frames_dropped",
                                                  "frame_width", "frame_height", "frames_per_second")}
                for s in getattr(st, "inbound", [])]
        except Exception as e:  # stats are context only
            summary["stats_error"] = repr(e)
    except ReactorError as e:
        rc = 2
        summary["error"] = {"class": type(e).__name__, "code": getattr(e, "code", None), "message": str(e),
                            "operation": getattr(e, "operation", None), "status": getattr(e, "status", None)}
        rec.log("exception", **summary["error"])
    except Exception as e:
        rc = 3
        summary["error"] = {"class": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        rec.log("exception", **summary["error"])
    finally:
        t = time.monotonic()
        try:
            await reactor.disconnect()
            summary["disconnect_s"] = round(time.monotonic() - t, 3)
            rec.log("disconnected", status=str(reactor.status))
        except Exception as e:
            summary["disconnect_error"] = repr(e)
            rec.log("disconnect_error", error=repr(e))
    return rc


def write_outputs(args: argparse.Namespace, rec: Recorder, summary: dict) -> list:
    n0 = summary.get("frames_before_start", 0)
    t_start = summary.pop("_t_start", None)
    gen = rec.frames[n0:] if t_start is not None else []
    summary["frames_total_received"] = len(rec.frames)
    summary["frames_after_start"] = len(gen)
    if not gen:
        return gen
    summary["frame_shapes"] = [list(s) for s in sorted({f.shape for _, f in gen})]
    span = gen[-1][0] - gen[0][0]
    summary["receive_span_s"] = round(span, 3)
    summary["fps_over_run"] = round((len(gen) - 1) / span, 3) if span > 0 else None
    buckets: dict[int, int] = {}
    for ts, _ in gen:
        k = int(ts - gen[0][0])
        buckets[k] = buckets.get(k, 0) + 1
    summary["frames_per_second_bucket"] = [buckets.get(i, 0) for i in range(max(buckets) + 1)]
    gaps = np.diff([ts for ts, _ in gen])
    summary["max_gap_s"] = round(float(gaps.max()), 3) if len(gaps) else None
    summary["mean_pixel_samples"] = [round(float(f.mean()), 1) for _, f in gen[:: max(1, len(gen) // 20)]]
    with imageio.get_writer(args.out / "output.mp4", fps=args.mp4_fps, codec="libx264",
                            quality=8, macro_block_size=8) as w:
        for _, f in gen:
            w.append_data(f)
    summary["mp4"] = {"path": "output.mp4", "fps_written": args.mp4_fps}
    for name, idx in (("still_first", 0), ("still_mid", len(gen) // 2), ("still_last", len(gen) - 1)):
        Image.fromarray(gen[idx][1]).save(args.out / f"{name}.png")
    return gen


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commands = load_commands(args)
    rec = Recorder(args.out)
    summary: dict = {"model": args.model, "api_url": args.api_url, "commands": commands,
                     "first_frame": str(args.first_frame) if args.first_frame else None,
                     "seconds_target": args.seconds}
    rc = asyncio.run(run(args, commands, rec, summary))
    gen = write_outputs(args, rec, summary)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in ("stats", "commands")},
                                  default=str), flush=True)
    rec.file.close()
    return rc if (gen or rc) else 4


if __name__ == "__main__":
    raise SystemExit(main())
