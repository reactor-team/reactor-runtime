"""Start several step sessions at once and report how fast each one stepped.

Run against a runtime serving the batch example::

    python bench.py --url http://localhost:8080 --sessions 4 --steps 300

Each session is a step session with its own seed. Its rate is read from the
``step_completed`` facts on its own event stream: the steps it completed after
its first, over the time between its first and its last. The aggregate is
every session's steps over the span from the first step of any session to the
last, and the sum of the sessions' own rates is reported beside it. With
``--digests`` the frame fingerprints of every step are fetched as well, for a
runtime whose manifest keeps step results.

Uses the standard library only, so it runs anywhere Python does.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
import uuid
from typing import Any


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result: dict[str, Any] = json.loads(response.read())
        return result


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        body: bytes = response.read()
        return body


def follow(base: str, sid: str, steps: int, out: dict[str, Any], ready: threading.Event) -> None:
    """Read *sid*'s event stream until it ends, recording each step's timestamp."""
    times: list[int] = []
    ended = False
    with urllib.request.urlopen(f"{base}/sessions/{sid}/events?since=0", timeout=600) as stream:
        ready.set()
        for raw in stream:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event["event"] == "step_completed" and len(times) < steps:
                times.append(event["ts"])
            if event["event"] == "cleanup_complete":
                ended = True
                break
    out[sid] = {"times_ms": times, "ended": ended}


def run(base: str, sessions: int, steps: int, *, digests: bool = False) -> dict[str, Any]:
    """Run *sessions* step sessions of *steps* steps at once and measure them."""
    sids = [str(uuid.uuid4()) for _ in range(sessions)]
    results: dict[str, Any] = {}
    threads = []
    for sid in sids:
        ready = threading.Event()
        thread = threading.Thread(target=follow, args=(base, sid, steps, results, ready))
        thread.start()
        ready.wait(timeout=10)
        threads.append(thread)
    for index, sid in enumerate(sids):
        _post(
            f"{base}/start_session",
            {"session_id": sid, "steps": steps, "starting_input": {"state": {"seed": index + 1}}},
        )
    for thread in threads:
        thread.join()
    per_session = []
    for index, sid in enumerate(sids):
        times = results[sid]["times_ms"]
        span = (times[-1] - times[0]) / 1000 if len(times) > 1 else 0.0
        entry: dict[str, Any] = {
            "session_id": sid,
            "seed": index + 1,
            "steps": len(times),
            "steps_per_s": (len(times) - 1) / span if span > 0 else None,
        }
        if digests:
            entry["digests"] = [
                _get(f"{base}/sessions/{sid}/steps/{step}/frame.sha256").decode()
                for step in range(1, steps + 1)
            ]
        per_session.append(entry)
    first = min(results[sid]["times_ms"][0] for sid in sids)
    last = max(results[sid]["times_ms"][-1] for sid in sids)
    total = sum(len(results[sid]["times_ms"]) for sid in sids)
    span = (last - first) / 1000
    rates = [entry["steps_per_s"] for entry in per_session if entry["steps_per_s"] is not None]
    return {
        "sessions": per_session,
        "aggregate_steps_per_s": total / span if span > 0 else None,
        "sum_of_session_steps_per_s": sum(rates),
    }


def main() -> None:
    """Parse the arguments, run the sessions, and print the measurement as JSON."""
    parser = argparse.ArgumentParser(description="Measure concurrent step sessions.")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--digests", action="store_true")
    args = parser.parse_args()
    started = time.time()
    report = run(args.url.rstrip("/"), args.sessions, args.steps, digests=args.digests)
    report["wall_s"] = time.time() - started
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
