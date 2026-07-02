"""The committed OpenAPI contract stays true to the code."""

from __future__ import annotations

import json
from pathlib import Path

from reactor_runtime.http.spec import render_openapi, render_spec_json

SPEC_PATH = Path(__file__).resolve().parents[3] / "api" / "openapi.json"


def test_committed_spec_is_fresh() -> None:
    """api/openapi.json matches a fresh render byte for byte.

    The committed spec is the contract artifact the gate diffs; a stale file
    would gate against a surface that no longer exists. Regenerate with
    ``mise run http-spec``.
    """
    assert SPEC_PATH.exists(), "api/openapi.json is missing — run `mise run http-spec`"
    assert SPEC_PATH.read_text() == render_spec_json(), (
        "api/openapi.json is stale — run `mise run http-spec` and commit the result"
    )


def test_rendered_spec_covers_the_served_surface() -> None:
    """The rendered document names every route group's paths.

    A sentinel path per mount site: the session routes, the egress routes, the
    upload routes, the recording routes, and the WebRTC transport router. A
    missing sentinel means the spec renderer no longer assembles the app the
    served process runs.
    """
    paths = render_openapi()["paths"]
    for sentinel in (
        "/start_session",
        "/events",
        "/sessions/{sid}/uploads",
        "/clips",
        "/sessions/{sid}/transport/webrtc/connections/{cid}/sdp_params",
    ):
        assert sentinel in paths, f"expected {sentinel} in the rendered spec"


def test_spec_json_is_canonical() -> None:
    """The serialization is byte-stable: sorted keys, fixed indent, one trailing newline."""
    rendered = render_spec_json()
    assert rendered.endswith("}\n")
    assert rendered == json.dumps(json.loads(rendered), indent=2, sort_keys=True) + "\n"
