import itertools
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from reactor_runtime.schema import main, render

_MODEL_SOURCE = '''\
from reactor_runtime import ModelMessage, Output, ReactorModel, Video, event


class LevelSet(ModelMessage):
    """The level now in effect."""

    level: int


class Out(Output):
    video: Video


class Demo(ReactorModel):
    """A demo model."""

    output: Out

    @event(name="set_level", description="Set the level.")
    def set_level(self, level: int = 1) -> LevelSet:
        return LevelSet(level=level)

    def load(self, config_path=None) -> None:
        raise AssertionError("rendering a schema must not load the model")

    async def run(self) -> None: ...
'''

_MODULE_NAMES = itertools.count()


@pytest.fixture(autouse=True)
def _restore_imports() -> Iterator[None]:
    """Remove model modules that a render imports."""
    saved_modules = set(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """Write a model and its manifest to a directory, and return the directory.

    Each call names the module uniquely, so importing it runs the declarations
    that fill the registries rather than hitting a cached module.
    """
    module = f"demo_{next(_MODULE_NAMES)}"
    (tmp_path / f"{module}.py").write_text(_MODEL_SOURCE)
    (tmp_path / "reactor.yaml").write_text(f"runtime:\n  import: {module}:Demo\n")
    return tmp_path


def test_render_emits_the_model_identity(model_dir: Path) -> None:
    doc = render(model_dir)

    assert doc["openapi"] == "3.1.0"
    assert doc["info"]["title"] == "demo"
    assert doc["info"]["description"] == "A demo model."
    assert doc["x-reactor"]["tracks"] == [{"name": "video", "kind": "video", "direction": "out"}]


def test_render_titles_the_document_with_the_published_name(tmp_path: Path) -> None:
    # The published name is the model's identity, and it carries characters a
    # class name cannot. Titling from the class would spell this one "demo".
    module = f"demo_{next(_MODULE_NAMES)}"
    (tmp_path / f"{module}.py").write_text(_MODEL_SOURCE)
    (tmp_path / "reactor.yaml").write_text(
        f"model:\n  name: mage-vl\nruntime:\n  import: {module}:Demo\n"
    )

    assert render(tmp_path)["info"]["title"] == "mage-vl"


def test_render_falls_back_to_the_class_when_the_manifest_names_none(model_dir: Path) -> None:
    assert render(model_dir)["info"]["title"] == "demo"


def test_render_answers_a_command_with_its_reply_component(model_dir: Path) -> None:
    doc = render(model_dir)

    responses = doc["paths"]["/events/set_level"]["post"]["responses"]
    assert responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/LevelSet"
    }
    assert doc["components"]["schemas"]["LevelSet"]["properties"]["level"] == {"type": "integer"}


def test_render_stamps_the_default_release_tag(model_dir: Path) -> None:
    assert render(model_dir)["info"]["version"] == "v0.0.0"


@pytest.mark.parametrize(("given", "emitted"), [("1.4.0", "v1.4.0"), ("v1.4.0", "v1.4.0")])
def test_render_stamps_a_release_tag_the_way_the_command_does(
    model_dir: Path, given: str, emitted: str
) -> None:
    assert render(model_dir, given)["info"]["version"] == emitted


def test_render_refuses_a_version_that_is_not_a_release_tag(model_dir: Path) -> None:
    with pytest.raises(ValueError, match="is not a release tag"):
        render(model_dir, "release-1.4.0")


def test_render_leaves_the_callers_streams_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The stdout-to-stderr reroute belongs to the command; a code generator
    # calling render() while writing its own progress to stdout keeps it there.
    module = f"loud_{next(_MODULE_NAMES)}"
    (tmp_path / f"{module}.py").write_text(f'print("CUDA device 0 ready")\n{_MODEL_SOURCE}')
    (tmp_path / "reactor.yaml").write_text(f"runtime:\n  import: {module}:Demo\n")

    doc = render(tmp_path)

    captured = capsys.readouterr()
    assert doc["info"]["title"] == "demo"
    assert "CUDA device 0 ready" in captured.out
    assert captured.err == ""


def test_importing_the_module_leaves_the_server_stack_out() -> None:
    # Rendering a schema must not drag in the ASGI server, the native media
    # engine, or the encoder — the module exists so a code generator can get
    # the document without paying for a transport stack it never uses.
    probe = (
        "import sys\n"
        "import reactor_runtime.schema\n"
        "heavy = [name for name in ('fastapi', 'uvicorn', 'reactor_webrtc', 'av')\n"
        "         if name in sys.modules]\n"
        "raise SystemExit(f'transitively imported: {heavy}' if heavy else 0)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_render_restores_the_import_path(model_dir: Path) -> None:
    previous = list(sys.path)

    render(model_dir)

    assert sys.path == previous


def test_render_refuses_a_directory_without_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"no reactor\.yaml found"):
        render(tmp_path)


def test_render_refuses_a_manifest_that_names_no_model(tmp_path: Path) -> None:
    (tmp_path / "reactor.yaml").write_text("model:\n  name: demo\n")

    with pytest.raises(SystemExit, match=r"missing runtime\.import"):
        render(tmp_path)


def test_render_preserves_a_model_import_error(tmp_path: Path) -> None:
    (tmp_path / "reactor.yaml").write_text("runtime:\n  import: absent:Demo\n")

    with pytest.raises(ModuleNotFoundError, match="absent"):
        render(tmp_path)


def test_main_keeps_a_failure_inside_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A model that reads a file as it imports must keep the traceback that names
    # the failing line, rather than collapsing into a one-line message.
    (tmp_path / "hungry.py").write_text("open('vocab.json')\n")
    (tmp_path / "reactor.yaml").write_text("runtime:\n  import: hungry:Demo\n")
    monkeypatch.setattr(sys, "argv", ["schema", "--path", str(tmp_path)])

    with pytest.raises(FileNotFoundError, match=r"vocab\.json"):
        main()


def test_main_prints_the_document_and_nothing_else(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["schema", "--path", str(model_dir)])

    main()

    # Standard output has to parse on its own, so a caller can redirect it.
    captured = capsys.readouterr()
    assert json.loads(captured.out)["info"]["title"] == "demo"
    assert captured.err == ""


def test_main_keeps_what_the_model_prints_off_standard_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = f"loud_{next(_MODULE_NAMES)}"
    (tmp_path / f"{module}.py").write_text(f'print("CUDA device 0 ready")\n{_MODEL_SOURCE}')
    (tmp_path / "reactor.yaml").write_text(f"runtime:\n  import: {module}:Demo\n")
    monkeypatch.setattr(sys, "argv", ["schema", "--path", str(tmp_path)])

    main()

    captured = capsys.readouterr()
    assert json.loads(captured.out)["info"]["title"] == "demo"
    assert "CUDA device 0 ready" in captured.err


def test_main_reads_the_working_directory_by_default(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(model_dir)
    monkeypatch.setattr(sys, "argv", ["schema"])

    main()

    assert json.loads(capsys.readouterr().out)["info"]["title"] == "demo"


def test_main_writes_the_document_to_out(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = model_dir / "generated" / "schema.json"
    monkeypatch.setattr(sys, "argv", ["schema", "--path", str(model_dir), "--out", str(out)])

    main()

    assert json.loads(out.read_text())["info"]["title"] == "demo"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(out) in captured.err


@pytest.mark.parametrize(
    ("given", "emitted"),
    [
        ("1.4.0", "v1.4.0"),
        ("v1.4.0", "v1.4.0"),
        ("1.4.0-gac767ec", "v1.4.0-gac767ec"),
    ],
)
def test_main_stamps_the_release_tag_with_its_prefix(
    model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    given: str,
    emitted: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["schema", "--path", str(model_dir), "--version", given])

    main()

    assert json.loads(capsys.readouterr().out)["info"]["version"] == emitted


@pytest.mark.parametrize("given", ["1.4", "1.4.0-dirty", "release-1.4.0", "1.4.0\n", ""])
def test_main_rejects_a_version_that_is_not_a_release_tag(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch, given: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["schema", "--path", str(model_dir), "--version", given])

    with pytest.raises(SystemExit, match="is not a release tag"):
        main()


def test_main_rejects_the_version_before_it_reads_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["schema", "--path", str(tmp_path), "--version", "1.4"])

    # The directory holds no manifest, so only argument-first validation can
    # produce the version error rather than the missing-manifest one.
    with pytest.raises(SystemExit, match="is not a release tag"):
        main()
