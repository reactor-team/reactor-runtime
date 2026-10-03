from pathlib import Path

import pytest

from reactor_runtime import ReactorApp
from reactor_runtime.core import RecordingConfig, StepResultsConfig
from reactor_runtime.manifest import import_model_class, load_config

_MANIFEST = """\
model:
  name: demo
  version: 0.1.0
runtime:
  import: pipeline:Demo
  config: config.yml
"""


def test_load_config_reads_the_model_reference_from_runtime_import(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    cfg = load_config(manifest)

    assert cfg.model_ref == "pipeline:Demo"


def test_load_config_reads_the_published_name_from_model_name(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    assert load_config(manifest).model_name == "demo"


def test_load_config_keeps_a_published_name_a_class_cannot_spell(tmp_path: Path) -> None:
    # A published name carries characters a Python class name cannot, which is
    # why the schema cannot be titled from the class.
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("model:\n  name: mage-vl\nruntime:\n  import: pipeline:Demo\n")

    assert load_config(manifest).model_name == "mage-vl"


@pytest.mark.parametrize(
    "document",
    [
        "runtime:\n  import: pipeline:Demo\n",
        "model:\n  version: 0.1.0\nruntime:\n  import: pipeline:Demo\n",
        "model: demo\nruntime:\n  import: pipeline:Demo\n",
        "model:\n  name: ''\nruntime:\n  import: pipeline:Demo\n",
    ],
)
def test_load_config_leaves_the_name_unset_when_the_manifest_states_none(
    tmp_path: Path, document: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(document)

    assert load_config(manifest).model_name is None


def test_load_config_resolves_runtime_config_against_the_manifest_dir(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    cfg = load_config(manifest)

    assert cfg.config_path == tmp_path / "config.yml"


def test_load_config_leaves_config_path_none_when_unset(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("runtime:\n  import: pipeline:Demo\n")

    cfg = load_config(manifest)

    assert cfg.config_path is None


def test_load_config_refuses_a_manifest_without_runtime_import(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("model:\n  name: demo\n")

    with pytest.raises(SystemExit):
        load_config(manifest)


def test_load_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    # A tab where YAML expects spaces is a syntax error, not a mapping problem.
    manifest.write_text("runtime:\n\timport: pipeline:Demo\n")

    with pytest.raises(SystemExit, match="invalid YAML"):
        load_config(manifest)


def test_load_config_rejects_a_document_that_is_not_a_mapping(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("- just\n- a\n- list\n")

    with pytest.raises(SystemExit, match="not a valid"):
        load_config(manifest)


def test_load_config_leaves_recording_at_defaults_when_absent(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("runtime:\n  import: pipeline:Demo\n")

    assert load_config(manifest).recording == RecordingConfig()


def test_load_config_reads_the_recording_block_nested_under_runtime(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n"
        "  import: pipeline:Demo\n"
        "  recording:\n"
        "    enabled: true\n"
        "    chunk_seconds: 4\n"
        "    video_track: video\n"
        "    video:\n"
        "      codec: libx264\n"
        "      crf: 20\n"
        "      target_width: 1280\n"
        "    audio:\n"
        "      bitrate_kbps: 96\n"
    )

    recording = load_config(manifest).recording

    assert recording.enabled is True
    assert recording.chunk_seconds == 4
    assert recording.video_track == "video"
    assert recording.video_codec == "libx264"
    assert recording.video_crf == 20
    assert recording.target_width == 1280
    assert recording.audio_bitrate_kbps == 96


def test_load_config_reads_a_top_level_recording_block(tmp_path: Path) -> None:
    # The legacy placement, still honored so older manifests keep working.
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n  import: pipeline:Demo\nrecording:\n  enabled: true\n  chunk_seconds: 4\n"
    )

    recording = load_config(manifest).recording

    assert recording.enabled is True
    assert recording.chunk_seconds == 4


def test_load_config_prefers_the_nested_recording_block(tmp_path: Path) -> None:
    # A manifest that declares both is ambiguous; the nested placement wins.
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n"
        "  import: pipeline:Demo\n"
        "  recording:\n"
        "    chunk_seconds: 9\n"
        "recording:\n"
        "  enabled: true\n"
        "  chunk_seconds: 4\n"
    )

    recording = load_config(manifest).recording

    assert recording.chunk_seconds == 9
    assert recording.enabled is False


def test_load_config_ignores_unknown_recording_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n  import: pipeline:Demo\nrecording:\n  enabled: true\n  from_the_future: 7\n"
    )

    assert load_config(manifest).recording.enabled is True


def test_load_config_leaves_step_results_off_when_absent(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)
    assert load_config(manifest).step_results == StepResultsConfig()
    assert load_config(manifest).step_results.enabled is False


def test_load_config_reads_the_step_results_block(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n"
        "  import: pipeline:Demo\n"
        "  step_results:\n"
        "    enabled: true\n"
        "    video: {codec: h265, preset: medium, crf: 28}\n"
        "    audio: {codec: aac, bitrate_kbps: 96}\n"
        "    queue: 3\n"
    )

    assert load_config(manifest).step_results == StepResultsConfig(
        enabled=True,
        video_codec="h265",
        video_preset="medium",
        video_crf=28,
        audio_codec="aac",
        audio_bitrate_kbps=96,
        queue=3,
    )


@pytest.mark.parametrize("queue", [0, -2])
def test_load_config_rejects_a_step_results_queue_below_one(tmp_path: Path, queue: int) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n  import: pipeline:Demo\n  step_results:\n"
        f"    enabled: true\n    queue: {queue}\n"
    )

    with pytest.raises(ValueError, match="queue must be at least 1"):
        load_config(manifest)


def test_load_config_fills_unset_step_results_settings_with_defaults(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("runtime:\n  import: pipeline:Demo\n  step_results:\n    enabled: true\n")

    assert load_config(manifest).step_results == StepResultsConfig(enabled=True)


@pytest.mark.parametrize(
    "block",
    [
        "  step_results: true\n",
        "  step_results: [enabled]\n",
        "  step_results:\n    enabled: true\n    video: fast\n    audio: [aac]\n",
    ],
)
def test_load_config_tolerates_a_step_results_block_of_the_wrong_shape(
    tmp_path: Path, block: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("runtime:\n  import: pipeline:Demo\n" + block)

    step_results = load_config(manifest).step_results

    defaults = StepResultsConfig()
    assert step_results.video_codec == defaults.video_codec
    assert step_results.audio_bitrate_kbps == defaults.audio_bitrate_kbps


def test_load_config_ignores_unknown_step_results_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n"
        "  import: pipeline:Demo\n"
        "  step_results:\n"
        "    enabled: true\n"
        "    tracks: [main_video]\n"
        "    video: {codec: h264, from_the_future: 1}\n"
    )

    assert load_config(manifest).step_results == StepResultsConfig(enabled=True)


def test_import_model_class_resolves_a_model_reference() -> None:
    assert import_model_class("reactor_runtime:ReactorApp") is ReactorApp


@pytest.mark.parametrize("ref", ["pipeline", ":Demo", "pipeline:", ""])
def test_import_model_class_rejects_a_malformed_reference(ref: str) -> None:
    with pytest.raises(ValueError, match="module:Class"):
        import_model_class(ref)


def test_import_model_class_rejects_a_class_that_is_no_model() -> None:
    with pytest.raises(TypeError, match="ReactorCore subclass"):
        import_model_class("pathlib:Path")
