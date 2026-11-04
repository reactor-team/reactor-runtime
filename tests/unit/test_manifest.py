from pathlib import Path
from typing import Any, cast

import pytest

from reactor_runtime import ReactorApp
from reactor_runtime.core import (
    DEFAULT_SHM_INPUT_BYTES,
    DEFAULT_SHM_OUTPUT_BYTES,
    LiveKitBlockError,
    LiveKitConfig,
    LiveKitRpcMethod,
    ModelServerConfig,
    RecordingConfig,
    RuntimeConfig,
    SharedMemoryConfig,
    StepResultsConfig,
)
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


def test_load_config_leaves_livekit_empty_when_absent(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    assert load_config(manifest).livekit == LiveKitConfig()


def test_load_config_reads_the_livekit_block(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n"
        "  import: pipeline:Demo\n"
        "  livekit:\n"
        "    rpc:\n"
        "      - method: lk.clear_buffer\n"
        "        command: clear_buffer\n"
        "      - method: lk.playback_finished\n"
        "        message: playback_finished\n"
        "    topics: [avatar, acme.avatar]\n"
        "    media_inputs:\n"
        "      lk.audio_stream: agent_audio\n"
    )

    assert load_config(manifest).livekit == LiveKitConfig(
        rpc=(
            LiveKitRpcMethod(method="lk.clear_buffer", command="clear_buffer"),
            LiveKitRpcMethod(method="lk.playback_finished", message="playback_finished"),
        ),
        topics=("avatar", "acme.avatar"),
        media_inputs={"lk.audio_stream": "agent_audio"},
    )


def test_load_config_allows_two_inbound_methods_for_one_command(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n  import: pipeline:Demo\n  livekit:\n    rpc:\n"
        "      - {method: clear_buffer, command: clear_buffer}\n"
        "      - {method: lk.clear_buffer, command: clear_buffer}\n"
    )

    assert len(load_config(manifest).livekit.rpc) == 2


def test_load_config_ignores_unknown_livekit_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(
        "runtime:\n  import: pipeline:Demo\n  livekit:\n    from_the_future: 1\n    rpc:\n"
        "      - {method: m, command: c, timeout: 3}\n"
    )

    assert load_config(manifest).livekit == LiveKitConfig(
        rpc=(LiveKitRpcMethod(method="m", command="c"),)
    )


@pytest.mark.parametrize(
    ("block", "reason"),
    [
        ("  livekit: true\n", "runtime.livekit must be a mapping"),
        ("  livekit:\n    rpc: {method: m}\n", "rpc must be a list"),
        ("  livekit:\n    rpc: [m]\n", r"rpc\[0\] must be a mapping"),
        ("  livekit:\n    rpc:\n      - {command: c}\n", r"rpc\[0\]\.method"),
        ("  livekit:\n    rpc:\n      - {method: m}\n", "exactly one of"),
        ("  livekit:\n    rpc:\n      - {method: m, command: c, message: d}\n", "exactly one of"),
        ("  livekit:\n    rpc:\n      - {method: m, command: ''}\n", r"rpc\[0\]\.command"),
        (
            "  livekit:\n    rpc:\n"
            "      - {method: m, command: a}\n"
            "      - {method: m, command: b}\n",
            "already runs a command",
        ),
        ("  livekit:\n    topics: avatar\n", "topics must be a list"),
        ("  livekit:\n    topics: ['']\n", "topics must be a list"),
        ("  livekit:\n    media_inputs: [lk.audio_stream]\n", "media_inputs must map"),
        ("  livekit:\n    media_inputs: {lk.audio_stream: 3}\n", "media_inputs must map"),
        # A key present with no value is refused, not read as an empty block.
        ("  livekit:\n", "runtime.livekit must be a mapping"),
        ("  livekit:\n    rpc:\n", "rpc must be a list"),
        ("  livekit:\n    topics:\n", "topics must be a list"),
        ("  livekit:\n    media_inputs:\n", "media_inputs must map"),
    ],
)
def test_load_config_refuses_a_livekit_block_of_the_wrong_shape(
    tmp_path: Path, block: str, reason: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text("runtime:\n  import: pipeline:Demo\n" + block)

    with pytest.raises(ValueError, match=reason):
        load_config(manifest)


def test_a_livekit_block_keeps_the_runtime_config_hashable() -> None:
    def config() -> RuntimeConfig:
        livekit = LiveKitConfig(media_inputs={"lk.audio_stream": "agent_audio"})
        return RuntimeConfig(model_ref="pipeline:Demo", livekit=livekit)

    assert config() == config()
    assert hash(config()) == hash(config())


def test_a_livekit_blocks_media_inputs_cannot_be_changed() -> None:
    sources = {"lk.audio_stream": "agent_audio"}
    livekit = LiveKitConfig(media_inputs=sources)
    sources["lk.other"] = "other"

    assert dict(livekit.media_inputs) == {"lk.audio_stream": "agent_audio"}
    with pytest.raises(TypeError):
        cast(Any, livekit.media_inputs)["lk.other"] = "other"


def test_livekit_check_accepts_a_block_naming_what_the_model_declares() -> None:
    config = LiveKitConfig(
        rpc=(
            LiveKitRpcMethod(method="lk.clear_buffer", command="clear_buffer"),
            LiveKitRpcMethod(method="lk.playback_finished", message="playback_finished"),
        ),
        media_inputs={"lk.audio_stream": "agent_audio"},
    )

    config.check(
        commands={"clear_buffer"},
        messages={"playback_finished"},
        input_tracks={"agent_audio"},
    )


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            LiveKitConfig(rpc=(LiveKitRpcMethod(method="m", command="missing"),)),
            "runs command 'missing'",
        ),
        (
            LiveKitConfig(rpc=(LiveKitRpcMethod(method="m", message="missing"),)),
            "message 'missing'",
        ),
        (LiveKitConfig(media_inputs={"lk.audio_stream": "missing"}), "input track 'missing'"),
    ],
)
def test_livekit_check_refuses_a_name_the_model_does_not_declare(
    config: LiveKitConfig, reason: str
) -> None:
    with pytest.raises(LiveKitBlockError, match=reason):
        config.check(commands={"c"}, messages={"d"}, input_tracks={"t"})


_SPLIT_MANIFEST = """\
model:
  name: acme/plasma
  resources:
    gpu: {type: NVIDIA_H100, count: 1}
    cpu: {request: "4", limit: "8"}
    memory: {request: 32Gi, limit: 48Gi}
    workers:
      count: 4
      cpu: {request: "1", limit: "2"}
      memory: {request: 2Gi, limit: 4Gi}
      shared_memory:
        input: 1Mi
        output: 24Mi
runtime:
  import: plasma:Plasma
  model_server:
    import: plasma_model:PlasmaModel
    max_batch: 2
    batch_window: 0.005
"""


def test_load_config_leaves_the_model_server_out_when_the_model_does_not_split(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_MANIFEST)

    cfg = load_config(manifest)

    assert cfg.model_server is None
    assert cfg.shared_memory == SharedMemoryConfig()


def test_load_config_reads_a_split_model_into_typed_settings(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_SPLIT_MANIFEST)

    cfg = load_config(manifest)

    assert cfg.model_server == ModelServerConfig(
        import_ref="plasma_model:PlasmaModel", max_batch=2, batch_window=0.005
    )
    assert cfg.shared_memory == SharedMemoryConfig(input_bytes=1 << 20, output_bytes=24 << 20)
    assert cfg.shared_memory.slot_bytes == 2 * (1 << 20) + (24 << 20)


@pytest.mark.parametrize(
    "block",
    [
        "model_server:\n    import: plasma_model:PlasmaModel",
        "model_server:\n    import: plasma_model:PlasmaModel\n    max_batch: 1",
    ],
)
def test_a_model_server_block_alone_serves_with_the_default_shared_memory(
    tmp_path: Path, block: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(f"runtime:\n  import: plasma:Plasma\n  {block}\n")

    cfg = load_config(manifest)

    assert cfg.model_server is not None
    assert (cfg.shared_memory.input_bytes, cfg.shared_memory.output_bytes) == (
        DEFAULT_SHM_INPUT_BYTES,
        DEFAULT_SHM_OUTPUT_BYTES,
    )


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ("model_server: plasma_model:PlasmaModel", "must be a block"),
        ("model_server:", "must be a block"),
        ("model_server:\n  import: plasma_model:PlasmaModel", "must be a block"),
        ("model_server:\n    max_batch: 2", "import must name the model half"),
        (
            "model_server:\n    import: plasma_model:PlasmaModel\n    shm_size: 1",
            r"unknown keys \['shm_size'\]; it takes import, max_batch, batch_window",
        ),
        (
            "model_server:\n    import: plasma_model:PlasmaModel\n    shm_output_bytes: 1048576",
            r"runtime\.model_server has unknown keys \['shm_output_bytes'\]",
        ),
        (
            "model_server:\n    import: plasma_model:PlasmaModel\n    max_batch: '2'",
            r"runtime\.model_server\.max_batch must be a positive integer",
        ),
        (
            "model_server:\n    import: plasma_model:PlasmaModel\n    max_batch: 0",
            "positive integer",
        ),
        (
            "model_server:\n    import: plasma_model:PlasmaModel\n    batch_window: -1",
            r"runtime\.model_server\.batch_window must be a number of seconds, at least 0",
        ),
        ("model_server:\n    import: plasma_model", "module:Class"),
        ("model_server:\n    import: 'plasma_model:'", "module:Class"),
        ("model_server:\n    import: ':PlasmaModel'", "module:Class"),
        ("model_server:\n    import: ':'", "module:Class"),
        ("model_server:\n    import: 3", "module:Class"),
    ],
)
def test_load_config_refuses_a_model_server_block_of_the_wrong_shape(
    tmp_path: Path, block: str, match: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(f"runtime:\n  import: plasma:Plasma\n  {block}\n")

    with pytest.raises(ValueError, match=match):
        load_config(manifest)


def _with_shared_memory(block: str) -> str:
    return f"model:\n  resources:\n    workers:\n      {block}\nruntime:\n  import: plasma:Plasma\n"


# Each value as the Kubernetes quantity grammar reads it, and the whole number of
# bytes it names; a deployment sizes each worker's volume from the same text.
@pytest.mark.parametrize(
    ("value", "size"),
    [
        ("1Mi", 1 << 20),
        ("24Mi", 24 << 20),
        ("1048576", 1 << 20),
        ("'1048576'", 1 << 20),
        ("1.5Mi", 3 << 19),
        ("0.5Ki", 512),
        ("1e6", 10**6),
        ("'1E6'", 10**6),
        ("2e+3", 2000),
        ("1000e-3", 1),
        ("1G", 10**9),
        ("1k", 1000),
        ("1Ki", 1024),
        ("1Gi", 1 << 30),
        ("1Ti", 1 << 40),
        ("1Pi", 1 << 50),
        ("1Ei", 1 << 60),
        ("2E", 2 * 10**18),
        ("2000m", 2),
        ("'+1Mi'", 1 << 20),
        ("1.Mi", 1 << 20),
        ("'.5Ki'", 512),
        ("007Ki", 7 << 10),
    ],
)
def test_a_shared_memory_size_is_read_as_a_quantity_of_whole_bytes(
    tmp_path: Path, value: str, size: int
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_with_shared_memory(f"shared_memory: {{input: {value}, output: 24Mi}}"))

    assert load_config(manifest).shared_memory == SharedMemoryConfig(
        input_bytes=size, output_bytes=24 << 20
    )


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        ("0", "0"),
        ("'0'", "'0'"),
        ("0Mi", "'0Mi'"),
        ("-1Mi", "'-1Mi'"),
        ("-1", "-1"),
        ("100m", "'100m'"),
        ("1500m", "'1500m'"),
        ("'0.5'", "'0.5'"),
        ("'1.5'", "'1.5'"),
        ("1001e-3", "'1001e-3'"),
        ("1e-30", "'1e-30'"),
        ("abc", "'abc'"),
        ("1MiB", "'1MiB'"),
        ("1mi", "'1mi'"),
        ("1K", "'1K'"),
        ("1 Mi", "'1 Mi'"),
        ("Mi", "'Mi'"),
        ("1e", "'1e'"),
        ("1e1.5", "'1e1.5'"),
        ("''", "''"),
        ("true", "True"),
        ("null", "None"),
        ("[1]", "[1]"),
    ],
)
def test_a_shared_memory_size_that_is_not_whole_bytes_is_refused_naming_its_key(
    tmp_path: Path, value: str, shown: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_with_shared_memory(f"shared_memory: {{output: {value}}}"))

    with pytest.raises(ValueError, match="must be a quantity of whole bytes") as refused:
        load_config(manifest)

    assert str(refused.value) == (
        "model.resources.workers.shared_memory.output must be a quantity of whole bytes, "
        f"at least 1, such as 24Mi; got {shown}"
    )


@pytest.mark.parametrize("value", ["0.5", "1.5", "2.0", "1.5e+6"])
def test_a_shared_memory_size_yaml_reads_as_a_float_is_refused_asking_for_a_string(
    tmp_path: Path, value: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_with_shared_memory(f"shared_memory: {{input: {value}}}"))

    with pytest.raises(
        ValueError,
        match=(
            r"^model\.resources\.workers\.shared_memory\.input must be a quantity of whole "
            r"bytes, at least 1, such as 24Mi; got [0-9.e+]+, which YAML reads as a float: "
            "write the quantity as a string"
        ),
    ):
        load_config(manifest)


@pytest.mark.parametrize("value", ["8Ei", "9223372036854775808", "1e19", "1e999999999"])
def test_a_shared_memory_size_beyond_a_64_bit_count_is_refused(tmp_path: Path, value: str) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_with_shared_memory(f"shared_memory: {{output: '{value}'}}"))

    with pytest.raises(
        ValueError,
        match=r"shared_memory\.output must be at most 9223372036854775807 bytes",
    ):
        load_config(manifest)


def test_the_largest_64_bit_count_is_a_size(tmp_path: Path) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_with_shared_memory("shared_memory: {output: 9223372036854775807}"))

    assert load_config(manifest).shared_memory.output_bytes == 2**63 - 1


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ("shared_memory:", r"shared_memory must be a block with `input` and `output` sizes"),
        ("shared_memory: 24Mi", r"shared_memory must be a block"),
        ("shared_memory: [1Mi, 24Mi]", r"shared_memory must be a block"),
        (
            "shared_memory: {input: 1Mi, outptu: 24Mi}",
            r"shared_memory has unknown keys \['outptu'\]; it takes input, output",
        ),
        (
            "shared_memory: {input_bytes: 1048576}",
            r"shared_memory has unknown keys \['input_bytes'\]",
        ),
    ],
)
def test_load_config_refuses_a_shared_memory_block_of_the_wrong_shape(
    tmp_path: Path, block: str, match: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_with_shared_memory(block))

    with pytest.raises(ValueError, match=rf"^model\.resources\.workers\.{match}"):
        load_config(manifest)


@pytest.mark.parametrize(
    "model",
    [
        "model: {name: plasma}",
        "model: {resources: 4}",
        "model: {resources: {gpu: {type: NVIDIA_H100, count: 1}}}",
        "model: {resources: {workers: 4}}",
        "model: {resources: {workers: {count: 4, cpu: {request: '1'}, memory: {limit: 4Gi}}}}",
        "model: {resources: {workers: {count: 4, gpu: banana, shm: 1}, storage: 1Ti}}",
    ],
)
def test_everything_under_model_resources_but_shared_memory_is_left_to_the_deployment(
    tmp_path: Path, model: str
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(f"{model}\nruntime:\n  import: plasma:Plasma\n")

    assert load_config(manifest).shared_memory == SharedMemoryConfig()


def test_a_shared_memory_block_may_size_one_block_and_leave_the_other_at_its_default(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "reactor.yaml"
    manifest.write_text(_with_shared_memory("shared_memory: {output: 64Mi}"))

    assert load_config(manifest).shared_memory == SharedMemoryConfig(
        input_bytes=DEFAULT_SHM_INPUT_BYTES, output_bytes=64 << 20
    )
    manifest.write_text(_with_shared_memory("shared_memory: {}"))
    assert load_config(manifest).shared_memory == SharedMemoryConfig()
