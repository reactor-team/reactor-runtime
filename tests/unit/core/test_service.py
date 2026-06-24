from pathlib import Path

from reactor_runtime.core import Health, RuntimeConfig, ServiceComponent


class FakeComponent:
    name = "runner"
    depends_on: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")

    async def drain(self) -> None:
        self.events.append("drain")

    async def stop(self) -> None:
        self.events.append("stop")

    def health(self) -> Health:
        return Health.healthy()


def test_fake_component_conforms_by_shape() -> None:
    assert isinstance(FakeComponent(), ServiceComponent)


def test_a_plain_object_does_not_conform() -> None:
    assert not isinstance(object(), ServiceComponent)


def test_runtime_config_requires_a_model_ref_and_defaults_the_rest() -> None:
    cfg = RuntimeConfig(model_ref="my_pkg.model:MyModel")
    assert cfg.model_ref == "my_pkg.model:MyModel"
    assert cfg.config_path is None
    assert cfg.port == 8080
    assert cfg.grace_period == 30.0
    assert cfg.recording_dir is None


def test_runtime_config_overrides() -> None:
    cfg = RuntimeConfig(
        model_ref="m:M",
        config_path=Path("/etc/model/config.yml"),
        port=9000,
        recording_dir="/tmp/clips",
    )
    assert cfg.config_path == Path("/etc/model/config.yml")
    assert cfg.port == 9000
    assert cfg.recording_dir == "/tmp/clips"
