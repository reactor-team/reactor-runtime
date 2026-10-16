import pytest

from reactor_runtime.core import LiveKitConfig, LiveKitRpcMethod
from reactor_runtime.transport.livekit.signals import RoomSignals, payload_args

CONFIG = LiveKitConfig(
    rpc=(
        LiveKitRpcMethod(method="lk.clear_buffer", command="clear_buffer"),
        LiveKitRpcMethod(method="clear", command="clear_buffer"),
        LiveKitRpcMethod(method="lk.playback_finished", message="playback_finished"),
        LiveKitRpcMethod(method="acme.done", message="playback_finished"),
    ),
    topics=("avatar", "acme.avatar"),
)


def test_each_inbound_method_runs_its_command() -> None:
    assert RoomSignals(CONFIG).inbound == {
        "lk.clear_buffer": "clear_buffer",
        "clear": "clear_buffer",
    }


def test_a_message_calls_every_method_mapped_to_it() -> None:
    assert RoomSignals(CONFIG).outbound == {
        "playback_finished": ("lk.playback_finished", "acme.done")
    }


@pytest.mark.parametrize(
    ("topic", "command"),
    [
        ("avatar.user_speaking_state", "user_speaking_state"),
        # The longest listed prefix wins.
        ("acme.avatar.user_speaking_state", "user_speaking_state"),
        ("avatar", None),
        ("avatar.", None),
        ("avatars.user_speaking_state", None),
        ("other.user_speaking_state", None),
        # A command an RPC method runs is reachable only through the method.
        ("avatar.clear_buffer", None),
    ],
)
def test_a_data_topic_runs_the_command_after_its_prefix(topic: str, command: str | None) -> None:
    assert RoomSignals(CONFIG).topic_command(topic) == command


def test_no_topics_run_nothing() -> None:
    assert RoomSignals(LiveKitConfig()).topic_command("avatar.anything") is None


@pytest.mark.parametrize(
    ("payload", "args"),
    [
        ("", {}),
        ("   ", {}),
        ('{"state": "speaking"}', {"state": "speaking"}),
        (b'{"state": "idle"}', {"state": "idle"}),
        ("[1, 2]", None),
        ("speaking", None),
        ("42", None),
    ],
)
def test_a_payload_is_a_json_object_or_nothing(payload: str | bytes, args: object) -> None:
    assert payload_args(payload) == args
