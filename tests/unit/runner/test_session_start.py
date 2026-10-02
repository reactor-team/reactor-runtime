from typing import Any

import pytest

from reactor_runtime.runner.session_start import (
    InvalidSessionStartError,
    SessionStart,
    StartingCommand,
    StartingInput,
    parse_session_start,
)


def test_a_body_without_the_keys_starts_a_session_as_before() -> None:
    assert parse_session_start({}) == SessionStart()
    assert parse_session_start({"session_id": "8c0e7a52"}) == SessionStart()


def test_null_keys_read_as_absent() -> None:
    parsed = parse_session_start({"starting_input": None, "steps": None})
    assert parsed == SessionStart()


def test_a_full_body_is_parsed() -> None:
    parsed = parse_session_start(
        {
            "session_id": "8c0e7a52",
            "starting_input": {
                "state": {"seed": 42},
                "commands": [
                    {"command": "set_canvas", "data": {"aspect": "16:9"}},
                    {"command": "start"},
                ],
            },
            "steps": 1,
        }
    )
    assert parsed == SessionStart(
        starting_input=StartingInput(
            state={"seed": 42},
            commands=(
                StartingCommand("set_canvas", {"aspect": "16:9"}),
                StartingCommand("start", {}),
            ),
        ),
        steps=1,
    )


def test_a_starting_input_may_hold_only_state_or_only_commands() -> None:
    assert parse_session_start({"starting_input": {"state": {"seed": 1}}}).starting_input == (
        StartingInput(state={"seed": 1})
    )
    assert parse_session_start({"starting_input": {}}).starting_input == StartingInput()


def test_command_order_is_kept() -> None:
    commands = [{"command": f"c{index}"} for index in range(5)]
    parsed = parse_session_start({"starting_input": {"commands": commands}})
    assert parsed.starting_input is not None
    assert [command.command for command in parsed.starting_input.commands] == [
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"starting_input": []}, "starting_input must be an object"),
        ({"starting_input": {"command": []}}, "starting_input has unknown keys ['command']"),
        ({"starting_input": {"state": [1]}}, "starting_input.state must be an object"),
        ({"starting_input": {"commands": {}}}, "starting_input.commands must be a list"),
        ({"starting_input": {"commands": ["start"]}}, "starting_input.commands[0] must be"),
        (
            {"starting_input": {"commands": [{"command": "a"}, {"name": "b"}]}},
            "starting_input.commands[1] has unknown keys ['name']",
        ),
        ({"starting_input": {"commands": [{"data": {}}]}}, "commands[0].command must be"),
        ({"starting_input": {"commands": [{"command": ""}]}}, "commands[0].command must be"),
        (
            {"starting_input": {"commands": [{"command": "a", "data": [1]}]}},
            "commands[0].data must be an object",
        ),
        ({"steps": 0}, "steps must be a positive integer"),
        ({"steps": -2}, "steps must be a positive integer"),
        ({"steps": "1"}, "steps must be a positive integer"),
        ({"steps": 1.0}, "steps must be a positive integer"),
        ({"steps": True}, "steps must be a positive integer"),
    ],
)
def test_a_key_of_the_wrong_shape_is_rejected(body: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidSessionStartError) as raised:
        parse_session_start(body)
    assert message in str(raised.value)
