"""The environment helpers every entry point reads its settings with."""

from __future__ import annotations

import pytest

from reactor_runtime.environment import bool_env


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_bool_env_reads_the_true_words(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("SOME_FLAG", raw)
    assert bool_env("SOME_FLAG", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "False", "no", "off"])
def test_bool_env_reads_the_false_words(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("SOME_FLAG", raw)
    assert bool_env("SOME_FLAG", True) is False


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_bool_env_falls_back_to_the_default_when_unset_or_empty(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv("SOME_FLAG", raising=False)
    else:
        monkeypatch.setenv("SOME_FLAG", raw)
    assert bool_env("SOME_FLAG", True) is True
    assert bool_env("SOME_FLAG", False) is False


def test_bool_env_refuses_any_other_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_FLAG", "enabled")
    with pytest.raises(SystemExit, match="SOME_FLAG 'enabled'"):
        bool_env("SOME_FLAG", False)
