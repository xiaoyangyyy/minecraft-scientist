import pytest

from minecraft_scientist.discovery import OpenAIHypothesisGenerator


def test_openai_adapter_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIHypothesisGenerator()
