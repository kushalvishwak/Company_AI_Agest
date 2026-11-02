import json

import pytest

from agent_stack.layer1_identity import GroqLLM


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Annual leave is 18 days.",
                        "tool_calls": [],
                    }
                }
            ],
            "usage": {"total_tokens": 12},
        }


def test_groq_client_uses_groq_api_and_default_hosted_model(monkeypatch):
    seen = {}

    def fake_post(url, *, headers, data, timeout):
        seen.update(url=url, headers=headers, payload=json.loads(data), timeout=timeout)
        return FakeResponse()

    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr("requests.post", fake_post)
    client = GroqLLM(api_key="test-key")

    response = client.complete(
        "Use supplied context.",
        "How many days of annual leave?",
        context="[leave_policy] Annual leave is 18 days.",
    )

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["payload"]["model"] == "openai/gpt-oss-120b"
    assert "[leave_policy]" in seen["payload"]["messages"][1]["content"]
    assert response.content == "Annual leave is 18 days."
    assert response.usage_tokens == 12


def test_groq_client_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqLLM()
