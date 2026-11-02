import json

import pytest

from agent_stack.layer1_identity import OpenAICompatibleLLM


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_compatible_chat_sends_context_and_reads_tool_call(monkeypatch):
    seen = {}

    def fake_post(url, *, headers, data, timeout):
        seen.update(url=url, headers=headers, payload=json.loads(data), timeout=timeout)
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "apply_leave",
                                        "arguments": '{"from":"2 Nov","to":"4 Nov"}',
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {"total_tokens": 16},
            }
        )

    monkeypatch.setattr("requests.post", fake_post)
    client = OpenAICompatibleLLM(
        api_key="test-key",
        base_url="https://api.example.test/v1",
        model="test-model",
    )

    response = client.complete(
        "Follow the rules.",
        "apply leave 2-4 Nov",
        context="[leave.pdf] Annual leave is 18 days.",
        tools=[{"name": "apply_leave", "description": "Submit leave"}],
    )

    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["headers"]["Authorization"]
    assert seen["payload"]["model"] == "test-model"
    assert "[leave.pdf]" in seen["payload"]["messages"][1]["content"]
    assert response.tool_call.name == "apply_leave"
    assert response.usage_tokens == 16
    assert seen["timeout"] >= 120


def test_compatible_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatibleLLM()
