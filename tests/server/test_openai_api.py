import json
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk # Used for type hint reference in mock

from src.server.app import app  # Assuming your FastAPI app instance is here
from src.server.openai_models import (
    OpenAIChatCompletionResponse,
    OpenAIError,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionStreamChoice,
    OpenAIChatCompletionChoiceDelta
)

client = TestClient(app)

# Helper for parsing SSE events
def parse_sse_events(sse_text):
    events = []
    for line in sse_text.splitlines():
        if line.startswith("data: "):
            data_json_str = line[len("data: "):]
            if data_json_str == "[DONE]":
                events.append({"done": True})
            else:
                try:
                    events.append(json.loads(data_json_str))
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {data_json_str}") # Should not happen in well-formed SSE
    return events

# Minimal mock for AIMessageChunk to be used by the mock generator
class MockAIMessageChunk:
    def __init__(self, content, id="some_id", role=None, response_metadata=None, tool_call_chunks=None, tool_calls=None):
        self.content = content
        self.id = id
        self.role = role
        self.response_metadata = response_metadata or {}
        self.tool_call_chunks = tool_call_chunks
        self.tool_calls = tool_calls


@pytest.fixture
def mock_graph_fixture():
    with patch('src.server.app.graph', new_callable=AsyncMock) as mock_graph:
        yield mock_graph

# --- Non-Streaming Tests ---

def test_successful_non_streaming_completion(mock_graph_fixture: AsyncMock):
    mock_graph_fixture.ainvoke.return_value = {"final_report": "This is the test report."}

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": False},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert data["model"] == "test-model"
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "This is the test report."
    assert choice["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 0 # Placeholder value
    assert data["usage"]["completion_tokens"] == 0 # Placeholder value
    assert data["usage"]["total_tokens"] == 0 # Placeholder value
    
    # Validate with Pydantic model
    OpenAIChatCompletionResponse(**data)


def test_non_streaming_with_graph_error(mock_graph_fixture: AsyncMock):
    mock_graph_fixture.ainvoke.side_effect = Exception("Graph processing error")

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": False},
    )

    assert response.status_code == 500
    data = response.json()
    
    assert "error" in data
    assert data["error"]["type"] == "internal_server_error"
    assert data["error"]["message"] == "Graph processing error"
    
    # Validate with Pydantic model
    OpenAIError(**data)


# --- Streaming Tests ---

async def mock_astream_success_gen(*args, **kwargs):
    # Simulate graph.astream yielding (agent_name_tuple, message_metadata_dict, event_data_tuple)
    # event_data_tuple is (message_chunk_object, message_metadata_dict_again_or_None)
    yield (("some_agent",), {}, (MockAIMessageChunk(content="Hello, "), {}))
    yield (("some_agent",), {}, (MockAIMessageChunk(content="world!"), {}))
    # Simulate other types of events that should be ignored by the SSE generator
    yield (("other_agent",), {}, (MockAIMessageChunk(content="ignore this"), {})) 
    yield (("some_agent",), {}, ("not_a_tuple_event_data", {})) # Should be ignored


def test_successful_streaming_completion(mock_graph_fixture: AsyncMock):
    mock_graph_fixture.astream.return_value = mock_astream_success_gen()

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model-stream", "messages": [{"role": "user", "content": "Hello Stream"}], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    events = parse_sse_events(response.text)
    
    assert len(events) >= 4 # At least: role, content1, content2, finish_reason, DONE
    
    # First chunk: role definition
    first_chunk_data = events[0]
    OpenAIChatCompletionChunk(**first_chunk_data) # Validate structure
    assert first_chunk_data["choices"][0]["delta"]["role"] == "assistant"
    assert first_chunk_data["choices"][0]["delta"].get("content") == "" # Initially empty content
    assert first_chunk_data["model"] == "test-model-stream"

    # Subsequent content chunks
    accumulated_content = ""
    content_chunks = [e for e in events if e.get("choices") and e["choices"][0]["delta"].get("content") is not None and e["choices"][0]["delta"].get("content") != ""]
    
    assert len(content_chunks) == 2
    accumulated_content += content_chunks[0]["choices"][0]["delta"]["content"]
    OpenAIChatCompletionChunk(**content_chunks[0]) # Validate
    
    accumulated_content += content_chunks[1]["choices"][0]["delta"]["content"]
    OpenAIChatCompletionChunk(**content_chunks[1]) # Validate

    assert accumulated_content == "Hello, world!"

    # Final content chunk with finish_reason
    finish_chunk_data = events[-2] # Second to last event should be the finish reason chunk
    OpenAIChatCompletionChunk(**finish_chunk_data) # Validate
    assert finish_chunk_data["choices"][0]["delta"] == {} # Empty delta content
    assert finish_chunk_data["choices"][0]["finish_reason"] == "stop"

    # DONE event
    assert events[-1] == {"done": True}


async def mock_astream_setup_error_gen(*args, **kwargs):
    # This generator itself will raise an error when iterated.
    raise Exception("Graph streaming setup error")
    yield # Unreachable, but makes it a generator

def test_streaming_with_pre_stream_error_in_generator_iteration(mock_graph_fixture: AsyncMock):
    # This tests an error that occurs when the stream *starts* to be processed.
    # The _openai_stream_generator has its own try-except.
    # If graph.astream() itself (the generator object) is problematic on first iteration.
    mock_graph_fixture.astream.return_value = mock_astream_setup_error_gen()
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model-stream-fail", "messages": [{"role": "user", "content": "Hello Stream Fail"}], "stream": True},
    )
    
    assert response.status_code == 200 # The headers are sent before the generator is fully consumed
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # The stream will contain the initial role chunk, then likely terminate prematurely.
    # The error "Graph streaming setup error" will be logged server-side.
    # The client will see a truncated stream.
    events = parse_sse_events(response.text)
    assert len(events) > 0 # At least the first chunk
    first_chunk_data = events[0]
    assert first_chunk_data["choices"][0]["delta"]["role"] == "assistant"
    # No standard OpenAI error is sent over SSE, the stream just breaks.
    # The [DONE] marker will NOT be present if an error occurs mid-stream like this.
    assert not any(event.get("done") for event in events)


def test_streaming_with_error_before_generator_is_called(mock_graph_fixture: AsyncMock):
    # This tests an error that occurs *before* StreamingResponse is even created
    # or before the generator starts.
    mock_graph_fixture.astream.side_effect = Exception("Synchronous graph.astream setup error")

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model-stream-sync-fail", "messages": [{"role": "user", "content": "Hello Stream Sync Fail"}], "stream": True},
    )

    assert response.status_code == 500 # Error happens before stream response is initiated
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "internal_server_error"
    assert data["error"]["message"] == "Synchronous graph.astream setup error"
    OpenAIError(**data) # Validate

# Test for invalid request format (e.g. stream not a bool, though Pydantic should catch this)
# Pydantic validation usually happens before the endpoint logic,
# so it would return a 422 Unprocessable Entity error.
# This test case is more about ensuring the final `else` in the endpoint is not hit.
def test_invalid_request_format_if_stream_not_bool(mock_graph_fixture: AsyncMock):
    # This test is somewhat artificial as Pydantic handles type validation.
    # We'd have to bypass Pydantic to truly test the endpoint's internal else.
    # However, if Pydantic was misconfigured for `stream`, this path could be hit.
    # For now, we assume Pydantic validation is active.
    # A request with `stream` as not a boolean would be caught by Pydantic.
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": "not-a-boolean"},
    )
    assert response.status_code == 422 # Unprocessable Entity due to Pydantic validation
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["type"] == "bool_type"
    assert data["detail"][0]["loc"] == ["body", "stream"]
