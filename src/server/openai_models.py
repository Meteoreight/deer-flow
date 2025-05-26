from typing import List, Optional

from pydantic import BaseModel, Field

class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # Add other common parameters as needed, e.g., top_p, n, stop, etc.

# For Streaming Responses (stream: true)
class OpenAIChatCompletionChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class OpenAIChatCompletionStreamChoice(BaseModel):
    index: int
    delta: OpenAIChatCompletionChoiceDelta
    finish_reason: Optional[str] = None

class OpenAIChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIChatCompletionStreamChoice]
    # Potentially add system_fingerprint: Optional[str] = None later if needed
    # Potentially add usage: Optional[OpenAIUsage] = None for the last chunk if stream_options.include_usage is true

# For Non-Streaming Responses (stream: false)
class OpenAIChatMessageOutput(BaseModel):
    role: str
    content: Optional[str] = None
    # tool_calls: Optional[List[dict]] = None # To be added later if needed

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatCompletionChoice(BaseModel):
    index: int
    message: OpenAIChatMessageOutput
    finish_reason: Optional[str] = None
    # logprobs: Optional[dict] = None # To be added later if needed

class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChatCompletionChoice]
    usage: OpenAIUsage
    # system_fingerprint: Optional[str] = None # To be added later if needed

# For Error Responses
class OpenAIErrorDetail(BaseModel):
    type: str
    message: str
    code: Optional[str] = None
    param: Optional[str] = None # Parameter that caused the error

class OpenAIError(BaseModel):
    error: OpenAIErrorDetail
