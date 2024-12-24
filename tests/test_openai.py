import asyncio
import json
from typing import Annotated, Literal, TypedDict, NamedTuple, cast
from enum import Enum
from dataclasses import dataclass

import pytest
from openai import OpenAI
from minitools._registry import ToolRegistry, ToolCallDict


class Priority(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class UserInfo:
    """User information for task creation."""

    name: str
    age: int


class TaskOptions(TypedDict):
    """Options for task creation."""

    due_date: str
    tags: list[str]


class Coordinates(NamedTuple):
    """A pair of coordinates."""

    latitude: float
    longitude: float


def get_weather(location: Annotated[str, "The location to get the weather for"]) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."


async def aget_weather(location: Annotated[str, "The location to get the weather for"]) -> str:
    """Get the weather for a location asynchronously."""
    return f"The weather in {location} is sunny."


def send_email(
    recipient: Annotated[str, "The email address of the recipient"],
    subject: Annotated[str, "The subject of the email"],
    message: Annotated[str, "The message to send"],
    *,
    importance: Annotated[
        Literal["low", "medium", "high"], "The importance of the email"
    ],
) -> str:
    """Send an email"""
    return "\n".join(
        [f"To: {recipient}", f"Subject: {subject}", "", f"{importance}: {message}"]
    )


def create_complex_task(
    user: UserInfo,
    priority: Priority,
    coordinates: Coordinates,
    status: Literal["pending", "in_progress", "done"],
    options: TaskOptions,
) -> str:
    """Create a task with complex type parameters."""
    return (
        f"Created task for {user.name} (age {user.age}) with {priority.value} priority at "
        f"coordinates {coordinates}, status: {status}, due: {options['due_date']}, "
        f"tags: {', '.join(options['tags'])}"
    )




@pytest.fixture
def client() -> OpenAI:
    """Create an OpenAI client."""
    return OpenAI()


@pytest.fixture(params=[True, False], ids=["strict", "non-strict"])
def registry(request) -> ToolRegistry:
    """Create a tool registry with strict mode parameterized.

    - strict=True: Enforces strict schema validation
    - strict=False: Allows additional properties in JSON schema
    """
    registry = ToolRegistry()
    strict = request.param
    registry.register_tool(get_weather, strict=strict)
    registry.register_tool(send_email, strict=strict)
    return registry


def test_openai(client: OpenAI) -> None:
    """Test OpenAI's API."""
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = client.chat.completions.create(messages=messages, model="gpt-4o-mini")
    answer = response.choices[0].message.content
    assert "Paris" in answer


def test_tool(client: OpenAI, registry: ToolRegistry) -> None:
    """Test call to a registered tool."""
    messages = [
        {"role": "user", "content": "What is the weather in London?"},
    ]
    registry.register_tool(get_weather)
    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )
    tool_calls = response.choices[0].message.tool_calls
    function_name = tool_calls[0].function.name
    assert function_name == "get_weather"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["location"] == "London"
    call_result = registry.call(function_name, **args)
    assert "The weather in London is sunny." in call_result


def test_multiple_tools(client: OpenAI, registry: ToolRegistry) -> None:
    """Test call to multiple registered tools."""
    messages = [
        {"role": "user", "content": "What is the weather in London and Paris?"},
    ]

    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )
    tool_calls = response.choices[0].message.tool_calls
    assert len(tool_calls) == 2
    function_name = tool_calls[0].function.name
    assert function_name == "get_weather"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["location"] == "London"
    call_result = registry.call(function_name, **args)
    assert "The weather in London is sunny." in call_result
    function_name = tool_calls[1].function.name
    assert function_name == "get_weather"
    args = json.loads(tool_calls[1].function.arguments)
    assert args["location"] == "Paris"
    call_result = registry.call(function_name, **args)
    assert "The weather in Paris is sunny." in call_result


def test_sequential_tools(client: OpenAI, registry: ToolRegistry) -> None:
    """Test call to tools in sequence."""
    messages = [
        {
            "role": "user",
            "content": "Find the weather in London and Paris and then send an email to foo@example.com with the results.",
        }
    ]

    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )

    message = response.choices[0].message
    messages.append(message)

    for tool_call in message.tool_calls:
        tool_call_dict: ToolCallDict = cast(ToolCallDict, tool_call.to_dict())
        tool_response = registry.tool_call(tool_call_dict)
        messages.append(tool_response)

    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )

    tool_calls = response.choices[0].message.tool_calls
    assert len(tool_calls) == 1
    tool_call_dict: ToolCallDict = cast(ToolCallDict, tool_calls[0].to_dict())
    email_result = registry.tool_call(tool_call_dict)
    assert "To: foo@example.com" in email_result["content"]
    assert "Paris" in email_result["content"]
    assert "London" in email_result["content"]


@pytest.mark.asyncio
async def test_async_tool(client: OpenAI) -> None:
    """Test async call to a registered tool."""
    registry = ToolRegistry()
    registry.register_tool(aget_weather)
    messages = [
        {"role": "user", "content": "What is the weather in Tokyo?"},
    ]
    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )
    tool_calls = response.choices[0].message.tool_calls
    function_name = tool_calls[0].function.name
    assert function_name == "aget_weather"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["location"] == "Tokyo"
    call_result = await registry.acall(function_name, **args)
    assert "The weather in Tokyo is sunny." in call_result


def test_complex_types(client: OpenAI, registry: ToolRegistry) -> None:
    """Test handling of complex Python types including tuples, literals, enums, dataclasses and TypedDict."""
    registry.register_tool(create_complex_task)

    messages = [
        {
            "role": "user",
            "content": "Create a high priority task for John (age 30) at coordinates (42.1, -71.1), "
            "mark it as in progress, due tomorrow with tags project and urgent",
        },
    ]

    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )

    tool_calls = response.choices[0].message.tool_calls
    assert len(tool_calls) == 1

    function_name = tool_calls[0].function.name
    assert function_name == "create_complex_task"

    args = json.loads(tool_calls[0].function.arguments)

    # Verify complex type handling
    assert args["user"] == {"name": "John", "age": 30}
    assert args["priority"] == "high"
    assert args["coordinates"] == {"latitude": 42.1, "longitude": -71.1}
    assert args["status"] == "in_progress"
    # Verify the tags are correct
    assert args["options"]["tags"] == ["project", "urgent"]
    # Verify due_date is present but don't check exact value since model may return actual dates
    assert "due_date" in args["options"]
    assert isinstance(args["options"]["due_date"], str)

    result = registry.call(function_name, **args)
    assert "John" in result
    assert "age 30" in result
    assert "high priority" in result
    assert "in_progress" in result
    assert "project" in result
    assert "urgent" in result
