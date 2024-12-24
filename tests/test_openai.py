import json
from typing import Annotated, Any, Literal

import pytest
from openai import OpenAI
from minitools._registry import ToolRegistry


def get_weather(location: Annotated[str, "The location to get the weather for"]) -> str:
    """Get the weather for a location."""
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


def test_tool(client: OpenAI) -> None:
    """Test call to a registered tool with both strict and non-strict modes.
    
    Tests both:
    - strict=True: Validates exact parameter matching
    - strict=False: Allows additional parameters in tool calls
    """
    messages = [
        {"role": "user", "content": "What is the weather in London?"},
    ]
    
    for strict in [True, False]:
        registry = ToolRegistry()
        registry.register_tool(get_weather, strict=strict)

    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )
    tool_calls = response.choices[0].message.tool_calls
    function_name = tool_calls[0].function.name
    assert function_name == "get_weather"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["location"] == "London"
    call_result = registry.call_tool(function_name, args)
    assert "The weather in London is sunny." in call_result


def test_multiple_tools(client: OpenAI, registry: ToolRegistry) -> None:
    """Test call to multiple registered tools.
    
    Uses parameterized registry to test:
    - strict=True: Enforces exact parameter schemas
    - strict=False: More lenient parameter validation
    """
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
    call_result = registry.call_tool(function_name, args)
    assert "The weather in London is sunny." in call_result
    function_name = tool_calls[1].function.name
    assert function_name == "get_weather"
    args = json.loads(tool_calls[1].function.arguments)
    assert args["location"] == "Paris"
    call_result = registry.call_tool(function_name, args)
    assert "The weather in Paris is sunny." in call_result


def test_sequential_tools(client: OpenAI, registry: ToolRegistry) -> None:
    """Test call to tools in sequence.
    
    Tests chained tool calls with:
    - strict=True: Strict schema validation for each tool
    - strict=False: Allows additional parameters in each call
    """
    messages = [
        {
            "role": "user",
            "content": "Find the weather in London and Paris and then send an email to foo@example.com with the results.",
        }
    ]

    print(json.dumps(registry.definitions(), indent=2))
    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )

    message = response.choices[0].message
    messages.append(message)

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        call_result = registry.call_tool(function_name, args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": call_result,
            }
        )

    response = client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", tools=registry.definitions()
    )

    tool_calls = response.choices[0].message.tool_calls
    assert len(tool_calls) == 1
    function_name = tool_calls[0].function.name
    assert function_name == "send_email"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["recipient"] == "foo@example.com"
    email_result = registry.call_tool(function_name, args)
    assert "To: foo@example.com" in email_result
    assert "Paris" in email_result
    assert "London" in email_result
