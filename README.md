# 🧰 Toolkitr

A lightweight Python library for creating and managing function tools that integrate with any LLM provider supporting function calling. Toolkitr provides type-safe function registration with automatic JSON Schema generation from Python type hints.

## ✨ Features

- Type-safe function tool registry system
- Automatic JSON Schema generation from Python type annotations
- Support for both synchronous and asynchronous tools
- Custom response serialization for both success and error responses
- Configurable exception handling
- Support for complex Python types:
  - Enums, Dataclasses, TypedDicts, NamedTuples
  - Lists, Tuples, and Dictionaries
  - Optional and Union types
  - Literal types
  - Annotated types with descriptions

## 📦 Installation

```bash
pip install toolkitr
```

## 🚀 Quick Start

The following example shows how to create a basic tool registry, register a function, and execute it:

```python
from typing import Annotated
from toolkitr import ToolRegistry

# Create a registry
registry = ToolRegistry()

# Register a function as a tool
@registry.tool()
def get_weather(location: Annotated[str, "The location to get weather for"]) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

# Get tool definitions for an LLM provider
tool_definitions = registry.definitions()

# Execute a tool directly
result = registry.call("get_weather", location="London")
print(result)  # "The weather in London is sunny."

# Execute a tool call from an LLM (similar to what an LLM provider would send)
tool_result = registry.tool_call({
    "id": "call_123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "London"}'
    }
})

# Access components of the tool result
print(tool_result.result)       # The raw function result
print(tool_result.message)      # The formatted message for the LLM
print(tool_result.success)      # True if call succeeded, False if it raised an exception
print(tool_result.tool.name)    # The name of the tool that was called
```

## ⚡ Working with Async Tools

Toolkitr seamlessly supports both synchronous and asynchronous functions. The `smart_call` method automatically handles both types:

```python
import asyncio

# Define an async tool
async def async_weather(location: str) -> str:
    await asyncio.sleep(0.1)  # Simulate API call
    return f"Weather in {location} is cloudy."

# Register and use it
registry.register_tool(async_weather)

# Use the unified interface for both sync and async tools
async def main():
    # Works with both sync and async functions
    sync_result = await registry.smart_call("get_weather", location="Paris")
    async_result = await registry.smart_call("async_weather", location="Tokyo")
    
    # Handle OpenAI-style tool calls
    tool_result = await registry.smart_tool_call({
        "id": "call_456",
        "type": "function",
        "function": {
            "name": "async_weather",
            "arguments": '{"location": "Berlin"}'
        }
    })
    
    print(tool_result.result)  # "Weather in Berlin is cloudy."
```

## 🛡️ Error Handling

Toolkitr provides robust error handling capabilities, allowing you to customize how exceptions are reported:

```python
# Configure error handling
registry = ToolRegistry(
    # Custom exception serializer
    exception_serializer=lambda exc: json.dumps({
        "error": {
            "type": type(exc).__name__,
            "message": str(exc)
        }
    }),
    # Set to False to let exceptions propagate
    catch_exceptions=True
)

@registry.tool()
def risky_function(input: str) -> str:
    if input == "fail":
        raise ValueError("Intentional failure")
    return f"Success: {input}"

# When a tool raises an exception, tool_result has:
# - tool_result.error: The exception that was raised
# - tool_result.result: None
# - tool_result.success: False
# - tool_result.message: Contains the serialized error
```

## 🤖 OpenAI Integration

Integrating with OpenAI is straightforward. Here's how to use your Toolkitr tools with OpenAI's chat completions:

```python
from openai import OpenAI

client = OpenAI()
messages = [
    {"role": "user", "content": "What's the weather in London?"}
]

# Create chat completion with tools
response = client.chat.completions.create(
    messages=messages,
    model="gpt-4",
    tools=registry.definitions()
)

# Handle tool calls
message = response.choices[0].message
messages.append(message)

for tool_call in message.tool_calls:
    # Get the tool result
    tool_result = registry.tool_call(tool_call.model_dump())
    # Add just the message to the conversation
    messages.append(tool_result.message)
```

## 🧩 Advanced Features

### Custom Serializers

You can customize how tool results are serialized, either at the registry level or for individual tools:

```python
# Registry-level serializer
registry = ToolRegistry(response_serializer=lambda result: json.dumps(result, indent=2))

# Per-tool serializer
@registry.tool(response_serializer=lambda x: f'"Custom: {x}"')
def special_tool(input: str) -> str:
    return input.upper()
```

### Human-friendly Tool Titles

Make your tools more user-friendly by providing clear titles that will appear in the LLM interface:

```python
@registry.tool(title="Get Weather Information")
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."
```

### Strict Mode

Control whether additional parameters are allowed or rejected:

```python
# Enable strict mode to prevent additional parameters
registry = ToolRegistry(strict=True)

# Override for specific tools
@registry.tool(strict=False)
def flexible_tool(param: str) -> str:
    """This tool allows additional parameters."""
    return f"Got {param}"
```

## 🧠 Complex Types

Toolkitr supports a wide range of Python data structures, automatically converting them to and from JSON:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal, TypedDict, NamedTuple

class Priority(Enum):
    LOW = "low"
    HIGH = "high"

@dataclass
class User:
    name: str
    age: int

class Options(TypedDict):
    tags: list[str]
    due_date: Optional[str]

class Point(NamedTuple):
    x: float
    y: float

@registry.tool()
def create_task(
    user: User,
    priority: Priority,
    location: Point,
    options: Options,
    status: Literal["pending", "done"] = "pending"
) -> str:
    """Create a new task."""
    return f"Created task for {user.name} with {priority.value} priority"
```

## ⚠️ Limitations

When using tuples with LLM providers, prefer:
- `NamedTuple` for fixed-length sequences with named fields
- `List` for variable-length sequences
- `dataclass` for structured data

## License

MIT License. See [LICENSE](LICENSE) file for details.
