# Minitools

A lightweight Python library for creating and managing function tools that can integrate with any LLM provider supporting function calling. Minitools provides type-safe function registration and automatic JSON Schema generation from Python type hints.

## Features

- LLM-agnostic function tool registry system
- Simple tool registration with function decorators
- Automatic JSON Schema generation from Python type annotations
- Unified interface for both synchronous and asynchronous tools
- Custom response serialization for both success and error responses
- Configurable exception handling
- Support for complex Python types including:
  - Enums
  - Dataclasses
  - TypedDicts
  - NamedTuples
  - Lists, Tuples, and Dictionaries
  - Optional and Union types
  - Literal types
  - Annotated types with descriptions
- Ready-to-use integration examples for popular LLM providers
- Type-safe conversion between JSON and Python objects

## Installation

You can install the package using pip:

```bash
pip install minitools
```

Or using Poetry:

```bash
poetry add minitools
```

## Quick Start

Here's a simple example of how to use Minitools:

```python
from typing import Annotated
from minitools import ToolRegistry

# Create a new registry with strict mode enabled
registry = ToolRegistry(strict=True)  # Enforces exact parameter matching
# Or with strict mode disabled (default)
registry = ToolRegistry(strict=False)  # Allows additional parameters


# Register a function as a tool
# Register with specific strict mode
@registry.tool(strict=True)  # Override registry's default strict mode
def get_weather(location: Annotated[str, "The location to get weather for"]) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."


# Get tool definitions - can be used with any LLM provider
tool_definitions = registry.definitions()

# Execute a tool call directly
result = registry.call("get_weather", location="London")

# Or handle OpenAI tool calls
result = registry.tool_call({
    "id": "call_123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "London"}'
    }
})
```

### Integration Examples

#### OpenAI Integration

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
    tools=registry.definitions()  # Tool definitions work with OpenAI
)

# Handle tool calls
message = response.choices[0].message
for tool_call in message.tool_calls:
    tool_response = registry.tool_call(tool_call.model_dump())
    messages.append(tool_response)  # Add tool response to conversation
```

## Unified Calling for Sync and Async Tools

Minitools provides a unified way to call tools regardless of whether they're synchronous or asynchronous:

```python
# Define synchronous and asynchronous tools
def sync_function(x: str) -> str:
    return f"Sync: {x}"

async def async_function(x: str) -> str:
    await asyncio.sleep(0.1)  # Async operation
    return f"Async: {x}"

# Register both types of tools
registry = ToolRegistry()
registry.register_tool(sync_function)
registry.register_tool(async_function)

# Use smart_call to handle both types uniformly
async def main():
    # Works with both sync and async functions
    sync_result = await registry.smart_call("sync_function", x="Hello")
    async_result = await registry.smart_call("async_function", x="World")
    
    # Similarly, smart_tool_call works for OpenAI-style tool calls
    response = await registry.smart_tool_call({
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "sync_function",
            "arguments": '{"x": "From OpenAI"}'
        }
    })
```

## Custom Response Serialization

Minitools allows customizing how tool results are serialized in responses:

```python
import json
from minitools import ToolRegistry, default_serializer

# Define a custom serializer
def pretty_serializer(result: any) -> str:
    if isinstance(result, str):
        return f'"{result} (serialized)"'
    return json.dumps(result, indent=2)

# Registry-level serializer
registry = ToolRegistry(response_serializer=pretty_serializer)

# Or per-tool serializer
@registry.tool(response_serializer=lambda x: f'"Custom: {x}"')
def special_tool(input: str) -> str:
    return input.upper()

# Default serializer is also available for import
from minitools import default_serializer
```

## Exception Handling

Minitools provides robust exception handling with customizable serialization:

```python
# Configure exception handling at registry level
registry = ToolRegistry(
    # Custom exception serializer
    exception_serializer=lambda exc: json.dumps({
        "error_type": type(exc).__name__,
        "message": str(exc),
        "custom_field": "Additional context"
    }),
    # Control whether exceptions are caught or bubble up
    catch_exceptions=True  # Set to False to let exceptions propagate
)

# Per-tool exception handling
@registry.tool(
    exception_serializer=lambda exc: json.dumps({
        "custom_error": str(exc)
    })
)
def risky_function(input: str) -> str:
    if input == "fail":
        raise ValueError("Intentional failure")
    return f"Success: {input}"

# When a tool raises an exception:
# - If catch_exceptions=True: Returns serialized error response
# - If catch_exceptions=False: Raises the exception normally
```

## Limitations

### Tuple Handling
When using tuples with LLM providers like OpenAI, you may encounter inconsistent handling of tuple types in function calls. This is because tuples are serialized to JSON arrays, and LLMs may not reliably maintain the fixed-length nature of tuples.

**Recommendation:** Instead of using plain tuples, prefer:
- `NamedTuple` for fixed-length sequences with named fields
- `List` for variable-length sequences
- `dataclass` for structured data

Example of recommended approach:
```python
# Instead of:
def process_point(point: Tuple[float, float]):
    x, y = point
    return f"Point at {x}, {y}"

# Prefer:
class Point(NamedTuple):
    x: float
    y: float

def process_point(point: Point):
    return f"Point at {point.x}, {point.y}"
```

## Advanced Usage

### Complex Types

Minitools supports a wide range of Python types that are automatically converted to JSON Schema:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal

class Priority(Enum):
    LOW = "low"
    HIGH = "high"

@dataclass
class User:
    name: str
    age: int

@registry.tool()
def create_task(
    user: User,
    priority: Priority,
    tags: Optional[list[str]] = None,
    status: Literal["pending", "done"] = "pending"
) -> str:
    """Create a new task for a user."""
    return f"Created task for {user.name} with {priority.value} priority"
```

### Function and Parameter Descriptions

Minitools supports multiple ways to specify descriptions for your functions and parameters:

#### Using Docstrings

Function descriptions can be automatically extracted from docstrings:

```python
@registry.tool()
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient.
    
    This function connects to the email server and delivers the message.
    """
    return f"Email sent to {recipient}"
```

For complex types like dataclasses and TypedDicts, their docstrings are also used:

```python
@dataclass
class EmailConfig:
    """Configuration for email sending.
    
    Contains all necessary settings for SMTP connection and delivery.
    """
    host: str
    port: int
    use_tls: bool

class MessageOptions(TypedDict):
    """Options for customizing message delivery."""
    priority: str
    retry_count: int
```

#### Using Annotations

Parameter descriptions can be specified using `Annotated`. These will override any docstring descriptions:

```python
@registry.tool()
def send_email(
    recipient: Annotated[str, "Email address of the recipient"],
    subject: Annotated[str, "Email subject line"],
    body: Annotated[str, "Email body content"],
    config: Annotated[EmailConfig, "Email server configuration"]
) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {recipient}. Subject: {subject}, Body: {body}, config: {config}"
```

#### Mixing Approaches

You can combine both approaches, using docstrings for general descriptions and annotations for specific parameter details:

```python
@registry.tool()
def create_user(
    name: Annotated[str, "Full name of the user"],
    role: str,
    settings: UserSettings
) -> User:
    """Create a new user in the system.
    
    This function validates the input, creates the user record,
    and initializes their default settings.
    """
    return User(name=name, role=role, settings=settings)
```

## Requirements

- Python 3.11 or higher
- Optional dependencies for specific LLM integrations (e.g., `openai` package for OpenAI integration)

### Strict Mode

Minitools supports strict mode for OpenAI function calling:

```python
# Global strict mode for all tools
registry = ToolRegistry(strict=True)

# Per-tool strict mode override
@registry.tool(strict=False)
def flexible_tool(param: str) -> str:
    """This tool allows additional parameters."""
    return f"Got {param}"

# Register with specific strict mode
registry.register_tool(my_func, strict=True)
```

When strict mode is enabled:
- OpenAI's function calling will be restricted to only use declared parameters
- Additional properties in the function schema are marked as not allowed
- Helps prevent hallucination of non-existent parameters by the LLM

When strict mode is disabled (default):
- OpenAI's function calling may include additional parameters
- More lenient schema validation
- Useful when you want to allow the LLM more flexibility in function calls

## Roadmap

- [x] Add support for `strict` mode and other flavour options for JSON Schema generation
- [x] Add unified interface for synchronous and asynchronous tools
- [x] Add custom serialization for tool responses
- [x] Add configurable exception handling
- [ ] Parameters documentation parsing
- [ ] Add support for more complex Python types (e.g., generics, Pydantic models)
- [ ] Hooks for customizing tool registration and execution
- [ ] Publish the registry as MCP (Model Context Protocol)

## Development

To set up the development environment:

1. Clone the repository
2. Install Poetry if you haven't already
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Run tests:
   ```bash
   poetry run pytest
   ```

## License

Pending
