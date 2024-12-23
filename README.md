# Minitools

A lightweight Python library for creating and managing function tools that integrate seamlessly with OpenAI's function calling API. Minitools provides type-safe function registration and automatic JSON Schema generation from Python type hints.

## Features

- Simple tool registration system with function decorators
- Automatic JSON Schema generation from Python type annotations
- Support for complex Python types including:
  - Enums
  - Dataclasses
  - TypedDicts
  - NamedTuples
  - Lists, Tuples, and Dictionaries
  - Optional and Union types
  - Literal types
  - Annotated types with descriptions
- Full integration with OpenAI's function calling API
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

# Create a new registry
registry = ToolRegistry()

# Register a function as a tool
@registry.tool()
def get_weather(location: Annotated[str, "The location to get weather for"]) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

# Use with OpenAI
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
tool_calls = response.choices[0].message.tool_calls
for tool_call in tool_calls:
    function_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = registry.call_tool(function_name, args)
```

## Advanced Usage

### Complex Types

Minitools supports a wide range of Python types that are automatically converted to JSON Schema:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal
from minitools import ToolRegistry

registry = ToolRegistry()

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

### Type Descriptions

Use `Annotated` to add descriptions to your parameters:

```python
from typing import Annotated
from minitools import ToolRegistry

registry = ToolRegistry()

@registry.tool()
def send_email(
    recipient: Annotated[str, "Email address of the recipient"],
    subject: Annotated[str, "Email subject line"],
    body: Annotated[str, "Email body content"]
) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {recipient}"
```

## Requirements

- Python 3.11 or higher
- OpenAI Python package (for integration with OpenAI's API)

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

[License information not provided in source files]

## Contributing

[Contribution guidelines not provided in source files]