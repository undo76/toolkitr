# Minitools

A lightweight Python library for creating and managing function tools that can integrate with any LLM provider supporting function calling. Minitools provides type-safe function registration and automatic JSON Schema generation from Python type hints.

## Features

- LLM-agnostic function tool registry system
- Simple tool registration with function decorators
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

# Create a new _registry
registry = ToolRegistry()

# Register a function as a tool
@registry.tool()
def get_weather(location: Annotated[str, "The location to get weather for"]) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

# Get tool definitions - can be used with any LLM provider
tool_definitions = registry.definitions()

# Execute a tool call
result = registry.call_tool("get_weather", {"location": "London"})
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
    return f"Email sent to {recipient}"
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