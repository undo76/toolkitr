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

### Strict Mode

Minitools supports strict mode for parameter validation:

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
- Exact parameter matching is enforced
- Additional properties in JSON schema are disallowed
- Tool calls must match the function signature exactly

When strict mode is disabled (default):
- Additional parameters are allowed in tool calls
- More lenient parameter validation
- Useful for tools that may receive extra context from LLMs

## Roadmap

- [x] Add support for `strict` mode and other flavour options for JSON Schema generation
- [ ] Parameters documentation parsing
- [ ] Add support for more complex Python types (e.g., generics, Pydantic models)
- [ ] Improve error handling and validation for tool calls
- [ ] Hooks for customizing tool registration and execution
- [ ] Improve async support for LLM integrations
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
