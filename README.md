# 🧰 Toolkitr

A lightweight Python library for creating and managing function tools that integrate with any LLM provider supporting function calling. Toolkitr provides type-safe function registration with automatic JSON Schema generation from Python type hints, enabling seamless integration between your Python functions and Large Language Models.

## ✨ Features

- **Type-safe function registry**: Build reliable AI applications with runtime type validation and conversion
- **Zero-config schema generation**: Automatically generate JSON Schema from Python type annotations with no additional configuration needed
- **Modern async support**: Seamlessly work with both synchronous and asynchronous functions through a unified API
- **Flexible serialization**: Customize how function results and errors are presented to your LLM with powerful serializers
- **Robust error handling**: Control exactly how exceptions are processed, formatted, and presented to the LLM
- **Rich type system support**:
  - Enums, Dataclasses, TypedDicts, NamedTuples for structured data
  - Lists, Tuples, and Dictionaries for collections
  - Optional and Union types for flexible inputs
  - Literal types for constrained values
  - Annotated types with human-readable descriptions

## 📦 Installation

```bash
pip install toolkitr
```

## 🚀 Quick Start

Getting started with Toolkitr is simple. The following example demonstrates how to create a tool registry, register a function with rich type information, and execute it in different contexts:

```python
from typing import Annotated
from toolkitr import ToolRegistry

# Create a registry - the central hub for all your tools
registry = ToolRegistry()

# Register a function as a tool with automatic documentation
@registry.tool()
def get_weather(location: Annotated[str, "The location to get weather for"]) -> str:
    """Get the weather for a location."""
    # In a real app, this would call a weather API
    return f"The weather in {location} is sunny with a high of 22°C."

# --- INTEGRATION WITH LLM PROVIDERS ---

# Get tool definitions ready to send to any LLM provider (OpenAI, Anthropic, etc.)
tool_definitions = registry.definitions()
# This generates a properly formatted JSON Schema that LLMs understand

# --- DIRECT EXECUTION ---

# Call the tool directly from your code
result = registry.call("get_weather", location="London")
print(result)  # "The weather in London is sunny with a high of 72°F."

# --- HANDLING LLM TOOL CALLS ---

# Process a tool call exactly as it would come from an LLM
tool_result = registry.tool_call({
    "id": "call_123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "London"}'
    }
})

# Access the rich result object with everything you need
print(tool_result.result)       # The raw function return value
print(tool_result.message)      # The formatted message ready to send back to the LLM
print(tool_result.success)      # True if the call succeeded, False if it raised an exception
print(tool_result.tool.name)    # The name of the tool that was called
```

## ⚡ Working with Async Tools

Modern Python applications often use asynchronous programming for better performance and resource utilization. Toolkitr provides first-class support for async functions while maintaining a clean, consistent API:

```python
import asyncio

# Define an async tool that could be calling an external API
async def async_weather(location: str) -> str:
    # In a real app, this would be an async API call
    await asyncio.sleep(0.1)  # Simulate network latency
    return f"Weather in {location} is cloudy with a chance of rain. Current temperature is 18°C."

# Register the async function just like a synchronous one
registry.register_tool(async_weather)

# Use the unified interface that works with BOTH sync and async functions
async def main():
    # The smart_call method automatically detects function type and handles it appropriately
    sync_result = await registry.smart_call("get_weather", location="Paris")
    async_result = await registry.smart_call("async_weather", location="Tokyo")
    
    # Similarly for OpenAI-style tool calls - works with both sync and async
    tool_result = await registry.smart_tool_call({
        "id": "call_456",
        "type": "function",
        "function": {
            "name": "async_weather",
            "arguments": '{"location": "Berlin"}'
        }
    })
    
    print(tool_result.result)  # "Weather in Berlin is cloudy with a chance of rain."

# Run in your async application
asyncio.run(main())
```

This means you can:
- Mix sync and async functions in the same registry
- Use a consistent API regardless of function type
- Integrate with async frameworks like FastAPI
- Make external API calls efficiently

## 🛡️ Error Handling

In real-world applications, errors are inevitable. Toolkitr provides comprehensive error handling that balances user experience, security, and debuggability:

```python
import json

# Configure error handling at registry creation
registry = ToolRegistry(
    # Define how exceptions should be presented to the LLM
    exception_serializer=lambda exc: json.dumps({
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "severity": "error" if isinstance(exc, ValueError) else "warning"
        }
    }, indent=2),
    # Control whether exceptions should be caught or propagated
    catch_exceptions=True  # Set to False in development for easier debugging
)

@registry.tool()
def database_query(query_params: str) -> str:
    """Query the database with the given parameters."""
    # Simulate potential errors in real applications
    if query_params == "invalid":
        raise ValueError("Invalid query parameters")
    elif query_params == "unauthorized":
        raise PermissionError("User not authorized to access this data")
    return f"Query results for: {query_params}"

# When you handle a tool call with an error:
result = registry.tool_call({
    "id": "db_query_123",
    "type": "function",
    "function": {
        "name": "database_query",
        "arguments": '{"query_params": "invalid"}'
    }
})

# You get a comprehensive result object:
print(result.success)      # False - indicating there was an error
print(result.error)        # The actual ValueError exception
print(result.result)       # None since the function didn't complete
print(result.message)      # LLM-friendly formatted error message with your custom serialization
```

Benefits:
- Prevent sensitive error details from leaking to the LLM
- Provide helpful error messages to guide the LLM's next actions
- Maintain full control over how errors are processed
- Excellent for debugging and production use cases

## 🤖 OpenAI Integration

Toolkitr is designed to work seamlessly with OpenAI's function calling capabilities. Here's a complete example showing how to integrate your tools with OpenAI chat completions:

```python
from openai import OpenAI
from toolkitr import ToolRegistry

# Set up your tool registry
registry = ToolRegistry()

@registry.tool(title="Get Current Weather")
def get_weather(location: str, units: str = "celsius") -> str:
    """Get the current weather in a given location"""
    # In a real app, call a weather API here
    return f"The weather in {location} is {22 if units.startswith('c') else 295}°{units[0].upper()}."

@registry.tool(title="Get Restaurant Recommendations")
def get_restaurants(cuisine: str, location: str, price_range: str = "moderate") -> str:
    """Find restaurants matching the requested criteria"""
    return f"Here are 3 {price_range} {cuisine} restaurants in {location}: [restaurant list]"

# Set up OpenAI client
client = OpenAI()
messages = [
    {"role": "user", "content": "What's the weather in London? Also, recommend some Italian restaurants there."}
]

# Create chat completion with tools
response = client.chat.completions.create(
    messages=messages,
    model="gpt-4-turbo",
    tools=registry.definitions()  # This automatically formats your tools for OpenAI
)

# Handle tool calls and continue the conversation
message = response.choices[0].message
messages.append(message.model_dump())  # Add assistant's response to conversation

if message.tool_calls:
    # Process each tool call the model requested
    for tool_call in message.tool_calls:
        # Execute the tool and get results
        tool_result = registry.tool_call(tool_call.model_dump())
        
        # Add the tool response to the conversation
        messages.append(tool_result.message)  # Adds as a role="tool" message
    
    # Get the final answer incorporating the tool results
    final_response = client.chat.completions.create(
        messages=messages,
        model="gpt-4-turbo"
    )
    
    print(final_response.choices[0].message.content)
```

This integration offers:
- Clean separation between tool logic and LLM interaction
- Type validation for all parameters
- Automatic conversion between Python and JSON data types
- Streamlined error handling with intelligent responses

## 🧩 Advanced Features

### Custom Serializers

Control exactly how your function results are presented to the LLM with custom serializers:

```python
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class WeatherReport:
    location: str
    temperature: float
    conditions: str
    humidity: int
    updated_at: datetime

# Registry-level serializer for all tools
def global_serializer(result):
    """Format all results with a timestamp and structured format"""
    if isinstance(result, dict) or hasattr(result, "__dict__"):
        # Convert to JSON with proper formatting
        if hasattr(result, "__dict__"):
            result = result.__dict__
        return json.dumps(result, indent=2, default=str)
    return str(result)

registry = ToolRegistry(response_serializer=global_serializer)

# Per-tool custom serializer for special formatting needs
def weather_serializer(report: WeatherReport) -> str:
    """Format weather data in a human-readable format"""
    return f"""Weather Report for {report.location}:
- Temperature: {report.temperature}°C
- Conditions: {report.conditions}
- Humidity: {report.humidity}%
- Last Updated: {report.updated_at.strftime('%H:%M:%S')}"""

@registry.tool(response_serializer=weather_serializer)
def get_detailed_weather(location: str) -> WeatherReport:
    """Get detailed weather information for a location."""
    # In real code, this would call a weather API
    return WeatherReport(
        location=location,
        temperature=22.5,  # Celsius
        conditions="Partly Cloudy",
        humidity=65,  # Percentage
        updated_at=datetime.now()
    )

# The result will be formatted using the custom serializer
# when returned to the LLM, making it more readable and useful
```

Benefits:
- Format complex objects in LLM-friendly ways
- Handle custom data types and date/time information properly
- Present information in the most useful format for the LLM to process
- Different serialization strategies for different tools

### Human-friendly Tool Titles

Improve the LLM's understanding and selection of tools by providing clear, descriptive titles:

```python
# Tools with explicit, human-friendly titles improve LLM understanding
@registry.tool(
    title="Get Current Weather Conditions", 
    description="Provides real-time weather data for any city worldwide"
)
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny with a high of 22°C."

@registry.tool(
    title="Search Knowledge Base Articles",
    description="Find help articles related to a specific topic or question"
)
def search_kb(query: str, max_results: int = 3) -> list[str]:
    """Search the knowledge base for articles matching the query."""
    # This would call your search API in practice
    return [f"Article {i}: Results for '{query}'" for i in range(max_results)]
```

When these tools are presented to the LLM:
- The titles appear in the LLM's interface, making it easier for the LLM to choose the right tool
- For UI-based systems (like ChatGPT), users see these friendly titles
- More descriptive titles lead to better tool selection by the model

### Strict Mode

Control whether parameters are mandatory or optional in the generated schema, as defined by OpenAI's function calling specification:

```python
# Global registry setting for parameter validation
registry = ToolRegistry(strict=True)  # All parameters are required, even those with defaults

# Specific tools can override the global setting
@registry.tool(strict=False)
def search_database(
    query: str, 
    limit: int = 10,
    # Other documented parameters...
) -> list[str]:
    """Flexible search tool where only explicit parameters are required.
    
    With strict=False, parameters with default values become optional in the schema.
    """
    return [f"Result {i} for '{query}'" for i in range(limit)]

@registry.tool(strict=True)  # Every parameter is required in the schema
def critical_operation(resource_id: str, action: str) -> str:
    """Performs critical operations where all parameters must be explicitly provided."""
    return f"Performed {action} on {resource_id}"
```

When to use each mode:
- **Strict mode (True)**: When you want to ensure the LLM provides all parameters explicitly, even those with default values
- **Flexible mode (False)**: When you want parameters with defaults to be optional in the schema, giving the LLM more flexibility

## 🧠 Complex Types

Toolkitr excels at handling complex Python data types, automatically converting between Python objects and JSON. This enables you to create tools with rich, structured inputs and outputs:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal, TypedDict, NamedTuple, List, Dict
from datetime import datetime

# --- RICH TYPE DEFINITIONS ---

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UserProfile:
    name: str
    email: str
    role: str
    department: Optional[str] = None
    joined_date: Optional[datetime] = None

class TaskAttachments(TypedDict, total=False):
    files: List[str]
    links: List[str]
    notes: str

class GeoCoordinate(NamedTuple):
    latitude: float
    longitude: float
    altitude: Optional[float] = None

# --- TOOL WITH COMPLEX TYPES ---

@registry.tool(title="Create New Task")
def create_task(
    assignee: UserProfile,
    priority: TaskPriority,
    location: Optional[GeoCoordinate] = None,
    attachments: Optional[TaskAttachments] = None,
    status: Literal["draft", "assigned", "in_progress", "review", "completed"] = "assigned",
    tags: List[str] = [],
    due_date: Optional[datetime] = None
) -> Dict[str, any]:
    """Create a new task in the project management system."""
    
    # In a real implementation, this would create a task in your system
    task_id = "TASK-123"
    
    # Return rich structured data
    return {
        "task_id": task_id,
        "summary": f"Task assigned to {assignee.name} with {priority.value} priority",
        "status": status,
        "details": {
            "assignee": assignee.__dict__,
            "due_date": due_date.isoformat() if due_date else None,
            "has_attachments": bool(attachments),
            "location_specified": location is not None,
            "tag_count": len(tags)
        }
    }

# --- LLM INTERACTION ---

# When used with an LLM, Toolkitr will:
# 1. Generate proper JSON Schema for all these complex types
# 2. Validate incoming JSON against these types
# 3. Convert JSON to proper Python objects (enums, dataclasses, etc.)
# 4. Execute your function with the correct Python types
# 5. Convert the result back to JSON for the LLM
```

Key benefits:
- Write truly Pythonic code with native data structures
- No need to manually serialize/deserialize between JSON and Python objects
- Strong type safety with proper validation
- Support for nested and recursive data structures
- Clean separation between your business logic and LLM integration

## ⚠️ Limitations and Best Practices

To get the most out of Toolkitr, keep these recommendations in mind:

- **Tuple handling**: When working with LLM providers:
  - Prefer `NamedTuple` for fixed-length sequences with named fields
  - Use `List` for variable-length sequences
  - Consider `dataclass` for structured data instead of regular tuples

- **Function design for LLMs**:
  - Use clear, descriptive parameter names
  - Add `Annotated` types with descriptions for better LLM understanding
  - Keep function purposes focused and singular
  - Provide reasonable defaults when possible

- **Error handling**:
  - Consider what information is safe to expose to the LLM in error messages
  - Use custom exception serializers for sensitive operations
  - During development, set `catch_exceptions=False` for easier debugging

- **Performance considerations**:
  - For high-frequency API services, consider caching tool definitions
  - Use async tools for I/O-bound operations

## License

MIT License. See [LICENSE](LICENSE) file for details.
