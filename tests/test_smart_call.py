import pytest
import asyncio
import json
from minitools import ToolRegistry, default_serializer


def example_function(x: str) -> str:
    """Test synchronous function."""
    return f"Hello, {x}!"


async def async_example_function(x: str) -> str:
    """Test asynchronous function."""
    await asyncio.sleep(0.01)  # Simulate async work
    return f"Async Hello, {x}!"


@pytest.fixture
def registry():
    registry = ToolRegistry()
    registry.register_tool(example_function)
    registry.register_tool(async_example_function)
    return registry


@pytest.mark.asyncio
async def test_smart_call_sync(registry):
    """Test smart_call with a synchronous function."""
    result = await registry.smart_call("example_function", x="World")
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_smart_call_async(registry):
    """Test smart_call with an asynchronous function."""
    result = await registry.smart_call("async_example_function", x="Async World")
    assert result == "Async Hello, Async World!"


@pytest.mark.asyncio
async def test_smart_tool_call_sync(registry):
    """Test smart_tool_call with a synchronous function."""
    tool_call = {
        "type": "function",
        "id": "call_123",
        "function": {
            "name": "example_function",
            "arguments": '{"x": "Tool World"}'
        }
    }
    
    response = await registry.smart_tool_call(tool_call)
    
    assert response["role"] == "tool"
    assert response["tool_call_id"] == "call_123"
    assert response["name"] == "example_function"
    assert response["content"] == '"Hello, Tool World!"'


@pytest.mark.asyncio
async def test_smart_tool_call_async(registry):
    """Test smart_tool_call with an asynchronous function."""
    tool_call = {
        "type": "function",
        "id": "call_456",
        "function": {
            "name": "async_example_function",
            "arguments": '{"x": "Async Tool World"}'
        }
    }
    
    response = await registry.smart_tool_call(tool_call)
    
    assert response["role"] == "tool"
    assert response["tool_call_id"] == "call_456"
    assert response["name"] == "async_example_function"
    assert response["content"] == '"Async Hello, Async Tool World!"'


@pytest.mark.asyncio
async def test_custom_serializer():
    """Test registry with custom serializer."""
    # Define a custom serializer that adds a prefix
    def custom_serializer(result):
        if isinstance(result, str):
            return f'"CUSTOM: {result}"'
        return json.dumps(result)
    
    registry = ToolRegistry(response_serializer=custom_serializer)
    registry.register_tool(example_function)
    
    tool_call = {
        "type": "function",
        "id": "call_789",
        "function": {
            "name": "example_function",
            "arguments": '{"x": "Serialized"}'
        }
    }
    
    response = await registry.smart_tool_call(tool_call)
    assert response["content"] == '"CUSTOM: Hello, Serialized!"'

@pytest.mark.asyncio
async def test_per_tool_serializer():
    """Test setting serializer per tool."""
    # Registry-level serializer
    def registry_serializer(result):
        return f'"REGISTRY: {result}"'
    
    # Tool-level serializer
    def tool_serializer(result):
        return f'"TOOL: {result}"'
    
    registry = ToolRegistry(response_serializer=registry_serializer)
    
    # Register two tools, one with custom serializer
    registry.register_tool(example_function)
    registry.register_tool(
        async_example_function, 
        response_serializer=tool_serializer
    )
    
    # Test tool with registry-level serializer
    tool_call1 = {
        "type": "function",
        "id": "call_reg",
        "function": {
            "name": "example_function",
            "arguments": '{"x": "Default"}'
        }
    }
    response1 = await registry.smart_tool_call(tool_call1)
    assert response1["content"] == '"REGISTRY: Hello, Default!"'
    
    # Test tool with tool-level serializer
    tool_call2 = {
        "type": "function",
        "id": "call_tool",
        "function": {
            "name": "async_example_function",
            "arguments": '{"x": "Custom"}'
        }
    }
    response2 = await registry.smart_tool_call(tool_call2)
    assert response2["content"] == '"TOOL: Async Hello, Custom!"'


@pytest.mark.asyncio
async def test_class_with_smart_call():
    """Test smart_call with class methods."""
    
    class ToolClass:
        def __init__(self, prefix: str):
            self.prefix = prefix
            
        def sync_method(self, message: str) -> str:
            return f"{self.prefix}: {message}"
            
        async def async_method(self, message: str) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Async {self.prefix}: {message}"
    
    registry = ToolRegistry()
    instance = ToolClass("Test")
    
    registry.register_tool(instance.sync_method, name="sync_test")
    registry.register_tool(instance.async_method, name="async_test")
    
    sync_result = await registry.smart_call("sync_test", message="Hello")
    assert sync_result == "Test: Hello"
    
    async_result = await registry.smart_call("async_test", message="World")
    assert async_result == "Async Test: World"
