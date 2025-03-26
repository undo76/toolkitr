import pytest
import asyncio
from minitools import ToolRegistry


def my_function(x: str) -> str:
    """Test synchronous function."""
    return f"Hello, {x}!"


async def async_my_function(x: str) -> str:
    """Test asynchronous function."""
    await asyncio.sleep(0.01)  # Simulate async work
    return f"Async Hello, {x}!"


@pytest.fixture
def registry():
    registry = ToolRegistry()
    registry.register_tool(my_function)
    registry.register_tool(async_my_function)
    return registry


@pytest.mark.asyncio
async def test_smart_call_sync(registry):
    """Test smart_call with a synchronous function."""
    result = await registry.smart_call("my_function", x="World")
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_smart_call_async(registry):
    """Test smart_call with an asynchronous function."""
    result = await registry.smart_call("async_my_function", x="Async World")
    assert result == "Async Hello, Async World!"


@pytest.mark.asyncio
async def test_smart_tool_call_sync(registry):
    """Test smart_tool_call with a synchronous function."""
    tool_call = {
        "type": "function",
        "id": "call_123",
        "function": {
            "name": "my_function",
            "arguments": '{"x": "Tool World"}'
        }
    }
    
    response = await registry.smart_tool_call(tool_call)
    
    assert response["role"] == "tool"
    assert response["tool_call_id"] == "call_123"
    assert response["name"] == "my_function"
    assert response["content"] == '"Hello, Tool World!"'


@pytest.mark.asyncio
async def test_smart_tool_call_async(registry):
    """Test smart_tool_call with an asynchronous function."""
    tool_call = {
        "type": "function",
        "id": "call_456",
        "function": {
            "name": "async_my_function",
            "arguments": '{"x": "Async Tool World"}'
        }
    }
    
    response = await registry.smart_tool_call(tool_call)
    
    assert response["role"] == "tool"
    assert response["tool_call_id"] == "call_456"
    assert response["name"] == "async_my_function"
    assert response["content"] == '"Async Hello, Async Tool World!"'


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
