import pytest

from minitools._registry import ToolRegistry


def test_init():
    """Test the initialization of the registry."""
    registry = ToolRegistry()
    assert registry._registry == {}
    assert list(registry) == []
    assert len(registry) == 0


def test_register_tool():
    """Test registering a tool."""
    registry = ToolRegistry()
    registry.register_tool(lambda x: x, name="echo", description="Echoes the input.")
    assert "echo" in registry._registry
    assert len(registry) == 1
    assert list(registry)[0].name == "echo"
    assert list(registry)[0].description == "Echoes the input."
    assert list(registry)[0].function("test") == "test"
    assert list(registry)[0].is_async is False
    assert list(registry)[0].parameters == {
        "properties": {"x": {}},
        "required": ["x"],
        "type": "object",
    }


def test_call_tool():
    """Test calling a tool."""
    registry = ToolRegistry()
    registry.register_tool(lambda x: x, name="echo", description="Echoes the input.")
    result = registry.call_tool("echo", {"x": "test"})
    assert result == "test"

    with pytest.raises(KeyError):
        registry.call_tool("unknown", {})

    with pytest.raises(TypeError):
        registry.call_tool("echo", {})
