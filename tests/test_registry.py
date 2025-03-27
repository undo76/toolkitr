from dataclasses import dataclass

import pytest

from llmtoolbox._registry import ToolRegistry


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
    result = registry.call("echo", **{"x": "test"})
    assert result == "test"

    with pytest.raises(KeyError):
        registry.call("unknown", **{})

    with pytest.raises(TypeError):
        registry.call("echo", **{})


def test_iteration():
    """Test iteration and container operations on registry."""
    registry = ToolRegistry()

    # Test empty registry
    assert len(registry) == 0
    assert list(registry) == []
    assert "echo" not in registry

    # Add a tool
    registry.register_tool(lambda x: x, name="echo", description="Echo tool")

    # Test non-empty registry
    assert len(registry) == 1
    tools = list(registry)
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].description == "Echo tool"
    assert "echo" in registry
    assert "unknown" not in registry


def test_method():
    """Test registering a method."""

    @dataclass
    class Test:
        name: str

        def echo(self, x: str):
            """Echoes the input."""
            return " ".join([self.name, x])

    registry = ToolRegistry()
    registry.register_tool(Test("Hi").echo)
    assert "echo" in registry
    assert len(registry) == 1
    assert list(registry)[0].name == "echo"
    assert list(registry)[0].description == "Echoes the input."
    assert list(registry)[0].function("test") == "Hi test"
    assert list(registry)[0].is_async is False
    assert list(registry)[0].parameters == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    result = registry.call("echo", **{"x": "test"})
    assert result == "Hi test"


def test_import_registry():
    """Test importing tools from another registry."""
    # Create source registry
    src_registry = ToolRegistry()
    src_registry.register_tool(lambda x: x, name="echo", description="Echoes the input.")
    
    # Create target registry
    target_registry = ToolRegistry()
    
    # Import without namespace
    target_registry.import_registry(src_registry)
    assert "echo" in target_registry
    assert target_registry.call("echo", x="test") == "test"
    
    # Create another target registry for namespaced import
    ns_registry = ToolRegistry()
    
    # Import with namespace
    ns_registry.import_registry(src_registry, namespace="utils")
    assert "utils.echo" in ns_registry
    assert "echo" not in ns_registry
    assert ns_registry.call("utils.echo", x="test") == "test"
    
    # Test overwrite behavior
    ns_registry.register_tool(lambda x: f"original: {x}", name="utils.echo")
    
    # Should raise an error when trying to import with the same name
    with pytest.raises(ValueError):
        ns_registry.import_registry(src_registry, namespace="utils")
    
    # Should work with overwrite=True
    ns_registry.import_registry(src_registry, namespace="utils", overwrite=True)
    assert ns_registry.call("utils.echo", x="test") == "test"  # Overwritten with imported function
