import inspect
import json
from typing import (
    Any,
    Callable,
    get_type_hints,
    Annotated,
    Optional,
    Literal,
    TypedDict, Iterator,
)
from enum import Enum
from dataclasses import dataclass

from minitools._schema import python_type_to_json_schema, json_to_python


class ToolFunctionDefinition(TypedDict, total=False):
    """A definition of a function tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool


class ToolDefinition(TypedDict):
    """A definition of a registered tool."""

    type: Literal["function"]
    function: ToolFunctionDefinition


@dataclass(frozen=True)
class ToolInfo:
    """Information about a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    function: Callable
    is_async: bool = False
    strict: bool = False

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            type="function",
            function=ToolFunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
                strict=self.strict,
            )
        )


class ToolRegistry:
    def __init__(self, strict: bool = False):
        self._registry: dict[str, ToolInfo] = {}
        self._default_strict = strict

    def __getitem__(self, tool_name: str):
        return self._registry[tool_name]

    def __iter__(self) -> Iterator[ToolInfo]:
        return iter(self._registry.values())

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, item) -> bool:
        return item in self._registry

    def register_tool(
        self,
        func: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        strict: Optional[bool] = None,
    ):
        strict = self._default_strict if strict is None else strict
        sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)

        tool_name = name if name else func.__name__
        tool_description = description if description else (func.__doc__ or "").strip()

        properties = {}
        required_fields = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            py_type = type_hints.get(param_name, Any)
            schema_prop = python_type_to_json_schema(py_type, strict=strict)
            properties[param_name] = schema_prop
            if param.default is inspect.Parameter.empty:
                required_fields.append(param_name)

        parameters_schema = {"type": "object", "properties": properties}
        if strict:
            parameters_schema["additionalProperties"] = False
        if required_fields:
            parameters_schema["required"] = required_fields

        is_async = inspect.iscoroutinefunction(func)

        tool_info = ToolInfo(
            name=tool_name,
            description=tool_description,
            parameters=parameters_schema,
            function=func,
            is_async=is_async,
            strict=strict,
        )

        self._registry[tool_name] = tool_info

    def call_tool(self, tool_name: str, json_args: dict):
        tool_info = self._registry[tool_name]
        func = tool_info.function

        sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)

        py_kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            py_type = type_hints.get(param_name, Any)
            if param_name in json_args:
                py_kwargs[param_name] = json_to_python(json_args[param_name], py_type)

        if tool_info.is_async:
            async def async_call():
                return await func(**py_kwargs)

            return async_call()
        else:
            return func(**py_kwargs)

    def definition(self, tool_name: str) -> dict[str, Any]:
        tool_info = self._registry[tool_name]
        return tool_info.definition

    def definitions(self) -> list[dict[str, Any]]:
        return [tool_info.definition for tool_info in self._registry.values()]

    def tool(self, *, name: str = None, description: str = None, strict: Optional[bool] = None):
        def decorator(func: Callable):
            self.register_tool(func, name=name, description=description, strict=strict)
            return func

        return decorator
