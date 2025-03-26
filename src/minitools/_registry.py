import inspect
import json
import asyncio
from typing import (
    Any,
    Callable,
    get_type_hints,
    Optional,
    Literal,
    TypedDict,
    Iterator, Coroutine,
    Union, get_origin, get_args,
)
from minitools._schema import NoneType
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
    response_serializer: Optional[Callable[[Any], str]] = None

    @property
    def definition(self) -> ToolDefinition:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            },
        }


class ToolCallFunctionDict(TypedDict):
    name: str
    arguments: str  # In JSON format


class ToolCallDict(TypedDict):
    type: Literal["function"]
    id: str
    function: ToolCallFunctionDict


class ToolCallMessageDict(TypedDict):
    """Message returned from a tool call execution."""

    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str


class ToolRegistry:
    def __init__(
        self, 
        strict: bool = False,
        response_serializer: Callable[[Any], str] = None
    ):
        self._registry: dict[str, ToolInfo] = {}
        self._default_strict = strict
        self._response_serializer = response_serializer or json.dumps
        
    @property
    def response_serializer(self) -> Callable[[Any], str]:
        return self._response_serializer
        
    @response_serializer.setter
    def response_serializer(self, serializer: Callable[[Any], str]):
        self._response_serializer = serializer

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
        response_serializer: Optional[Callable[[Any], str]] = None,
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
            
            # In strict mode, add all parameters to required fields
            # In non-strict mode, only add if not optional and has no default
            if strict:
                required_fields.append(param_name)
            else:
                # Check if the parameter is optional
                is_optional = False
                origin = get_origin(py_type)
                if origin is Union:
                    args = get_args(py_type)
                    if len(args) == 2 and NoneType in args:
                        is_optional = True
                
                # Only add to required if it has no default value and is not Optional
                if param.default is inspect.Parameter.empty and not is_optional:
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
            response_serializer=response_serializer,
        )

        self._registry[tool_name] = tool_info

    @staticmethod
    def _build_arguments(func: Callable, arguments: dict[str, Any]) -> dict[str, Any]:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)
        py_kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self" and inspect.ismethod(func):
                continue
            py_type = type_hints.get(param_name, Any)
            if param_name in arguments:
                py_kwargs[param_name] = json_to_python(arguments[param_name], py_type)
        return py_kwargs

    def call(self, name: str, **arguments: any):
        tool_info = self[name]
        func = tool_info.function
        py_kwargs = self._build_arguments(func, arguments)
        return func(**py_kwargs)

    async def acall(self, name: str, **arguments: any):
        tool_info = self[name]
        func = tool_info.function
        py_kwargs = self._build_arguments(func, arguments)
        return await func(**py_kwargs)

    def tool_call(self, tool_call: ToolCallDict) -> ToolCallMessageDict:
        function = tool_call["function"]
        function_name = function["name"]
        arguments = json.loads(function["arguments"])
        call_result = self.call(function_name, **arguments)
        
        # Use tool-specific serializer if available, otherwise use registry default
        tool_info = self[function_name]
        serializer = tool_info.response_serializer or self._response_serializer
        
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": function_name,
            "content": serializer(call_result),
        }

    async def atool_call(self, tool_call: ToolCallDict) -> ToolCallMessageDict:
        function = tool_call["function"]
        function_name = function["name"]
        arguments = json.loads(function["arguments"])
        call_result = await self.acall(function_name, **arguments)
        
        # Use tool-specific serializer if available, otherwise use registry default
        tool_info = self[function_name]
        serializer = tool_info.response_serializer or self._response_serializer
        
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": function_name,
            "content": serializer(call_result),
        }
        
    async def smart_call(self, name: str, **arguments: any):
        """Call a tool regardless of whether it's sync or async.
        
        This method automatically detects if the tool is synchronous or asynchronous
        and calls it appropriately. Synchronous tools are executed in a thread pool
        to prevent blocking the event loop.
        
        Args:
            name: The name of the tool to call
            **arguments: The arguments to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        tool_info = self[name]
        if tool_info.is_async:
            return await self.acall(name, **arguments)
        else:
            # Run synchronous function in thread pool to prevent blocking
            return await asyncio.to_thread(self.call, name, **arguments)
    
    async def smart_tool_call(self, tool_call: ToolCallDict) -> ToolCallMessageDict:
        """Handle tool calls for both sync and async tools automatically.
        
        This method automatically detects if the tool is synchronous or asynchronous
        and processes the tool call appropriately. Synchronous tools are executed in
        a thread pool to prevent blocking the event loop.
        
        Args:
            tool_call: The tool call dictionary in OpenAI format
            
        Returns:
            A tool call message dictionary in OpenAI format
        """
        function_name = tool_call["function"]["name"]
        if self[function_name].is_async:
            return await self.atool_call(tool_call)
        else:
            # Run synchronous function in thread pool to prevent blocking
            return await asyncio.to_thread(self.tool_call, tool_call)

    def definitions(self) -> list[dict[str, Any]]:
        return [tool_info.definition for tool_info in self._registry.values()]

    def tool(
        self,
        *,
        name: str = None,
        description: str = None,
        strict: Optional[bool] = None,
        response_serializer: Optional[Callable[[Any], str]] = None,
    ):
        def decorator(func: Callable):
            self.register_tool(
                func, 
                name=name, 
                description=description, 
                strict=strict,
                response_serializer=response_serializer
            )
            return func

        return decorator
