import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Required,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from toolkitr._schema import NoneType, json_to_python, python_type_to_json_schema
from toolkitr._serializer import default_exception_serializer


class ToolFunctionDefinition(TypedDict, total=False):
    """A definition of a function tool."""

    name: Required[str]
    description: str
    parameters: dict[str, object]
    strict: bool | None


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
    title: Optional[str] = None  # Human-friendly name
    response_serializer: Optional[Callable[[Any], str]] = None
    exception_serializer: Optional[Callable[[Exception], str]] = None

    def __post_init__(self):
        if self.title is None:
            # Set title to function name if not provided, as it is frozen we need to use object.__setattr__
            object.__setattr__(self, "title", self.name)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            {
                "type": "function",
                "function": ToolFunctionDefinition(
                    {
                        "name": self.name,
                        "description": self.description,
                        "parameters": self.parameters,
                        "strict": self.strict,
                    }
                ),
            }
        )

    def __call__(self, *args, **kwargs):
        if self.is_async:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self.function(*args, **kwargs)
            else:
                return asyncio.run(self.function(*args, **kwargs))
        else:
            return self.function(*args, **kwargs)

    def __acall__(self, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        """Call the t ol function with the provided arguments asynchronously."""
        if self.is_async:
            return self.function(*args, **kwargs)
        else:
            return asyncio.to_thread(self.function, *args, **kwargs)


class ToolCallFunctionDict(TypedDict):
    name: str
    arguments: str  # In JSON format


class ToolCallDict(TypedDict):
    type: Literal["function"]
    id: str
    function: ToolCallFunctionDict


class ContentPartTextDict(TypedDict, total=True):
    text: Required[str]
    type: Required[Literal["text"]]


class ToolCallMessageDict(TypedDict, total=True):
    """Message returned from a tool call execution."""

    role: Required[Literal["tool"]]
    tool_call_id: Required[str]
    content: Required[str | Iterable[ContentPartTextDict]]


@dataclass
class ToolCallResult:
    """Result of a tool call execution."""

    # Information about the tool that was called
    tool: ToolInfo

    # The formatted response message for LLM
    message: ToolCallMessageDict

    # The result from the tool function (None if it failed)
    result: Optional[Any] = None

    # The exception if the call failed (None if successful)
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        """Returns True if the tool call was successful."""
        return self.error is None


class ToolRegistry:
    def __init__(
        self,
        strict: bool = False,
        response_serializer: Callable[[Any], str] = json.dumps,
        exception_serializer: Callable[[Exception], str] = default_exception_serializer,
        catch_exceptions: bool = True,
    ):
        self._registry: dict[str, ToolInfo] = {}
        self._default_strict = strict
        self._response_serializer = response_serializer or json.dumps
        self._exception_serializer = (
            exception_serializer or default_exception_serializer
        )
        self._catch_exceptions = catch_exceptions

    @property
    def response_serializer(self) -> Callable[[Any], str]:
        return self._response_serializer

    @response_serializer.setter
    def response_serializer(self, serializer: Callable[[Any], str]):
        self._response_serializer = serializer

    @property
    def exception_serializer(self) -> Callable[[Exception], str]:
        return self._exception_serializer

    @exception_serializer.setter
    def exception_serializer(self, serializer: Callable[[Exception], str]):
        self._exception_serializer = serializer

    @property
    def catch_exceptions(self) -> bool:
        return self._catch_exceptions

    @catch_exceptions.setter
    def catch_exceptions(self, value: bool):
        self._catch_exceptions = value

    def __getitem__(self, tool_name: str):
        return self._registry[tool_name]

    def __iter__(self) -> Iterator[ToolInfo]:
        return iter(self._registry.values())

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, item) -> bool:
        return item in self._registry

    def import_registry(
        self,
        registry: "ToolRegistry",
        namespace: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Import tools from another registry.

        Args:
            registry: The registry to import tools from
            namespace: Optional prefix to add to imported tool names (e.g., "math")
            overwrite: Whether to overwrite existing tools with the same name

        Example:
            # Create registries
            math_registry = ToolRegistry()
            math_registry.register_tool(add)
            math_registry.register_tool(subtract)

            main_registry = ToolRegistry()
            main_registry.import_registry(math_registry, namespace="math")

            # Call imported tools
            result = main_registry.call("math.add", a=1, b=2)  # Returns 3
        """
        for tool_info in registry:
            if namespace:
                name = f"{namespace}.{tool_info.name}"
            else:
                name = tool_info.name

            if name in self and not overwrite:
                raise ValueError(
                    f"Tool with name '{name}' already exists. Use overwrite=True to replace it."
                )

            # Create a new ToolInfo with the updated name
            imported_tool = ToolInfo(
                name=name,
                description=tool_info.description,
                parameters=tool_info.parameters,
                function=tool_info.function,
                is_async=tool_info.is_async,
                strict=tool_info.strict,
                title=tool_info.title,
                response_serializer=tool_info.response_serializer,
                exception_serializer=tool_info.exception_serializer,
            )

            # Add to registry
            self._registry[name] = imported_tool

    def register_tool(
        self,
        func: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        title: Optional[str] = None,
        strict: Optional[bool] = None,
        response_serializer: Optional[Callable[[Any], str]] = None,
        exception_serializer: Optional[Callable[[Exception], str]] = None,
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
            title=title,
            response_serializer=response_serializer,
            exception_serializer=exception_serializer,
        )

        self._registry[tool_name] = tool_info

    @staticmethod
    def _build_arguments(func: Callable, arguments: dict[str, Any]) -> dict[str, Any]:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)
        py_kwargs = {}
        for param_name in sig.parameters.keys():
            if param_name == "self" and inspect.ismethod(func):
                continue
            py_type = type_hints.get(param_name, Any)
            if param_name in arguments:
                py_kwargs[param_name] = json_to_python(arguments[param_name], py_type)
        return py_kwargs

    def _handle_exception(self, exc: Exception, tool_info: ToolInfo) -> str:
        """Handle an exception from a tool call."""
        if not self._catch_exceptions:
            raise exc
        serializer = tool_info.exception_serializer or self._exception_serializer
        return serializer(exc)

    def _create_tool_response(
        self, tool_call_id: str, content: str
    ) -> ToolCallMessageDict:
        """Create a tool call response."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def _call(self, name: str, /, **arguments: Any):
        tool_info = self[name]
        py_kwargs = self._build_arguments(tool_info.function, arguments)
        result = None
        error = None
        try:
            result = tool_info.__call__(**py_kwargs)
            serializer = tool_info.response_serializer or self._response_serializer
            content = serializer(result)
        except Exception as exc:
            error = exc
            content = self._handle_exception(exc, tool_info)
        return content, result, error

    async def _acall(self, name: str, /, **arguments: Any):
        tool_info = self[name]
        py_kwargs = self._build_arguments(tool_info.function, arguments)
        result = None
        error = None
        try:
            result = await tool_info.__acall__(**py_kwargs)
            serializer = tool_info.response_serializer or self._response_serializer
            content = serializer(result)
        except Exception as exc:
            error = exc
            content = self._handle_exception(exc, tool_info)
        return content, result, error

    def call(self, name: str, /, **arguments: Any):
        return self._call(name, **arguments)[0]

    async def acall(self, name: str, /, **arguments: Any) -> Any:
        return (await self._acall(name, **arguments))[0]

    async def smart_call(self, name: str, /, **arguments: Any):
        """Call a tool regardless of whether it's sync or async.

        This method automatically detects if the tool is synchronous or asynchronous
        and calls it appropriately. Synchronous tools are executed in a thread pool
        to prevent blocking the event loop.

        Args:
            name: The name of the tool to call
            **arguments: The arguments to pass to the tool

        Returns:
            The result of the tool exelution
        """
        return await self.acall(name, **arguments)

    def tool_call(self, tool_call: ToolCallDict) -> ToolCallResult:
        """Execute a synchronous tool call.

        Args:
            tool_call: The tool call dictionary in OpenAI format

        Returns:
            A ToolCallResult object containing the result or error, tool info, and formatted message
        """
        function = tool_call["function"]
        function_name = function["name"]
        tool = self[function_name]
        tool_call_id = tool_call["id"]
        arguments = json.loads(tool_call["function"]["arguments"])

        content, result, error = self._call(function_name, **arguments)
        message = self._create_tool_response(tool_call_id, content)

        return ToolCallResult(result=result, error=error, tool=tool, message=message)

    async def atool_call(self, tool_call: ToolCallDict) -> ToolCallResult:
        """Execute an asynchronous tool call.

        Args:
            tool_call: The tool call dictionary in OpenAI format

        Returns:
            A ToolCallResult object containing the result or error, tool info, and formatted message
        """
        function = tool_call["function"]
        function_name = function["name"]
        tool = self[function_name]
        tool_call_id = tool_call["id"]
        arguments = json.loads(tool_call["function"]["arguments"])

        content, result, error = await self._acall(function_name, **arguments)
        message = self._create_tool_response(tool_call_id, content)

        return ToolCallResult(result=result, error=error, tool=tool, message=message)

    async def smart_tool_call(self, tool_call: ToolCallDict) -> ToolCallResult:
        """Handle tool calls for both sync and async tools automatically.

        This method automatically detects if the tool is synchronous or asynchronous
        and processes the tool call appropriately. Synchronous tools are executed in
        a thread pool to prevent blocking the event loop.

        Args:
            tool_call: The tool call dictionary in OpenAI format

        Returns:
            A ToolCallResult object containing the result or error, tool info, and formatted message
        """
        function_name = tool_call["function"]["name"]
        tool = self[function_name]
        tool_call_id = tool_call["id"]
        arguments = json.loads(tool_call["function"]["arguments"])

        if tool.is_async:
            content, result, error = await self._acall(function_name, **arguments)
        else:
            # Run synchronous function in thread pool
            content, result, error = await asyncio.to_thread(
                self._call, function_name, **arguments
            )

        message = self._create_tool_response(tool_call_id, content)

        return ToolCallResult(result=result, error=error, tool=tool, message=message)

    def definitions(self) -> list[ToolDefinition]:
        return [tool_info.definition for tool_info in self._registry.values()]

    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        title: str | None = None,
        strict: bool | None = None,
        response_serializer: Optional[Callable[[Any], str]] = None,
        exception_serializer: Optional[Callable[[Exception], str]] = None,
    ):
        def decorator(func: Callable):
            self.register_tool(
                func,
                name=name,
                description=description,
                title=title,
                strict=strict,
                response_serializer=response_serializer,
                exception_serializer=exception_serializer,
            )
            return func

        return decorator
