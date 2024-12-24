import inspect
from dataclasses import is_dataclass, fields, MISSING
from enum import Enum
from types import UnionType
from typing import (
    Any,
    Dict,
    get_origin,
    Annotated,
    get_args,
    Literal,
    Union,
    List,
    Tuple,
    Dict as TypingDict,
    NamedTuple,
    Type,
    cast,
)

NoneType = type(None)

type_map = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    NoneType: "null",
}


def is_named_tuple_type(t: Any) -> bool:
    return (
        isinstance(t, type)
        and issubclass(t, tuple)
        and hasattr(t, "__annotations__")
        and hasattr(t, "_fields")
    )


def is_typed_dict(obj: Any) -> bool:
    return (
        isinstance(obj, type)
        and issubclass(obj, dict)
        and hasattr(obj, "__annotations__")
        and isinstance(getattr(obj, "__annotations__", None), dict)
    )


def is_autogenerated_namedtuple_doc(namedtuple_cls: Type[NamedTuple]) -> bool:
    expected_doc: str = (
        f"{namedtuple_cls.__name__}({', '.join(namedtuple_cls._fields)})"
    )
    return namedtuple_cls.__doc__ == expected_doc


def python_type_to_json_schema(py_type: Any, strict: bool = False) -> Dict[str, Any]:
    origin = get_origin(py_type)

    # Initialize base schema with ordered keys
    schema: Dict[str, Any] = {}

    # Handle Annotated by extracting description first
    description = None
    if origin is Annotated:
        args = get_args(py_type)
        underlying_type = args[0]
        metadata = args[1:]
        if metadata and isinstance(metadata[0], str):
            description = metadata[0]
        py_type = underlying_type
        origin = get_origin(underlying_type)

    # Handle Literal
    if origin is Literal:
        values = get_args(py_type)
        first_type = type(values[0])
        if all(isinstance(v, first_type) for v in values):
            schema["type"] = type_map.get(first_type, None)
        if description:
            schema["description"] = description
        schema["enum"] = list(values)
        return schema

    # Enums
    if inspect.isclass(py_type) and issubclass(py_type, Enum):
        schema["type"] = "string"
        if description:
            schema["description"] = description
        schema["enum"] = [member.value for member in py_type]
        return schema

    # Union (including Optional)
    if origin is Union or isinstance(py_type, UnionType):
        args = get_args(py_type)
        # Optional[T]
        if len(args) == 2 and NoneType in args:
            non_none_type = args[0] if args[1] is NoneType else args[1]
            sub_schema = python_type_to_json_schema(non_none_type, strict=strict)
            schema["oneOf"] = [sub_schema, {"type": "null"}]
            if description:
                schema["description"] = description
            return schema
        else:
            if description:
                schema["description"] = description
            schema["oneOf"] = [python_type_to_json_schema(a, strict=strict) for a in args]
            return schema

    # Lists (List[T])
    if origin in (list, List):
        schema["type"] = "array"
        if description:
            schema["description"] = description
        (item_type,) = get_args(py_type)
        schema["items"] = python_type_to_json_schema(item_type, strict=strict)
        return schema

    # Tuples
    if origin in (tuple, Tuple):
        schema["type"] = "array"
        if description:
            schema["description"] = description

        args_ = get_args(py_type)
        if len(args_) == 2 and args_[1] is Ellipsis:
            # variable-length tuple
            item_schema = python_type_to_json_schema(args_[0], strict=strict)
            schema["items"] = item_schema
        else:
            # Fixed-length tuple using prefixItems
            # AI warn the user that this doesn't work well and to use NamedTuples instead
            items = [python_type_to_json_schema(a, strict=strict) for a in args_]
            length = len(items)
            schema["prefixItems"] = items
            schema["items"] = False  # Disallow additional items
            schema["minItems"] = length
            schema["maxItems"] = length
        return schema

    # Dicts (Dict[str, T])
    if origin in (dict, TypingDict):
        schema["type"] = "object"
        if description:
            schema["description"] = description
        key_type, value_type = get_args(py_type)
        schema["additionalProperties"] = python_type_to_json_schema(value_type, strict=strict)
        if strict:
            schema["additionalProperties"] = False
        return schema

    # TypedDict
    if is_typed_dict(py_type):
        schema["type"] = "object"
        if description:
            schema["description"] = description
        elif py_type.__doc__:
            schema["description"] = py_type.__doc__.strip()

        props = {}
        td_annotations = py_type.__annotations__
        for k, v in td_annotations.items():
            props[k] = python_type_to_json_schema(v, strict=strict)
        schema["properties"] = props

        required_fields = list(props.keys())
        if hasattr(py_type, "__optional_keys__"):
            for opt_key in py_type.__optional_keys__:
                if opt_key in required_fields:
                    required_fields.remove(opt_key)
        if strict:
            schema["additionalProperties"] = False
        if required_fields:
            schema["required"] = required_fields

        return schema

    # NamedTuple
    if is_named_tuple_type(py_type):
        schema["type"] = "object"
        if description:
            schema["description"] = description
        elif py_type.__doc__ and not is_autogenerated_namedtuple_doc(
            cast(type[NamedTuple], py_type)
        ):
            schema["description"] = py_type.__doc__.strip()

        nt_fields = py_type.__annotations__
        props = {
            fname: python_type_to_json_schema(ftype, strict=strict)
            for fname, ftype in nt_fields.items()
        }
        if strict:
            schema["additionalProperties"] = False
        schema["properties"] = props
        schema["required"] = list(props.keys())
        return schema

    # Dataclass
    if is_dataclass(py_type):
        schema["type"] = "object"
        if description:
            schema["description"] = description
        elif py_type.__doc__ and not py_type.__doc__.startswith(py_type.__name__):
            schema["description"] = py_type.__doc__.strip()

        props = {}
        required = []
        for f in fields(py_type):
            f_schema = python_type_to_json_schema(f.type)
            props[f.name] = f_schema
            if f.default is MISSING and f.default_factory is MISSING:
                required.append(f.name)
        schema["properties"] = props
        if required:
            schema["required"] = required
        if strict:
            schema["additionalProperties"] = False
        return schema

    # Any
    if py_type is Any:
        if description:
            schema["description"] = description
        return schema

    # Primitive fallback
    schema["type"] = type_map.get(py_type, "string")
    if description:
        schema["description"] = description
    return schema


def json_to_python(value: Any, py_type: Any) -> Any:
    origin = get_origin(py_type)
    if origin is Annotated:
        py_type = get_args(py_type)[0]
        origin = get_origin(py_type)

    if origin is Literal:
        return value

    if isinstance(py_type, type) and issubclass(py_type, Enum):
        return py_type(value)

    if origin is Union or isinstance(py_type, UnionType):
        args = get_args(py_type)
        if len(args) == 2 and NoneType in args:
            if value is None:
                return None
            non_none_type = args[0] if args[1] is NoneType else args[1]
            return json_to_python(value, non_none_type)
        else:
            for candidate in args:
                try:
                    return json_to_python(value, candidate)
                except Exception:
                    pass
            raise ValueError("No union variant matched the value")

    if origin in (list, List):
        (item_type,) = get_args(py_type)
        return [json_to_python(v, item_type) for v in value]

    if origin in (tuple, Tuple):
        args_ = get_args(py_type)
        if len(args_) == 2 and args_[1] is Ellipsis:
            item_type = args_[0]
            return tuple(json_to_python(v, item_type) for v in value)
        else:
            if len(value) != len(args_):
                raise ValueError("Tuple length mismatch")
            return tuple(json_to_python(v, t) for v, t in zip(value, args_))

    if origin in (dict, TypingDict):
        key_type, val_type = get_args(py_type)
        return {k: json_to_python(v, val_type) for k, v in value.items()}

    if is_typed_dict(py_type):
        td_annotations = py_type.__annotations__
        result = {}
        for k, val_type in td_annotations.items():
            if k in value:
                result[k] = json_to_python(value[k], val_type)
        return result

    if is_named_tuple_type(py_type):
        nt_fields = py_type.__annotations__
        kwargs = {
            fname: json_to_python(value[fname], ftype)
            for fname, ftype in nt_fields.items()
        }
        return py_type(**kwargs)

    if is_dataclass(py_type):
        cls_fields = fields(py_type)
        kwargs = {}
        for f in cls_fields:
            if f.name in value:
                kwargs[f.name] = json_to_python(value[f.name], f.type)
            elif f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
            else:
                raise ValueError(f"Missing required field {f.name}")
        return py_type(**kwargs)

    from typing import Any as AnyType

    if py_type is AnyType:
        return value

    return value
