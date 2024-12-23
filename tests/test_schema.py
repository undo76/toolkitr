import pytest
from enum import Enum
from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Literal,
    Optional,
    TypedDict,
    NamedTuple,
    Any,
    Annotated,
)
from dataclasses import dataclass

from minitools._schema import python_type_to_json_schema, json_to_python


# Test fixtures
class Color(Enum):
    RED = "red"
    GREEN = "green"


@dataclass
class Person:
    name: str
    age: int


class UserDict(TypedDict, total=False):
    name: str
    role: Optional[str]


class UserTuple(NamedTuple):
    id: int
    username: str


def test_primitive_types():
    assert python_type_to_json_schema(str) == {"type": "string"}
    assert python_type_to_json_schema(int) == {"type": "integer"}
    assert python_type_to_json_schema(float) == {"type": "number"}
    assert python_type_to_json_schema(bool) == {"type": "boolean"}


def test_enum():
    schema = python_type_to_json_schema(Color)
    assert schema == {"type": "string", "enum": ["red", "green"]}

    # Test conversion
    assert json_to_python("red", Color) == Color.RED


def test_annotated():
    schema = python_type_to_json_schema(Annotated[str, "A description"])
    assert schema == {"type": "string", "description": "A description"}


def test_literal():
    schema = python_type_to_json_schema(Literal["a", "b"])
    assert schema == {"type": "string", "enum": ["a", "b"]}


def test_optional():
    schema = python_type_to_json_schema(Optional[str])
    assert schema == {"oneOf": [{"type": "string"}, {"type": "null"}]}

    # Test conversion
    assert json_to_python(None, Optional[str]) is None
    assert json_to_python("test", Optional[str]) == "test"


def test_list():
    schema = python_type_to_json_schema(List[int])
    assert schema == {"type": "array", "items": {"type": "integer"}}

    # Test conversion
    assert json_to_python([1, 2, 3], List[int]) == [1, 2, 3]


def test_tuple():
    schema = python_type_to_json_schema(Tuple[int, str])
    assert schema == {
        "type": "array",
        "items": [{"type": "integer"}, {"type": "string"}],
        "minItems": 2,
        "maxItems": 2,
    }

    # Test conversion
    assert json_to_python([1, "test"], Tuple[int, str]) == (1, "test")


def test_dict():
    schema = python_type_to_json_schema(Dict[str, int])
    assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}

    # Test conversion
    assert json_to_python({"a": 1}, Dict[str, int]) == {"a": 1}


def test_typed_dict():
    schema = python_type_to_json_schema(UserDict)
    assert schema == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"oneOf": [{"type": "string"}, {"type": "null"}]},
        },
    }

    # Test conversion
    data = {"name": "Alice", "role": "admin"}
    assert json_to_python(data, UserDict) == data


def test_named_tuple():
    schema = python_type_to_json_schema(UserTuple)
    assert schema == {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "username": {"type": "string"}},
        "required": ["id", "username"],
    }

    # Test conversion
    result = json_to_python({"id": 1, "username": "alice"}, UserTuple)
    assert isinstance(result, UserTuple)
    assert result.id == 1
    assert result.username == "alice"


def test_dataclass():
    schema = python_type_to_json_schema(Person)
    assert schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    # Test conversion
    result = json_to_python({"name": "Bob", "age": 30}, Person)
    assert isinstance(result, Person)
    assert result.name == "Bob"
    assert result.age == 30


def test_any():
    schema = python_type_to_json_schema(Any)
    assert schema == {}

    # Test conversion
    assert json_to_python(42, Any) == 42
    assert json_to_python("test", Any) == "test"

