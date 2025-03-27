import json
from enum import Enum
from typing import Any


def default_serializer(obj: Any) -> str:
    """Smart serializer that handles various Python types appropriately.
    
    - Complex data structures (lists, dicts) use JSON
    - Custom objects try to convert to dict or string representation
    - Handles dataclasses, enums, and other common types
    """
    # Handle None explicitly
    if obj is None:
        return "null"

    # For strings, just return them within JSON (keeps quotes and escaping)
    # For other primitives, use standard JSON serialization
    if isinstance(obj, (str, int, float, bool, type(None))):
        return json.dumps(obj)

    # Complex data structures use pretty JSON
    try:
        # Custom encoder function for handling non-standard JSON types
        def json_encoder(o):
            # Handle dataclasses
            if hasattr(o, "__dataclass_fields__"):
                return {f: getattr(o, f) for f in o.__dataclass_fields__}

            # Handle Enums
            if isinstance(o, Enum):
                return o.value

            # Handle objects with __dict__
            if hasattr(o, "__dict__"):
                return o.__dict__

            # Last resort: convert to string
            return str(o)

        return json.dumps(obj, default=json_encoder, ensure_ascii=False)
    except:
        # Fallback to basic string representation if JSON fails
        return f'"{str(obj)}"'  # Quote the string to make valid JSON


def default_exception_serializer(exc: Exception) -> str:
    """Default serializer for exceptions.
    
    Returns a JSON object with the exception type and message.
    """
    return json.dumps({
        "error": {
            # "type": type(exc).__name__,
            "message": str(exc),
            # Include traceback for debugging but without full paths
            # "traceback": traceback.format_exc().splitlines()
        }
    })
