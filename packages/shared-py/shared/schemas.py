"""Helpers to load JSON Schemas emitted from packages/contracts.

Each Python stage does:
    from shared.schemas import validate
    validate(body, "separate_request")
and gets a ValidationError if the body doesn't match the contract.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, ValidationError

_CONTRACTS_ROOT = (
    Path(__file__).resolve().parents[3] / "packages" / "contracts" / "json-schema"
)


@cache
def load(schema_name: str) -> dict[str, Any]:
    """Load packages/contracts/json-schema/<schema_name>.json (snake_case)."""
    path = _CONTRACTS_ROOT / f"{schema_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"contract schema not found: {path}")
    return json.loads(path.read_text())


@cache
def validator(schema_name: str) -> Draft7Validator:
    return Draft7Validator(load(schema_name))


def validate(body: Any, schema_name: str) -> None:
    """Raise jsonschema.ValidationError if body doesn't conform."""
    validator(schema_name).validate(body)


__all__ = ["load", "validator", "validate", "ValidationError"]
