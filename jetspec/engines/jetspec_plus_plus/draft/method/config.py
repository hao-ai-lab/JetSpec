"""Pure five-category launch configs; no environment or engine dependencies."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


CATEGORIES = ("method", "model", "engine", "hardware", "benchmark")


@dataclass(frozen=True)
class Schema:
    """A category owns its leaf paths; checks receive the complete config.

    Fields map dotted paths to exact Python types, enum tuples or callables.
    Callables raise ValueError or return False to reject a value/config.
    A plain field dict is shorthand for a Schema with no required keys/checks.
    """
    fields: dict
    required: tuple = ()
    checks: tuple = ()
    atomic: tuple = ()


def _canonical_json(resolved):
    """The stable JSON representation used by launch/checkpoint identities."""
    return json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _config_hash(resolved):
    return hashlib.sha256(_canonical_json(resolved).encode()).hexdigest()


def _merge(destination, source):
    """Merge objects recursively, copying replaced values."""
    for key, value in source.items():
        if isinstance(value, dict):
            _merge(destination.setdefault(key, {}), value)
        else:
            destination[key] = deepcopy(value)


def _read_entry(root, category, name):
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category}")
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
        raise ValueError(f"invalid {category} id {name!r}")
    path = Path(root) / (category + ".json")
    entries = json.loads(path.read_text()) if path.is_file() else {}
    if name not in entries:
        raise ValueError(f"unknown {category} key {name}")
    payload = entries[name]
    if not isinstance(payload, dict):
        raise ValueError(f"{category}.{name} must contain an object")
    return payload


def _load(root, category, name):
    """Read an entry with at most one same-category inheritance level."""
    payload = _read_entry(root, category, name)
    if "extends" not in payload:
        return payload
    parent = payload.pop("extends")
    if parent == name:
        raise ValueError(f"extends cycle in {category}.{name}")
    base = _read_entry(root, category, parent)
    if "extends" in base:
        if base["extends"] == name or base["extends"] == parent:
            raise ValueError(f"extends cycle in {category}.{name}")
        raise ValueError(f"extends allows only one level in {category}.{name}")
    _merge(base, payload)
    return base


def _compose(root, names, overrides=()):
    """Select all five entries and apply category.field=JSON overrides."""
    if set(names) != set(CATEGORIES):
        raise ValueError("names must contain method, model, engine, hardware and benchmark")
    resolved = {category: _load(root, category, names[category]) for category in CATEGORIES}
    for override in overrides:
        field, separator, raw = override.partition("=")
        parts = field.split(".")
        if not separator or len(parts) < 2 or parts[0] not in resolved or not all(parts):
            raise ValueError(f"invalid override {override!r}")
        value = json.loads(raw)
        payload = resolved[parts[0]]
        for key in parts[1:-1]:
            payload = payload.setdefault(key, {})
            if not isinstance(payload, dict):
                raise ValueError(f"override path is not an object: {field}")
        payload[parts[-1]] = value
    return resolved


def _flatten(payload, prefix="", *, atomic=()):
    """Flatten nested objects to leaf paths, retaining declared atomic objects."""
    if not isinstance(payload, dict):
        raise ValueError(f"{prefix or 'layer'} must be an object")
    result = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key or "." in key:
            raise ValueError(f"invalid field name: {key!r}")
        path = prefix + key
        if isinstance(value, dict) and path not in atomic:
            if not value:
                raise ValueError(f"empty object at {path}")
            result.update(_flatten(value, path + ".", atomic=atomic))
        else:
            result[path] = value
    return result


def _validate_ownership(schemas):
    """Each leaf path has exactly one owner: its schema's category."""
    owners = {}
    for category, fields in schemas.items():
        if isinstance(fields, Schema):
            fields = fields.fields
        for key in fields:
            if key in owners:
                raise ValueError(f"key {key} declared by two layers: {owners[key]}, {category}")
            owners[key] = category
    return owners


def _validate_fields(payload, fields, required, label, *, owners=None, atomic=()):
    """Validate a closed set of leaf paths; validators are types or callables."""
    values = _flatten(payload, atomic=atomic)
    for key, value in values.items():
        if key not in fields:
            owner = (owners or {}).get(key)
            hint = f"; owned by {owner}" if owner else ""
            raise ValueError(f"unknown {label} key {key}{hint}")
        rule = fields[key]
        if isinstance(rule, type):
            valid = type(value) is rule
        elif isinstance(rule, tuple):
            valid = any(type(value) is type(option) and value == option for option in rule)
        else:
            valid = rule(value) is not False
        if not valid:
            raise ValueError(f"{label}.{key} has invalid type or value")
    missing = set(required) - values.keys()
    if missing:
        raise ValueError(f"{label} required keys missing: {sorted(missing)}")


def _validate(resolved, schemas):
    """Validate all five categories against closed schemas owned per leaf."""
    if set(resolved) != set(CATEGORIES) or set(schemas) != set(CATEGORIES):
        raise ValueError("resolved config and schemas must contain all five categories")
    owners = _validate_ownership(schemas)
    normalized = {category: schema if isinstance(schema, Schema) else Schema(schema)
                  for category, schema in schemas.items()}
    for category, schema in normalized.items():
        _validate_fields(resolved[category], schema.fields, schema.required, category,
                         owners=owners, atomic=schema.atomic)
    for category, schema in normalized.items():
        for check in schema.checks:
            if check(resolved) is False:
                raise ValueError(f"{category} cross-category check failed")


# Generic config utilities use aliases so the method-step audit tracks algorithms.
canonical_json = _canonical_json
config_hash = _config_hash
merge = _merge
load = _load
compose = _compose
flatten = _flatten
validate_ownership = _validate_ownership
validate_fields = _validate_fields
validate = _validate
