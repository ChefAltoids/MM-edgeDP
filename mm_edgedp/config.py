from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

ConfigDict = Dict[str, Any]


def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge dictionaries without mutating inputs."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_json(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_parent_path(parent_ref: str, child_path: Path, repo_root: Path) -> Path:
    parent = Path(parent_ref)
    if parent.is_absolute():
        return parent

    local_candidate = (child_path.parent / parent).resolve()
    if local_candidate.exists():
        return local_candidate

    return (repo_root / parent).resolve()


def _load_recursive(path: Path, repo_root: Path, seen: List[Path]) -> ConfigDict:
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ValueError(f"Config extends cycle detected: {chain}")

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    payload = _load_json(path)
    parents = payload.get("extends", [])
    if not isinstance(parents, list):
        raise TypeError(f"Config field 'extends' must be a list in {path}")

    merged: ConfigDict = {}
    for parent_ref in parents:
        parent_path = _resolve_parent_path(parent_ref, path, repo_root)
        parent_payload = _load_recursive(parent_path, repo_root, [*seen, path])
        merged = deep_merge(merged, parent_payload)

    payload = deepcopy(payload)
    payload.pop("extends", None)
    return deep_merge(merged, payload)


def load_config(config_path: str | Path, repo_root: str | Path | None = None) -> ConfigDict:
    path = Path(config_path).resolve()
    root = Path(repo_root).resolve() if repo_root else path.parent.resolve()
    return _load_recursive(path, root, seen=[])


def _parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None

    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    if raw.startswith("[") or raw.startswith("{") or raw.startswith('"'):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    return raw


def apply_overrides(config: ConfigDict, overrides: Iterable[str]) -> ConfigDict:
    """Apply CLI overrides in dot-path style: section.key=value."""
    updated = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format, got: {item}")

        key_path, raw_value = item.split("=", 1)
        keys = [k for k in key_path.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Invalid override key path: {item}")

        current: ConfigDict = updated
        for key in keys[:-1]:
            existing = current.get(key)
            if existing is None:
                current[key] = {}
            elif not isinstance(existing, dict):
                raise TypeError(f"Cannot set nested key on non-dict field: {'.'.join(keys[:-1])}")
            current = current[key]

        current[keys[-1]] = _parse_override_value(raw_value.strip())

    return updated


def dump_config(config: ConfigDict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
