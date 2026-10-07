"""Canonical configuration serialization and experiment fingerprints."""
from typing import Any
import hashlib
import json


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def config_sha256(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode('utf-8')).hexdigest()


__all__ = ['canonical_json', 'config_sha256']
