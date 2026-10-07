"""Canonical experiment-protocol serialization and fingerprints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by experiment manifests."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def config_sha256(config: Any) -> str:
    """Return the SHA-256 fingerprint of a configuration value."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def aggregate_runtime_source_manifest(files: list[dict[str, str]]) -> str:
    """Hash sorted ``path + NUL + file_sha256 + newline`` manifest records."""
    normalized = sorted(
        ({"path": str(item["path"]), "sha256": str(item["sha256"])} for item in files),
        key=lambda item: item["path"],
    )
    digest = hashlib.sha256()
    for item in normalized:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def runtime_source_manifest(root: str | Path) -> dict[str, Any]:
    """Fingerprint runtime Python sources under algorithm/ and env/ only."""
    project_root = Path(root).resolve()
    files: list[dict[str, str]] = []
    for top_level in ("algorithm", "env"):
        source_root = project_root / top_level
        if not source_root.is_dir():
            raise FileNotFoundError(f"runtime source directory is missing: {source_root}")
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(project_root)
            if "__pycache__" in relative.parts:
                continue
            files.append({
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    files.sort(key=lambda item: item["path"])
    if not files:
        raise RuntimeError("runtime source manifest is empty")
    return {
        "runtime_source_manifest_sha256": aggregate_runtime_source_manifest(files),
        "runtime_source_manifest_file_count": len(files),
        "runtime_source_manifest_files": files,
    }


def runtime_source_branch_provenance(
    source_checkpoint: dict[str, Any], destination_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Describe source availability honestly and require destination identity."""
    destination_sha = destination_manifest.get("runtime_source_manifest_sha256")
    if not isinstance(destination_sha, str) or len(destination_sha) != 64:
        raise RuntimeError("destination runtime source manifest is missing")
    source_sha = source_checkpoint.get("extra", {}).get("runtime_source_manifest_sha256")
    if source_sha is not None and (not isinstance(source_sha, str) or len(source_sha) != 64):
        raise RuntimeError("source checkpoint runtime source manifest is malformed")
    return {
        "destination_runtime_source_manifest_sha256": destination_sha,
        "source_runtime_source_manifest_sha256": source_sha,
        "source_runtime_source_provenance_available": source_sha is not None,
    }


__all__ = [
    "aggregate_runtime_source_manifest",
    "canonical_json",
    "config_sha256",
    "runtime_source_branch_provenance",
    "runtime_source_manifest",
]
