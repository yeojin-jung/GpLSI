"""Content-addressed immutable stage outputs with process locks and atomic commit."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile
import time
import uuid

import numpy as np

from .config import canonical_json, fingerprint


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """Explicit strings preserve infinite scientific scores, unlike JSON null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def stage_lock(directory, *, blocking=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Kernel-managed advisory locks release even after SIGKILL; never delete a
    # lock file while peers may hold an open handle to its inode.
    with open(directory / ".lock", "a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield directory
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def stage_key(specification, *, code_hash, input_hashes, parent_keys):
    return fingerprint({"schema": 2, "specification": specification, "code_hash": code_hash,
                        "input_hashes": input_hashes, "parent_keys": sorted(parent_keys)})


def compatible_completed(directory, key, *, verify_hashes=True):
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    try:
        metadata = json.loads(marker.read_text())
        if metadata["cache_key"] != key or metadata["state"] != "complete":
            return False
        for relative, digest in metadata["files"].items():
            path = directory / relative
            if not path.is_file() or (verify_hashes and sha256_file(path) != digest):
                return False
    except (KeyError, ValueError, OSError):
        return False
    return True


def commit_stage(directory, key, metadata, files):
    directory = Path(directory)
    for path in files:
        if not (directory / path).is_file():
            raise FileNotFoundError(directory / path)
    record = {**metadata, "cache_key": key, "state": "complete", "committed_at": time.time(),
              "files": {str(path): sha256_file(directory / path) for path in files}}
    atomic_json(directory / "complete.json", record)
    return record


def record_failed_attempt(directory, exception, metadata):
    # Failed attempts do not replace completion markers or successful outputs.
    path = Path(directory) / "attempts" / (uuid.uuid4().hex + ".json")
    atomic_json(path, {**metadata, "state": "failed", "type": type(exception).__name__,
                       "error": str(exception), "recorded_at": time.time()})
    return path
