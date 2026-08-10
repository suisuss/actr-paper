"""Atomic file-write helpers.

A direct write to a final path can be interrupted (full disk, kill,
crash) and leave a partial file that exists() reports as valid --- the
artifact then shadows checkpoints and fails on every later load until
someone deletes it by hand. These helpers write to a per-process tmp
name in the same directory, rename into place on success (rename is
atomic on POSIX within a filesystem), and remove the tmp on failure so
a full disk is not made fuller by orphaned partials.
"""
import json
import os
from pathlib import Path

import numpy as np


def atomic_savez(path, arrays: dict, compressed: bool = True) -> Path:
    """np.savez[_compressed] to `path` via tmp + rename. Raises on failure
    with the previous file (if any) left intact and the tmp removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # keep the .npz suffix so numpy doesn't append a second one
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
    try:
        (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_json_dump(path, obj, **json_kwargs) -> Path:
    """json.dump to `path` via tmp + rename, same guarantees as above."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, **json_kwargs)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
