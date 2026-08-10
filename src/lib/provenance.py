"""Code-state provenance for experiment results.

A results JSON that says "auc 0.78" is only reproducible if it also says
which code produced it. collect() captures the git commit (and whether
the working tree was dirty --- a dirty run is NOT reproducible from the
commit alone), a sha256 of the producing script, the exact argv, and the
interpreter/host identity. The results registry merges this into every
saved result's _provenance block.

All fields degrade to None rather than raising: provenance must never be
the reason a result fails to save.
"""
import hashlib
import socket
import subprocess
import sys
from pathlib import Path

# src/lib/provenance.py -> repo root two levels up from src/
REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(args, repo_root) -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo_root), *args],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def file_sha256(path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def collect(script_path=None, repo_root=None) -> dict:
    """Snapshot of the code state producing a result. Never raises."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    commit = _git(["rev-parse", "HEAD"], root)
    dirty = None
    if commit is not None:
        # git_dirty must reflect uncommitted changes to tracked CODE/inputs,
        # NOT the tracked outputs the pipeline itself regenerates - otherwise
        # every rerun self-reports dirty. That is the results tree (new result
        # JSONs are untracked; results_registry.json is modified) AND the
        # paper figures/tables, which the actr_figures/tables stages
        # rewrite (matplotlib emits byte-different SVGs each run), dirtying the
        # tree for any provenance-stamping stage that runs after them. Ignore
        # untracked files and exclude all three so a pure rerun reads clean.
        status = _git(["status", "--porcelain", "--untracked-files=no",
                       "--", ".", ":(exclude)data/results",
                       ":(exclude)data/results",
                       ":(exclude,glob)paper/figures/**",
                       ":(exclude,glob)paper/tables/**"], root)
        dirty = bool(status) if status is not None else None
    return {
        "git_commit": commit,
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
                      if commit is not None else None,
        "git_dirty": dirty,
        "script_sha256": file_sha256(script_path) if script_path else None,
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
    }
