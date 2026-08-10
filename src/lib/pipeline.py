"""Detached, restart-recoverable pipeline runner.

Replaces ad-hoc bash chains (`cmd1; cmd2; ...` with sentinel greps) which
proved fragile in practice: working-directory drift, self-matching pkill,
watcher processes dying with the agent session, and no resume after a host
or session restart.

Spec (JSON):
    {
      "name": "wave2",
      "workdir": "/abs/path",              # optional; default: spec's dir
      "log_dir": "/abs/path/logs",         # optional; default: <state>.logs/
      "stages": [
        {"name": "multiseed", "cmd": ["python3", "x.py"],
         "depends_on": []},
        {"name": "eval", "cmd": [...], "depends_on": ["multiseed"]}
      ]
    }

Commands:
    python -m lib.pipeline run <spec.json>      # execute (resumes if state exists)
    python -m lib.pipeline launch <spec.json>   # detach via setsid; survives the
                                                # launching session's death
    python -m lib.pipeline status <spec.json>   # human-readable stage table

Semantics:
    - Stages run serially in dependency order (list order among ready stages).
    - State is persisted atomically to <spec>.state.json after every
      transition; a re-run skips "done" stages and re-executes failed or
      interrupted ones.
    - A stage found in state "running" at startup is treated as interrupted
      (the previous supervisor died) and re-executed.
    - Stage stdout/stderr stream to <log_dir>/<stage>.log (append on retry).
    - A failing stage stops the pipeline with a nonzero exit; dependents are
      not run.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _state_path(spec_path: Path) -> Path:
    return spec_path.with_suffix(".state.json")


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    tmp.rename(path)


def _validate(spec: dict) -> list:
    stages = spec.get("stages")
    if not stages:
        raise ValueError("spec has no stages")
    names = [s["name"] for s in stages]
    if len(set(names)) != len(names):
        raise ValueError("duplicate stage names")
    known = set(names)
    for s in stages:
        for d in s.get("depends_on", []):
            if d not in known:
                raise ValueError(f"stage {s['name']} depends on unknown {d}")
        if not isinstance(s.get("cmd"), list) or not s["cmd"]:
            raise ValueError(f"stage {s['name']} has no cmd list")
    return stages


def _ready(stage: dict, state: dict) -> bool:
    return all(state.get(d, {}).get("status") == "done"
               for d in stage.get("depends_on", []))


def run(spec_path: Path) -> int:
    spec_path = spec_path.resolve()
    spec = _load_json(spec_path)
    stages = _validate(spec)
    # One supervisor per spec: an exclusive lock on <state>.lock, released
    # in the finally below. A second concurrent run() would otherwise race
    # the state file and double-run stages. Open O_RDWR (no truncate) so a
    # losing contender never clobbers the holder's recorded pid.
    lock_path = _state_path(spec_path).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.fdopen(os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644),
                        "r+")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fd.close()
        print(f"[pipeline] another supervisor is already running this spec "
              f"(lock: {lock_path}) - refusing to start", flush=True)
        return 3
    try:
        lock_fd.seek(0)
        lock_fd.truncate()
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return _run_locked(spec, stages, spec_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _run_locked(spec: dict, stages: list, spec_path: Path) -> int:
    """Execute the pipeline; caller holds the per-spec supervisor lock."""
    workdir = Path(spec.get("workdir") or spec_path.parent)
    # spec-level env applied to every stage's subprocess. Declares thread
    # policy in the spec instead of relying on the launching shell - e.g.
    # OMP/BLAS=1 so sklearn's joblib owns parallelism and BLAS does not
    # oversubscribe cores (the failure that wedged a PRISM stage for 90min).
    stage_env = {**os.environ, **{str(k): str(v)
                                  for k, v in spec.get("env", {}).items()}}
    state_path = _state_path(spec_path)
    log_dir = Path(spec.get("log_dir") or
                   state_path.with_suffix(".logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    state = _load_json(state_path) if state_path.exists() else {}
    # a stage left "running" means the previous supervisor died mid-stage
    for name, st in state.items():
        if st.get("status") == "running":
            st["status"] = "interrupted"
    _atomic_write(state_path, state)

    remaining = [s for s in stages
                 if state.get(s["name"], {}).get("status") != "done"]
    done_skipped = len(stages) - len(remaining)
    if done_skipped:
        print(f"[pipeline] resuming: {done_skipped} stage(s) already done",
              flush=True)

    while remaining:
        stage = next((s for s in remaining if _ready(s, state)), None)
        if stage is None:
            print("[pipeline] no runnable stage (failed dependency?) - stopping",
                  flush=True)
            return 2
        name = stage["name"]
        state[name] = {"status": "running", "started": time.time()}
        _atomic_write(state_path, state)
        print(f"[pipeline] {name}: starting", flush=True)
        log = open(log_dir / f"{name}.log", "a")
        proc = subprocess.run(stage["cmd"], cwd=str(workdir), env=stage_env,
                              stdout=log, stderr=subprocess.STDOUT)
        log.close()
        state[name].update(status="done" if proc.returncode == 0 else "failed",
                           exit=proc.returncode, ended=time.time())
        _atomic_write(state_path, state)
        print(f"[pipeline] {name}: "
              f"{'done' if proc.returncode == 0 else f'FAILED ({proc.returncode})'}",
              flush=True)
        if proc.returncode != 0:
            return 1
        remaining = [s for s in remaining if s["name"] != name]
    print("[pipeline] all stages done", flush=True)
    return 0


def launch(spec_path: Path) -> int:
    """Detach a supervisor with setsid so it survives this session."""
    spec_path = spec_path.resolve()
    sup_log = open(_state_path(spec_path).with_suffix(".supervisor.log"), "a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "lib.pipeline", "run", str(spec_path)],
        stdout=sup_log, stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent.parent),
        start_new_session=True,           # own session: survives parent death
        env={**os.environ},
    )
    print(f"[pipeline] launched supervisor pid={proc.pid} "
          f"(detached; state: {_state_path(spec_path)})")
    return 0


def status(spec_path: Path) -> int:
    spec_path = spec_path.resolve()
    spec = _load_json(spec_path)
    state_path = _state_path(spec_path)
    state = _load_json(state_path) if state_path.exists() else {}
    for s in spec["stages"]:
        st = state.get(s["name"], {})
        line = f"{s['name']:24s} {st.get('status', 'pending'):12s}"
        if "exit" in st:
            line += f" exit={st['exit']}"
        if st.get("started") and st.get("ended"):
            line += f" ({st['ended'] - st['started']:.0f}s)"
        print(line)
    return 0


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2 or argv[0] not in ("run", "launch", "status"):
        print("usage: python -m lib.pipeline {run|launch|status} <spec.json>")
        return 64
    return {"run": run, "launch": launch, "status": status}[argv[0]](Path(argv[1]))


if __name__ == "__main__":
    sys.exit(main())
