"""
Experiment runtime with resource limits.

Controls CPU, memory, and process priority so experiments don't starve
the rest of the system.

Usage:
    from lib.runtime import init_runtime

    # Call at top of any experiment script
    init_runtime()

    # Or with custom limits
    init_runtime(max_ram_gb=4, max_cores=2, nice=10)

Configuration via environment variables (override defaults):
    EXPERIMENT_MAX_RAM_GB=4
    EXPERIMENT_MAX_CORES=2
    EXPERIMENT_NICE=10
    EXPERIMENT_AUTO_INIT=0   # disable the soft guardrails on `import lib`

Auto-guardrails: `lib/__init__.py` calls soft_init() on import, so any
process that touches the library gets a generous RAM watchdog (25% of
system RAM, floor DEFAULT_MAX_RAM_GB) and a SIGTERM grace handler even
if the script forgot init_runtime(). soft_init() deliberately does NOT
cap cores or change niceness --- those would silently fight explicit
taskset pinning and priority choices. Call init_runtime() for the full
treatment.

Graceful termination: register flush callbacks with on_terminate(); they
run on SIGTERM and on the RAM-watchdog abort path, before the process
exits. Callbacks must be fast and idempotent (e.g. "write the current
checkpoint"); they may run on the watchdog thread while the main thread
is mid-computation.
"""

import os
import resource
import multiprocessing
import signal


# Defaults: conservative limits that leave headroom for other work
DEFAULT_MAX_RAM_GB = 6
DEFAULT_MAX_CORES = max(1, multiprocessing.cpu_count() // 2)
DEFAULT_NICE = 10  # Lower priority (0=normal, 19=lowest)

# Soft (auto) init uses a generous ceiling: a forgotten init_runtime()
# should still die predictably, but must not strangle a legitimate job.
SOFT_RAM_FRACTION = 0.25

# Warn at startup when free disk drops below this (override with
# EXPERIMENT_MIN_DISK_GB). A near-full disk stops checkpoint/results
# writes mid-run; better to hear it before hours of compute, not after.
LOW_DISK_GB = 5

_state = {
    "initialized": None,      # None | "soft" | "full"
    "max_bytes": None,        # current watchdog ceiling (mutable; the
    "max_gb": None,           # single watchdog thread re-reads it)
    "watchdog_started": False,
}
_terminate_callbacks = []


def on_terminate(callback):
    """Register a flush callback to run before termination.

    Runs on SIGTERM and on RAM-watchdog abort. Usable as a decorator.
    """
    _terminate_callbacks.append(callback)
    return callback


def _safe_print(msg: str):
    """print that cannot kill the guardrail: on a full disk, writing to a
    redirected stdout raises OSError, which must not end the watchdog
    thread or block the abort path."""
    try:
        print(msg, flush=True)
    except OSError:
        pass


def _run_terminate_callbacks(reason: str):
    for cb in list(_terminate_callbacks):
        try:
            cb()
        except Exception as e:  # a failing callback must not block the rest
            _safe_print(f"[runtime] terminate callback failed ({reason}): {e}")


def _install_sigterm_handler():
    def _handler(signum, frame):
        if _terminate_callbacks:
            _safe_print(f"[runtime] SIGTERM: running "
                        f"{len(_terminate_callbacks)} flush callback(s)")
        _run_terminate_callbacks("SIGTERM")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)  # re-raise -> exit 143
    try:
        signal.signal(signal.SIGTERM, _handler)
    except ValueError:
        pass  # not the main thread; skip


def _check_disk(path=None, threshold_gb: float = None):
    """Warn when free disk is below the threshold. Returns True (enough),
    False (low, warned), or None (could not determine)."""
    import shutil
    if threshold_gb is None:
        threshold_gb = float(os.environ.get("EXPERIMENT_MIN_DISK_GB",
                                            LOW_DISK_GB))
    try:
        free_gb = shutil.disk_usage(
            path if path is not None else __file__).free / 1e9
    except OSError:
        return None
    if free_gb < threshold_gb:
        _safe_print(f"[runtime] WARNING: only {free_gb:.1f} GB free disk "
                    f"(threshold {threshold_gb:.0f} GB) - long runs will "
                    f"halt at the next checkpoint write; free space before "
                    f"starting expensive work")
        return False
    return True


def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        return 32.0


def soft_init():
    """Minimal guardrails, auto-applied on `import lib`.

    RAM watchdog at max(DEFAULT_MAX_RAM_GB, 25% of system RAM) --- or
    EXPERIMENT_MAX_RAM_GB when set --- plus a SIGTERM grace handler.
    Silent, idempotent, and a no-op after init_runtime() or when
    EXPERIMENT_AUTO_INIT=0.
    """
    if _state["initialized"] is not None:
        return
    if os.environ.get("EXPERIMENT_AUTO_INIT", "1") == "0":
        return
    env_gb = os.environ.get("EXPERIMENT_MAX_RAM_GB")
    max_gb = (float(env_gb) if env_gb else
              max(DEFAULT_MAX_RAM_GB, SOFT_RAM_FRACTION * total_ram_gb()))
    _limit_ram(max_gb)
    _install_sigterm_handler()
    _check_disk()
    _state["initialized"] = "soft"


def init_runtime(
    max_ram_gb: float = None,
    max_cores: int = None,
    nice: int = None,
    silent: bool = False,
):
    """Initialize resource-limited runtime for experiments.

    Args:
        max_ram_gb: Max RAM in GB. Process gets SIGKILL if exceeded.
        max_cores: Max CPU cores for numpy/torch/sklearn parallelism.
        nice: Process niceness (0-19). Higher = lower priority.
        silent: Suppress info messages.
    """
    max_ram_gb = max_ram_gb or float(os.environ.get("EXPERIMENT_MAX_RAM_GB", DEFAULT_MAX_RAM_GB))
    max_cores = max_cores or int(os.environ.get("EXPERIMENT_MAX_CORES", DEFAULT_MAX_CORES))
    nice = nice if nice is not None else int(os.environ.get("EXPERIMENT_NICE", DEFAULT_NICE))

    _limit_ram(max_ram_gb)
    _limit_cores(max_cores)
    _set_priority(nice)
    _install_sigterm_handler()
    _check_disk()
    _state["initialized"] = "full"

    if not silent:
        total_cores = multiprocessing.cpu_count()
        print(f"[runtime] RAM limit: {max_ram_gb:.1f} GB | "
              f"Cores: {max_cores}/{total_cores} | "
              f"Nice: {nice}")


def _limit_ram(max_gb: float):
    """Limit resident memory usage.

    RLIMIT_AS (virtual memory) is too aggressive --- shared library mappings
    count toward it, so scipy/sklearn fail to load even at 4 GB. Instead we
    use RLIMIT_DATA which limits heap growth without blocking mmap, and spawn
    a lightweight watchdog thread that checks RSS periodically.

    The ceiling lives in _state and the watchdog re-reads it each cycle, so
    a later init_runtime() call retargets the existing thread instead of
    stacking a second one.
    """
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    _state["max_bytes"] = max_bytes
    _state["max_gb"] = max_gb

    # RLIMIT_DATA limits the data segment (heap), not mmap regions.
    # This lets shared libraries load while still capping allocations.
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))
    except (ValueError, OSError):
        pass

    if _state["watchdog_started"]:
        return

    # Watchdog thread: check RSS every 10s, warn at 80%, abort at 100%.
    import threading

    def _watchdog():
        import time
        warned = False
        while True:
            time.sleep(10)
            limit = _state["max_bytes"]
            try:
                with open("/proc/self/statm") as f:
                    pages = int(f.read().split()[1])  # RSS in pages
            except (FileNotFoundError, ValueError):
                return  # Not on Linux, stop watchdog
            except Exception:
                continue  # transient read failure must not end monitoring
            rss = pages * resource.getpagesize()
            if rss > limit:
                _safe_print(f"\n[runtime] RSS {rss / 1e9:.1f} GB exceeds "
                            f"limit {_state['max_gb']:.1f} GB --- aborting "
                            f"to prevent OOM")
                _run_terminate_callbacks("RAM watchdog")
                os._exit(137)
            elif rss > limit * 0.8 and not warned:
                _safe_print(f"\n[runtime] WARNING: RSS {rss / 1e9:.1f} GB "
                            f"approaching limit {_state['max_gb']:.1f} GB")
                warned = True

    t = threading.Thread(target=_watchdog, daemon=True,
                         name="lib-runtime-watchdog")
    t.start()
    _state["watchdog_started"] = True


def _limit_cores(max_cores: int):
    """Restrict parallelism for numpy, torch, sklearn, and OpenMP.

    Uses both environment variable hints AND hard CPU affinity pinning
    so libraries cannot ignore the limit.
    """
    cores_str = str(max_cores)

    # Hard limit: pin process to specific CPU cores
    try:
        available = sorted(os.sched_getaffinity(0))
        pinned = set(available[:max_cores])
        os.sched_setaffinity(0, pinned)
    except (AttributeError, OSError):
        pass

    # Soft limits: environment variable hints for thread pools
    os.environ["OMP_NUM_THREADS"] = cores_str
    os.environ["MKL_NUM_THREADS"] = cores_str
    os.environ["OPENBLAS_NUM_THREADS"] = cores_str
    os.environ["NUMEXPR_NUM_THREADS"] = cores_str
    os.environ["TORCH_NUM_THREADS"] = cores_str
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if max_cores <= 2 else "true"

    # PyTorch runtime limit
    try:
        import torch
        torch.set_num_threads(max_cores)
        torch.set_num_interop_threads(max(1, max_cores // 2))
    except (ImportError, RuntimeError):
        pass


def _set_priority(nice_value: int):
    """Lower process priority so other tasks get CPU first."""
    nice_value = max(0, min(19, nice_value))
    try:
        os.nice(nice_value)
    except OSError:
        pass

    # Also try to set I/O scheduling class to idle
    try:
        import subprocess
        pid = os.getpid()
        subprocess.run(
            ["ionice", "-c", "3", "-p", str(pid)],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
