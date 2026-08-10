"""
Results registry for tracking and loading experiment results.

Mirrors the encoding registry pattern: JSON index + immutable result files.
Results are never overwritten — each run creates a timestamped artifact.

Usage in experiment scripts:

    # Simple single-step experiment
    from lib.results import experiment

    with experiment("word_order", "minilm_g0.g1", "paws", "default",
                    params={"n_folds": 5}, tags=["paper1"]) as results:
        results["g0"] = {"auc": 0.789, "std": 0.019, "folds": [...]}
    # auto-saved, auto-registered

    # Multi-step experiment with checkpointing
    from lib.results import ExperimentCheckpoint

    ckpt = ExperimentCheckpoint("paper2_suite", "minilm_g0.g1", "sgd", "pre_tool",
                                params={"n_folds": 5}, tags=["paper2"])

    if not ckpt.is_done("exp1_tool_selection"):
        result = run_tool_selection(...)
        ckpt.save_step("exp1_tool_selection", result)

    key = ckpt.finalize()

Query results:

    from lib.results import query_results, get_latest, load_result

    query_results(experiment="word_order")
    get_latest("word_order", config="minilm_g0.g1")
    load_result("word_order:minilm_g0.g1:paws:default:20260326T143022")
"""

import json
import hashlib
import inspect
import os
import time
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4

import numpy as np

from . import provenance


def get_results_root() -> Path:
    override = os.environ.get("EXPERIMENT_RESULTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent.parent.parent / "data" / "results"


def get_results_registry_path() -> Path:
    return get_results_root() / "results_registry.json"


def _json_convert(obj):
    """Convert numpy and other non-serializable types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _caller_script() -> str:
    """Get the calling script's path relative to src/."""
    work_root = Path(__file__).parent.parent
    for frame_info in inspect.stack():
        frame_path = Path(frame_info.filename)
        if frame_path.suffix == '.py' and frame_path != Path(__file__):
            try:
                return str(frame_path.relative_to(work_root))
            except ValueError:
                return str(frame_path)
    return "unknown"


class ResultsRegistry:
    """
    Central registry for experiment results. Mirrors EncodingRegistry.

    Results are immutable — each run creates a timestamped file.
    The registry JSON is the queryable index.

    Directory layout:
        data/results/{experiment}/{dataset}/{config}/{timestamp}.json
    """

    def __init__(self):
        self.root = get_results_root()
        self.registry_path = get_results_registry_path()
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "results": {}
        }

    def _save_registry(self):
        self._registry["updated"] = datetime.now().isoformat()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        tmp = self.registry_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self._registry, f, indent=2, default=_json_convert)
        tmp.rename(self.registry_path)

    def _make_key(self, experiment: str, config: str, dataset: str,
                  variant: str, timestamp: str) -> str:
        return f"{experiment}:{config}:{dataset}:{variant}:{timestamp}"

    def _parse_key(self, key: str) -> Tuple[str, str, str, str, str]:
        parts = key.split(':')
        if len(parts) != 5:
            raise ValueError(f"Invalid result key format: {key}")
        return tuple(parts)

    def _compute_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            h.update(f.read(1024 * 1024))
        return f"sha256:{h.hexdigest()[:16]}"

    def _summarize_results(self, results: Dict) -> Dict:
        """Extract summary metrics from results for the registry entry."""
        summary = {}
        best_metric = None
        best_model = None

        for model_name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
            for metric_name in ('auc', 'accuracy', 'f1', 'auc_roc', 'macro_auc'):
                if metric_name in model_results:
                    val = model_results[metric_name]
                    if isinstance(val, (int, float)):
                        summary[f"{model_name}_{metric_name}"] = round(val, 4)
                        if best_metric is None or val > best_metric:
                            best_metric = val
                            best_model = model_name

        if best_model:
            summary["best_model"] = best_model
            summary["best_metric"] = best_metric

        return summary

    def save(
        self,
        experiment: str,
        config: str,
        dataset: str,
        variant: str,
        params: Dict,
        results: Dict,
        tags: Optional[List[str]] = None,
        script: Optional[str] = None,
        duration: Optional[float] = None,
        n_samples: Optional[int] = None,
        filename_stem: Optional[str] = None,
    ) -> str:
        """
        Save results as an immutable artifact and register.

        Args:
            experiment: Experiment name (e.g., 'word_order_discrimination')
            config: Encoding config used (e.g., 'minilm_g0.g1')
            dataset: Dataset name (e.g., 'paws', 'sgd')
            variant: Dataset variant (e.g., 'default', 'pre_tool')
            params: Experiment parameters (n_folds, seed, classifier, etc.)
            results: The actual results dict
            tags: Optional tags for filtering (e.g., ['paper1', 'signal'])
            script: Script that produced this (auto-detected if None)
            duration: Runtime in seconds
            n_samples: Number of samples used
            filename_stem: Optional prefix for the run file, making it
                self-describing outside its directory:
                {filename_stem}_{timestamp}.json instead of
                {timestamp}.json. Campaign scripts pass their experiment
                leaf name (see scripts/campaign_io.py).

        Returns:
            Registry key
        """
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        script_name = script or _caller_script()
        script_abs = Path(script_name)
        if not script_abs.is_absolute():
            script_abs = Path(__file__).parent.parent / script_name

        # Build result file with provenance (incl. code state: git commit,
        # dirty flag, script hash, argv - see lib/provenance.py)
        result_file = {
            "_provenance": {
                "experiment": experiment,
                "config": config,
                "dataset": dataset,
                "variant": variant,
                "n_samples": n_samples,
                "params": params,
                "timestamp": timestamp,
                "script": script_name,
                "duration_seconds": duration,
                **provenance.collect(
                    script_path=script_abs if script_abs.exists() else None),
            },
            "results": results,
        }

        # Save immutable file: results/{experiment}/{dataset}/{config}/{timestamp}.json
        result_dir = self.root / experiment / dataset / config
        result_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{filename_stem}_{timestamp}" if filename_stem else timestamp
        path = result_dir / f"{stem}.json"

        # Never overwrite — add suffix for sub-second collisions
        if path.exists():
            path = result_dir / f"{stem}_{uuid4().hex[:6]}.json"

        # Atomic write
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(result_file, f, indent=2, default=_json_convert)
        tmp.rename(path)

        # Register
        key = self._make_key(experiment, config, dataset, variant, timestamp)
        entry = {
            "experiment": experiment,
            "config": config,
            "dataset": dataset,
            "variant": variant,
            "timestamp": timestamp,
            "path": str(path.relative_to(self.root)),
            "params": params,
            "summary": self._summarize_results(results),
            "tags": tags or [],
            "checksum": self._compute_checksum(path),
            "size_bytes": path.stat().st_size,
            "duration_seconds": duration,
            "script": script or _caller_script(),
        }
        if n_samples is not None:
            entry["n_samples"] = n_samples

        self._registry["results"][key] = entry
        self._save_registry()

        print(f"Result saved: {key}")
        print(f"  Path: {path}")
        return key

    def query(
        self,
        experiment: Optional[str] = None,
        config: Optional[str] = None,
        dataset: Optional[str] = None,
        variant: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Query results by any combination of fields."""
        matches = []
        for key, entry in self._registry["results"].items():
            if experiment and entry.get("experiment") != experiment:
                continue
            if config and entry.get("config") != config:
                continue
            if dataset and entry.get("dataset") != dataset:
                continue
            if variant and entry.get("variant") != variant:
                continue
            if tags:
                entry_tags = set(entry.get("tags", []))
                if not set(tags).issubset(entry_tags):
                    continue
            entry_with_key = dict(entry)
            entry_with_key["_key"] = key
            matches.append(entry_with_key)

        # Sort by timestamp descending (most recent first)
        matches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return matches

    def get_latest(
        self,
        experiment: str,
        config: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> Optional[Dict]:
        """Get the most recent result for an experiment."""
        matches = self.query(experiment=experiment, config=config, dataset=dataset)
        return matches[0] if matches else None

    def load(self, key: str) -> Dict:
        """Load full result file by registry key."""
        entry = self._registry["results"].get(key)
        if not entry:
            raise KeyError(f"Result not found: {key}")

        path = self.root / entry["path"]
        if not path.exists():
            raise FileNotFoundError(f"Result file missing: {path}")

        with open(path) as f:
            return json.load(f)

    def load_latest(
        self,
        experiment: str,
        config: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> Optional[Dict]:
        """Load the most recent result file for an experiment."""
        latest = self.get_latest(experiment, config, dataset)
        if latest:
            return self.load(latest["_key"])
        return None

    def list_experiments(self) -> List[str]:
        """List all unique experiment names."""
        experiments = set()
        for entry in self._registry["results"].values():
            experiments.add(entry["experiment"])
        return sorted(experiments)

    def list_all(self) -> List[Dict]:
        """List all registered results."""
        return list(self._registry["results"].values())

    def scan(self) -> int:
        """
        Scan filesystem for unregistered result files.
        Looks for JSON files matching the directory structure:
            results/{experiment}/{dataset}/{config}/{timestamp}.json

        Returns number of newly discovered results.
        """
        discovered = 0
        existing_paths = set()
        for entry in self._registry["results"].values():
            existing_paths.add(entry.get("path"))

        for json_path in self.root.rglob("*.json"):
            # Skip registry file, checkpoints, and tmp files
            if json_path.name in ("results_registry.json",):
                continue
            if json_path.name.startswith(".") or json_path.suffix == '.tmp':
                continue

            try:
                rel_path = json_path.relative_to(self.root)
                parts = rel_path.parts

                # Must be: experiment/dataset/config/timestamp.json
                if len(parts) != 4:
                    # Try legacy format: experiment/timestamp.json (2 parts)
                    # Skip — we only index the new structure
                    continue

                rel_str = str(rel_path)
                if rel_str in existing_paths:
                    continue

                experiment = parts[0]
                dataset = parts[1]
                config = parts[2]
                timestamp = parts[3].replace('.json', '')

                # Try to load and extract metadata
                with open(json_path) as f:
                    data = json.load(f)

                provenance = data.get("_provenance", {})
                results = data.get("results", data)

                key = self._make_key(experiment, config, dataset,
                                     provenance.get("variant", "default"), timestamp)

                if key not in self._registry["results"]:
                    entry = {
                        "experiment": experiment,
                        "config": config,
                        "dataset": dataset,
                        "variant": provenance.get("variant", "default"),
                        "timestamp": timestamp,
                        "path": rel_str,
                        "params": provenance.get("params", {}),
                        "summary": self._summarize_results(results),
                        "tags": [],
                        "checksum": self._compute_checksum(json_path),
                        "size_bytes": json_path.stat().st_size,
                        "duration_seconds": provenance.get("duration_seconds"),
                        "script": provenance.get("script", "unknown"),
                    }
                    self._registry["results"][key] = entry
                    discovered += 1
                    print(f"  Discovered: {experiment}/{dataset}/{config}/{timestamp}")

            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        if discovered > 0:
            self._save_registry()

        return discovered

    def print_status(self):
        """Print registry status."""
        print(f"\n{'='*60}")
        print("RESULTS REGISTRY STATUS")
        print(f"{'='*60}")
        print(f"Registry: {self.registry_path}")
        print(f"Updated: {self._registry.get('updated', 'N/A')}")
        print(f"Total results: {len(self._registry['results'])}")

        if not self._registry['results']:
            print("\n  (no results registered)")
            return

        # Group by experiment
        by_experiment = {}
        for key, entry in self._registry['results'].items():
            exp = entry['experiment']
            if exp not in by_experiment:
                by_experiment[exp] = []
            by_experiment[exp].append(entry)

        for exp_name in sorted(by_experiment.keys()):
            entries = by_experiment[exp_name]
            print(f"\n  {exp_name}/ ({len(entries)} runs)")
            for entry in sorted(entries, key=lambda x: x.get('timestamp', ''), reverse=True):
                ts = entry.get('timestamp', '?')
                dataset = entry.get('dataset', '?')
                config = entry.get('config', '?')
                summary = entry.get('summary', {})
                duration = entry.get('duration_seconds')

                # Format summary metrics
                metric_str = ""
                if 'best_model' in summary:
                    metric_str = f" best={summary['best_model']}({summary.get('best_metric', '?'):.4f})"
                elif summary:
                    first_key = next(iter(summary))
                    metric_str = f" {first_key}={summary[first_key]}"

                dur_str = f" ({duration:.0f}s)" if duration else ""
                tags_str = f" [{','.join(entry.get('tags', []))}]" if entry.get('tags') else ""

                print(f"    {ts}  {dataset}/{config}{metric_str}{dur_str}{tags_str}")


class ExperimentCheckpoint:
    """
    Checkpoint for multi-step experiments.

    Saves progress after each step so experiments can resume if interrupted.
    On finalize(), saves the complete result as an immutable artifact and
    deletes the checkpoint.

    Usage:
        ckpt = ExperimentCheckpoint("paper2_suite", "minilm_g0.g1", "sgd", "pre_tool",
                                    params={"n_folds": 5}, tags=["paper2"])

        if not ckpt.is_done("exp1_tool_selection"):
            result = run_tool_selection(...)
            ckpt.save_step("exp1_tool_selection", result)

        if not ckpt.is_done("exp2_domain"):
            result = run_domain_classification(...)
            ckpt.save_step("exp2_domain", result)

        key = ckpt.finalize()
    """

    def __init__(
        self,
        experiment: str,
        config: str,
        dataset: str,
        variant: str = "default",
        params: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        self.experiment = experiment
        self.config = config
        self.dataset = dataset
        self.variant = variant
        self.params = params or {}
        self.tags = tags or []
        self._start_time = time.time()

        self.root = get_results_root()
        self.checkpoint_dir = self.root / experiment / dataset / config
        self.checkpoint_path = self.checkpoint_dir / ".checkpoint.json"

        self._completed: Dict[str, Any] = {}
        self._step_times: Dict[str, float] = {}
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load existing checkpoint if present."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            self._completed = data.get("completed", {})
            self._step_times = data.get("step_times", {})
            n = len(self._completed)
            if n > 0:
                print(f"Resumed checkpoint: {n} steps completed "
                      f"({', '.join(self._completed.keys())})")

    def _save_checkpoint(self):
        """Save checkpoint (atomic write)."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "experiment": self.experiment,
            "config": self.config,
            "dataset": self.dataset,
            "variant": self.variant,
            "params": self.params,
            "tags": self.tags,
            "completed": self._completed,
            "step_times": self._step_times,
            "updated": datetime.now().isoformat(),
        }
        tmp = self.checkpoint_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=_json_convert)
        tmp.rename(self.checkpoint_path)

    def is_done(self, step_name: str) -> bool:
        """Check if a step has already been completed."""
        return step_name in self._completed

    def save_step(self, step_name: str, step_results: Dict):
        """Save one step's results and checkpoint."""
        self._completed[step_name] = step_results
        self._step_times[step_name] = time.time() - self._start_time
        self._save_checkpoint()
        print(f"  Checkpoint: {step_name} done ({len(self._completed)} steps total)")

    def get_step(self, step_name: str) -> Optional[Dict]:
        """Get a completed step's results."""
        return self._completed.get(step_name)

    def finalize(self) -> str:
        """
        All steps done. Save final result, register, delete checkpoint.

        Returns registry key.
        """
        duration = time.time() - self._start_time
        registry = _get_results_registry()
        key = registry.save(
            experiment=self.experiment,
            config=self.config,
            dataset=self.dataset,
            variant=self.variant,
            params=self.params,
            results=self._completed,
            tags=self.tags,
            duration=duration,
        )

        # Delete checkpoint
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

        return key


# --- Singleton ---

_results_registry_instance = None


def _get_results_registry() -> ResultsRegistry:
    global _results_registry_instance
    if _results_registry_instance is None:
        _results_registry_instance = ResultsRegistry()
    return _results_registry_instance


# --- Context manager ---

@contextmanager
def experiment(
    name: str,
    config: str,
    dataset: str,
    variant: str = "default",
    params: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
):
    """
    Context manager that auto-saves experiment results.

    Usage:
        with experiment("word_order", "minilm_g0.g1", "paws", "default",
                        params={"n_folds": 5}, tags=["paper1"]) as results:
            results["g0"] = {"auc": 0.789, "std": 0.019}
            results["g0_g1"] = {"auc": 0.812, "std": 0.015}
        # auto-saved on context exit
    """
    result_container = {}
    start = time.time()

    yield result_container

    duration = time.time() - start
    registry = _get_results_registry()
    registry.save(
        experiment=name,
        config=config,
        dataset=dataset,
        variant=variant,
        params=params or {},
        results=result_container,
        tags=tags,
        duration=duration,
        n_samples=n_samples,
    )


# --- Convenience functions ---

def save_result(
    experiment: str,
    config: str,
    dataset: str,
    variant: str,
    params: Dict,
    results: Dict,
    tags: Optional[List[str]] = None,
    duration: Optional[float] = None,
    n_samples: Optional[int] = None,
) -> str:
    """Save a result directly (without context manager)."""
    registry = _get_results_registry()
    return registry.save(
        experiment=experiment,
        config=config,
        dataset=dataset,
        variant=variant,
        params=params,
        results=results,
        tags=tags,
        duration=duration,
        n_samples=n_samples,
    )


def query_results(
    experiment: Optional[str] = None,
    config: Optional[str] = None,
    dataset: Optional[str] = None,
    variant: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict]:
    """Query the results registry."""
    return _get_results_registry().query(
        experiment=experiment, config=config,
        dataset=dataset, variant=variant, tags=tags,
    )


def get_latest(
    experiment: str,
    config: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Optional[Dict]:
    """Get the most recent result for an experiment."""
    return _get_results_registry().get_latest(experiment, config, dataset)


def load_result(key: str) -> Dict:
    """Load a full result file by registry key."""
    return _get_results_registry().load(key)


def list_experiments() -> List[str]:
    """List all unique experiment names."""
    return _get_results_registry().list_experiments()


def results_status():
    """Print results registry status."""
    _get_results_registry().print_status()


def scan_results() -> int:
    """Scan for unregistered result files."""
    return _get_results_registry().scan()
