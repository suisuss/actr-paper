"""
Experiment framework: manifest-driven dispatch, discovery, and index generation.

Every experiment script defines a MANIFEST dict and a run() function:

    MANIFEST = {
        "id": "word_order",
        "name": "Word Order Discrimination",
        "category": "signal",
        "description": "Tests f1 coefficient word-order sensitivity",
        "modes": {"default": "Per-pair cosine similarity"},
        "configs": ["minilm_g0.g1"],   # or "auto" or None
        "datasets": ["paws"],           # or None
        "params": {},
        "tags": ["paper1"],
    }

    def run(mode, config, dataset, **params):
        return {"auc": 0.85}

    if __name__ == "__main__":
        from lib.experiment import main
        main(run)

CLI dispatch:
    python -m lib.cli run word_order --mode default --config minilm_g0.g1
    python -m lib.cli run paper2_suite --mode tool_selection --all
    python -m lib.cli experiments
    python -m lib.cli index
"""

import ast
import sys
import time
import inspect
import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST-based manifest extraction (no imports, safe for discovery)
# ---------------------------------------------------------------------------

def _extract_manifest_ast(script_path: Path) -> Optional[Dict]:
    """Extract MANIFEST dict from a script using AST parsing.

    Safe: does not import the module or execute any code.
    Only works if MANIFEST is a literal dict (strings, lists, dicts, bools, ints, None).
    """
    try:
        source = script_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return None

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MANIFEST":
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        return None
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _experiments_root() -> Path:
    return Path(__file__).parent.parent / "experiments"


def discover_experiments() -> Dict[str, Dict]:
    """Walk experiments/*/*.py and collect MANIFEST dicts via AST parsing.

    Returns dict mapping experiment_id -> {manifest, script_path, category}.
    """
    root = _experiments_root()
    results = {}

    if not root.exists():
        return results

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('_'):
            continue
        for script in sorted(category_dir.glob("*.py")):
            if script.name == "__init__.py":
                continue
            manifest = _extract_manifest_ast(script)
            if manifest and "id" in manifest:
                exp_id = manifest["id"]
                results[exp_id] = {
                    "manifest": manifest,
                    "script_path": str(script),
                    "script_rel": str(script.relative_to(root.parent)),
                    "category": category_dir.name,
                }

    # Also check root-level scripts
    for script in sorted(root.glob("*.py")):
        if script.name == "__init__.py":
            continue
        manifest = _extract_manifest_ast(script)
        if manifest and "id" in manifest:
            results[manifest["id"]] = {
                "manifest": manifest,
                "script_path": str(script),
                "script_rel": str(script.relative_to(root.parent)),
                "category": manifest.get("category", "root"),
            }

    return results


def get_experiment(experiment_id: str) -> Dict:
    """Look up a single experiment by manifest ID. Raises KeyError if not found."""
    experiments = discover_experiments()
    if experiment_id not in experiments:
        available = sorted(experiments.keys())
        raise KeyError(
            f"Experiment '{experiment_id}' not found. "
            f"Available: {', '.join(available)}"
        )
    return experiments[experiment_id]


# ---------------------------------------------------------------------------
# main() - script-level CLI dispatcher
# ---------------------------------------------------------------------------

def main(run_fn: Callable, manifest: Optional[Dict] = None):
    """Standard entry point for experiment scripts.

    Reads MANIFEST from the calling module if not provided.
    Builds argparse, validates inputs, calls run_fn, and auto-saves results.
    """
    # Ensure src/ is on sys.path for imports
    work_dir = str(Path(__file__).parent.parent)
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)

    # Get manifest from caller module if not provided
    if manifest is None:
        frame = inspect.stack()[1]
        caller_globals = frame[0].f_globals
        manifest = caller_globals.get("MANIFEST")
        if manifest is None:
            print("ERROR: No MANIFEST dict found in calling module")
            sys.exit(1)

    # Init runtime
    from .runtime import init_runtime
    init_runtime()

    parser = _build_parser(manifest)
    args = parser.parse_args()

    # Resolve what to run
    modes = [args.mode] if args.mode else (
        list(manifest["modes"].keys()) if args.run_all else
        [next(iter(manifest["modes"]))]  # first mode as default
    )

    configs = _resolve_run_configs(manifest, args)
    datasets = _resolve_run_datasets(manifest, args)

    # Collect custom params
    params = {}
    for param_name, param_spec in manifest.get("params", {}).items():
        if hasattr(args, param_name):
            params[param_name] = getattr(args, param_name)

    # Run
    save_mode = manifest.get("save_mode", "auto")
    all_results = {}

    for mode in modes:
        for config in configs:
            for dataset in datasets:
                label = _run_label(manifest["id"], mode, config, dataset)
                print(f"\n{'='*60}")
                print(f"RUN: {label}")
                print(f"{'='*60}")

                start = time.time()
                try:
                    result = run_fn(mode=mode, config=config, dataset=dataset, **params)
                except Exception:
                    logger.exception(f"Failed: {label}")
                    continue
                duration = time.time() - start

                if result is not None and save_mode == "auto":
                    _auto_save(manifest, mode, config, dataset, params, result, duration)

                all_results[label] = result
                print(f"\nCompleted: {label} ({duration:.1f}s)")

    return all_results


def _build_parser(manifest: Dict) -> argparse.ArgumentParser:
    """Build CLI parser from manifest."""
    parser = argparse.ArgumentParser(
        description=f"{manifest['name']}: {manifest.get('description', '')}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode_names = list(manifest.get("modes", {"default": ""}).keys())
    if len(mode_names) > 1:
        parser.add_argument(
            "--mode", "-m",
            choices=mode_names,
            default=None,
            help=f"Experiment mode (default: {mode_names[0]})",
        )
    else:
        parser.add_argument("--mode", "-m", default=mode_names[0], help=argparse.SUPPRESS)

    # Config selection
    configs_spec = manifest.get("configs")
    if configs_spec is not None and configs_spec != "auto":
        parser.add_argument(
            "--config", "-c",
            choices=configs_spec,
            default=None,
            help="Encoding config",
        )
    elif configs_spec == "auto":
        parser.add_argument("--config", "-c", default=None, help="Encoding config (auto-discovered)")
    else:
        parser.add_argument("--config", "-c", default=None, help=argparse.SUPPRESS)

    # Dataset selection
    datasets_spec = manifest.get("datasets")
    if datasets_spec is not None:
        parser.add_argument(
            "--dataset", "-d",
            choices=datasets_spec,
            default=None,
            help="Dataset",
        )
    else:
        parser.add_argument("--dataset", "-d", default=None, help=argparse.SUPPRESS)

    # Run all combinations
    parser.add_argument(
        "--all", dest="run_all",
        action="store_true",
        help="Run all modes/configs/datasets",
    )

    # Custom params from manifest
    for param_name, param_spec in manifest.get("params", {}).items():
        kwargs = {"help": param_spec.get("help", "")}

        ptype = param_spec.get("type", "str")
        if ptype == "bool":
            kwargs["action"] = "store_true"
            kwargs["default"] = param_spec.get("default", False)
        else:
            type_map = {"int": int, "float": float, "str": str}
            kwargs["type"] = type_map.get(ptype, str)
            kwargs["default"] = param_spec.get("default")
            if "choices" in param_spec:
                kwargs["choices"] = param_spec["choices"]

        flag = f"--{param_name.replace('_', '-')}"
        parser.add_argument(flag, dest=param_name, **kwargs)

    return parser


def _resolve_run_configs(manifest: Dict, args) -> List[Optional[str]]:
    """Resolve configs to iterate over."""
    configs_spec = manifest.get("configs")

    if args.run_all and configs_spec == "auto":
        return [None]  # auto-discover scripts handle their own iteration
    if args.run_all and isinstance(configs_spec, list):
        return configs_spec
    if hasattr(args, "config") and args.config:
        return [args.config]
    if isinstance(configs_spec, list) and len(configs_spec) == 1:
        return configs_spec
    if configs_spec is None or configs_spec == "auto":
        return [None]
    if isinstance(configs_spec, list):
        return configs_spec  # run all configs by default
    return [None]


def _resolve_run_datasets(manifest: Dict, args) -> List[Optional[str]]:
    """Resolve datasets to iterate over."""
    datasets_spec = manifest.get("datasets")

    if hasattr(args, "dataset") and args.dataset:
        return [args.dataset]
    if isinstance(datasets_spec, list) and len(datasets_spec) == 1:
        return datasets_spec
    if datasets_spec is None:
        return [None]
    if args.run_all and isinstance(datasets_spec, list):
        return datasets_spec
    if isinstance(datasets_spec, list):
        return datasets_spec  # run all datasets by default
    return [None]


def _run_label(exp_id: str, mode: str, config: Optional[str], dataset: Optional[str]) -> str:
    """Human-readable label for a run."""
    parts = [exp_id]
    if mode != "default":
        parts.append(mode)
    if config:
        parts.append(config)
    if dataset:
        parts.append(dataset)
    return " | ".join(parts)


def _auto_save(manifest, mode, config, dataset, params, result, duration):
    """Save result via the results registry."""
    from .results import save_result

    exp_name = manifest["id"]
    if mode != "default":
        exp_name = f"{exp_name}/{mode}"

    key = save_result(
        experiment=exp_name,
        config=config or "none",
        dataset=dataset or "none",
        variant="default",
        params=params,
        results=result,
        tags=manifest.get("tags", []),
        duration=duration,
    )
    print(f"Saved: {key}")


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "foundation", "signal", "basis", "analysis",
    "tool_prediction", "prediction", "trajectory",
    "oos_detection", "circular", "paper1", "paper2", "utility",
]

CATEGORY_LABELS = {
    "foundation": "Foundation",
    "signal": "Signal Analysis",
    "basis": "Basis Comparison",
    "analysis": "Analysis",
    "tool_prediction": "Tool Prediction",
    "prediction": "Prediction",
    "trajectory": "Trajectory Analysis",
    "oos_detection": "OOS Detection",
    "circular": "Circular Polar Encoding",
    "paper1": "Paper 1 Validation",
    "paper2": "Paper 2 Suite",
    "utility": "Utility",
}


def generate_index(output_path: Optional[Path] = None) -> str:
    """Auto-generate INDEX.md from discovered experiment manifests."""
    experiments = discover_experiments()

    by_category = {}
    for exp_id, exp_info in experiments.items():
        cat = exp_info["category"]
        by_category.setdefault(cat, []).append(exp_info)

    lines = [
        "# Experiment Index",
        "",
        "Auto-generated from experiment MANIFEST dicts. Do not edit manually.",
        f"Run `python -m lib.cli index` to regenerate.",
        "",
    ]

    # Ordered categories first, then any remaining
    ordered = [c for c in CATEGORY_ORDER if c in by_category]
    remaining = sorted(set(by_category.keys()) - set(CATEGORY_ORDER))
    all_categories = ordered + remaining

    for cat in all_categories:
        exps = sorted(by_category[cat], key=lambda e: e["script_rel"])
        label = CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())

        lines.append(f"## {label}")
        lines.append("")
        lines.append("| ID | Name | Script | Modes | Datasets | Configs | Tags |")
        lines.append("|----|------|--------|-------|----------|---------|------|")

        for exp in exps:
            m = exp["manifest"]
            script = exp["script_rel"]
            n_modes = len(m.get("modes", {}))

            datasets_spec = m.get("datasets")
            if datasets_spec is None:
                ds_str = "self-loaded"
            else:
                ds_str = ", ".join(datasets_spec)

            configs_spec = m.get("configs")
            if configs_spec is None:
                cfg_str = "none"
            elif configs_spec == "auto":
                cfg_str = "auto"
            else:
                cfg_str = f"{len(configs_spec)}"

            tags = ", ".join(m.get("tags", []))
            doc = m.get("doc_ref", "")
            name = m["name"]
            if doc:
                name = f"[{name}]({doc})"

            lines.append(
                f"| `{m['id']}` | {name} | `{script}` "
                f"| {n_modes} | {ds_str} | {cfg_str} | {tags} |"
            )

        lines.append("")

    # Mode details for multi-mode experiments
    multi_mode = [
        (eid, einfo) for eid, einfo in experiments.items()
        if len(einfo["manifest"].get("modes", {})) > 1
    ]
    if multi_mode:
        lines.append("## Mode Details")
        lines.append("")
        for exp_id, exp_info in sorted(multi_mode):
            m = exp_info["manifest"]
            lines.append(f"### {m['name']} (`{exp_id}`)")
            lines.append("")
            lines.append("| Mode | Description |")
            lines.append("|------|-------------|")
            for mode_name, mode_desc in m["modes"].items():
                lines.append(f"| `{mode_name}` | {mode_desc} |")
            lines.append("")

    md = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(md)
        print(f"Index written to {output_path}")

    return md
