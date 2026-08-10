#!/usr/bin/env python3
"""One-command reproduction driver for the flagship paper.

Composes the paper's full evidence chain into a single resumable pipeline
and runs it: turn embeddings (skipped when the cached npz already exist),
the tracked stage specs under data/pipelines/, and the standalone
generators that reuse the caches those specs populate, ending with the
figures, the evidence tables, the draft-vs-artifact gate, and the PDF.

Resumable. Stage status is persisted by lib/pipeline.py to
.reproduce/main/pipeline.state.json after every transition; re-run this
driver and completed stages are skipped, an interrupted stage re-runs
(long scripts checkpoint internally under data/experiment-caches/). A
second concurrent supervisor on the same state is refused (a file lock).

    python main.py               run / resume
    python main.py launch        detached run (survives the shell)
    python main.py status        stage table
    python main.py reset         forget stage state
    python main.py --fetch-data  download the corpora first, then run

Raw corpora are public but never downloaded implicitly: pass --fetch-data
(PRISM needs a HuggingFace token in HF_TOKEN). The judge stages replay the
frozen committed ratings (data/results/trace_h2h/judge_ratings*.json) -
zero API calls.

Requirements: numpy/scipy/scikit-learn always; sentence-transformers +
torch only for stages that (re)build embeddings; pandoc/xelatex for the
final PDF. See README.md.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
DATA = REPO / "data"
PIPELINES = DATA / "pipelines"
EMBEDDINGS = DATA / "embeddings"
RAW = DATA / "raw"
RESULTS = DATA / "results" / "trace_h2h"
STATE_DIR = REPO / ".reproduce" / "main"
SPEC_PATH = STATE_DIR / "pipeline.json"
STATE_PATH = STATE_DIR / "pipeline.state.json"

PIPELINE_NAME = "flagship_reproduce"

# BLAS pinned to one thread so sklearn's joblib owns RF parallelism.
ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

# Dataset sources, pinned so two reproductions see identical data.
USS_CLONE = "https://github.com/sunnweiwei/user-satisfaction-simulation"
USS_COMMIT = "8ba06bd92570fd94d3ec0cb828aad45172be71b2"
PRISM_REPO = "HannahRoseKirk/prism-alignment"
PRISM_REVISION = "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6"
SGD_CLONE = ("https://github.com/google-research-datasets/"
             "dstc8-schema-guided-dialogue")
SGD_COMMIT = "e852981ae34990f4358979625854259302feaa78"

# (name, argv, deps, output files under data/embeddings/ whose joint
# existence marks the stage as already satisfied)
EMBED_STAGES = [
    ("embed_uss", ["embed_uss.py"], [],
     ["uss_sgd_minilm.npz", "uss_mwoz_minilm.npz",
      "uss_sgd_meta.json", "uss_mwoz_meta.json"]),
    ("embed_prism", ["embed_prism.py"], [],
     ["prism_minilm.npz", "prism_meta.json"]),
    ("embed_gte", ["embed_multimodel.py", "gte"], ["embed_uss", "embed_prism"],
     ["uss_sgd_gte.npz", "uss_mwoz_gte.npz", "prism_gte.npz"]),
    ("embed_bge", ["embed_multimodel.py", "bge"], ["embed_uss", "embed_prism"],
     ["uss_sgd_bge.npz", "uss_mwoz_bge.npz", "prism_bge.npz"]),
    ("build_sgd_map", ["build_sgd_map.py"], [],
     ["uss_sgd_original_map.json"]),
    ("embed_hierarchical", ["embed_hierarchical.py"],
     ["embed_uss", "embed_prism"],
     ["uss_sgd_hier.npz", "uss_mwoz_hier.npz", "prism_hier.npz"]),
]
ST_STAGES = {"embed_uss", "embed_prism", "embed_gte", "embed_bge",
             "embed_hierarchical"}

# Tracked-spec producers dropped when their derived caches already exist.
GUARDED_TRACKED = {
    "cross_encoder_encode": ["uss_sgd_xenc.npz", "uss_mwoz_xenc.npz",
                             "prism_xenc.npz"],
}

# Ordering edges beyond what the tracked specs declare.
EXTRA_DEPS = {
    "multimodel_eval": ["embed_gte", "embed_bge"],
    "rolesplit_embedders": ["embed_gte", "embed_bge"],
    "service_transfer": ["build_sgd_map"],
    "descriptive_profiles": ["build_sgd_map"],
    "hier_eval": ["embed_hierarchical"],
    "g0_combo_prism": ["build_prism_meta"],
    "basis_probes": ["multiseed_probes", "eqdim_probes"],
}

TRACKED_SPECS = ["paper_evidence", "framework_evidence", "eqdim_evidence",
                 "basis_evidence"]

# Standalone generators covered by no tracked spec; they reuse the
# multiseed / rolesplit_graft OOF caches the specs populate. The judge
# stages replay the frozen committed ratings - zero API calls.
STANDALONE_TAIL = [
    ("cross_encoder_eval", ["cross_encoder_eval.py", "eval"],
     ["cross_encoder_encode"]),
    ("ledger_stats", ["ledger_stats.py"], []),
    ("scoreboard_arms", ["scoreboard_arms.py"], ["build_prism_meta"]),
    ("annotator_ceiling", ["annotator_ceiling.py"], []),
    ("roleblind_probes", ["roleblind_probes.py"], []),
    ("capacity_uniqueness", ["capacity_uniqueness.py"], ["scoreboard_arms"]),
    ("uniqueness_mwoz", ["uniqueness_mwoz.py"], []),
    ("actr_figures", ["actr_figures.py"],
     ["scoreboard_arms", "ledger_stats", "rolesplit_graft",
      "rolesplit_embedders", "containment_graft_sgd", "containment_graft_mwoz"]),
    ("llm_judge_eval", ["llm_judge.py", "eval"], []),
    ("judge_stack", ["judge_stack.py"], ["annotator_ceiling"]),
    ("judge_template1", ["judge_template1.py", "eval"],
     ["annotator_ceiling", "judge_stack"]),
    ("actr_tables", ["actr_tables.py"],
     ["judge_stack", "judge_template1", "scoreboard_arms", "eqdim_probes",
      "basis_calibration", "roleblind_probes", "capacity_uniqueness",
      "interleaved_degree", "deconfound_length", "dims_control"]),
    # Draft-vs-artifact gate: fails the run if a regenerated table cell
    # diverges from paper.md. Runs after the tables are regenerated.
    ("verify_draft_tables", ["verify_draft_tables.py"], ["actr_tables"]),
    # Terminal stage: the preview PDF. Fails resumably with install
    # instructions when the pandoc/xelatex toolchain is absent.
    ("build_pdf", ["build_pdf.py"],
     ["actr_figures", "actr_tables", "verify_draft_tables"]),
]
FIGURE_STAGES = {"actr_figures"}
PDF_TOOLS = ("pandoc", "xelatex", "rsvg-convert")


def _load_module(name, path):
    """Load a module by file path, bypassing the eager lib/__init__ (keeps
    status/reset dependency-free: lib/pipeline.py is stdlib-only)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pinned_clone(url, dest, commit):
    """Shallow-fetch a pinned commit of a dataset repo. Falls back to a
    plain shallow clone and warns loudly if its HEAD drifts from the pin."""
    dest = Path(dest)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin",
                        url], check=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1",
                        "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q",
                        "FETCH_HEAD"], check=True)
        return
    except subprocess.CalledProcessError:
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[fetch] pinned fetch of {commit[:12]} failed; falling back "
              f"to a plain shallow clone of {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        print(f"[fetch] WARNING: {url} HEAD {head[:12]} differs from the "
              f"pinned commit {commit[:12]} - upstream moved; the data may "
              "not match the paper's exactly")


# ------------------------------------------------------------- spec assembly

def _add(stages, included, name, argv, deps):
    stages.append({"name": name, "cmd": [sys.executable] + argv,
                   "depends_on": [d for d in deps if d in included]})
    included.add(name)


def _add_tracked(stages, included, emb, spec_name):
    spec = json.loads((PIPELINES / f"{spec_name}.json").read_text())
    for st in spec["stages"]:
        guard = GUARDED_TRACKED.get(st["name"])
        if guard and all((EMBEDDINGS / f).exists() for f in guard):
            continue
        argv = st["cmd"][1:] if st["cmd"][0] in ("python3", "python") \
            else st["cmd"]
        _add(stages, included, st["name"], argv,
             list(st.get("depends_on", [])) + emb
             + EXTRA_DEPS.get(st["name"], []))


def build_spec():
    stages, included = [], set()

    for name, argv, deps, outputs in EMBED_STAGES:
        if not all((EMBEDDINGS / f).exists() for f in outputs):
            _add(stages, included, name, argv, deps)

    # embed_prism writes prism_meta.json WITHOUT duration_per_turn;
    # build_prism_meta enriches it. Guard on CONTENT, not existence.
    meta_f = EMBEDDINGS / "prism_meta.json"
    needs_meta = True
    if "embed_prism" not in included and meta_f.exists():
        try:
            needs_meta = "duration_per_turn" not in json.loads(
                meta_f.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    if needs_meta:
        _add(stages, included, "build_prism_meta", ["build_prism_meta.py"],
             ["embed_prism"])

    emb = sorted(n for n in included if n in
                 ("embed_uss", "embed_prism", "embed_gte", "embed_bge"))

    for spec_name in TRACKED_SPECS:
        _add_tracked(stages, included, emb, spec_name)
    _add(stages, included, "basis_calibration", ["basis_calibration.py"],
         ["basis_probes"])
    _add_tracked(stages, included, emb, "regime_cells")

    for name, argv, deps in STANDALONE_TAIL:
        _add(stages, included, name, argv,
             deps + ["multiseed_probes", "rolesplit_graft"])

    # Descriptive probe (draft 3.2): registry-backs the PRISM goal-proxy
    # numbers from the raw corpus + encoded turns. Independent of the OOF
    # caches, so it is not part of the tail's dependency chain.
    _add(stages, included, "goal_proxy_prism", ["goal_proxy_prism.py"],
         ["embed_prism"])

    # Issue #97 model-identity confound controls (PRISM is the only
    # multi-model corpus; USS is single-system and cannot carry this
    # confound). A is self-contained; B re-scores the eqdim / rolesplit_graft
    # OOF caches under a same-model pair filter (no retraining); C1 retrains
    # the identity-augmented arms and replays the plain ones from the eqdim
    # cache, keeping its baseline bit-identical to Table 2.
    _add(stages, included, "identity_probes_prism",
         ["identity_probes_prism.py"], ["build_prism_meta"])
    _add(stages, included, "identity_controls_prism",
         ["identity_controls_prism.py"],
         ["build_prism_meta", "eqdim_probes", "rolesplit_graft"])
    _add(stages, included, "identity_augment_prism",
         ["identity_augment_prism.py"], ["build_prism_meta", "eqdim_probes"])

    return {"name": PIPELINE_NAME, "workdir": str(SRC),
            "log_dir": str(STATE_DIR / "logs"), "env": ENV, "stages": stages}


# ------------------------------------------------------------------ preflight

def _pending(names):
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    return [n for n in names if state.get(n, {}).get("status") != "done"]


def preflight(fetch, stages):
    errors = []

    uss = RAW / "uss_repo" / "dataset"
    if not ((uss / "SGD.txt").exists() and (uss / "MWOZ.txt").exists()):
        if fetch:
            pinned_clone(USS_CLONE, RAW / "uss_repo", USS_COMMIT)
        else:
            errors.append(f"USS corpus missing at {uss} - re-run with "
                          f"--fetch-data (pinned commit {USS_COMMIT[:12]})")

    prism = RAW / "prism" / "conversations.jsonl"
    if not prism.exists():
        if fetch:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                errors.append("--fetch-data for PRISM needs huggingface_hub "
                              "(pip install huggingface_hub) and HF_TOKEN")
            else:
                p = hf_hub_download(repo_id=PRISM_REPO, repo_type="dataset",
                                    filename="conversations.jsonl",
                                    revision=PRISM_REVISION)
                prism.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, prism)
        else:
            errors.append(f"PRISM corpus missing at {prism} - re-run with "
                          "--fetch-data (gated dataset; needs HF_TOKEN)")

    sgd_orig = RAW / "sgd_original"
    if not (EMBEDDINGS / "uss_sgd_original_map.json").exists() \
            and not (sgd_orig / "train").exists():
        if fetch:
            pinned_clone(SGD_CLONE, sgd_orig, SGD_COMMIT)
        else:
            errors.append(f"original SGD corpus missing at {sgd_orig} (needed "
                          "once by build_sgd_map; ~600MB) - re-run with "
                          f"--fetch-data (pinned commit {SGD_COMMIT[:12]})")

    for f in ("judge_ratings.json", "judge_ratings_t1.json"):
        if not (RESULTS / f).exists():
            errors.append(f"frozen judge ratings missing: {RESULTS / f} "
                          "(committed raw data - restore from the repo)")

    for mod in ("numpy", "scipy", "sklearn"):
        if importlib.util.find_spec(mod) is None:
            errors.append(f"missing package '{mod}' "
                          "(pip install -r requirements.txt)")

    if importlib.util.find_spec("sentence_transformers") is None:
        st_pending = _pending([s["name"] for s in stages
                               if s["name"] in ST_STAGES])
        if st_pending:
            errors.append("sentence-transformers is required by pending "
                          f"embedding stages {st_pending} "
                          "(pip install -r requirements.txt)")

    if importlib.util.find_spec("matplotlib") is None \
            and _pending([s["name"] for s in stages
                          if s["name"] in FIGURE_STAGES]):
        errors.append("matplotlib is required by the figure stage "
                      "(pip install -r requirements.txt)")

    if _pending(["build_pdf"]):
        missing = [t for t in PDF_TOOLS if shutil.which(t) is None]
        if missing:
            print("[preflight] warning: PDF tools missing "
                  f"({', '.join(missing)}) - every artifact, figure, and "
                  "table still builds; the terminal build_pdf stage will "
                  "fail with install instructions until they are present "
                  "(run ./install_pdf_tools.sh)")

    return errors


# ------------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reproduce every cited artifact of the flagship paper "
                    "(resumable; see README.md)")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "launch", "status", "reset"])
    ap.add_argument("--fetch-data", action="store_true",
                    help="download USS / PRISM / SGD (needs HF_TOKEN) if "
                    "missing, then run")
    args = ap.parse_args(argv)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR.parent / ".gitignore").write_text("*\n")

    spec = build_spec()
    SPEC_PATH.write_text(json.dumps(spec, indent=1))

    pipeline = _load_module("pipeline", SRC / "lib" / "pipeline.py")

    if args.command == "status":
        return pipeline.status(SPEC_PATH)
    if args.command == "reset":
        if STATE_PATH.exists():
            STATE_PATH.unlink()
            print(f"cleared {STATE_PATH} - next run re-executes every stage "
                  "(experiment caches and saved artifacts are untouched, so "
                  "most stages replay their caches)")
        else:
            print("no state to clear")
        return 0

    errors = preflight(args.fetch_data, spec["stages"])
    if errors:
        print("[preflight] FAILED:", file=sys.stderr)
        for e in errors:
            print(f" - {e}", file=sys.stderr)
        return 2

    n_pending = len(_pending([s["name"] for s in spec["stages"]]))
    print(f"[reproduce] {len(spec['stages'])} stages ({n_pending} pending); "
          f"state: {STATE_PATH}; logs: {STATE_DIR / 'logs'}")
    if args.command == "launch":
        return pipeline.launch(SPEC_PATH)
    return pipeline.run(SPEC_PATH)


if __name__ == "__main__":
    sys.exit(main())
