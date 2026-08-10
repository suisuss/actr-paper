"""Campaign module contract - the experiment API for src/.

Every campaign experiment script is an importable module exporting three
callables (importing a module never executes work):

    run(**params) -> dict
        The experiment entry point: loads its data, computes, persists
        through save_result below (flat [src:] file + provenance-stamped
        registry copy; scripts import it via the campaign_io shim), and
        returns the results dict. Standard optional params where they
        apply: dataset=..., mode=..., max_samples=None.

    run_sample(max_samples, **params) -> dict
        Alias for run(max_samples=max_samples, **params), built with
        make_run_sample() below. Sampled in-process runs are EPHEMERAL:
        they must not call save_result and must bypass checkpoint caches
        (full-N cache entries would otherwise be silently reused, and
        sampled numbers must never enter the canonical registry or flat
        artifacts). Modules whose data access cannot be capped in memory
        (raw-file readers, embedding builders) and modules that are
        full-corpus-bound by construction (the judge replays, whose
        frozen ratings are index-aligned to full corpora) raise
        ValueError pointing at the drivers' sandbox flag
        (reproduce.py --max-samples N), which caps at the raw layer
        instead (lib/corpus_sample.py).

    main(argv=None) -> int
        Thin CLI shim preserving the script's historical argv exactly,
        called from the `if __name__ == "__main__"` guard - pipeline
        specs and the reproduction drivers keep invoking
        `python scripts/<name>.py ...` unchanged.

max_samples semantics are first-N dialogues per corpus - the in-memory
mirror of the raw-layer truncation the drivers' sandbox mode performs
(first-N preserves every position-aligned structure; see
lib/corpus_sample.py). Apply cap() immediately after loading.
"""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent.parent


def cap(emb, offsets, meta, max_samples):
    """Slice a loaded corpus triple to its first max_samples dialogues.

    emb: (n_utterances, d) array; offsets: (n_dialogues+1,) int array;
    meta: dict whose per-dialogue values are lists of length
    n_dialogues (any other value passes through untouched).
    Returns the triple unchanged when max_samples is None.
    """
    if max_samples is None:
        return emb, offsets, meta
    n_dialogues = len(offsets) - 1
    n = min(int(max_samples), n_dialogues)
    offsets_c = np.asarray(offsets[:n + 1])
    emb_c = emb[:int(offsets_c[-1])]
    meta_c = {k: (v[:n] if isinstance(v, list) and len(v) == n_dialogues
                  else v)
              for k, v in meta.items()}
    return emb_c, offsets_c, meta_c


def make_run_sample(run):
    """Standard run_sample factory: ephemeral capped invocation."""
    def run_sample(max_samples, **params):
        return run(max_samples=max_samples, **params)
    run_sample.__doc__ = (f"run(max_samples=N, ...) alias; ephemeral - "
                          "no save_result, no checkpoint writes. "
                          "See lib/campaign.py.")
    return run_sample


class Checkpoint:
    """The house OOF-checkpoint pattern, in one place.

    One dir per script under data/experiment-caches/<name>/, one npz per
    tag, atomic tmp+rename writes. File layout is identical to the
    historical per-script ckpt_load/ckpt_save helpers, so pre-existing
    caches stay warm. Sampled runs must not touch checkpoints (the
    run_sample contract); callers gate on their sampled flag.

    Staleness: a PRODUCER constructs Checkpoint(name, script=__file__).
    save() stamps the producing script's sha256 into <dir>/producer.json
    (idempotently, before writing the npz); when the script later
    changes, load() MISSes any entry stamped under a different sha
    (stale-but-warm entries are never silently replayed) and the next
    save() clears the stale npz files and re-stamps. load() is
    read-only. A dir predating stamping (no producer.json) is adopted:
    reads are allowed and the next save() writes the stamp without
    wiping (today's caches match today's code - the 2026-07-13
    equivalence tests). READERS of another script's cache construct
    Checkpoint(name) with no script: passive reads that never
    invalidate, since freshness is the producer's job (pipeline
    depends_on ordering runs the producer first).
    """

    def __init__(self, name, root=None, script=None):
        self.dir = Path(root) / name if root else \
            WORK.parent / "data" / "experiment-caches" / name
        self._script = Path(script) if script else None

    def _stamp_file(self):
        return self.dir / "producer.json"

    def _source_digest(self):
        """Digest of the script's LOGIC, used only for cache invalidation:
        a hash of the parsed AST, so comment- and formatting-only edits
        leave it unchanged and do not wipe cached checkpoints. Only a
        change to the code itself (or a docstring) re-invalidates. Falls
        back to a byte hash if the source does not currently parse (e.g.
        mid-edit). This is deliberately separate from the exact file
        sha256 recorded in each artifact's _provenance for external
        verifiability (lib/provenance.py::file_sha256)."""
        import ast
        import hashlib
        src = self._script.read_bytes()
        try:
            payload = ast.dump(ast.parse(src)).encode()
        except SyntaxError:
            payload = src
        return hashlib.sha256(payload).hexdigest()

    def _stamped_sha(self):
        stamp = self._stamp_file()
        if stamp.exists():
            # legacy stamps used the file-bytes key "script_sha256"; a
            # missing "source_digest" reads as None and is adopted, not
            # refused (see _ensure_stamp / load).
            return json.loads(stamp.read_text()).get("source_digest")
        return None

    def _ensure_stamp(self):
        """Producer-only, idempotent: stamp the dir with the current
        script sha, clearing stale npz entries when a different sha was
        stamped. No-op for readers (script=None)."""
        if self._script is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        cur = self._source_digest()
        stamped = self._stamped_sha()
        if stamped == cur:
            return
        if stamped is not None:            # producer logic changed - wipe stale
            for f in self.dir.glob("*.npz"):
                f.unlink()
            print(f"[cache] {self.dir.name}: producer script changed - "
                  "cleared stale checkpoints", flush=True)
        stamp = self._stamp_file()
        tmp = stamp.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"script": self._script.name, "source_digest": cur}, indent=1))
        tmp.rename(stamp)

    def load(self, tag):
        # a producer never replays a cache stamped by a different script
        # version; a missing stamp (legacy dir) is adopted, not refused.
        if self._script is not None:
            stamped = self._stamped_sha()
            if stamped is not None and stamped != self._source_digest():
                return None
        f = self.dir / f"{tag}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            return {k: z[k] for k in z.files}
        return None

    def save(self, tag, arrays):
        self._ensure_stamp()
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / f"{tag}.npz"
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.rename(f)


def load_corpus(name, max_samples=None):
    """Canonical three-corpus loader for campaign scripts.

    Returns (emb, offsets, meta, groups): the campaign one-off MiniLM
    turn embeddings (data/embeddings/{corpus}_minilm.npz + meta json),
    capped to the first max_samples dialogues when set. groups is
    meta["user_ids"] for PRISM (user-grouped CV), None for USS.
    """
    emb_dir = WORK.parent / "data" / "embeddings"
    if name == "prism":
        stem = "prism"
    elif name.startswith("uss_"):
        stem = name
    else:
        raise ValueError(f"unknown corpus {name!r}; expected 'prism', "
                         "'uss_sgd', or 'uss_mwoz'")
    npz = np.load(emb_dir / f"{stem}_minilm.npz")
    with open(emb_dir / f"{stem}_meta.json") as f:
        meta = json.load(f)
    emb, offsets, meta = cap(npz["embeddings"], npz["offsets"], meta,
                             max_samples)
    groups = meta.get("user_ids") if name == "prism" else None
    return emb, offsets, meta, groups


def save_result(name, results, *, dataset, config="minilm", variant="none",
                params=None, tags=("trace_h2h",), n_samples=None,
                duration=None, also_flat=True, script=None):
    """Persist a campaign result: flat [src:] file + provenance-stamped
    registry copy (trace_h2h/<name>/<dataset>/<config>/<name>_<ts>.json).

    Moved here from scripts/campaign_io.py (which remains as a shim).
    script: repo-relative path of the producing script; auto-detected
    from the caller when omitted.
    """
    import inspect
    import os
    from lib.ioutils import atomic_json_dump
    from lib.results import ResultsRegistry

    if also_flat:
        atomic_json_dump(WORK.parent / "data" / "results" / "trace_h2h"
                         / f"{name}.json", results, indent=1)
    if script is None:
        caller = inspect.stack()[1].filename
        try:
            script = os.path.relpath(caller, WORK)
        except ValueError:
            script = caller
    key = ResultsRegistry().save(
        experiment=f"trace_h2h/{name}", config=config, dataset=dataset,
        variant=variant, params=params or {}, results=results,
        tags=list(tags), script=script, n_samples=n_samples,
        duration=duration, filename_stem=name)
    print(f"registered {key}", flush=True)
    return key


def verify_provenance(results_root=None):
    """The README.md gate as a callable: every canonical
    campaign artifact must come from a clean pinned checkout. Returns a
    list of offending paths (empty = gate passes); the campaign/
    back-fill is excluded by construction."""
    root = Path(results_root) if results_root else \
        WORK.parent / "data" / "results"
    bad = []
    for f in sorted((root / "trace_h2h").glob("*/**/*.json")):
        if "/campaign/" in str(f):
            continue
        p = json.loads(f.read_text()).get("_provenance", {})
        if p.get("git_dirty") is not False or not (
                p.get("git_commit") and p.get("script_sha256")):
            bad.append(str(f))
    return bad


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        offenders = verify_provenance()
        for f in offenders:
            print(f"DIRTY/UNSTAMPED: {f}")
        print(f"{'FAIL' if offenders else 'PASS'}: "
              f"{len(offenders)} offending artifact(s)")
        sys.exit(1 if offenders else 0)
    print("usage: python -m lib.campaign verify")
    sys.exit(64)
