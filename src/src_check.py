#!/usr/bin/env python3
"""Verify the <!-- src: SCRIPT --> back-trace tags in the paper.

For paper/paper.md, checks:

  1. every `src:` names a script that exists in src/;
  2. every campaign script's artifact exists and is provenance-clean
     (latest timestamped registry copy has git_dirty == False) - reported
     as a warning, not a gate;
  3. coverage: every ![](figures/*.svg) reference carries a following tag.

Generators (actr_figures / actr_tables / trace_h2h) produce
figures/tables/mode-dirs rather than a flat <stem>.json, so only their
existence is checked.

    python src_check.py [paper/paper.md]

Exit 0 = clean; non-zero prints the failures. Read-only, stdlib-only.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
RESULTS = REPO / "data" / "results" / "trace_h2h"
DEFAULT_DRAFT = REPO / "paper" / "paper.md"

# scripts that emit figures/tables/mode-dirs, not a flat <stem>.json artifact
GENERATORS = {"actr_figures", "actr_tables", "trace_h2h"}


def find_script(name):
    hits = glob.glob(str(SRC / "**" / name), recursive=True)
    return os.path.relpath(hits[0], REPO) if hits else None


def artifact_provenance(stem):
    """(status, detail) for a script's registry artifact(s). Handles the
    per-corpus case (script foo.py -> foo_uss_sgd.json / foo_prism.json / ...)
    by matching the exact stem dir plus any <stem>_<suffix> dirs; every matched
    artifact's latest copy must be provenance-clean."""
    dirs = [d for d in glob.glob(str(RESULTS / f"{stem}*")) if os.path.isdir(d)
            and (os.path.basename(d) == stem
                 or os.path.basename(d).startswith(stem + "_"))]
    latest = []
    for d in dirs:
        tcs = sorted(glob.glob(d + "/**/*.json", recursive=True))
        if tcs:
            latest.append((os.path.basename(d), tcs[-1]))
    if not latest:
        if (RESULTS / f"{stem}.json").exists():
            return ("flat-only", "flat artifact, no provenance copy yet")
        return ("MISSING", "no artifact")
    bad = []
    commits = set()
    for name, f in latest:
        p = json.load(open(f)).get("_provenance", {})
        commits.add((p.get("git_commit") or "?")[:7])
        if not (p.get("git_dirty") is False and p.get("git_commit")):
            bad.append(f"{name} dirty={p.get('git_dirty')}")
    if bad:
        return ("DIRTY", "; ".join(bad))
    return ("clean", f"{'/'.join(sorted(commits))} ({len(latest)} artifact(s))")


def main():
    draft = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DRAFT
    lines = draft.read_text().split("\n")

    errors, warnings, rows = [], [], []
    for i, line in enumerate(lines, 1):
        if "tags name the producing script" in line:
            continue
        # a single line may carry several inline src tags (paragraphs are
        # unwrapped to one line each), so validate every tag on the line.
        for m in re.finditer(r"<!--\s*src:\s*([^>]+?)\s*-->", line):
            for name in [s.strip() for s in m.group(1).split(",") if s.strip()]:
                if not name.endswith(".py"):
                    errors.append(f"L{i}: '{name}' is not a script (.py) name")
                    continue
                path = find_script(name)
                if path is None:
                    errors.append(f"L{i}: script {name} not found under src/")
                    continue
                stem = name[:-3]
                if stem in GENERATORS:
                    rows.append((i, name, "generator", path))
                    continue
                status, detail = artifact_provenance(stem)
                rows.append((i, name, status, detail))
                if status in ("MISSING", "DIRTY"):
                    warnings.append(f"L{i}: {name} -> artifact {status} ({detail})")

    # coverage: every figure reference must be immediately followed by a tag
    for i, line in enumerate(lines):
        if re.search(r"\]\(figures/[a-z_]+\.svg\)", line):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "src:" not in nxt:
                errors.append(f"L{i + 1}: figure ref has no following src tag")

    print(f"=== {draft.name}: {len(rows)} src tags ===")
    for i, name, status, detail in rows:
        print(f"  L{i:<5} {name:<24} {status:<12} {detail}")
    print()
    # Tag correctness (scripts exist, figures covered) is the gate; artifact
    # provenance is reported as a warning - a dirty/missing canonical artifact
    # is a data-state issue (regenerate + promote), not a mis-tag.
    if warnings:
        print(f"ARTIFACT WARNINGS: {len(warnings)} cited artifact(s) not "
              "provenance-clean on canonical (regenerate/promote):")
        for w in warnings:
            print(f"  ! {w}")
        print()
    if errors:
        print(f"TAG FAIL: {len(errors)} issue(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"TAG PASS: all {len(rows)} tags resolve to existing scripts; "
          "every figure tagged"
          + (f" ({len(warnings)} artifact-provenance warnings above)"
             if warnings else "; all cited artifacts provenance-clean"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
