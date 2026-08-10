"""PRISM goal-proxy check (draft section 3.2), registry-backed.

Backs the section 3.2 numbers with a provenance-stamped artifact instead
of an ad-hoc check. The goal embedding G is set to the first user turn on
every corpus. On PRISM this is exact rather than a substitution, because
the `opening_prompt` field is a byte-identical copy of the first user
turn; the two documented goal degeneracies then hold to float precision,
and PRISM's guided/unguided composition motivates the proxy's weakest
fidelity there.

Measures, over the released PRISM data:
- opening_prompt == first user turn, byte-identical, over all raw conversations
- share of values-/controversy-guided conversations
- goal_initial_prompt_dist identically 0 (max abs residual; float32 inputs)
- model_goal_dist_mean identically model_initial_prompt_dist (correlation,
  max abs difference)

The two degeneracy quantities are read from the paper's own feature
computation (lib.trace_features, G defaulting to the first user turn), so
they are the operationalization the paper reports, not a reimplementation.
Deterministic: no RNG.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from lib.campaign import load_corpus, make_run_sample
from lib.trace_features import FEATURE_NAMES, features_matrix

RAW_PRISM = WORK.parent / "data" / "raw" / "prism" / "conversations.jsonl"


def _raw_scan():
    """opening_prompt vs the first user turn (byte-identical) and the
    conversation-type composition, over the raw conversations and over the
    analyzed subset (helpfulness score present and >= 2 turns - the paper's
    PRISM preprocessing filter, which recovers the 7,951 analyzed count)."""
    total = first_is_user = byte_identical = 0
    mismatch_ids = []
    types = {}
    types_analyzed = {}
    with open(RAW_PRISM) as f:
        for line in f:
            d = json.loads(line)
            total += 1
            t = d["conversation_type"]
            types[t] = types.get(t, 0) + 1
            hp = (d.get("performance_attributes") or {}).get("helpfulness")
            if hp is not None and (d.get("conversation_turns") or 0) >= 2:
                types_analyzed[t] = types_analyzed.get(t, 0) + 1
            ch = d.get("conversation_history") or []
            if not ch or str(ch[0].get("role")).lower() != "user":
                continue
            first_is_user += 1
            if ch[0].get("content") == d.get("opening_prompt"):
                byte_identical += 1
            elif len(mismatch_ids) < 5:
                mismatch_ids.append(d.get("conversation_id"))
    return (total, first_is_user, byte_identical, mismatch_ids,
            types, types_analyzed)


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError("goal_proxy_prism reads the full raw PRISM corpus "
                         "and cannot cap in memory")
    if not RAW_PRISM.exists():
        raise FileNotFoundError(f"raw PRISM missing at {RAW_PRISM}; run "
                                "python main.py --fetch-data")

    (total, first_is_user, byte_identical, mismatch_ids,
     types, types_analyzed) = _raw_scan()

    def guided_share(counts):
        n = sum(counts.values())
        g = counts.get("values guided", 0) + counts.get("controversy guided", 0)
        return n, g, (round(g / n, 4) if n else 0.0)

    raw_n, raw_g, raw_frac = guided_share(types)
    ana_n, ana_g, ana_frac = guided_share(types_analyzed)

    emb, offsets, meta, _ = load_corpus("prism", None)
    X = features_matrix(emb, offsets, meta["roles"])
    gip = X[:, FEATURE_NAMES.index("goal_initial_prompt_dist")]
    mgd = X[:, FEATURE_NAMES.index("model_goal_dist_mean")]
    mip = X[:, FEATURE_NAMES.index("model_initial_prompt_dist")]
    gip_max_abs = float(np.max(np.abs(gip)))
    pair_max_abs = float(np.max(np.abs(mgd - mip)))
    pair_r = float(np.corrcoef(mgd, mip)[0, 1])

    results = {
        "opening_prompt_equals_first_user_turn": {
            "conversations": total,
            "first_turn_is_user": first_is_user,
            "byte_identical": byte_identical,
            "byte_identical_fraction": round(byte_identical / total, 6),
            "mismatch_example_ids": mismatch_ids,
        },
        "composition": {
            "raw": {"counts": types, "n": raw_n,
                    "values_or_controversy_guided": raw_g,
                    "values_or_controversy_fraction": raw_frac},
            "analyzed": {"counts": types_analyzed, "n": ana_n,
                         "values_or_controversy_guided": ana_g,
                         "values_or_controversy_fraction": ana_frac,
                         "filter": "helpfulness present and >= 2 turns"},
        },
        "goal_degeneracies": {
            "n_dialogues_encoded": int(len(offsets) - 1),
            "goal_initial_prompt_dist_max_abs": gip_max_abs,
            "model_goal_vs_initial_prompt_corr": pair_r,
            "model_goal_vs_initial_prompt_max_abs_diff": pair_max_abs,
        },
        "note": ("G := first user turn on every corpus. On PRISM the "
                 "opening_prompt is a byte-identical copy of that turn, so "
                 "the choice is exact, not a substitution; the two goal "
                 "degeneracies then hold to float precision "
                 "(goal_initial_prompt_dist identically 0; "
                 "model_goal_dist_mean identically model_initial_prompt_dist). "
                 "Deterministic; float32 embedding inputs."),
    }

    print(f"[goal_proxy_prism] opening_prompt == first user turn: "
          f"{byte_identical}/{total} byte-identical", flush=True)
    print(f"[goal_proxy_prism] values/controversy-guided: raw {raw_g}/{raw_n} "
          f"({raw_frac:.1%}); analyzed {ana_g}/{ana_n} ({ana_frac:.1%})",
          flush=True)
    print(f"[goal_proxy_prism] goal_initial_prompt_dist max|x|={gip_max_abs:.2e}; "
          f"model_goal_dist_mean vs model_initial_prompt_dist r={pair_r:.13f}, "
          f"max|diff|={pair_max_abs:.2e}", flush=True)

    from campaign_io import save_result
    save_result("goal_proxy_prism", results, dataset="prism", config="minilm",
                params={"deterministic": True, "n_raw_conversations": total})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
