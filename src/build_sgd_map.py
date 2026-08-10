"""Issue #1 remaining task: map USS-SGD dialogues back to original SGD by
utterance text; recover dialogue_id + services metadata."""
import json
import sys
from collections import Counter
from pathlib import Path

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from lib.campaign import make_run_sample
from lib.uss import load_uss

SGD = WORK.parent / "data" / "raw" / "sgd_original"


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "build_sgd_map parses the raw SGD/USS files directly and cannot "
            "cap in memory; use the sandbox driver instead: reproduce.py "
            "--max-samples N caps the raw layer (lib/corpus_sample.py).")
    index = {}
    for split in ["train", "dev", "test"]:
        for f in sorted((SGD / split).glob("dialogues_*.json")):
            for d in json.load(open(f)):
                utts = [t["utterance"].strip() for t in d["turns"]]
                key = (utts[0], utts[1] if len(utts) > 1 else "")
                index[key] = (split, d["dialogue_id"], tuple(d["services"]))

    seqs, meta = load_uss("sgd")
    matched, mapping = 0, []
    for i, seq in enumerate(seqs):
        key = (seq[0].strip(), seq[1].strip() if len(seq) > 1 else "")
        hit = index.get(key)
        if hit:
            matched += 1
            mapping.append({"uss_idx": i, "split": hit[0], "dialogue_id": hit[1],
                            "services": list(hit[2])})
        else:
            mapping.append({"uss_idx": i, "split": None, "dialogue_id": None,
                            "services": []})

    print(f"matched {matched}/{len(seqs)} USS-SGD dialogues to original SGD")
    print("splits:", Counter(m["split"] for m in mapping if m["split"]))
    svc = Counter(s for m in mapping for s in m["services"])
    print("top services:", svc.most_common(8))
    out = WORK.parent / "data" / "embeddings" / "uss_sgd_original_map.json"
    json.dump(mapping, open(out, "w"))
    print("saved", out)
    return {"matched": matched, "n_dialogues": len(seqs), "output": str(out)}


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
