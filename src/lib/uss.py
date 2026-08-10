"""
USS (User Satisfaction Simulation) dataset loader.

Sun et al., "Simulating User Satisfaction for the Evaluation of Task-oriented
Dialogue Systems", SIGIR'21. https://github.com/sunnweiwei/user-satisfaction-simulation

File format (tab-separated, dialogues separated by blank lines):
    ROLE \t text \t action \t ratings
where ROLE is USER or SYSTEM, ratings is comma-separated per-annotator scores
(1-5, USER lines only). The final line of each dialogue is a pseudo-turn
"USER \t OVERALL \t \t ratings" carrying the dialogue-level satisfaction.

Labels are third-party annotator judgments of user satisfaction, not
first-person ratings (unlike TRACE's corpus).
"""

import statistics
from pathlib import Path
from typing import List, Tuple, Dict, Any

USS_DATASETS = ["sgd", "mwoz", "redial", "ccpe", "jddc"]

_FILENAMES = {
    "sgd": "SGD.txt",
    "mwoz": "MWOZ.txt",
    "redial": "ReDial.txt",
    "ccpe": "CCPE.txt",
    "jddc": "JDDC.txt",
}


def get_uss_root() -> Path:
    return Path(__file__).parent.parent.parent / "data" / "raw" / "uss_repo" / "dataset"


def parse_uss_file(path: Path) -> List[Dict[str, Any]]:
    """Parse one USS file into a list of dialogue dicts.

    Each dialogue: {
        'turns': [{'role', 'text', 'action', 'ratings': [int, ...]}],
        'overall': [int, ...],   # per-annotator dialogue-level ratings
    }
    """
    dialogues = []
    turns = []
    overall = None

    def flush():
        nonlocal turns, overall
        if turns:
            dialogues.append({"turns": turns, "overall": overall or []})
        turns = []
        overall = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            role = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            action = parts[2].strip() if len(parts) > 2 else ""
            ratings_str = parts[3].strip() if len(parts) > 3 else ""
            ratings = [int(r) for r in ratings_str.split(",") if r.strip().isdigit()]
            if text == "OVERALL":
                overall = ratings
            else:
                turns.append({"role": role, "text": text, "action": action,
                              "ratings": ratings})
    flush()
    return dialogues


def load_uss(dataset: str = "sgd") -> Tuple[List[List[str]], Dict[str, Any]]:
    """Load a USS subset following lib.datasets loader conventions.

    Returns:
        sequences: List[List[str]] - turn texts per dialogue
        metadata: dict with per-dialogue parallel lists:
            roles: List[List[str]]
            actions: List[List[str]]
            turn_ratings: List[List[List[int]]] (per user turn; empty for SYSTEM)
            overall: List[List[int]] - per-annotator dialogue ratings
            overall_mean: List[float]
            overall_median: List[float]
            n_turns: List[int]
    """
    path = get_uss_root() / _FILENAMES[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"USS data not found at {path}\n"
            f"Clone: git clone --depth 1 "
            f"https://github.com/sunnweiwei/user-satisfaction-simulation "
            f"{get_uss_root().parent}"
        )
    dialogues = parse_uss_file(path)

    sequences = []
    meta: Dict[str, Any] = {k: [] for k in
                            ["roles", "actions", "turn_ratings", "overall",
                             "overall_mean", "overall_median", "n_turns"]}
    for d in dialogues:
        if not d["overall"]:
            continue  # unusable without a dialogue-level label
        sequences.append([t["text"] for t in d["turns"]])
        meta["roles"].append([t["role"] for t in d["turns"]])
        meta["actions"].append([t["action"] for t in d["turns"]])
        meta["turn_ratings"].append([t["ratings"] for t in d["turns"]])
        meta["overall"].append(d["overall"])
        meta["overall_mean"].append(statistics.mean(d["overall"]))
        meta["overall_median"].append(statistics.median(d["overall"]))
        meta["n_turns"].append(len(d["turns"]))
    meta["dataset"] = f"uss_{dataset}"
    meta["n_dialogues"] = len(sequences)
    return sequences, meta
