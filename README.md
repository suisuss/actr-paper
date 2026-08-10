# actr-paper — reproduction kit

**Analysis of Conversation Trajectory Representations: Orthogonal Projections versus Scalar Geometry for Satisfaction Prediction**

Jacob Sussmilch, Heya Enterprises Pty Ltd

Everything needed to read, check, and re-run the paper: the manuscript source (`paper/paper.md`, built to `paper/paper.pdf`), every result artifact it cites, the ~40 scripts that produced them, and a one-command pipeline that regenerates the lot from public data.

This is the **representation study** of a three-paper line on reading conversations as trajectories through embedding space: embed each turn, and a conversation becomes a path of points that can be summarized as a fixed-size vector for a simple learner. *Which summary* is the question this paper studies: hand-crafted scalar geometry, or an orthogonal-polynomial projection of the trajectory.

*Status: preprint in preparation. The kit is complete and gated; the manuscript has not yet been posted or peer-reviewed.*

## Start here

| | |
|---|---|
| **1. Read the paper** | `paper/paper.pdf` (built from `paper/paper.md`, which is the source of truth) |
| **2. Check it against its artifacts** | `./run.sh check` — seconds, needs no datasets, no torch, no GPU. Five gates: all 407 printed table cells regenerate from `data/results/`, every artifact is provenance-clean, all 83 `<!-- src: -->` tags resolve to real scripts, the committed PDF rebuilds with the same words. The same gates run in CI on every push and pull request (`.github/workflows/validate.yml`). |
| **3. Spot-check a claim** | [Claim -> script -> artifact](#claim---script---artifact) maps every headline number to the one script and the one JSON artifact behind it. Open the artifact; its `_provenance` block names the commit, script sha256, argv, and seeds that produced it. |
| **4. Re-run it** | `./run.sh` — the full reproduction from raw text (many hours). Resumable. Then `git diff`: a clean diff *is* the reproduction, since regenerated artifacts overwrite the committed ones in place. |

Two facts about the setup, stated up front: the comparison suite (TRACE-geometric) is **our reimplementation** from the published specification, with every deviation documented in the paper's Appendix A; and the LLM-judge arms **make zero API calls** — the ratings are frozen raw data, committed here, and replayed.

Unexplained mismatches belong in an issue, with the failing gate's output and the `git diff` of the artifact in question.

## Contents

- [The paper in brief](#the-paper-in-brief)
- [Where this sits: the three-paper line](#where-this-sits-the-three-paper-line)
- [The encoding: from conversation to coefficient vector](#the-encoding-from-conversation-to-coefficient-vector)
- [Data](#data)
- [Reproducing the paper](#reproducing-the-paper)
- [Claim -> script -> artifact](#claim---script---artifact)
- [How the kit keeps the paper honest](#how-the-kit-keeps-the-paper-honest)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Citation and license](#citation-and-license)

## The paper in brief

### The question

A growing line of work scores conversation quality from the geometry of how a dialogue moved. The most prominent instance is TRACE (Gooding and Grefenstette, 2025), which predicts conversation-level satisfaction from roughly thirty hand-crafted scalar features — self-similarity of model turns, drift from a stated goal, turn-to-turn volatility, trends in relevance — fed to a random forest.

Every such feature embodies the same unexamined decision: it is a *scalarization* of an underlying vector object, almost always through a distance or a cosine. The trajectory — a sequence of points in `R^d` — is reduced feature by feature to hand-picked one-dimensional summaries before any learning happens. The alternative is older and simpler than feature engineering: fix an orthogonal basis, project, and hand the learner the *coefficient vectors*. No selection step, a quantified residual, and no way to lose direction information you did not know you needed.

### What it found

1. **Which representation wins is regime-dependent.** The projection beats TRACE's geometric features by 5-12 points on all three corpora — but on short, first-person-rated chats the trajectory-shape channels collapse and hand-crafted scalars become a genuine complement. This is a difference of corpus and label construct, not of conversation length (length is ruled out by an explicit deconfound arm).
2. **The projection's structure matters more than its basis.** Read as one interleaved sequence, the projection blends both speakers into shared low-degree coefficients. Splitting the trajectory by speaker — giving the learner each speaker's content on its own path — is the recommended representation: training-free, and never distinguishably worse than any measured alternative in either regime.
3. **What the split buys is the content separation.** The speaker-alternation aliasing it also removes is a representation-fidelity fact, not the source of the accuracy.
4. **The advantage is informational, not capacity.** Compressed to TRACE's own 26 dimensions the role-split projection still beats it, while a random 26-dim projection is significantly worse.

### What it does not claim

Section 8 of the paper lists twelve limitations, two of them declared scope decisions. The material ones:

- **This is offline prediction, not reward modeling — and our winners are plausibly the more gameable ones.** A reward built on content (mean pooling, or the projection coefficients) is more directly Goodhart-hackable than content-agnostic scalar geometry, so the offline ranking need not transfer to the reward setting TRACE was built for, and may invert. That hand-off is the subject of [`mtgcr-paper`](https://github.com/suisuss/mtgcr-paper).
- **TRACE-geometric is not TRACE.** Four of thirty features (all Temporal Dynamics) need timestamps that public corpora lack; the deviations and the proxy used are documented feature by feature in Appendix A.
- **The regime boundary is corpus-attributed, not fully causal.** Length is ruled out; label construct, provenance, and goal grounding remain bundled, and separating them needs a corpus crossing rater type with provenance — which our set does not contain.
- **Dimensionality is not matched in the headline comparison** (1,536-d vectors vs 26 scalars); the capacity-honest contrasts and the PCA-26 control carry that burden.
- **No learned-aggregation baseline.** The comparison class is fixed representations; a recurrent or attention encoder over the same turn embeddings is untested, so "never distinguishably worse" is a claim within that class, not about the best achievable representation.
- **One language, one judge family, one primary encoder** (English; a single judge model's frozen ratings; MiniLM primary, with the role-split result replicated under GTE-small and bge-base).

## Where this sits: the three-paper line

Each repo is a self-contained, public reproduction kit built on the same harness.

| Repo | Role in the line | What it establishes |
|---|---|---|
| [`gp-paper`](https://github.com/suisuss/gp-paper) | Method | The Gram projection: what the encoding *is*, its exact identities, and that the readable basis costs nothing against the DCT. |
| **`actr-paper`** (this repo) | **Representation** | Which trajectory representation predicts satisfaction, the role-split recommendation, and the regime boundary where it stops holding. |
| [`mtgcr-paper`](https://github.com/suisuss/mtgcr-paper) | Reward | Answers this paper's closing warning: the naive geometric reward is gameable by selection alone; a role-restricted, task-conditioned construction is not. |

This paper **adopts** the encoding rather than deriving it — the construction, the exact identities, and the basis comparison are `gp-paper`'s subject, restated here only where an arm or figure needs them — and **hands off** its closing caution (that the best satisfaction predictors are plausibly the most gameable rewards; see [What it does not claim](#what-it-does-not-claim)) to `mtgcr-paper`.

## The encoding: from conversation to coefficient vector

The construction is `gp-paper`'s; what this paper's arms need is below.

Let `e_1, ..., e_T` in `R^d` be the L2-normalized turn embeddings of a conversation. The degree-`k` Gram coefficient vectors `G_0, ..., G_K` summarize the trajectory on the turn grid, with `K = min(3, T-1)`:

```
G_k = sum_i phi_k(i) * e_i  /  sum_i phi_k(i)^2
```

where `phi_k` is the degree-`k` Gram polynomial on the uniform grid `{1, ..., T}`. Two identities carry most of the interpretive weight: **`G_0` is exactly the mean** (mean pooling is the degree-0 case) and **`G_1` is exactly the least-squares drift**.

Gram is adopted over the DCT (a measured equivalent) or the DST (a priceable boundary) for per-coefficient interpretability: a trend is a single `G_1`, whereas no single DCT coefficient is a slope.

**The representations compared** (all consumed by the same regressor):

| Arm | Dim | What it is |
|---|---|---|
| Interleaved projection | 1,536 | `[G_0 .. G_3]` concatenated over the whole dialogue — the paper's whole-dialogue representation |
| **Role-split(1)** | 1,536 | Per-role degrees 0-1: each speaker's own mean and drift, concatenated. **The recommendation.** |
| Mean pooling | 384 | Identically `G_0` — the content-only ablation |
| Shape only | 1,152 | `G_1`-`G_3` — the fairest shape-vs-shape contrast against TRACE-geometric |
| TRACE-geometric | ~26-30 | The hand-crafted scalar suite, reimplemented (`lib/trace_features.py`) |
| Projected TRACE series | 20 | Gram degrees 0-3 of TRACE's own five scalar series — the subsumption arm |
| Scalarized coefficients | 9 | Norms and angles of our vector coefficients — our representation deliberately scalarized |

**Role-split and the aliasing it removes.** The interleaved sequence `u_1, m_1, u_2, m_2, ...` alternates between two regions of embedding space — the highest frequency the grid supports — so a low-degree global polynomial cannot represent it and it aliases into the low-degree coefficients. Interleaved `G_1` therefore confounds "the conversation drifted" with "user and model occupy different regions." Role-split removes the artifact by construction: project each speaker's subsequence on its own grid and concatenate. The midline and speaker offset are recovered exactly, `c_k = (G_k^u + G_k^m)/2` and `d_k = G_k^u - G_k^m`.

**Causal per-role surprise.** The split makes per-role surprise well-defined: `pred_j` is the degree-1 extrapolation of a party's own path fit on its prefix only, `surprise_j = 1 - cos(pred_j, actual_j)`, with a static-prediction `naive_j` baseline and `trend_gain` their difference. The prefix-only fit is what makes it causal.

## Data

Three source corpora, downloaded on demand — nothing is fetched implicitly. `python main.py --fetch-data` acquires whatever is missing, each pinned to the exact revision the paper used.

| Corpus | Source | Size | Pinned revision | Terms |
|---|---|---|---|---|
| USS (SGD + MWOZ satisfaction labels) | [sunnweiwei/user-satisfaction-simulation](https://github.com/sunnweiwei/user-satisfaction-simulation) | 18MB | `8ba06bd92570fd94d3ec0cb828aad45172be71b2` | No license file in the upstream repo |
| PRISM | [HannahRoseKirk/prism-alignment](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) | 58MB | `18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6` | Gated — accept its terms on the HF page, then set `HF_TOKEN` |
| Original SGD | [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | 619MB | `e852981ae34990f4358979625854259302feaa78` | Upstream repository's own terms; needed once, to build a text-match map |

Those downloads yield the paper's three evaluation corpora — and the regime contrast that produces its central result:

| Corpus | Conversations | Turns | Label construct | CV |
|---|---|---|---|---|
| USS-SGD | 1,000 | 25,666 (mean 25.7) | 5-point satisfaction, mean over 3+ third-party annotators | 5-fold by dialogue |
| USS-MWOZ | 1,000 | 22,108 (mean 22.1) | same | 5-fold by dialogue |
| PRISM | 7,951 (1,395 users) | 54,212 (mean 6.8) | first-person helpfulness, rescaled 1-5 | `GroupKFold(5)` by user |

USS labels are third-party judgments made *from transcript text*, which plausibly inflates the content channel; PRISM labels are first-person but singular, so its annotator ceiling is unknowable. The USS ceiling is measured (leave-one-annotator-out agreement 0.943/0.948 at gap 0.5). That label-construct difference, not conversation length, is what the regime result turns on.

**None of the three corpora is vendored here, by decision**: the USS repository carries no license file, so it is not ours to redistribute; PRISM is gated behind its own terms, which each user must accept in their own name; and original SGD is 619MB. Each is fetched from the original source at the pinned revision, so a reproduction reads exactly the bytes the paper read, under the original terms.

**Embedding models.** Turn embeddings are the deliberately least-engineered component: identical off-the-shelf embeddings feed every arm, TRACE-geometric included, so no result can be attributed to turn-representation quality.

| Model | Dim | Used for |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Primary encoder — every headline arm |
| `thenlper/gte-small` | 384 | Multi-embedder replication of the role-split result |
| `BAAI/bge-base-en-v1.5` | 768 | Multi-embedder replication of the role-split result |

**Protocol.** `RandomForestRegressor` (300 trees, `min_samples_leaf` 2) on each arm's representation; 5-fold out-of-fold predictions (user-grouped on PRISM); pairwise preference accuracy over conversation pairs with mean-rating gap >= 0.5 (secondary) or 1.0 (primary). Five-seed replication; inferential contrasts use seed-ensembled predictions with m-out-of-n dialogue-resampling bootstrap CIs. Measured instrument resolution: median bootstrap CI half-width ±2.3-2.6 points seed-ensembled.

**Vendored in this repo**: the canonical result artifacts the paper cites (`data/results/`, ~2MB of JSON — including the frozen LLM-judge ratings, which are raw data and are *replayed*, never regenerated: the whole pipeline makes **zero API calls**), the five tracked pipeline specs, and the reference figures and tables the paper was built from.

## Reproducing the paper

### Getting started

One script does everything — sets up the Python environment, installs the PDF toolchain, downloads the datasets, runs every experiment in `src/`, and writes the paper to `paper/paper.pdf`:

```bash
git clone https://github.com/suisuss/actr-paper.git && cd actr-paper   # a git clone, not a zip
export HF_TOKEN=hf_...                     # PRISM is a gated dataset (see Data)
./run.sh
```

That is the full reproduction — many hours across ~40 stages, and much faster with a GPU. It is **resumable**: re-run `./run.sh` after any interruption and completed stages are skipped. `./run.sh status` shows the stage table without running anything.

To validate the paper **without** reproducing it — every table matches the shipped artifacts, every `<!-- src: -->` tag resolves to a real script, the artifacts are provenance-clean, and the committed `paper/paper.pdf` rebuilds with the same words — run `./run.sh check`. It takes seconds and needs no datasets, torch, or GPU (only `requirements-core.txt` and, for the PDF, the toolchain). How each gate works: [the five gates](#integrity-checking--the-five-gates).

`run.sh` in order: (1) creates `.venv` and installs `requirements.txt`; (2) installs the PDF toolchain via `install_pdf_tools.sh` (needs sudo — skip it and everything but the final PDF still builds); (3) runs `python main.py --fetch-data`, which downloads the corpora and runs the whole pipeline, ending with the PDF. To drive the pipeline directly instead of through `run.sh`, use `python main.py` (same subcommands: `status`, `launch`, `reset`).

### Run modes

`./run.sh` (equivalently `python main.py --fetch-data`) is the full reproduction: it re-embeds every corpus from raw text, regenerates every artifact, figure, and table, verifies them against the paper, and builds the PDF. Many hours; much faster with a GPU. There is one alternative:

**From cached embeddings** — the embedding stages and the 619MB original-SGD download skip themselves when their outputs already exist, so a prebuilt embeddings bundle (~1.1GB) extracted into `data/embeddings/` makes `requirements-core.txt` sufficient — no torch, much faster. No bundle is published yet: `embeddings_bundle` in `KIT_PROVENANCE.json` holds `null` for both URL and sha256, and will carry them if one is released. Note that `run.sh` installs the full `requirements.txt` regardless; to use the lighter set, run `pip install -r requirements-core.txt` and `python main.py` yourself.

### Requirements

- Linux or macOS (POSIX file locking; on Windows use WSL).
- Python >= 3.10 (developed on 3.13).
- A **git clone** — provenance stamping and dataset pinning use `.git`; GitHub's zip download will not work.
- Disk: ~700MB of downloaded corpora (or the ~1.1GB embeddings bundle instead, see Run modes), plus the Python environment and caches.
- A HuggingFace account with access to the gated [PRISM dataset](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) (accept its terms on that page), and `HF_TOKEN` set.
- GPU is optional. With an NVIDIA GPU visible, embedding stages use it automatically (~6x faster); everything else is CPU-bound either way, and the pipeline's statistics are identical on both. Force a device with `EXPERIMENT_DEVICE=cpu` or `EXPERIMENT_DEVICE=cuda`.
- No API access is needed at any point: the LLM-judge ratings are frozen in `data/results/` and replayed.
- PDF stage only: pandoc, XeLaTeX, and rsvg-convert — `./install_pdf_tools.sh` installs them (apt/brew). Without them every artifact, figure, and table still builds; the terminal `build_pdf` stage fails with instructions and resumes once they are installed.

### Reading your reproduction

Regenerated artifacts overwrite the vendored ones in the git working tree, so **`git diff` after a full run is a direct diff of your numbers against the paper's**. That is the intended way to read a reproduction — not a log to scan, but a diff that is empty when you have reproduced the paper.

**Expectation management**: with these exact package versions the run reproduces the printed cells. Re-embedding on different hardware or BLAS/torch builds can shift final decimals; the verify gate is strict cell equality, so treat a near-miss diff there as toolchain variance to inspect via `git diff`, not as a refutation.

One cross-paper printing convention to know: tables here print per-seed mean accuracies, while `gp-paper`'s tables print seed-ensembled accuracies, which run about a point higher. Same runs, two conventions — not a discrepancy.

## Claim -> script -> artifact

Every headline number in the paper traces to one stage script and one canonical artifact under `data/results/trace_h2h/`. Scripts live flat in `src/`; run them from `src/` (`cd src && python <script>`), or let `main.py` orchestrate everything.

| Result | Script | Artifact (`trace_h2h/...`) |
|---|---|---|
| Head-to-head, floors, designs, hybrid (Table 1 rows) | `trace_h2h.py --mode {features,floors,designs,hybrid}` | `features/`, `floors/`, `designs/`, `hybrid/` |
| Multi-seed + Holm families, linear probes | `multiseed_probes.py` | `multiseed_probes.json` |
| Table 1 core five-seed arms | `scoreboard_arms.py` | `scoreboard_arms.json` |
| Multi-embedder replication (GTE, bge) | `multimodel_eval.py` | `multimodel_eval.json` |
| Feature kill-switch ablations | `feature_ablation.py {uss_sgd,uss_mwoz}` | `feature_ablation_uss_*.json` |
| Combo arms (mean pooling + scalars) | `g0_combo.py {uss_sgd,uss_mwoz,prism}` | `g0_combo_*.json` |
| PRISM reversal + length deconfound | `prism_reversal.py`, `deconfound_length.py` | `prism_reversal.json`, `deconfound_length.json` |
| PCA-26 capacity control, dimension frontier | `dims_control.py`, `dim_frontier.py` | `dims_control.json`, `dim_frontier.json` |
| Role-split second task (service classification) | `service_transfer.py` | `service_transfer.json` |
| Feature extensions (hier, cross-encoder, surprise) | `extension_arms.py`, `hier_eval.py`, `cross_encoder_eval.py` (encode, then `eval`), `role_surprise.py` | `extension_arms_uss.json`, `hier_eval.json`, `cross_encoder_eval.json`, `role_surprise_sgd.json` |
| Degree ablation, MDL degree selection | `degree_ablation.py`, `mdl_degree.py` | `degree_ablation.json`, `mdl_degree_*.json` |
| Matched-dimensionality role-split(1) | `eqdim_probes.py` | `eqdim_probes.json` |
| §6.3 degree-matched interleaved control (prose result) | `interleaved_degree.py` | `interleaved_degree.json` |
| Containment + grafting | `containment_graft.py` | `containment_graft_uss_*.json` |
| Basis invariance (DCT/DST arms, subspace) | `basis_probes.py` | `basis_probes.json`, `basis_subspace.json` |
| Instrument resolution, seed-ensembled contrasts | `basis_calibration.py` | `basis_calibration.json` |
| Role-blind factorial cell | `roleblind_probes.py` | `roleblind_probes.json` |
| Judge stacking | `judge_stack.py` | `judge_stack.json` |
| Descriptive profiles (FDR) | `descriptive_profiles.py` | `descriptive_profiles.json` |
| Graft + role-split-vs-combo head-to-head | `rolesplit_graft.py` | `rolesplit_graft.json` |
| Role-split(1) across embedders | `rolesplit_embedders.py` | `rolesplit_embedders.json` |
| Corpus stats, role geometry | `ledger_stats.py` | `ledger_stats.json` |
| Annotator ceiling | `annotator_ceiling.py` | `annotator_ceiling.json` |
| LLM-judge hybrid (frozen ratings replay) | `llm_judge.py eval` | `llm_judge_eval.json` (ratings frozen in `judge_ratings.json`) |
| Template-1 prompt sensitivity | `judge_template1.py eval` | `judge_template1.json` |
| PCA-9 capacity control, SGD uniqueness | `capacity_uniqueness.py` | `capacity_uniqueness.json` |
| MWOZ uniqueness | `uniqueness_mwoz.py` | `uniqueness_mwoz.json` |
| §3.2 goal-proxy degeneracies (PRISM prose result) | `goal_proxy_prism.py` | `goal_proxy_prism.json` |
| §6.4 model-identity confound control (PRISM prose result) | `identity_probes_prism.py`, `identity_controls_prism.py`, `identity_augment_prism.py` | `identity_probes_prism.json`, `identity_controls_prism.json`, `identity_augment_prism.json` |
| All figures | `actr_figures.py` | `paper/figures/` |
| All evidence tables + draft check | `actr_tables.py` (`--check`), `verify_draft_tables.py` | `paper/tables/` |
| The PDF | `build_pdf.py` | `paper/paper.pdf` |

## How the kit keeps the paper honest

The premise of the kit: a paper should be confident it reproduces itself *while being developed*, not just at the end. Four mechanisms in `src/lib/` carry that — **resumability, caching, provenance, and integrity checking** — each enforced by the pipeline rather than left to discipline.

### The stage contract

Everything else hangs off this. Every script in `src/` is an importable module exporting up to three callables:

| Callable | Role |
|---|---|
| `run(**params) -> dict` | Computes and persists via `lib.campaign.save_result()` |
| `run_sample(max_samples, ...)` | Optional ephemeral sandbox alias — never writes results or caches |
| `main(argv)` | Thin CLI shim |

Importing a module never executes work. That is what lets `main.py` compile the stage DAG, and `src_check.py` resolve every claim in the paper to a script, without running anything.

### Resumability — a killed run is always safe to rerun

`lib/pipeline.py` is a detached, restart-recoverable runner. Stage state lives in `.reproduce/main/`: `pipeline.json` is the compiled stage list, `pipeline.state.json` tracks per-stage status, and `logs/` holds per-stage output.

Re-run `./run.sh` after any interruption and completed stages are skipped; `status` / `reset` / `launch` drive it directly. A file lock refuses a second concurrent supervisor on the same state, `lib/ioutils.py` writes every file atomically — an interrupted write cannot leave a partial file that `exists()` later reports as good — and `lib/runtime.py` caps CPU, memory, and process priority so a long run does not starve its host.

### Caching and invalidation — caches that expire themselves

Every expensive stage checkpoints under `data/experiment-caches/<script>/` through `lib.campaign.Checkpoint`, stamped in `producer.json` with the producing script's **logic digest**: a hash of the parsed AST, not of the file bytes. Comment- and formatting-only edits leave it unchanged and do not wipe cached checkpoints; only a change to the code itself re-invalidates. If the source does not currently parse (mid-edit), it falls back to a byte hash.

The logic digest is separate from the file sha256 recorded in each artifact's `_provenance`: one is for *cache correctness*, the other for *audit*.

**Never hand-delete a cache to force freshness**: cache validity derives from the code, not from memory of what you last edited. Embedding caches are the one exception — produce-once file guards; delete the `.npz` to force a re-embed.

### Provenance — every number knows which code produced it

`lib/provenance.py` stamps every artifact with the git commit, the dirty flag, the script sha256, the argv, and the seeds. `save_result()` writes both the canonical flat copy the tables read and an immutable provenance-stamped copy in the run history:

```
data/results/trace_h2h/<experiment>.json                      canonical latest (tracked)
data/results/trace_h2h/<experiment>/<dataset>/<config>/       immutable run history
```

Verify the whole set with `cd src && python -m lib.campaign verify`. The gate is that every artifact was built from a clean checkout: `git_dirty` must be `false`, and the commit and script sha256 must be present. **A number from a dirty tree does not count.**

These stamps come from the development history that produced the numbers; this repository ships as a single commit, so a shipped artifact's recorded commit and `script_sha256` identify the code that generated it rather than resolving against the files beside it. Regenerating an artifact (`./run.sh`) is what establishes that the shipped code reproduces the numbers — after which the artifact carries this repository's own commit.

### Integrity checking — the five gates

`./run.sh check` runs all five in seconds, on core dependencies only. CI runs them on every push and pull request (`.github/workflows/validate.yml`), so the paper cannot silently drift from its artifacts.

| Gate | What it proves |
|---|---|
| 1. Environment | `.venv` builds from `requirements-core.txt` alone — no torch, no GPU, no datasets needed to validate |
| 2. `verify_draft_tables.py` | Every table cell regenerated from `data/results/` equals the cell printed in `paper.md` (407 cells across 12 tables) |
| 3. `lib.campaign verify` | Every artifact is provenance-clean (clean checkout, commit and script sha256 present) |
| 4. `src_check.py` | Every `<!-- src: -->` tag in the paper resolves to a real script, every figure is tagged, every cited artifact is clean (83 tags) |
| 5. `verify_pdf_matches.py` | The committed `paper/paper.pdf` rebuilds from source with matching content |

Two HTML-comment conventions in `paper/paper.md` are what make gates 2 and 4 mechanical rather than manual: `<!-- src: script.py -->` after every generated figure or table names its producing script, and `<!-- table: name -->` before a generated table lets the gate look it up and diff a fresh regeneration against the printed cells.

The PDF gate is content-fidelity, not a byte diff: XeLaTeX output is not byte-reproducible across TeX Live versions, so `verify_pdf_matches.py` compares the *multiset of words* in the rebuild against the committed PDF — metadata, whitespace, hyphenation, word order, and pagination all normalize away, so it fails exactly when a word changed, i.e. when the PDF is stale. The rebuild must use the paper's main font (XCharter, installed by `install_pdf_tools.sh`), since `pdftotext` extraction of math subscripts and ligatures is font-dependent.

## Repository layout

```
run.sh                       one-command entry point (env, data, pipeline, PDF)
install_pdf_tools.sh         installs pandoc / xelatex / rsvg-convert
main.py                      the reproduction driver run.sh calls; STAGES is the DAG
KIT_PROVENANCE.json          dataset pins, embeddings-bundle checksum
requirements.txt             full deps (torch, sentence-transformers)
requirements-core.txt        light deps -- what `./run.sh check` and CI install

paper/
  paper.md                   the manuscript (source of truth for the PDF)
  paper.pdf                  built artifact, tracked and gated
  figures/                   generated SVGs, each tagged with its producing script
  tables/                    generated tables, regenerated and diffed by the gate

src/                         ~40 stage scripts, flat; run as `cd src && python <script>`
  trace_h2h.py               the head-to-head analysis core (features/floors/designs/hybrid)
  scoreboard_arms.py         Table 1's core five-seed arms
  rolesplit_*.py, eqdim_probes.py, interleaved_degree.py
                             the role-split result and its degree-matched controls
  basis_probes.py, basis_calibration.py
                             basis invariance and the measured instrument resolution
  dims_control.py, dim_frontier.py, capacity_uniqueness.py
                             capacity vs information controls
  llm_judge.py, judge_stack.py, judge_template1.py
                             LLM-judge arms -- frozen ratings, replayed, zero API calls
  multimodel_eval.py, rolesplit_embedders.py
                             GTE-small / bge-base replication
  descriptive_profiles.py    the descriptive lane (Spearman + FDR)
  actr_tables.py             generates every table the paper prints
  actr_figures.py            generates every figure
  verify_draft_tables.py     the draft-vs-artifact gate
  verify_pdf_matches.py      the committed-PDF content check
  src_check.py               resolves every <!-- src: --> tag to a real script
  build_pdf.py               pandoc/XeLaTeX build
  lib/                       shared infrastructure (below)

data/
  pipelines/                 tracked stage specs
  results/trace_h2h/         canonical artifacts (tracked; what tables/figures read)
  raw/, embeddings/          downloaded corpora and embeddings (regenerated)
  experiment-caches/         checkpointed intermediates, keyed by script digest
```

### `src/lib/` — the shared infrastructure

| Module | What it does |
|---|---|
| `encoder.py` | **The projection core.** Polynomial/frequency decomposition of turn embeddings across bases — Chebyshev (`c`), Fourier magnitude (`f`), DCT (`d`), DST (`s`), Gram (`g`) — with the `{basis}{degree}{type}` component naming (e.g. `g0`, `c1a`) the configs use. |
| `config.py` | Encoding configuration schema: embedding model plus component combination, and the registry of every named arm. |
| `trace_features.py` | TRACE-geometric, reimplemented from Appendix A (Table 3) of Gooding and Grefenstette — the comparison suite. |
| `uss.py` | USS corpus loader (Sun et al., 2021). |
| `embedding_cache.py` | Per-sequence raw embedding cache, so different pooling strategies never re-embed. |
| `campaign.py` | The stage contract: `run()` / `run_sample()` / `main()`, plus `save_result()` and `Checkpoint`. Read its docstring before writing a stage. |
| `experiment.py` | Manifest-driven dispatch, discovery, and index generation. |
| `pipeline.py` | Detached, restart-recoverable pipeline runner — what makes a killed run safe to rerun. |
| `provenance.py` | Stamps git commit, dirty flag, script sha256, and argv on every result. |
| `results.py` | Results registry: JSON index plus immutable per-run result files. |
| `runtime.py` | CPU/memory/priority limits so a run does not starve the host. |
| `ioutils.py` | Atomic file writes, so an interrupted run cannot leave a partial file that `exists()` reports as good. |
| `device.py` | GPU-if-present device selection for the embedding stages. |

## Troubleshooting

- **PRISM 401/403**: accept the dataset terms on its HF page while logged in, then `export HF_TOKEN=` a token of that account.
- **Huge torch download**: CPU-only machines should use the extra index in the `requirements.txt` header; embeddings-bundle users need no torch.
- **"pipeline already running"**: another supervisor holds the lock (`.reproduce/.../pipeline.state.lock`); `status` shows progress.
- **PDF stage fails**: `./install_pdf_tools.sh`, then re-run — it resumes at that stage. Without the XCharter font (texlive-fonts-extra) the build falls back to pandoc's default font.
- **Provenance errors about git**: you are not in a git clone, or the repo has no commits — re-clone, don't unzip.
- **A gate fails after an honest edit**: expected — it means the committed artifact or PDF is now stale. Re-run the pipeline and commit what it regenerates.

## Citation and license

The preprint is in preparation; this entry will be replaced with the posted version when it is available.

```bibtex
@unpublished{sussmilch2026trajectory,
  title  = {Analysis of Conversation Trajectory Representations:
            Orthogonal Projections versus Scalar Geometry for
            Satisfaction Prediction},
  author = {Sussmilch, Jacob},
  year   = {2026},
  note   = {Preprint in preparation. Reproduction kit:
            https://github.com/suisuss/actr-paper}
}
```

When citing a reproduction, cite the commit you ran — the artifacts and the PDF are gated against each other at every commit.

Code and manuscript are MIT-licensed (see `LICENSE`), copyright Heya Enterprises Pty Ltd. The corpora are not redistributed here and remain under their own terms (see [Data](#data)). TRACE-geometric is our reimplementation of the feature suite described by Gooding and Grefenstette (2025), not their code.
