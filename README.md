# actr-paper reproduction kit

This repository reproduces Heya's 'Analysis of Conversation Trajectory Representations: Orthogonal Projections versus Scalar Geometry for Satisfaction Prediction' paper and its findings. It holds the manuscript source (`paper/paper.md`, built to `paper/paper.pdf`), every result artifact the paper cites, the ~40 scripts that produced them, and a pipeline that regenerates all of it from public data — plus gates that fail whenever a printed number and its artifact disagree.

## Run it

```bash
git clone https://github.com/suisuss/actr-paper.git && cd actr-paper   # a git clone, not a zip
export HF_TOKEN=hf_...     # PRISM is a gated dataset (see Data)
./run.sh check             # seconds: validate the paper against its shipped artifacts
./run.sh                   # many hours: regenerate every artifact from raw text, then the PDF
```

`./run.sh` creates `.venv`, installs the PDF toolchain (`install_pdf_tools.sh`, needs sudo — skip it and everything but the final PDF still builds), fetches the three pinned corpora, runs the ~40 stages, and writes `paper/paper.pdf`. It is **resumable**: re-run after any interruption and completed stages are skipped. `python main.py` drives the pipeline directly (`status`, `launch`, `reset`).

Regenerated artifacts overwrite the committed ones in place, so **`git diff` after a full run is your reproduction** — an empty diff means you reproduced the paper. Different hardware or BLAS builds can shift final decimals; the gate is strict cell equality, so read a near-miss as toolchain variance in `git diff`, not a refutation.

`./run.sh check` needs no datasets, no torch, and no GPU — only `requirements-core.txt`. It runs five gates, the same ones CI runs on every push and pull request (`.github/workflows/validate.yml`):

| Gate | What it proves |
|---|---|
| 1. Environment | `.venv` builds from `requirements-core.txt` alone |
| 2. `verify_draft_tables.py` | Every table cell regenerated from `data/results/` equals the cell printed in `paper.md` (407 cells across 12 tables) |
| 3. `lib.campaign verify` | Every artifact is provenance-clean (clean checkout, commit and script sha256 present) |
| 4. `src_check.py` | Every `<!-- src: -->` tag in the paper resolves to a real script, every figure is tagged, every cited artifact is clean (83 tags) |
| 5. `verify_pdf_matches.py` | The committed `paper/paper.pdf` rebuilds from source with matching content |

Unexplained mismatches belong in an issue, with the failing gate's output and the `git diff` of the artifact in question.

### Requirements

- Linux or macOS (POSIX file locking; on Windows use WSL); Python >= 3.10 (developed on 3.13).
- A **git clone** — provenance stamping and dataset pinning use `.git`; GitHub's zip download will not work.
- A HuggingFace account with access to the gated [PRISM dataset](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) (accept its terms on that page), and `HF_TOKEN` set.
- Disk: ~700MB of downloaded corpora, plus the Python environment and caches.
- GPU optional: embedding stages use it automatically, everything else is CPU-bound, and the statistics are identical either way. Force a device with `EXPERIMENT_DEVICE=cpu|cuda`.
- No API access at any point: the LLM-judge ratings are frozen in `data/results/` and replayed, so the pipeline makes zero API calls.
- PDF stage only: pandoc, XeLaTeX, rsvg-convert — `./install_pdf_tools.sh` installs them (apt/brew).

## The paper

### What it argues

A growing line of work scores conversation quality from the geometry of how a dialogue moved, most prominently TRACE (Gooding and Grefenstette, 2025), which predicts conversation-level satisfaction from roughly thirty hand-crafted scalar features — self-similarity of model turns, drift from a stated goal, turn-to-turn volatility, trends in relevance — fed to a random forest. Every such feature embodies the same unexamined decision: it is a *scalarization* of an underlying vector object, almost always through a distance or a cosine, so the trajectory is reduced feature by feature to hand-picked one-dimensional summaries before any learning happens. The paper argues for the older and simpler alternative — fix an orthogonal basis, project, and hand the learner the *coefficient vectors*, with no selection step, a quantified residual, and no way to lose direction information you did not know you needed — and that the structure of that projection, splitting the trajectory by speaker, matters more than any of the choices usually argued about.

For L2-normalized turn embeddings `e_1, ..., e_T`, with `phi_k` the degree-`k` Gram polynomial on the uniform grid and `K = min(3, T-1)`:

```
G_k = sum_i phi_k(i) * e_i  /  sum_i phi_k(i)^2
```

Two identities carry the interpretive weight: **`G_0` is exactly the mean** (mean pooling is the degree-0 case) and **`G_1` is exactly the least-squares drift**. The **role-split** variant projects each speaker's subsequence on its own grid and concatenates, recovering the joint midline and speaker offset exactly (`c_k = (G_k^u + G_k^m)/2`, `d_k = G_k^u - G_k^m`); this also removes the speaker-alternation aliasing that a single interleaved fit spends its low-order structure on. The arms compared, all consumed by the same regressor:

| Arm | Dim | What it is |
|---|---|---|
| Interleaved projection | 1,536 | `[G_0 .. G_3]` over the whole dialogue |
| **Role-split(1)** | 1,536 | Per-role degrees 0-1: each speaker's own mean and drift. **The recommendation.** |
| Mean pooling | 384 | Identically `G_0` — the content-only ablation |
| Shape only | 1,152 | `G_1`-`G_3` — the fairest shape-vs-shape contrast against TRACE-geometric |
| TRACE-geometric | ~26-30 | The hand-crafted scalar suite, reimplemented (`lib/trace_features.py`) |
| Projected TRACE series | 20 | Gram degrees 0-3 of TRACE's own five scalar series — the subsumption arm |
| Scalarized coefficients | 9 | Norms and angles of the vector coefficients — the representation deliberately scalarized |

Code: `src/lib/encoder.py` and `src/lib/config.py` (the registry of every named arm). Full treatment: `paper/paper.md` §4, with the TRACE-geometric reimplementation in §3 and documented feature by feature in Appendix A.

### What it finds

1. **Which representation wins is regime-dependent.** The projection beats TRACE's geometric features by 5-12 points on all three corpora — but on short, first-person-rated chats the trajectory-shape channels collapse and hand-crafted scalars become a genuine complement. This is a difference of corpus and label construct, not of conversation length: length is ruled out by an explicit deconfound arm.
2. **The projection's structure matters more than its basis.** Read as one interleaved sequence, the projection blends both speakers into shared low-degree coefficients. Splitting by speaker — giving the learner each speaker's content on its own path — is the recommended representation: training-free, and never distinguishably worse than any measured alternative in either regime.
3. **What the split buys is the content separation.** The speaker-alternation aliasing it also removes is a representation-fidelity fact, not the source of the accuracy.
4. **The advantage is informational, not capacity.** Compressed to TRACE's own 26 dimensions the role-split projection still beats it, while a random 26-dim projection is significantly worse.
5. **The winners here are plausibly the more gameable ones.** This is offline prediction, not reward modeling: a reward built on content (mean pooling, or the projection coefficients) is more directly Goodhart-hackable than content-agnostic scalar geometry, so the ranking measured here need not transfer to the reward setting, and may invert.

Scope and limitations are in Section 8 of the paper.

## Data

Three source corpora, downloaded on demand. Nothing is fetched implicitly; `python main.py --fetch-data` acquires whatever is missing, each pinned in `KIT_PROVENANCE.json` to the exact revision the paper used.

| Corpus | Source | Size | Pinned revision | Terms |
|---|---|---|---|---|
| USS (SGD + MWOZ satisfaction labels) | [sunnweiwei/user-satisfaction-simulation](https://github.com/sunnweiwei/user-satisfaction-simulation) | 18MB | `8ba06bd92570fd94d3ec0cb828aad45172be71b2` | No license file in the upstream repo |
| PRISM | [HannahRoseKirk/prism-alignment](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) | 58MB | `18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6` | Gated — accept its terms on the HF page, then set `HF_TOKEN` |
| Original SGD | [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | 619MB | `e852981ae34990f4358979625854259302feaa78` | Upstream repository's own terms; needed once, to build a text-match map |

They yield the paper's three evaluation corpora, whose label constructs are what the regime finding turns on:

| Corpus | Conversations | Turns | Label construct | CV |
|---|---|---|---|---|
| USS-SGD | 1,000 | 25,666 (mean 25.7) | 5-point satisfaction, mean over 3+ third-party annotators | 5-fold by dialogue |
| USS-MWOZ | 1,000 | 22,108 (mean 22.1) | same | 5-fold by dialogue |
| PRISM | 7,951 (1,395 users) | 54,212 (mean 6.8) | first-person helpfulness, rescaled 1-5 | `GroupKFold(5)` by user |

**None of the three is vendored here, by decision**: the USS repository carries no license file, so it is not ours to redistribute; PRISM is gated behind terms each user must accept in their own name; and original SGD is 619MB. Each is fetched from the original source at the pinned revision, so a reproduction reads exactly the bytes the paper read.

Turn embeddings are the deliberately least-engineered component — identical off-the-shelf embeddings feed every arm, TRACE-geometric included, so no result can be attributed to turn-representation quality. `sentence-transformers/all-MiniLM-L6-v2` (384-d) is the primary encoder for every headline arm; `thenlper/gte-small` (384-d) and `BAAI/bge-base-en-v1.5` (768-d) replicate the role-split result. The protocol throughout: `RandomForestRegressor` (300 trees, `min_samples_leaf` 2) on each arm's representation, 5-fold out-of-fold predictions (user-grouped on PRISM), pairwise preference accuracy over conversation pairs with mean-rating gap >= 1.0 (primary) or 0.5 (secondary), five seeds, with inferential contrasts on seed-ensembled predictions and m-out-of-n dialogue-resampling bootstrap CIs — a measured instrument resolution of ±2.3-2.6 points.

**Vendored here**: the canonical result artifacts the paper cites (`data/results/`, ~2MB of JSON, including the frozen LLM-judge ratings — raw data, replayed and never regenerated), the five tracked pipeline specs, and the reference figures and tables. Embedding stages skip themselves when their outputs exist, so a prebuilt bundle dropped into `data/embeddings/` makes `requirements-core.txt` sufficient; none is published yet, hence the `null` `embeddings_bundle` entry in `KIT_PROVENANCE.json`.

## Claim to Script to Artifact

Every headline number traces to one stage script and one canonical artifact under `data/results/trace_h2h/`. Scripts live flat in `src/`; run them from `src/` (`cd src && python <script>`), or let `main.py` orchestrate everything. Each artifact's `_provenance` block names the commit, script sha256, argv, and seeds that produced it.

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

Tables here print per-seed mean accuracies, which run about a point below the seed-ensembled convention.

## How the kit keeps the paper honest

The premise: a paper should be confident it reproduces itself *while being developed*, not just at the end. Four mechanisms in `src/lib/` carry that, enforced by the pipeline rather than left to discipline.

**The stage contract.** Every script in `src/` is an importable module exporting up to three callables: `run(**params) -> dict` computes and persists via `lib.campaign.save_result()`; `run_sample(max_samples, ...)` is an optional ephemeral sandbox alias that never writes results or caches; `main(argv)` is a thin CLI shim. Importing a module never executes work — which is what lets `main.py` compile the stage DAG, and `src_check.py` resolve every claim in the paper to a script, without running anything.

**Resumability.** `lib/pipeline.py` is a detached, restart-recoverable runner; stage state lives in `.reproduce/main/` (compiled stage list, per-stage status, per-stage logs). A file lock refuses a second concurrent supervisor, `lib/ioutils.py` writes every file atomically so an interrupted write cannot leave a partial file that `exists()` later reports as good, and `lib/runtime.py` caps CPU, memory, and process priority so a long run does not starve its host. A killed run is always safe to rerun.

**Caching that expires itself.** Expensive stages checkpoint under `data/experiment-caches/<script>/` through `lib.campaign.Checkpoint`, stamped in `producer.json` with the producing script's **logic digest**: a hash of the parsed AST, not of the file bytes, so comment- and formatting-only edits do not wipe cached checkpoints while a change to the code itself re-invalidates (a file that does not currently parse falls back to a byte hash). That digest is for *cache correctness*; the file sha256 in each artifact's `_provenance` is for *audit*. **Never hand-delete a cache to force freshness**: validity derives from the code, not from memory of what you last edited. Embedding caches are the one exception — produce-once file guards; delete the `.npz` to force a re-embed.

**Provenance.** `lib/provenance.py` stamps every artifact with the git commit, dirty flag, script sha256, argv, and seeds. `save_result()` writes both the canonical flat copy the tables read (`data/results/trace_h2h/<experiment>.json`) and an immutable stamped copy in the run history (`.../<experiment>/<dataset>/<config>/`). Verify the set with `cd src && python -m lib.campaign verify`: `git_dirty` must be `false`, and the commit and script sha256 present. **A number from a dirty tree does not count.** The kit ships as a single commit, so a shipped artifact's recorded commit identifies the code that generated it rather than resolving against the files beside it; regenerating it is what establishes that the shipped code reproduces the number.

**Integrity checking.** Two HTML-comment conventions in `paper.md` make gates 2 and 4 mechanical rather than manual: `<!-- src: script.py -->` after every generated figure or table names its producing script, and `<!-- table: name -->` before a generated table lets the gate look it up and diff a fresh regeneration against the printed cells. The PDF gate is content-fidelity, not a byte diff — XeLaTeX output is not byte-reproducible across TeX Live versions, so `verify_pdf_matches.py` compares the *multiset of words* in the rebuild against the committed PDF; metadata, whitespace, hyphenation, word order, and pagination all normalize away, so it fails exactly when a word changed. The rebuild must use the paper's main font (XCharter, installed by `install_pdf_tools.sh`), since `pdftotext` extraction of math subscripts and ligatures is font-dependent.

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
  experiment-caches/         checkpointed intermediates, keyed by script logic digest
```

For `src/lib/`:
- `encoder.py` is the projection core — polynomial/frequency decomposition of turn embeddings across bases (Chebyshev `c`, Fourier magnitude `f`, DCT `d`, DST `s`, Gram `g`), with the `{basis}{degree}{type}` component naming the configs use; `config.py` is the encoding schema and the registry of every named arm.
- `trace_features.py` is TRACE-geometric, reimplemented from Appendix A (Table 3) of Gooding and Grefenstette — the comparison suite.
- `campaign.py` holds the stage contract, `save_result()`, and `Checkpoint` — read its docstring before writing a stage.
- `experiment.py` (manifest-driven dispatch), `pipeline.py` (the restart-recoverable runner), `provenance.py`, `results.py` (registry), `runtime.py` (CPU/memory/priority caps), `ioutils.py` (atomic writes), and `device.py` (GPU-if-present) are the harness.
- `uss.py` loads the USS corpus (Sun et al., 2021); `embedding_cache.py` is the per-sequence raw embedding cache, so different pooling strategies never re-embed.

## Troubleshooting

- **PRISM 401/403**: accept the dataset terms on its HF page while logged in, then `export HF_TOKEN=` a token of that account.
- **Huge torch download**: CPU-only machines should use the extra index in the `requirements.txt` header.
- **"pipeline already running"**: another supervisor holds the lock (`.reproduce/.../pipeline.state.lock`); `status` shows progress.
- **PDF stage fails**: `./install_pdf_tools.sh`, then re-run — it resumes at that stage. Without the XCharter font (texlive-fonts-extra) the build falls back to pandoc's default font.
- **Provenance errors about git**: you are not in a git clone, or the repo has no commits — re-clone, don't unzip.
- **A gate fails after an honest edit**: expected — the committed artifact or PDF is now stale. Re-run the pipeline and commit what it regenerates.

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
