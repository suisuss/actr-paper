#!/usr/bin/env bash
#
# Reproduce the paper, or validate what is already shipped.
#
#   ./run.sh            full reproduction (many hours; needs HF_TOKEN for PRISM):
#                       env + PDF toolchain + datasets + every experiment in
#                       src/, ending with the PDF at paper/paper.pdf.
#   ./run.sh check      validation only (seconds, no datasets/torch/GPU): checks
#                       the shipped paper against the shipped artifacts and
#                       rebuilds the PDF from them.
#   ./run.sh status     show the stage table without running.
#   ./run.sh reset      forget stage state.
#
# The full run is resumable: re-run after any interruption and completed
# stages are skipped. See README.md.
set -euo pipefail
cd "$(dirname "$0")"

# --------------------------------------------------------------- validation
if [ "${1:-}" = "check" ]; then
    echo "==> [1/5] Python environment (.venv, core dependencies only)"
    [ -d .venv ] || python3 -m venv .venv
    # shellcheck disable=SC1091
    . .venv/bin/activate
    python -m pip install --quiet --upgrade pip
    pip install --quiet -r requirements-core.txt

    rc=0
    echo "==> [2/5] Tables match the paper (data/results -> paper/paper.md)"
    ( cd src && python verify_draft_tables.py ) || rc=1

    echo "==> [3/5] Artifacts are provenance-clean (git commit + script sha256)"
    ( cd src && python -m lib.campaign verify ) || rc=1

    echo "==> [4/5] Paper src-tags resolve (every [src:] names a real script)"
    ( cd src && python src_check.py | tail -1 ) || rc=1

    echo "==> [5/5] Committed PDF reproduces (rebuild matches paper/paper.pdf)"
    if ./install_pdf_tools.sh >/dev/null 2>&1 \
            || command -v pandoc >/dev/null 2>&1; then
        ( cd src && python verify_pdf_matches.py ) || rc=1
    else
        echo "    skipped: PDF toolchain absent (run ./install_pdf_tools.sh)"
    fi

    echo
    if [ "$rc" -eq 0 ]; then
        echo "VALIDATION PASSED - the paper is consistent with its artifacts."
    else
        echo "VALIDATION FAILED - see the messages above." >&2
    fi
    exit "$rc"
fi

# ---------------------------------------------------------- full reproduction
echo "==> [1/4] Python environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
pip install -r requirements.txt

echo "==> [2/4] PDF toolchain"
if ! ./install_pdf_tools.sh; then
    echo "    PDF tools could not be installed - every artifact, figure, and" >&2
    echo "    table still builds; only the final PDF stage needs them, and it" >&2
    echo "    resumes once they are present." >&2
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "    note: HF_TOKEN is not set - PRISM is a gated dataset and its" >&2
    echo "    download will fail without it (see README.md, Data)." >&2
fi

echo "==> [3/4] Datasets + full experiment pipeline"
python main.py --fetch-data "$@"

echo "==> [4/4] Done"
if [ -f paper/paper.pdf ]; then
    echo "    Paper: paper/paper.pdf"
fi
