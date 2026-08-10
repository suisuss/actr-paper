#!/usr/bin/env bash
# Install the PDF build toolchain for the terminal build_pdf stage:
# pandoc, XeLaTeX, the SVG rasterizer, and the paper's fonts (XCharter
# ships in texlive-fonts-extra; DejaVu Sans Mono is the mono font).
# poppler-utils (pdftotext/pdfinfo) drives verify_pdf_matches.py, the
# check that the committed paper/paper.pdf reproduces from source.
#
# The pipeline runs fine without any of this - every artifact, figure,
# and table still builds; only the final PDF stage needs these tools,
# and it resumes cleanly once they are installed.
#
# Note: texlive-fonts-extra is large (several hundred MB). Skip it and
# the build falls back to pandoc's default main font (content identical,
# layout shifts slightly).
set -euo pipefail

PKGS_APT="pandoc texlive-xetex texlive-fonts-recommended texlive-fonts-extra librsvg2-bin fonts-dejavu-core poppler-utils"

# Idempotent: nothing to do if the tools are already on PATH (so run.sh
# and re-runs never invoke sudo apt unnecessarily).
if command -v pandoc >/dev/null 2>&1 \
        && command -v xelatex >/dev/null 2>&1 \
        && command -v rsvg-convert >/dev/null 2>&1 \
        && command -v pdftotext >/dev/null 2>&1; then
    echo "PDF toolchain already present (pandoc, xelatex, rsvg-convert, pdftotext)."
    exit 0
fi

if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    $SUDO apt-get update
    # shellcheck disable=SC2086
    $SUDO apt-get install -y $PKGS_APT
elif command -v brew >/dev/null 2>&1; then
    brew install pandoc librsvg poppler
    echo "For XeLaTeX on macOS install MacTeX (or BasicTeX + collection-fontsextra):"
    echo "  brew install --cask mactex-no-gui"
else
    echo "No supported package manager found (apt-get or brew)." >&2
    echo "Install manually: pandoc, XeLaTeX (TeX Live), rsvg-convert (librsvg)," >&2
    echo "the XCharter font, and DejaVu Sans Mono." >&2
    exit 1
fi

echo
echo "PDF toolchain installed. Re-run the driver to build the PDF:"
echo "  python main.py"
