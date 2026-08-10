"""Verify the committed paper/paper.pdf reproduces from source.

Rebuilds the PDF with the same toolchain build_pdf.py uses, then
checks that the rebuild carries the SAME WORDS as the committed
paper/paper.pdf - which come from paper.md, the tables, and the figure
captions. The gate fails exactly when the shipped PDF is stale with
respect to the sources it is built from.

This is a content-fidelity gate, not a byte-identity gate: XeLaTeX PDFs
are not byte-reproducible across TeX Live versions (font subset tags,
object layout, the trailer /ID) and carry per-build metadata
(creation/mod dates), so a raw diff - even minus metadata - fails
spuriously. The comparison instead normalizes each side to a multiset of
words: NFKC (decomposes ligatures and space variants), curly->straight
quotes, drop every whitespace/hyphen/dash character, split, and count.
Word order, line breaks, and pagination are all irrelevant; a changed,
added, or removed word is not. (A pure permutation of identical words is
the one thing this does not flag - no honest paper.md edit produces one.)

It is NOT fully font-independent, and cannot be: glyph-level pdftotext
extraction is font-dependent for math subscripts (G_u + G_k reads as
different characters under a fallback font) and for the Unicode encoding
of ligatures. The build must therefore use the paper's main font,
XCharter - install_pdf_tools.sh installs it (texlive-fonts-extra) and CI
registers it with fontconfig. When XCharter is genuinely absent the build
falls back and this check would report spurious word differences; the
fix is to install the font, not to loosen the check.

Exit 0 = words match (or the toolchain is absent -> skipped).
Exit 1 = the committed PDF is stale.

Run from src/:  python verify_pdf_matches.py
"""
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import build_pdf as bpf

REPO = Path(__file__).resolve().parent.parent
PDF = REPO / "paper" / "paper.pdf"

# curly -> straight quotes (pandoc smart-quote rendering can vary by
# version); every whitespace, hyphen, dash, and minus variant is stripped.
_QUOTES = str.maketrans({"‘": "'", "’": "'",
                         "“": '"', "”": '"'})
_DASHES = re.compile(r"[-­‐‑‒–—―−]")


def _pdftotext(path):
    return subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"],
                          capture_output=True, text=True, check=True).stdout


def _pagecount(path):
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True,
                         text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return None


def _words(text):
    """Font/order/metadata-independent multiset of words.

    NFKC decomposes font-dependent glyph encodings that carry the same
    text (Latin ligatures the fallback font emits as single codepoints;
    non-breaking / thin spaces). Dropping dashes absorbs end-of-line
    hyphenation, which varies by font.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _DASHES.sub("", text)
    return Counter(text.split())


def _fmt(counter, limit=30):
    items = sorted(counter.items())
    shown = ", ".join(f"{w!r}×{n}" if n > 1 else repr(w)
                      for w, n in items[:limit])
    if len(items) > limit:
        shown += f", ... (+{len(items) - limit} more)"
    return shown


def main():
    if not PDF.exists():
        print(f"[pdf-match] committed PDF missing: {PDF}", file=sys.stderr)
        return 1

    missing = [t for t in ("pandoc", "xelatex", "rsvg-convert")
               if shutil.which(t) is None]
    if missing:
        print("[pdf-match] skipped: PDF toolchain absent (" +
              ", ".join(missing) + "); run ./install_pdf_tools.sh")
        return 0
    if shutil.which("pdftotext") is None or shutil.which("pdfinfo") is None:
        print("[pdf-match] skipped: poppler-utils absent (pdftotext/pdfinfo); "
              "install poppler-utils to enable the check")
        return 0

    committed = PDF.read_bytes()
    try:
        want_words, want_pages = _words(_pdftotext(PDF)), _pagecount(PDF)
        if bpf.main() != 0:                       # rebuilds paper.pdf in place
            print("[pdf-match] rebuild failed", file=sys.stderr)
            return 1
        got_words, got_pages = _words(_pdftotext(PDF)), _pagecount(PDF)
    finally:
        PDF.write_bytes(committed)                # leave the working tree clean

    # Page count is font-dependent (a fallback font repaginates), so it is
    # informational, not a failure condition.
    if want_pages != got_pages:
        print(f"[pdf-match] note: page count differs "
              f"(committed={want_pages}, rebuilt={got_pages}) - expected when "
              f"the build falls back from XCharter; comparing words only")

    if want_words != got_words:
        only_committed = want_words - got_words
        only_rebuilt = got_words - want_words
        print("[pdf-match] WORDS differ (order, whitespace, hyphenation, "
              "ligatures, pagination all ignored):", file=sys.stderr)
        if only_committed:
            print("  committed has, rebuilt lacks: " + _fmt(only_committed),
                  file=sys.stderr)
        if only_rebuilt:
            print("  rebuilt has, committed lacks: " + _fmt(only_rebuilt),
                  file=sys.stderr)
        print("[pdf-match] FAIL: committed paper.pdf is stale - rebuild and "
              "commit it (python main.py, or ./run.sh check).", file=sys.stderr)
        return 1

    print(f"[pdf-match] OK: committed paper.pdf reproduces "
          f"({want_pages} pages, {sum(want_words.values())} words match "
          f"its sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
