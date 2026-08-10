"""Build the preview PDF (paper.pdf) - terminal pipeline stage.

Python port of this paper's half of paper/build_pdf.sh (the shell
script builds both papers at once; pipeline stages are single-purpose
and run as [interpreter, script], so the stage needs a python entry
point). Requires pandoc, xelatex (texlive-xetex) and rsvg-convert
(librsvg, for the SVG figures); exits nonzero with install
instructions when they are missing. This is the last stage of the
chain, so a failure here leaves every artifact, figure, and table
already regenerated, and re-running the driver resumes exactly here.

The XCharter main font is probed rather than assumed: when neither
fontconfig nor kpathsea can find it, the build falls back to pandoc's
default main font (layout shifts slightly; content is identical).
"""
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
PAPERS = WORK.parent / "paper"
PAPER = PAPERS

# tool -> Debian/Ubuntu package(s) that provide it
TOOLS = {"pandoc": "pandoc",
         "xelatex": "texlive-xetex texlive-fonts-recommended",
         "rsvg-convert": "librsvg2-bin"}


def _have_xcharter():
    for probe in (["fc-list", ":family"], ["kpsewhich", "XCharter-Roman.otf"]):
        try:
            out = subprocess.run(probe, capture_output=True, text=True,
                                 timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        if "XCharter" in out:
            return True
    return False


def main(argv=None):
    missing = [t for t in TOOLS if shutil.which(t) is None]
    if missing:
        pkgs = " ".join(TOOLS[t] for t in missing)
        print("[build_pdf] missing tool(s): " + ", ".join(missing) + "\n"
              "  install (Debian/Ubuntu):\n"
              f"    sudo apt install {pkgs} fonts-dejavu-core\n"
              "  then re-run the reproduction driver - every other stage is\n"
              "  already done, so it resumes at this stage.", file=sys.stderr)
        return 1

    opts = ["--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            "-V", "monofont=DejaVu Sans Mono",
            "--include-in-header", str(PAPERS / "pandoc-tableshrink.tex"),
            "--include-in-header", str(PAPERS / "letterhead.tex"),
            "--lua-filter", str(PAPERS / "tablenote.lua")]
    if _have_xcharter():
        opts += ["-V", "mainfont=XCharter"]
    else:
        print("[build_pdf] XCharter font not found - using pandoc's default "
              "main font (install texlive-fonts-extra for XCharter)")

    r = subprocess.run(["pandoc", "paper.md", "-o", "paper.pdf"] + opts,
                       cwd=PAPER)
    if r.returncode == 0:
        print(f"[build_pdf] built {PAPER / 'paper.pdf'}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
