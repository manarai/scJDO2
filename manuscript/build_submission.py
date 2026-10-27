"""Build bioRxiv submission files (PDF + DOCX) for manuscript v57.

Embeds each figure (600-dpi PNG) above its legend in the "Figure legends" section and
writes manuscript/submission/scJDO_manuscript_v57.{pdf,docx}. Requires pandoc and tectonic.

    python manuscript/build_submission.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "scJDO_manuscript_v57.md"
OUT = HERE / "submission"
FIG = HERE / "figures"


LATEX = "```{{=latex}}\n{}\n```"


def with_figures(text: str) -> str:
    """Put every figure on its own page above its legend.

    The legend is set in footnote size and the image height is capped by the legend
    length, so image and legend fit on one page (PDF only; DOCX ignores the LaTeX)."""
    def img(m: re.Match) -> str:
        supp, num, legend = bool(m.group(1)), m.group(2), m.group(0)
        path = FIG / ("supplementary" if supp else "main") / f"Fig{'S' if supp else ''}{num}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        # ~150 characters per footnote-size line on letter paper, ~1.65 % of text height each, plus slack
        share = min(0.80, max(0.35, 0.95 - 0.0165 * (len(legend) / 150 + 4)))
        return (f"\\newpage\n\n![]({path}){{width=100% height={share:.0%}}}\n\n"
                f"{LATEX.format(chr(92) + 'begingroup' + chr(92) + 'footnotesize')}\n\n{legend}\n\n"
                f"{LATEX.format(chr(92) + 'par' + chr(92) + 'endgroup')}")
    return re.sub(r"^\*\*(Supplementary )?Figure S?(\d+) \|.*$", img, text, flags=re.M)


def unbreakable_references(text: str) -> str:
    """Keep each reference on one page (no DOI split across a page break)."""
    return text.replace("\n## References\n",
                        "\n## References\n\n" + LATEX.format(chr(92) + "interlinepenalty=10000") + "\n", 1)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    body = unbreakable_references(with_figures(SRC.read_text())).replace("ᵀ", "^T^")
    tmp = OUT / "_build.md"
    tmp.write_text(body)
    common = ["pandoc", str(tmp), "--from", "markdown+pipe_tables+tex_math_dollars", "--resource-path", str(HERE)]
    subprocess.run(common + ["-o", str(OUT / "scJDO_manuscript_v57.pdf"), "--pdf-engine=tectonic",
                             "-V", "mainfont=Arial Unicode MS", "-V", "geometry:margin=2.2cm",
                             "-V", "fontsize=10pt", "-V", "linestretch=1.15", "-V", "colorlinks=true"], check=True)
    subprocess.run(common + ["-o", str(OUT / "scJDO_manuscript_v57.docx")], check=True)
    tmp.unlink()
    print("written:", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
