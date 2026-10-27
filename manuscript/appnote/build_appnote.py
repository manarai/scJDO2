"""Build the Bioinformatics Application Note (PDF + DOCX) with Fig. 1 placed after Section 3.

    python manuscript/appnote/build_appnote.py      # needs pandoc and tectonic
"""
from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "scJDO2_flow_appnote.md"
FIG = HERE.parent / "figures" / "appnote" / "Fig1.png"


def main() -> None:
    text = SRC.read_text()
    legend = text.split("## Figure legend\n", 1)[1].split("\n## ", 1)[0].strip()
    text = text.replace("## Figure legend\n\n" + legend + "\n\n", "")
    figure = f"![]({FIG}){{width=100%}}\n\n{legend}\n\n"
    text = text.replace("## 4 Usage notes and limitations", figure + "## 4 Usage notes and limitations", 1)
    tmp = HERE / "_build.md"
    tmp.write_text(text.replace("ᵀ", "^T^"))
    base = ["pandoc", str(tmp), "--from", "markdown+pipe_tables+tex_math_dollars"]
    subprocess.run(base + ["-o", str(HERE / "scJDO2_flow_appnote.pdf"), "--pdf-engine=tectonic",
                           "-V", "mainfont=Arial Unicode MS", "-V", "geometry:margin=2.2cm",
                           "-V", "fontsize=10pt", "-V", "linestretch=1.15", "-V", "colorlinks=true"], check=True)
    subprocess.run(base + ["-o", str(HERE / "scJDO2_flow_appnote.docx")], check=True)
    tmp.unlink()
    print("written:", HERE / "scJDO2_flow_appnote.pdf")


if __name__ == "__main__":
    main()
