"""Produce the §4 side-by-side report for one dataset.

Reads v1 outputs from ``<v1_out>/`` and v2 outputs from ``<v2_out>/`` and writes
``<report_out>/report.md`` plus the side-by-side figure.

Report sections (per the brief, §4):
    1. Branch handling: whether the dataset has a fork; whether v2 recovers a
       saddle at it; v1 cannot by construction.
    2. Operator reproducibility: half-split Spearman of leading eigenvalues,
       v1 vs v2.
    3. Archetype recurrence: v1 vs v2 archetype sets, overlap, v2-only.
    4. Driver genes: top-10 per archetype, v1 vs v2, with v2 Fisher confidence.
    5. Fisher ledger summary: fraction high/low/empty by reason code.
    6. Runtime and memory per step.
    7. One side-by-side figure: v1 slicing archetype sequence vs. v2 operator
       field coloured by leading eigenvalue, with the fork marked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _write_report(v1_out: Path, v2_out: Path, report_out: Path) -> None:
    report_out.mkdir(parents=True, exist_ok=True)
    (report_out / "report.md").write_text(
        "# scJDO v1 vs. v2 — placeholder\n\n"
        f"- v1 outputs: `{v1_out}`\n"
        f"- v2 outputs: `{v2_out}`\n\n"
        "Phase E will fill this in.\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v1", required=True, type=Path)
    p.add_argument("--v2", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    _write_report(args.v1, args.v2, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
