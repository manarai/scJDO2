"""Re-derive the PREREG_aging_hlca §6 outcome header and stiffness
qualifier from the primary per-type table.

    python scripts/hlca_outcome_check.py [--ref GIT_REF]

Reads ``benchmarks/aging_hlca/tests_primary.csv`` (one row per eligible
type: ``iso_pass``, ``ant_pass``, ``stiff_slope``, ``stiff_p``) and
``benchmarks/aging_hlca/RESULT.md``; fails if RESULT.md does not carry
the derived "header — qualifier" verbatim or asserts another header.
Used as a review command in coord/hlca/tasks/H04.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = "benchmarks/aging_hlca/tests_primary.csv"
RESULT = "benchmarks/aging_hlca/RESULT.md"

HEADERS = {  # PREREG_aging_hlca §6, verbatim
    (True, True): "directional, anticipatory loss",
    (True, False): "specific loosening without anticipation",
    (False, False): "isotropic noise increase; no aging claim beyond stiffness",
    (False, True): "anticipation without specific loosening (corner case; report as such)",
}
QUALIFIERS = {
    "loss": "— with stiffness loss in the majority of types",
    "gain": "— with stiffness gain in the majority of types",
    "none": "— no majority stiffness change",
}


def load(path: str, ref: str | None) -> str | None:
    if ref:
        p = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True)
        return p.stdout if p.returncode == 0 else None
    f = ROOT / path
    return f.read_text() if f.exists() else None


def truthy(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "pass"}


def derive(rows: list[dict]) -> dict:
    n = len(rows)
    iso = sum(truthy(r["iso_pass"]) for r in rows)
    ant = sum(truthy(r["ant_pass"]) for r in rows)
    loss = sum(float(r["stiff_slope"]) < 0 and float(r["stiff_p"]) <= 0.05 for r in rows)
    gain = sum(float(r["stiff_slope"]) > 0 and float(r["stiff_p"]) <= 0.05 for r in rows)
    header = HEADERS[(iso / n > 0.5, ant / n > 0.5)] if n else None
    q = "loss" if loss / n > 0.5 else "gain" if gain / n > 0.5 else "none"
    return {"n_eligible": n, "iso_pass": iso, "ant_pass": ant, "stiff_loss": loss,
            "stiff_gain": gain, "header": header, "qualifier": QUALIFIERS[q],
            "outcome": f"{header} {QUALIFIERS[q]}" if header else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref")
    a = ap.parse_args()
    table = load(TABLE, a.ref)
    if table is None:
        print(f"FAIL  {TABLE} not found"); return 1
    rows = list(csv.DictReader(io.StringIO(table)))
    need = {"ann_level_3", "iso_pass", "ant_pass", "stiff_slope", "stiff_p"}
    if not rows or need - set(rows[0]):
        print(f"FAIL  {TABLE} missing columns: {sorted(need - set(rows[0] if rows else []))}"); return 1
    out = derive(rows)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    result = load(RESULT, a.ref) or ""
    fails = []
    if out["outcome"] is None:
        fails.append("no eligible types")
    elif out["outcome"] not in result and f"**{out['header']}** {out['qualifier']}" not in result:
        fails.append(f"RESULT.md does not contain the derived outcome: {out['outcome']!r}")
    for h in HEADERS.values():
        if h != out["header"] and f"**{h}**" in result:
            fails.append(f"RESULT.md asserts a different header: {h!r}")
    for f in fails:
        print(f"FAIL  {f}")
    print("PASS" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
