"""Re-derive the PREREG §6 outcome header for the aging arm.

    python scripts/aging_outcome_check.py [--ref GIT_REF]

Reads ``benchmarks/aging/compare_pooled.csv`` and
``benchmarks/aging/RESULT.md`` (from the working tree, or from
``--ref``), computes isotropy and anticipation pass fractions over
non-artifact eligible types (pooled and per tissue), picks the §6 row,
and fails if RESULT.md does not carry that header or asserts another.
Used as a review command in coord/aging/tasks/A03.md.
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
POOLED = "benchmarks/aging/compare_pooled.csv"
RESULT = "benchmarks/aging/RESULT.md"

# PREREG §6 / spec §6, verbatim.
HEADERS = {
    (True, True): "directional, anticipatory loss",
    (True, False): "specific loosening without anticipation",
    (False, False): "isotropic noise increase; no aging claim beyond stiffness",
    (False, True): "anticipation without specific loosening "
                   "(corner case; report as such)",
}


def load(path: str, ref: str | None) -> str | None:
    if ref:
        p = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                           capture_output=True, text=True)
        return p.stdout if p.returncode == 0 else None
    f = ROOT / path
    return f.read_text() if f.exists() else None


def truthy(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "pass"}


def derive(rows: list[dict]) -> dict:
    eligible = [x for x in rows if not truthy(x.get("correction_artifact", "False"))]
    by_tissue: dict[str, list[dict]] = {}
    for x in eligible:
        by_tissue.setdefault(x["tissue"], []).append(x)
    out: dict = {}
    for name, sub in [("pooled", eligible), *sorted(by_tissue.items())]:
        n = len(sub)
        out[name] = {"n": n,
                     "iso_frac": sum(truthy(x["iso_pass"]) for x in sub) / n if n else None,
                     "ant_frac": sum(truthy(x["ant_pass"]) for x in sub) / n if n else None}
    out["n_correction_artifact"] = len(rows) - len(eligible)
    p = out["pooled"]
    out["header"] = HEADERS[(p["iso_frac"] > 0.5, p["ant_frac"] > 0.5)] if p["n"] else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref")
    a = ap.parse_args()
    fails = []
    pooled = load(POOLED, a.ref)
    if pooled is None:
        print(f"FAIL  {POOLED} not found")
        return 1
    rows = list(csv.DictReader(io.StringIO(pooled)))
    need = {"tissue", "cell_type", "correction_artifact", "iso_pass", "ant_pass"}
    missing = need - set(rows[0].keys() if rows else [])
    if missing:
        print(f"FAIL  compare_pooled.csv missing columns: {sorted(missing)}")
        return 1
    out = derive(rows)
    print(json.dumps(out, indent=2))
    result = load(RESULT, a.ref) or ""
    if out["header"] is None:
        fails.append("no eligible non-artifact types")
    elif out["header"] not in result:
        fails.append(f"RESULT.md does not contain the derived header: {out['header']!r}")
    for h in HEADERS.values():
        if h != out["header"] and f"**{h}**" in result:
            fails.append(f"RESULT.md asserts a different header: {h!r}")
    for f in fails:
        print(f"FAIL  {f}")
    print("PASS" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
