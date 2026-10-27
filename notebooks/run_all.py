#!/usr/bin/env python
"""Execute every v57 figure / report notebook in order, in place.

Usage (from the repository root):
    .venv-fig/bin/python notebooks/run_all.py [--timeout 600] [names...]

Fails loudly (non-zero exit, traceback) on the first notebook error.
Writes per-notebook wall times to manuscript/figures/run_log.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent
REPO = NB_DIR.parent
ORDER = [
    "fig1_framework",
    "fig2_benchmark",
    "fig3_flow_operator",
    "fig4_dyngen",
    "fig5_real_scale",
    "fig6_score_operator",
    "fig7_archetypes",
    "supp_s10_diffusion",
    "supp_s21_three_timepoints",
    "supp_s13_whitening",
    "supp_s11_operator_agreement",
    "tables_v57",
    "numbers_v57",
    "build_reports",
]


def fill_runtime(log: dict) -> None:
    """Write the measured runtime into manuscript/REPRODUCE_v57.md (between the runtime markers)."""
    rep = REPO / "manuscript" / "REPRODUCE_v57.md"
    if not rep.exists():
        return
    rows = ["| notebook | wall time (s) |", "|---|---|"]
    rows += [f"| `{n['notebook']}` | {n['seconds']} |" for n in log["notebooks"]]
    block = (f"Last full run: {log['total_seconds']} s in total (Python {log['python']}, {log['platform']}).\n\n"
             + "\n".join(rows))
    text = rep.read_text()
    a, b = "<!-- runtime:start -->", "<!-- runtime:end -->"
    i, j = text.index(a) + len(a), text.index(b)
    rep.write_text(text[:i] + "\n" + block + "\n" + text[j:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600, help="per-cell timeout (s)")
    ap.add_argument("names", nargs="*", help="subset of notebooks (default: all, in order)")
    args = ap.parse_args()
    names = args.names or ORDER
    # make the kernel use this interpreter (the venv), not a global python
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    full_run = not args.names
    if full_run:
        prov = REPO / "manuscript" / "figures" / "provenance.json"
        if prov.exists():
            prov.unlink()  # rebuilt from scratch by the notebooks
    log = {"python": sys.version.split()[0], "platform": platform.platform(),
           "timeout_per_cell_s": args.timeout, "notebooks": []}
    t_all = time.time()
    for name in names:
        path = NB_DIR / f"{name}.ipynb"
        if not path.exists():
            print(f"MISSING notebook: {path}", file=sys.stderr)
            return 2
        print(f"[run_all] executing {path.name} ...", flush=True)
        t0 = time.time()
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
        except Exception as exc:  # noqa: BLE001
            nbformat.write(nb, path)  # keep the failing outputs for inspection
            print(f"[run_all] FAILED in {path.name}: {exc}", file=sys.stderr)
            return 1
        nbformat.write(nb, path)
        dt = time.time() - t0
        log["notebooks"].append({"notebook": f"notebooks/{path.name}", "seconds": round(dt, 1)})
        print(f"[run_all] ok {path.name} ({dt:.1f} s)", flush=True)
    log["total_seconds"] = round(time.time() - t_all, 1)
    if full_run:
        out = REPO / "manuscript" / "figures" / "run_log.json"
        out.write_text(json.dumps(log, indent=1) + "\n")
        fill_runtime(log)
    print(f"[run_all] all done in {log['total_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
