"""Mechanical checks for the Claude ⇄ Manus protocol (coord/PROTOCOL.md).

    python scripts/coord_check.py board
    python scripts/coord_check.py check A01 [--head origin/manus/A01] [--base <sha>]
    python scripts/coord_check.py outcome [--ref HEAD]

``check`` verifies a task branch against the task's json contract:
changed paths inside ``allowed_paths``, protected files untouched,
``required_files`` committed, report json complete, protected symbols
unchanged, no large committed files. ``outcome`` re-derives the
PREREG §6 header from ``compare_pooled.csv`` and checks RESULT.md
carries it verbatim. Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "coord" / "tasks"
MAX_COMMITTED_BYTES = 10 * 1024 * 1024
ALWAYS_PROTECTED = ["PREREG_aging_basins.md", "coord/tasks/*",
                    "coord/reviews/*", "coord/BOARD.md", "coord/PROTOCOL.md",
                    "coord/MANUS.md", "MANUS_INSTRUCTIONS_aging_basins.md"]

# PREREG §6 / spec §6, verbatim.
HEADERS = {
    (True, True): "directional, anticipatory loss",
    (True, False): "specific loosening without anticipation",
    (False, False): "isotropic noise increase; no aging claim beyond stiffness",
    (False, True): "anticipation without specific loosening "
                   "(corner case; report as such)",
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=ROOT,
                          capture_output=True).returncode == 0


def read_task(task_id: str) -> tuple[dict, dict]:
    text = (TASKS / f"{task_id}.md").read_text()
    fm_match = re.match(r"---\n(.*?)\n---", text, re.S)
    front = {}
    if fm_match:
        for line in fm_match.group(1).splitlines():
            key, _, val = line.partition(":")
            front[key.strip()] = val.strip()
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    if not blocks:
        sys.exit(f"{task_id}: no json contract block")
    return front, json.loads(blocks[-1])


def matches(path: str, globs: list[str]) -> bool:
    return any(fnmatch(path, g) for g in globs)


def symbol_body(source: str, signature: str) -> str | None:
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(signature):
            j = i + 1
            while j < len(lines) and not re.match(r"(def |class |@)", lines[j]):
                j += 1
            return "\n".join(lines[i:j]).rstrip()
    return None


class Report:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def fail(self, msg: str) -> None:
        self.fails.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)

    def finish(self, title: str) -> int:
        print(f"== {title}")
        for m in self.fails:
            print(f"FAIL  {m}")
        for m in self.warns:
            print(f"WARN  {m}")
        print("PASS" if not self.fails else f"{len(self.fails)} failure(s)")
        return 1 if self.fails else 0


def cmd_check(args: argparse.Namespace) -> int:
    front, contract = read_task(args.task)
    head = args.head or f"origin/manus/{args.task}"
    base = args.base or front.get("base_commit", "")
    if not base or base == "SET-AT-ISSUE":
        base = "aging-basins"
    r = Report()
    if not git_ok("rev-parse", "--verify", head):
        r.fail(f"head ref {head} not found (git fetch?)")
        return r.finish(f"check {args.task}")

    changed = [p for p in git("diff", "--name-only", f"{base}...{head}").split("\n") if p]
    allowed = contract.get("allowed_paths", [])
    for p in changed:
        if front.get("owner") == "manus" and matches(p, ALWAYS_PROTECTED):
            r.fail(f"protected file changed: {p}")
        elif not matches(p, allowed):
            r.fail(f"outside allowed_paths: {p}")

    for p in contract.get("required_files", []):
        if not git_ok("cat-file", "-e", f"{head}:{p}"):
            r.fail(f"required file missing on branch: {p}")

    for line in git("ls-tree", "-r", "-l", head).splitlines():
        meta, path = line.split("\t", 1)
        size = meta.split()[3]
        if path in changed and size.isdigit() and int(size) > MAX_COMMITTED_BYTES:
            r.fail(f"large file committed ({int(size) >> 20} MB): {path}")

    for path, sigs in contract.get("protected_symbols", {}).items():
        before = git("show", f"{base}:{path}") if git_ok("cat-file", "-e", f"{base}:{path}") else ""
        after = git("show", f"{head}:{path}") if git_ok("cat-file", "-e", f"{head}:{path}") else ""
        for sig in sigs:
            if symbol_body(before, sig) != symbol_body(after, sig):
                r.fail(f"protected symbol changed: {path} :: {sig}")

    rpath = f"coord/reports/{args.task}.json"
    if front.get("owner") == "manus":
        if git_ok("cat-file", "-e", f"{head}:{rpath}"):
            try:
                rep = json.loads(git("show", f"{head}:{rpath}"))
            except json.JSONDecodeError as e:
                r.fail(f"report json invalid: {e}")
                rep = {}
            status = rep.get("status")
            if status not in {"done", "blocked", "partial"}:
                r.fail(f"report status invalid: {status!r}")
            if status == "blocked" and not rep.get("questions"):
                r.fail("status blocked but no questions")
            metrics = rep.get("metrics", {})
            for k in contract.get("required_metrics", []):
                if k not in metrics:
                    r.fail(f"required metric missing: {k}")
                elif metrics[k] is None and not rep.get("deviations"):
                    r.fail(f"metric {k} is null with no deviation note")
            for a in rep.get("artifacts", []):
                if not a.get("committed") and not a.get("regenerate"):
                    r.warn(f"uncommitted artifact without regenerate command: {a.get('path')}")
            budget = contract.get("compute_budget_cpu_hours")
            used = (rep.get("compute") or {}).get("cpu_hours")
            if budget is not None and isinstance(used, (int, float)) and used > budget:
                r.fail(f"compute {used} CPU-h exceeds budget {budget}")
            rep_head = rep.get("head_commit", "")
            if rep_head and not git("rev-parse", head).startswith(rep_head[:7]):
                r.warn(f"report head_commit {rep_head[:7]} != branch head "
                       "(report committed after its own sha is expected; check parent)")

        log = git("log", "--format=%H%x00%s%x00%b%x1e", f"{base}..{head}")
        for entry in filter(str.strip, log.split("\x1e")):
            sha, subj, body = (entry.strip().split("\x00") + ["", ""])[:3]
            if "Co-Authored-By: Manus" not in body:
                r.warn(f"commit {sha[:7]} lacks Manus co-author footer: {subj}")
            if args.task in {"A01", "A02", "A03", "A04"} and "§" not in subj and "coord" not in subj:
                r.warn(f"commit {sha[:7]} subject has no spec section: {subj}")

    print(f"base {base}  head {head}  changed {len(changed)} file(s)")
    return r.finish(f"check {args.task}")


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "pass"}


def derive_outcome(rows: list[dict]) -> dict:
    out: dict = {}
    eligible = [x for x in rows if not _truthy(x.get("correction_artifact", "False"))]
    by_tissue: dict[str, list[dict]] = {}
    for x in eligible:
        by_tissue.setdefault(x["tissue"], []).append(x)
    for name, sub in [("pooled", eligible), *sorted(by_tissue.items())]:
        n = len(sub)
        iso = sum(_truthy(x["iso_pass"]) for x in sub)
        ant = sum(_truthy(x["ant_pass"]) for x in sub)
        out[name] = {"n": n, "iso_frac": iso / n if n else None,
                     "ant_frac": ant / n if n else None}
    out["n_correction_artifact"] = len(rows) - len(eligible)
    p = out["pooled"]
    if p["n"]:
        out["header"] = HEADERS[(p["iso_frac"] > 0.5, p["ant_frac"] > 0.5)]
    else:
        out["header"] = None
    return out


def cmd_outcome(args: argparse.Namespace) -> int:
    r = Report()
    pooled_path = "benchmarks/aging/compare_pooled.csv"
    result_path = "benchmarks/aging/RESULT.md"

    def load(path: str) -> str | None:
        if args.ref:
            return git("show", f"{args.ref}:{path}") if git_ok("cat-file", "-e", f"{args.ref}:{path}") else None
        f = ROOT / path
        return f.read_text() if f.exists() else None

    pooled = load(pooled_path)
    if pooled is None:
        r.fail(f"{pooled_path} not found")
        return r.finish("outcome")
    rows = list(csv.DictReader(io.StringIO(pooled)))
    need = {"tissue", "cell_type", "correction_artifact", "iso_pass", "ant_pass"}
    missing = need - set(rows[0].keys() if rows else [])
    if missing:
        r.fail(f"compare_pooled.csv missing columns: {sorted(missing)}")
        return r.finish("outcome")
    out = derive_outcome(rows)
    print(json.dumps(out, indent=2))
    if out["header"] is None:
        r.fail("no eligible non-artifact types")
    result = load(result_path) or ""
    if out["header"] and out["header"] not in result:
        r.fail(f"RESULT.md does not contain the derived header: {out['header']!r}")
    for h in HEADERS.values():
        if h != out["header"] and f"**{h}**" in result:
            r.fail(f"RESULT.md asserts a different header: {h!r}")
    return r.finish("outcome")


def cmd_board(_: argparse.Namespace) -> int:
    for f in sorted(TASKS.glob("*.md")):
        front, _c = read_task(f.stem)
        rep = ROOT / "coord" / "reports" / f"{f.stem}.json"
        rev = ROOT / "coord" / "reviews" / f"{f.stem}.md"
        verdict = ""
        if rev.exists():
            m = re.search(r"^verdict:\s*(\S+)", rev.read_text(), re.M)
            verdict = m.group(1) if m else "?"
        print(f"{f.stem}  {front.get('owner', ''):6}  {front.get('status', ''):9}  "
              f"base={front.get('base_commit', '')[:10]:10}  "
              f"report={'yes' if rep.exists() else '-':3}  review={verdict or '-'}  "
              f"{front.get('title', '')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("task")
    c.add_argument("--head")
    c.add_argument("--base")
    o = sub.add_parser("outcome")
    o.add_argument("--ref", help="read files from this git ref instead of the working tree")
    sub.add_parser("board")
    args = ap.parse_args()
    return {"check": cmd_check, "outcome": cmd_outcome, "board": cmd_board}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
