"""Mechanical checks for the Claude ⇄ Manus protocol (coord/PROTOCOL.md).

Generic: identical in every project. Run from the repository root.

    python coord/coord_check.py board
    python coord/coord_check.py check <workstream> <ID> [--head REF] [--base REF] [--run-reviews]

``check`` verifies a task branch against the task's json contract:
changed paths inside ``allowed_paths``; coord files and the
workstream's ``protected_paths`` untouched; ``required_files``
committed; report json complete; ``protected_symbols`` unchanged; no
large committed files; with ``--run-reviews``, every
``review_commands`` entry exits 0. Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COORD = ROOT / "coord"
MAX_COMMITTED_BYTES = 10 * 1024 * 1024
# Only reports are Manus's to write under coord/.
COORD_PROTECTED = ["coord/*.md", "coord/*.py", "coord/*/WORKSTREAM.md",
                   "coord/*/tasks/*", "coord/*/reviews/*"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=ROOT,
                          capture_output=True).returncode == 0


def front_matter(text: str) -> dict:
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            key, _, val = line.partition(":")
            out[key.strip()] = val.strip()
    return out


def json_block(text: str, where: str) -> dict:
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    if not blocks:
        sys.exit(f"{where}: no json contract block")
    return json.loads(blocks[-1])


def read_workstream(ws: str) -> tuple[dict, dict]:
    path = COORD / ws / "WORKSTREAM.md"
    if not path.exists():
        sys.exit(f"no workstream {ws!r} ({path})")
    text = path.read_text()
    return front_matter(text), json_block(text, str(path))


def read_task(ws: str, task_id: str) -> tuple[dict, dict]:
    path = COORD / ws / "tasks" / f"{task_id}.md"
    if not path.exists():
        sys.exit(f"no task {ws}/{task_id} ({path})")
    text = path.read_text()
    return front_matter(text), json_block(text, str(path))


def glob_re(pattern: str) -> re.Pattern:
    """Path glob: ``*`` and ``?`` stay within one path segment, ``**``
    spans segments."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*"); i += 2
        elif pattern[i] == "*":
            out.append("[^/]*"); i += 1
        elif pattern[i] == "?":
            out.append("[^/]"); i += 1
        else:
            out.append(re.escape(pattern[i])); i += 1
    return re.compile("".join(out) + r"\Z")


def matches(path: str, globs: list[str]) -> bool:
    return any(glob_re(g).match(path) for g in globs)


def symbol_body(source: str, signature: str) -> str | None:
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(signature):
            j = i + 1
            while j < len(lines) and not re.match(r"(def |class |@)", lines[j]):
                j += 1
            return "\n".join(lines[i:j]).rstrip()
    return None


def show(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}") if git_ok("cat-file", "-e", f"{ref}:{path}") else ""


class Result:
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
    ws_front, ws_contract = read_workstream(args.workstream)
    front, contract = read_task(args.workstream, args.task)
    title = f"check {args.workstream}/{args.task}"
    head = args.head or f"manus/{args.workstream}-{args.task}"
    # Three-dot diff from the merge base, so coord commits Claude makes on
    # the base branch after the task branch was cut never count here.
    base = args.base or ws_front.get("base_branch") or "main"
    r = Result()
    for ref in (head, base):
        if not git_ok("rev-parse", "--verify", ref):
            r.fail(f"ref {ref} not found")
    if r.fails:
        return r.finish(title)

    manus = front.get("owner") == "manus"
    changed = [p for p in git("diff", "--name-only", f"{base}...{head}").split("\n") if p]
    protected = COORD_PROTECTED + ws_contract.get("protected_paths", [])
    allowed = contract.get("allowed_paths", [])
    for p in changed:
        if manus and matches(p, protected):
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
        before, after = show(base, path), show(head, path)
        for sig in sigs:
            if symbol_body(before, sig) != symbol_body(after, sig):
                r.fail(f"protected symbol changed: {path} :: {sig}")

    if manus:
        rpath = f"coord/{args.workstream}/reports/{args.task}.json"
        if git_ok("cat-file", "-e", f"{head}:{rpath}"):
            try:
                rep = json.loads(show(head, rpath))
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

        log = git("log", "--no-merges", "--format=%H%x00%s%x00%b%x1e", f"{base}..{head}")
        for entry in filter(str.strip, log.split("\x1e")):
            sha, subj, body = (entry.strip().split("\x00") + ["", ""])[:3]
            if "Co-Authored-By: Manus" not in body:
                r.warn(f"commit {sha[:7]} lacks Manus co-author footer: {subj}")

    if args.run_reviews:
        for cmd in contract.get("review_commands", []):
            cmd = cmd.replace("{head}", head)
            proc = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
            print(f"-- review command: {cmd}\n{proc.stdout}{proc.stderr}".rstrip())
            if proc.returncode != 0:
                r.fail(f"review command failed (exit {proc.returncode}): {cmd}")

    print(f"base {base}  head {head}  changed {len(changed)} file(s)")
    return r.finish(title)


def cmd_board(_: argparse.Namespace) -> int:
    for ws_dir in sorted(p for p in COORD.iterdir() if (p / "WORKSTREAM.md").exists()):
        ws = ws_dir.name
        ws_front, _c = read_workstream(ws)
        print(f"[{ws}]  base {ws_front.get('base_branch', '?')}")
        for f in sorted((ws_dir / "tasks").glob("*.md")):
            front = front_matter(f.read_text())
            rep = ws_dir / "reports" / f"{f.stem}.json"
            rev = ws_dir / "reviews" / f"{f.stem}.md"
            verdict = "-"
            if rev.exists():
                m = re.search(r"^verdict:\s*(\S+)", rev.read_text(), re.M)
                verdict = m.group(1) if m else "?"
            branch = f"manus/{ws}-{f.stem}"
            on_branch = git_ok("rev-parse", "--verify", branch)
            ahead = (git("rev-list", "--count", f"{ws_front.get('base_branch', 'main')}..{branch}").strip()
                     if on_branch else "-")
            print(f"  {f.stem:5} {front.get('owner', ''):6} {front.get('status', ''):9} "
                  f"commits={ahead:>3}  review={verdict:7} {front.get('title', '')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("workstream")
    c.add_argument("task")
    c.add_argument("--head")
    c.add_argument("--base")
    c.add_argument("--run-reviews", action="store_true")
    sub.add_parser("board")
    args = ap.parse_args()
    return {"check": cmd_check, "board": cmd_board}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
