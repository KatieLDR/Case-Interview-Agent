#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_migrate.py  —  REFACTOR_PLAN.md §4 Step 7 (Directory split, LAST step).

MECHANICAL ONLY. No logic changes. This script:
  1. creates the three new packages   backend/agents/ backend/tools/ backend/knowledge/
  2. `git mv`s nine files into them (history-preserving, so the gate sees RENAMES)
  3. rewrites every intra-project import across the whole tree (CRLF-preserved)

It does NOT touch:
  - backend/llm.py, backend/logger.py            (stay at backend root — decision ②)
  - backend/domain/  backend/interaction/  backend/logging/   (already-layered)
  - any *logic* line anywhere

Run from the REPO ROOT (the dir containing `backend/` and `frontend/`):

    python step7_migrate.py            # do the migration
    python step7_migrate.py --dry-run  # print the plan, change nothing

Idempotency / safety: aborts if backend/agents already exists (already migrated)
or if the pre-move files are missing (wrong cwd / partial state).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

DRY = "--dry-run" in sys.argv

# ── 1. File moves: (old path, new path) ──────────────────────────────────────
MOVES = [
    ("backend/base.py",              "backend/agents/base.py"),
    ("backend/black_box_agent.py",   "backend/agents/black_box.py"),
    ("backend/explainable_agent.py", "backend/agents/explainable.py"),
    ("backend/hitl_agent.py",        "backend/agents/hitl.py"),
    ("backend/concept_swap.py",      "backend/tools/concept_swap.py"),
    ("backend/rag_explainer.py",     "backend/tools/rag_explainer.py"),
    ("backend/knowledge_base.py",    "backend/knowledge/knowledge_base.py"),
    ("backend/knowledge_base.json",  "backend/knowledge/knowledge_base.json"),
    ("backend/cases.py",             "backend/knowledge/cases.py"),
]

NEW_PACKAGES = ["backend/agents", "backend/tools", "backend/knowledge"]

# ── 2. Import rewrite map:  old backend-relative module -> (subpkg, new module name)
#    Renames (*_agent -> bare) are folded in here.
RENAMES = {
    "base":              ("agents",    "base"),
    "black_box_agent":   ("agents",    "black_box"),
    "explainable_agent": ("agents",    "explainable"),
    "hitl_agent":        ("agents",    "hitl"),
    "concept_swap":      ("tools",     "concept_swap"),
    "rag_explainer":     ("tools",     "rag_explainer"),
    "knowledge_base":    ("knowledge", "knowledge_base"),
    "cases":             ("knowledge", "cases"),
}


def build_substitutions():
    """Return list of (compiled_regex, replacement) for all three import forms."""
    subs = []
    for old, (pkg, new) in RENAMES.items():
        oe = re.escape(old)
        # Form 1:  from backend.<old> import ...   ->  from backend.<pkg>.<new> import ...
        subs.append((re.compile(rf"from backend\.{oe} import"),
                     f"from backend.{pkg}.{new} import"))
        # Form 2:  from backend import <old>[ as X] ->  from backend.<pkg> import <new>[ as X]
        subs.append((re.compile(rf"from backend import {oe}\b"),
                     f"from backend.{pkg} import {new}"))
        # Form 3:  import backend.<old>[ as X]      ->  import backend.<pkg>.<new>[ as X]
        subs.append((re.compile(rf"\bimport backend\.{oe}\b"),
                     f"import backend.{pkg}.{new}"))
    return subs


SUBS = build_substitutions()


def run_git(args):
    res = subprocess.run(["git"] + args, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"[ABORT] git {' '.join(args)} failed:\n{res.stderr}")
    return res.stdout


# ── Guards ────────────────────────────────────────────────────────────────────
def preflight():
    if not Path("backend").is_dir():
        raise SystemExit("[ABORT] No ./backend dir. Run from the repo root.")
    if not Path(".git").is_dir():
        raise SystemExit("[ABORT] No ./.git. Step 7 gate needs git rename tracking.")
    if Path("backend/agents").exists():
        raise SystemExit("[ABORT] backend/agents already exists — already migrated?")
    missing = [src for src, _ in MOVES if not Path(src).exists()]
    if missing:
        raise SystemExit("[ABORT] expected pre-move files missing:\n  "
                         + "\n  ".join(missing))
    print("[OK] preflight passed — at repo root, git present, nothing pre-migrated.")


# ── 3. CRLF-safe import rewrite ───────────────────────────────────────────────
def rewrite_file(path: Path) -> int:
    """Apply SUBS to one file, preserving exact line endings. Returns #lines changed."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")               # \r\n preserved inside the string
    new = text
    for rx, repl in SUBS:
        new = rx.sub(repl, new)
    if new == text:
        return 0
    if not DRY:
        path.write_bytes(new.encode("utf-8"))   # write bytes -> no newline translation
    # count changed lines for the report
    changed = sum(1 for a, b in zip(text.splitlines(), new.splitlines()) if a != b)
    return changed


def all_py_files():
    for root, dirs, files in os.walk("."):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    preflight()
    mode = "DRY-RUN (no changes)" if DRY else "APPLYING"
    print(f"\n=== Step 7 directory split — {mode} ===\n")

    # 1+2. create packages + git mv
    for pkg in NEW_PACKAGES:
        print(f"  mkdir {pkg}/  + __init__.py")
        if not DRY:
            Path(pkg).mkdir(parents=True, exist_ok=True)
            init = Path(pkg) / "__init__.py"
            init.write_bytes(b"")            # empty: no logic, just package marker
            run_git(["add", str(init)])
    print()
    for src, dst in MOVES:
        print(f"  git mv {src}  ->  {dst}")
        if not DRY:
            run_git(["mv", src, dst])
    print()

    # 3. rewrite imports tree-wide
    total_files, total_lines = 0, 0
    for py in sorted(all_py_files()):
        n = rewrite_file(py)
        if n:
            total_files += 1
            total_lines += n
            print(f"  imports  {py}  ({n} line(s))")

    print(f"\n[SUMMARY] {len(MOVES)} files moved, "
          f"{total_files} files had imports rewritten ({total_lines} import lines).")
    if DRY:
        print("[DRY-RUN] nothing was changed.")
    else:
        print("\nNext:")
        print("  git status                 # expect: renames + import-line edits only")
        print("  git diff -M --stat         # rename detection should pair every move")
        print("  python -c 'import frontend.app'   # smoke import (or boot Chainlit)")


if __name__ == "__main__":
    main()
