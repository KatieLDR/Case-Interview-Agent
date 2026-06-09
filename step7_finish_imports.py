#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_finish_imports.py  —  recovery finisher for Step 7.

Use when the file MOVES already happened (backend/agents, backend/tools,
backend/knowledge exist) but imports were NOT rewritten — i.e. the app still
fails with `ModuleNotFoundError: No module named 'backend.black_box_agent'`.

This does ONLY the import rewrite (no moves, no git). Same substitutions as
step7_migrate.py, already verified against every real import line. CRLF-preserved.

Run from the repo root:
    python step7_finish_imports.py --dry-run    # preview
    python step7_finish_imports.py              # apply
Then:
    git add -A
    git diff --cached -M --stat                 # expect: renames + import lines
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DRY = "--dry-run" in sys.argv

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


def build_subs():
    subs = []
    for old, (pkg, new) in RENAMES.items():
        oe = re.escape(old)
        subs.append((re.compile(rf"from backend\.{oe} import"),
                     f"from backend.{pkg}.{new} import"))
        subs.append((re.compile(rf"from backend import {oe}\b"),
                     f"from backend.{pkg} import {new}"))
        subs.append((re.compile(rf"\bimport backend\.{oe}\b"),
                     f"import backend.{pkg}.{new}"))
    return subs


SUBS = build_subs()


def preflight():
    if not Path("backend").is_dir():
        raise SystemExit("[ABORT] No ./backend. Run from the repo root.")
    for d in ("backend/agents", "backend/tools", "backend/knowledge"):
        if not Path(d).is_dir():
            raise SystemExit(f"[ABORT] {d} missing — moves not done. "
                             "This script only rewrites imports; do the moves first.")
    # ensure the new package markers exist (step7_migrate created them; be safe)
    for d in ("backend/agents", "backend/tools", "backend/knowledge"):
        init = Path(d) / "__init__.py"
        if not init.exists():
            print(f"[WARN] {init} missing — creating empty package marker.")
            if not DRY:
                init.write_bytes(b"")
    print("[OK] preflight: moves present, package markers ok.")


def rewrite_file(path: Path) -> int:
    raw = path.read_bytes()
    text = raw.decode("utf-8")            # CRLF preserved in-string
    new = text
    for rx, repl in SUBS:
        new = rx.sub(repl, new)
    if new == text:
        return 0
    if not DRY:
        path.write_bytes(new.encode("utf-8"))   # bytes out -> no newline translation
    return sum(1 for a, b in zip(text.splitlines(), new.splitlines()) if a != b)


def all_py():
    for root, dirs, files in os.walk("."):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def main():
    preflight()
    print(f"\n=== Step 7 import rewrite — {'DRY-RUN' if DRY else 'APPLYING'} ===\n")
    files, lines = 0, 0
    for py in sorted(all_py()):
        n = rewrite_file(py)
        if n:
            files += 1
            lines += n
            print(f"  {py}  ({n} line(s))")
    print(f"\n[SUMMARY] {files} files rewritten, {lines} import lines.")
    if DRY:
        print("[DRY-RUN] nothing changed.")
    else:
        print("\nNext:\n  git add -A\n  git diff --cached -M --stat   # renames + import lines\n"
              "  poetry run chainlit run frontend/app.py   # boot")


if __name__ == "__main__":
    main()
