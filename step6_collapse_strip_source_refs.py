#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_collapse_strip_source_refs.py
F-ARCH2 follow-up (Step 6 latent bug fix). SEPARATE commit from the Step 7 move.

Collapses the 4 duplicate `_strip_source_refs` definitions to ONE canonical copy
in backend/domain/grounding.py. After this:
  - grounding.py            : canonical def (UNCHANGED)
  - domain/matching.py      : def removed -> re-exports from grounding
                              (keeps `matching._strip_source_refs` for explainable + gate;
                               KEEPS its own _REF_RE, still used by the normalizer)
  - agents/black_box.py     : def + _REF_RE removed; bare calls -> grounding._strip_source_refs
  - agents/hitl.py          : def + _REF_RE removed; bare calls -> grounding._strip_source_refs
  - agents/base.py          : bare calls -> grounding._strip_source_refs   (fixes the NameError)
  - agents/explainable.py   : UNTOUCHED (uses matching._strip_source_refs re-export;
                               its _INLINE_REF_RE is a different util — leave it)

Behavior-preserving: regex identical everywhere; grounding's version is null-safe
(`text or ""`), which also removes a latent HITL-only crash-on-None (I-1 win).

Run from repo root AFTER the Step 7 moves + import rewrite, on a clean-ish tree:
    python step6_collapse_strip_source_refs.py --dry-run
    python step6_collapse_strip_source_refs.py
Aborts loudly if any file/shape isn't what's expected (no half-state).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

DRY = "--dry-run" in sys.argv

GROUNDING = Path("backend/domain/grounding.py")
MATCHING  = Path("backend/domain/matching.py")
BLACKBOX  = Path("backend/agents/black_box.py")
HITL      = Path("backend/agents/hitl.py")
BASE      = Path("backend/agents/base.py")

REEXPORT = "from backend.domain.grounding import _strip_source_refs  # F-ARCH2: single source\n"
# bare-call -> grounding-qualified, guarded so it never matches `x._strip_source_refs`
# or an already-qualified `grounding._strip_source_refs` (idempotent).
CALL_RX = re.compile(r"(?<![\w.])_strip_source_refs\(")
DEF_RX  = re.compile(r"\ndef _strip_source_refs\(text: str\) -> str:\r?\n"
                     r"    return _REF_RE\.sub\(\"\", text(?: or \"\")?\)\.strip\(\)\r?\n")
REFRE_RX = re.compile(r'_REF_RE = re\.compile\(r"\\s\*\\\[\[a-z\]\\\]"\)\r?\n')

errors: list[str] = []


def read(p: Path) -> str:
    return p.read_bytes().decode("utf-8")          # CRLF preserved in-string


def write(p: Path, s: str) -> None:
    if not DRY:
        p.write_bytes(s.encode("utf-8"))           # bytes out -> no newline translation


def need(p: Path):
    if not p.exists():
        raise SystemExit(f"[ABORT] missing {p} — run from repo root, after Step 7.")


def has_grounding_import(text: str) -> bool:
    return ("from backend.domain import grounding" in text
            or "from backend.domain import matching, grounding" in text
            or "import grounding" in text)


def main():
    for p in (GROUNDING, MATCHING, BLACKBOX, HITL, BASE):
        need(p)

    # 0. canonical must exist
    g = read(GROUNDING)
    if "def _strip_source_refs(" not in g:
        raise SystemExit("[ABORT] canonical def not found in grounding.py — nothing to collapse onto.")
    print("[OK] canonical def present in grounding.py (unchanged).")

    report = []

    # 1. matching: drop dup def, add re-export, KEEP _REF_RE
    m = read(MATCHING)
    if REEXPORT.strip() in m:
        report.append("matching.py: already re-exports (skip)")
    else:
        m2, n = DEF_RX.subn("\n" + REEXPORT, m, count=1)
        if n != 1:
            errors.append("matching.py: dup def shape not found (expected exactly 1).")
        else:
            if "_REF_RE.sub" not in m2:
                errors.append("matching.py: would orphan _REF_RE — the normalizer use is gone?!")
            else:
                write(MATCHING, m2)
                report.append("matching.py: def -> re-export; _REF_RE kept for normalizer")

    # 2+3. black_box / hitl: remove _REF_RE + def, qualify bare calls
    for p, name in ((BLACKBOX, "black_box.py"), (HITL, "hitl.py")):
        t = read(p)
        if not has_grounding_import(t):
            errors.append(f"{name}: no `grounding` import in scope — cannot qualify calls.")
            continue
        t2, nd = DEF_RX.subn("\n", t, count=1)
        t2, nr = REFRE_RX.subn("", t2, count=1)
        t3 = CALL_RX.sub("grounding._strip_source_refs(", t2)
        if "def _strip_source_refs(" in t3:
            errors.append(f"{name}: a _strip_source_refs def still remains after removal.")
        write(p, t3)
        calls = len(CALL_RX.findall(t)) - (1 if nd else 0)  # minus the def line itself
        report.append(f"{name}: removed def({nd}) + _REF_RE({nr}); qualified ~{max(calls,0)} call(s)")

    # 4. base: qualify bare calls (no def, no _REF_RE here)
    b = read(BASE)
    if not has_grounding_import(b):
        errors.append("base.py: no `grounding` import in scope — cannot qualify calls.")
    else:
        before = len(CALL_RX.findall(b))
        b2 = CALL_RX.sub("grounding._strip_source_refs(", b)
        write(BASE, b2)
        report.append(f"base.py: qualified {before} bare call(s) -> grounding. (fixes NameError)")

    print("\n=== collapse plan ({}) ===".format("DRY-RUN" if DRY else "APPLIED"))
    for r in report:
        print("  " + r)

    if errors:
        print("\n[ABORT] shape mismatch — NOTHING was written if dry-run; if applied, review:")
        for e in errors:
            print("  ! " + e)
        raise SystemExit(1)

    print("\n[OK] one definition remains (grounding). Verify:")
    print("  grep -rn 'def _strip_source_refs' backend/   # expect ONLY grounding.py")
    print("  poetry run chainlit run frontend/app.py       # framework render no longer NameErrors")


if __name__ == "__main__":
    main()
