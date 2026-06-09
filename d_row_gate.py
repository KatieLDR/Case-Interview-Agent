#!/usr/bin/env python3
"""d_row_gate.py — Step 3 gate (REFACTOR_PLAN §4/§5, §5.1 table B, catalog §4 D-rows).

Two independently-runnable parts:

  PART A — LIVE ROUTER GATE (needs GEMINI_API_KEY).
    Runs the SHARED router `backend.interaction.intents.classify_intent` against the
    catalog D-rows plus contrast rows. Because every arm will call this same router
    (I-1/I-2), "identical routing across arms" holds by construction; this validates
    the substantive half — that the router is CORRECT, the "else" bug is fixed
    (D6: filler never becomes a concept), and ask_agent_to_suggest is distinguished
    from add (named concept) and advance (bare "next").

        GEMINI_API_KEY=...  python3 d_row_gate.py
        python3 d_row_gate.py --runs 5        # majority over N runs (§5 determinism)

  PART B — WIRING CHECK (no key, pure source inspection).
    After you wire an arm to the shared router, confirms the switchover happened:
    the arm imports + calls classify_intent, and the old per-arm entry points are gone
    (W3: HITL's dual path; W4: redo handlers). Use it to verify BlackBox first, then
    Explainable, then HITL.

        python3 d_row_gate.py --arm blackbox
        python3 d_row_gate.py --arm explainable
        python3 d_row_gate.py --arm hitl

NOT covered here (by design — these are Step 4, not the router; see W6):
  * D7 (accept a suggestion -> stays ask_agent_suggestion, agent-originated) needs the
    pending-suggestion state in the handler.
  * source attribution (user_spontaneous | user_elicited, A9/B6) is handler/logging.
"""
import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── catalog D-rows + contrast rows ───────────────────────────────────────────
# expectation:
#   intent  : exact intent the router must return
#   detail  : "null"  -> detail MUST be None (D-rows: filler never a concept)
#             "any"   -> detail must be non-None (an add/remove/revisit target)
#             None    -> not asserted
#   parent  : substring the parent must contain (A6), else None (not asserted)
ROWS = [
    # ---- D1–D6: ask_agent_to_suggest, detail MUST be null (the "else"-bug guard) ----
    ("D1", "What else should we consider?",       "ask_agent_to_suggest", "null", None),
    ("D2", "Any suggestions from you?",            "ask_agent_to_suggest", "null", None),
    ("D3", "What would you add here?",             "ask_agent_to_suggest", "null", None),
    ("D4", "Am I missing anything?",               "ask_agent_to_suggest", "null", None),
    ("D5", "What should come next?",               "ask_agent_to_suggest", "null", None),
    ("D6a", "anything else?",                       "ask_agent_to_suggest", "null", None),
    ("D6b", "what else?",                           "ask_agent_to_suggest", "null", None),
    ("D6c", "What else could we add?",              "ask_agent_to_suggest", "null", None),
    # ---- contrast rows: the router must NOT collapse everything to suggest ----
    ("C1", "what about regulatory risk?",          "add",       "any",  None),   # names a concept -> add, NOT suggest
    ("C2", "We should add a point about change management.", "add", "any", None),  # A1
    ("C3", "Under Feasibility, add a note on vendor lock-in.", "add", "any", "feasibility"),  # A6 parent
    ("C4", "next",                                  "advance",   "null", None),   # G1: bare next != "what's next?"
    ("C5", "move on",                               "advance",   "null", None),
    ("C6", "remove Feasibility",                    "remove",    "any",  None),
    ("C7", "why is Feasibility here?",              "question",  None,   None),
    ("C8", "Let's go back to Strategic Fit.",       "revisit",   "any",  None),   # E1 (needs passed-pillar ctx)
    ("C9", "I'm not sure this one fits.",           "doubt",     "null", None),   # G4
    ("C10", "start over",                           "none",      "null", None),   # redo removed
]

# Per-row context overrides. Some rows only make sense in a specific walkthrough
# state — e.g. a revisit row needs the named pillar to be ALREADY PASSED (catalog E1
# is state=passed), so we put `current_pillar` on a LATER pillar. Without this, "go
# back to Strategic Fit" while ON Strategic Fit is a no-op the model rightly calls none.
CTX_OVERRIDE = {
    "C8": dict(current_pillar="Feasibility", current_bullets="(none)"),
}


def _context():
    """Realistic turn context from the live KB (router takes it by value)."""
    try:
        from backend import knowledge_base as kb
        shown = kb.get_shown_pillars()
        cur = shown[0]["name"] if shown else "Strategic Fit"
        bullets = "\n".join(f"- {b}" for b in (shown[0].get("sub_bullets", []) if shown else [])) or "(none)"
        pillars = ", ".join(p["name"] for p in shown) or "(none)"
    except Exception:
        cur, bullets, pillars = "Strategic Fit", "(none)", "(none)"
    return dict(current_pillar=cur, current_bullets=bullets,
                walkthrough_pillars=pillars, last_agent="(walkthrough in progress)")


def _detail_ok(res, want):
    if want is None:
        return True
    if want == "null":
        return res.detail is None
    if want == "any":
        return res.detail is not None
    return True


def _parent_ok(res, want):
    if want is None:
        return True
    return bool(res.parent) and want.lower() in res.parent.lower()


def run_part_a(runs: int) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set — PART A calls the live classifier. "
                 "Set it and re-run from the project root (or use --arm for the no-key wiring check).")

    from backend.interaction import intents as I

    ctx = _context()
    print(f"PART A — live router gate ({runs} run(s) per row), current_pillar={ctx['current_pillar']!r}")
    print(f"{'row':5} {'message':46} {'intent':22} {'detail/parent':22} result")
    print("-" * 118)

    passed = 0
    gateable = 0          # rows with at least one non-error draw (503s don't count)
    flaky = []
    transient = []        # rows where ALL draws were LLM errors (e.g. 503) — re-run, not a fail
    for rid, msg, want_intent, want_detail, want_parent in ROWS:
        row_ctx = {**ctx, **CTX_OVERRIDE.get(rid, {})}
        draws = [I.classify_intent(msg, **row_ctx) for _ in range(runs)]
        ok = [r for r in draws if not r.error]          # F-I3: error draws are infra blips,
        n_err = len(draws) - len(ok)                    # NOT routing outcomes — exclude them.

        if not ok:
            # Every draw 503'd / errored. This says nothing about the router — flag to re-run.
            transient.append(rid)
            print(f"{rid:5} {msg[:45]:46} {'(LLM error)':22} {'—':22} "
                  f"ERR transient {n_err}/{runs} — re-run")
            continue

        gateable += 1
        intents_seen = Counter(r.intent for r in ok)
        maj_intent, maj_n = intents_seen.most_common(1)[0]
        last = ok[-1]
        stable = maj_n / len(ok) >= 0.90
        if not stable:
            flaky.append((rid, dict(intents_seen)))
        good = (maj_intent == want_intent
                and _detail_ok(last, want_detail)
                and _parent_ok(last, want_parent))
        passed += good
        dp = f"d={last.detail!r}" + (f" p={last.parent!r}" if last.parent else "")
        flag = "" if stable else f" ⚠{maj_n}/{len(ok)}"
        if n_err:
            flag += f" ({n_err} err draw dropped)"
        print(f"{rid:5} {msg[:45]:46} {maj_intent[:21]:22} {dp[:21]:22} "
              f"{'PASS' if good else 'FAIL exp ' + want_intent}{flag}")
        if not good:
            # Evidence, not guesswork: show the RAW classifier output so a routing FAIL
            # (wrong intent? right intent but empty detail the targetless-guard demoted?)
            # is diagnosable before any prompt change. (Only reached on a non-error draw.)
            try:
                from backend.llm import classify_json
                raw = classify_json(I.INTENT_ROUTER_PROMPT.format(user_msg=msg, **row_ctx))
                print(f"      └─ raw classifier: {raw}")
            except Exception as e:
                print(f"      └─ (raw probe failed — transient: {e})")
    print("-" * 118)
    print(f"{passed}/{gateable} resolved rows route as the catalog expects"
          + (f"  ({len(transient)} transient/ERR row(s) — re-run: {', '.join(transient)})" if transient else ""))
    for rid, dist in flaky:
        print(f"  ⚠ {rid} unstable across runs: {dist} — sharpen the router prompt (§5).")
    if transient:
        print("Inconclusive — LLM was briefly unavailable (503). Re-run; use --runs 3 to ride out blips.")
        return 2
    if passed != gateable:
        print("Gate NOT met — inspect the FAIL rows (router prompt wording).")
        return 1
    print("Step-3 router gate MET (shared classify_intent correct; 'else' bug fixed). "
          "Per-arm routing converges once each arm is wired (W1–W9) + passes --arm.")
    return 0


# ── PART B — per-arm wiring check (no key; static source inspection) ──────────
_ARM_FILE = {
    "blackbox":    "backend/black_box_agent.py",
    "explainable": "backend/explainable_agent.py",
    "hitl":        "backend/hitl_agent.py",
}
# Old per-arm intent entry points that the switchover must retire (call sites, not
# necessarily the def — a transitional dead def is fine, a live CALL is not).
_RETIRED_CALLS = {
    "blackbox":    [r"self\._detect_override\(", r"self\._classify_question\("],
    "explainable": [r"INTENT_ROUTER_PROMPT", r"self\._classify_intent\("],
    "hitl":        [r"self\._detect_override\(", r"self\._classify_intent\("],  # BOTH (W3)
}
# Redo HANDLING residue (W4) — a routing branch or the handler itself. NOT prompt
# vocabulary: OVERRIDE_CLASSIFIER_PROMPT legitimately lists "redo" while it is still
# inherited by an un-wired HITL, so we match the branch/handler, never the word.
_REDO = [r'==\s*"redo"', r"_handle_redo"]


def run_part_b(arm: str) -> int:
    path = _ARM_FILE[arm]
    if not os.path.exists(path):
        sys.exit(f"{path} not found — run from the project root.")
    src = open(path, encoding="utf-8").read()

    print(f"PART B — wiring check: {arm}  ({path})")
    checks = []

    # 1. imports + calls the shared router
    imports = bool(re.search(r"from backend\.interaction(\.intents)? import|"
                             r"from backend\.interaction import intents|"
                             r"import backend\.interaction\.intents", src))
    # lookbehind excludes the OLD method `self._classify_intent(` — we want the
    # bare/qualified shared call `classify_intent(` or `intents.classify_intent(`.
    calls = bool(re.search(r"(?<!_)classify_intent\(", src))
    checks.append(("imports interaction.intents", imports))
    checks.append(("calls classify_intent()", calls))

    # 2. old entry points retired (no live call sites)
    for pat in _RETIRED_CALLS[arm]:
        hits = re.findall(pat, src)
        checks.append((f"retired call {pat!r} (0 expected)", len(hits) == 0))

    # 3. redo handlers gone (W4)
    redo_hits = sum(len(re.findall(p, src)) for p in _REDO)
    checks.append(("redo handlers removed (W4)", redo_hits == 0))

    # 4. buttons map via intent_for_button where the arm has buttons (HITL only)
    if arm == "hitl":
        checks.append(("uses intent_for_button (D-Q1)", "intent_for_button" in src))

    width = max(len(n) for n, _ in checks)
    passed = 0
    for name, good in checks:
        passed += good
        print(f"  {'PASS' if good else 'FAIL'}  {name:{width}}")
    print(f"{passed}/{len(checks)} wiring checks pass for {arm}.")
    if passed != len(checks):
        print("Wiring incomplete — see FAILs (W3/W4). The arm is not fully switched over.")
        return 1
    print(f"{arm} switched over to the shared router. Now run PART A (live) to confirm routing.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Step-3 intent router gate.")
    ap.add_argument("--arm", choices=list(_ARM_FILE), help="run the no-key wiring check for one arm (PART B)")
    ap.add_argument("--runs", type=int, default=1, help="PART A: runs per row for the majority check (§5)")
    args = ap.parse_args()

    if args.arm:
        sys.exit(run_part_b(args.arm))
    sys.exit(run_part_a(max(1, args.runs)))


if __name__ == "__main__":
    main()
