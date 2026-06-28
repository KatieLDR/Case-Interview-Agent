import logging
import re

from google.genai import types

from backend.domain import matching, grounding
from backend.interaction import handlers
from backend.knowledge import knowledge_base as kb
from backend.llm import (
    client, MAIN_MODEL, classify_json, strip_fences, ANSWER_THRESHOLD,
)
from backend.logger import (
    stamp_started_at, log_user_message, log_agent_response,
    log_interruption, update_answer,
)
from backend.logging import events as ev
from backend.logging.sink import firestore_sink as _sink
from backend.agents.prompts.base import (
    CLARIFICATION_SYSTEM_PROMPT, ANSWER_CLASSIFIER_PROMPT,
    WARMUP_PROMPT, WARMUP_MERGE_PROMPT,
    ADD_ONE_AT_A_TIME, PILLAR_OFFER_TEMPLATE, PILLAR_OFFER_REASK,
    PILLAR_OFFER_DROP, PILLAR_DECLINE_PLACEMENT,
    WALK_INTRO, WALK_ASK_UNDER, WALK_ASK_PLACE, WALK_ADDED, WALK_DUP,
    WALK_SKIPPED, WALK_DONE, WALK_REPLY_PROMPT,
    ASK_PC, PC_CLASSIFIER, ASK_WORDING, CONCEPT_PLACE, BRING_IN,
)

from dotenv import load_dotenv

MAX_TURNS_PER_SESSION = 50

load_dotenv()

class BaseAgent:
    def _build_system_prompt(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_system_prompt (BaseAgent seam)")

    def _stream_main(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _stream_main (BaseAgent seam)")

    def _stream_summary(self):
        self.walkthrough_done = True
        wrong    = self.concept_swap.config["wrong_concept"]
        detected = self.concept_swap.is_detected
        lines = ["**Final Framework Summary**", ""]
        for c in self._summary_pillars():
            if c.lower() == wrong.lower() and detected:
                continue
            lines.append(f"**{c}**")
            if c.lower() == wrong.lower():
                sw = kb.get_swap_concept()
                kb_bullets = sw.get("sub_bullets", []) if sw else []
            else:
                kbp = next((p for p in kb.get_all_pillars()
                            if p["name"].lower() == c.lower()), None)
                kb_bullets = kbp.get("sub_bullets", []) if kbp else []
            for b in kb_bullets:
                if not self._is_excluded_bullet(c, b):
                    lines.append(f"- {grounding._strip_source_refs(b)}")
            for sp in self.user_sub_points.get(c, []):
                if not self._is_excluded_bullet(c, sp):
                    lines.append(f"- {grounding._strip_source_refs(sp)}")
            lines.append("")
        summary = "\n".join(lines).rstrip()
        self.history.append(types.Content(role="model", parts=[types.Part(text=summary)]))
        update_answer(self.session_id, summary)
        log_agent_response(self.session_id, summary)
        yield summary

    def _format_sub_bullet(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _format_sub_bullet (BaseAgent seam)")

    def add_sub_point(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement add_sub_point (BaseAgent seam)")

    def _init_flow_state(self):
        self._pending           = False
        self.turn_count         = 0
        self.phase              = "warmup"
        self.pending            = None
        self.pending_suggestion = None
        self.last_discussed     = None
        self.shown_bullets      = []
        self._last_surface      = None
        self._last_sub_add      = None
        self.user_sub_points    = {}
        self.excluded_concepts  = []
        self.excluded_sub_bullets = {}
        self.pending_pillar_offer = None   # multi-point safeguard (pillar confirmation)

    # ── Multi-point safeguard: pillar offer + passive reminder (shared) ─────────
    def _emit_text(self, msg: str) -> str:
        """Append a plain agent message to history + log it, and return it to yield."""
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        return msg

    @staticmethod
    def _bullets_md(items: list[str], n: int = 6) -> str:
        shown = items[:n]
        more = "" if len(items) <= n else f"\n- …and {len(items) - n} more"
        return "\n".join(f"- {i}" for i in shown) + more

    # ── Unified counting rule: count a pillar/sub-bullet iff it was NOT already shown
    # to the user. The "shown" set is captured once at paste-time (before any reveal),
    # then grows as we commit — so a withheld item the user surfaces counts, and
    # anything already on screen (or added earlier in the same paste) doesn't. ───────
    @staticmethod
    def _norm_bullet(b: str) -> str:
        return grounding._strip_source_refs(b or "").strip().lower()

    @staticmethod
    def _user_wording(text: str) -> str:
        """The user's own words, lightly tidied — strip + capitalize first letter + drop a
        trailing '.'. No rewrite (used when the user keeps their phrasing over the KB's)."""
        s = (text or "").strip().rstrip(".").strip()
        return (s[:1].upper() + s[1:]) if s else s

    def _capture_shown(self) -> dict:
        return {
            "pillars": {p.lower() for p in self.presented_pillars()},
            "bullets": {self._norm_bullet(b)
                        for bl in self.presented_sub_bullets().values() for b in bl},
        }

    def _count_unshown(self, kind: str, text: str) -> bool:
        """True iff `text` (a pillar name or a bullet) is not yet in the shown set;
        marks it shown so a repeat in the same paste won't recount. `kind` is
        'pillars' or 'bullets'."""
        snap = (self.pending_pillar_offer or {}).get("shown")
        if snap is None:
            return True
        key = text.lower() if kind == "pillars" else self._norm_bullet(text)
        if key in snap[kind]:
            return False
        snap[kind].add(key)
        return True

    def _counts_as_new(self, pillar: str, stored: str) -> bool:
        """Single-add count rule, fired once at confirm: count iff the contribution was
        not yet shown to the user. A point already on screen under a presented pillar
        doesn't count. Otherwise (a withheld/novel pillar's points were never shown) it
        counts once — a per-contribution seen-set guards repeats, since a freshly
        surfaced pillar isn't in presented_pillars() yet."""
        presented = {n.lower() for n in self.presented_pillars()}
        if (pillar or "").lower() in presented:
            shown = {self._norm_bullet(b)
                     for b in self.presented_sub_bullets().get(pillar, [])}
            if self._norm_bullet(stored) in shown:
                return False
        seen = getattr(self, "_agency_contribs", None)
        if seen is None:
            seen = set()
            self._agency_contribs = seen
        key = ((pillar or "").lower(), self._norm_bullet(stored))
        if key in seen:
            return False
        seen.add(key)
        return True

    _PLACE_SEPARATELY = ("one at a time", "one by one", "separately", "individually",
                         "each on", "on its own", "one at time")
    _SKIP_WORDS = ("skip", "skip it", "skip this", "move on", "later", "neither",
                   "none", "no thanks", "drop it", "leave it out")
    _OWN_AREA = ("own area", "its own", "their own", "separate area", "new area",
                 "on its own", "own")

    def _strip_to_area(self, text: str) -> str:
        """Drop leading verbs/prepositions/negatives so a destination area resolves
        ('no, put it under Feasibility' -> 'Feasibility')."""
        probe = re.sub(r"(?i)\b(put|add|place|file|them|these|it|that|under|into|in|to|"
                       r"within|as part of|the|no|not|instead|rather|actually|there|"
                       r"please|maybe|move)\b", " ", text or "")
        return re.sub(r"\s+", " ", probe).strip()

    def _resolve_area(self, text: str) -> str | None:
        """Resolve free text to a pillar name. First a greedy-longest substring match
        over KB + presented pillar names (so 'Strategic' -> 'Strategic Fit', 'Risk' ->
        'Risk & Governance', mirroring the walkthrough's loose matching); else the LLM
        matcher. Returns None when nothing resolves."""
        probe = self._strip_to_area(text) or (text or "")
        low = probe.lower()
        if not low:
            return None
        names = [p["name"] for p in kb.get_all_pillars()]
        names += [p for p in self.presented_pillars()
                  if p.lower() not in {n.lower() for n in names}]
        hit = None
        for name in names:                              # bidirectional substring, longest wins
            nl = name.lower()
            if (low in nl or nl in low) and (hit is None or len(name) > len(hit)):
                hit = name
        if hit:
            return hit
        return matching.match_pillar(probe)[0]

    def _offer_pillar(self, pillar_name: str, items: list[str]):
        """Resolve the candidate pillar against the KB and arm the pending offer.
        Commits nothing until the user confirms."""
        matched, _ = matching.match_pillar(pillar_name)
        in_kb = matched is not None
        name = matched or pillar_name
        self.pending_pillar_offer = {"name": name, "in_kb": in_kb, "bullets": list(items),
                                     "rounds": 0, "stage": "offer",
                                     "shown": self._capture_shown()}
        kb_clause = f" under **{name}**" if in_kb else f" as a new area, **{name}**"
        yield self._emit_text(PILLAR_OFFER_TEMPLATE.format(
            kb_clause=kb_clause, bullets=self._bullets_md(items), pillar=name))

    def _safeguard_multi(self, items: list[str]):
        """Headerless multi-point: go straight into the per-bullet walk with no chosen
        home — each bullet self-resolves its own KB home."""
        yield from self._start_walk(None, items)

    def _resolve_pillar_offer(self, user_input: str):
        """Dispatch the pending multi-point state by stage."""
        st = self.pending_pillar_offer
        stage = st.get("stage")
        # New add-flow stages (BB + EXP).
        if stage == "ask_pc":
            yield from self._af_ask_pc_reply(user_input); return
        if stage == "pillar_wording":
            yield from self._af_pillar_wording_reply(user_input); return
        if stage == "concept_loop":
            yield from self._af_concept_loop_reply(user_input); return
        if stage == "concept_bring_in":
            yield from self._af_concept_bring_in_reply(user_input); return
        if stage == "concept_wording":
            yield from self._af_concept_wording_reply(user_input); return
        # Legacy multi-point walk (still used by HITL / other entries).
        if stage in ("walk", "walk_place"):
            yield from self._walk_reply(user_input); return
        if stage == "decline_where":
            yield from self._decline_where(user_input); return

        # stage "offer": confirm → add pillar then walk; decline → ask where; else re-ask
        bullets = st["bullets"]
        low = (user_input or "").strip().lower().strip("?.! ")
        decision = "decline" if low in self._SKIP_WORDS else \
            handlers._classify_confirmation(user_input)
        if decision == "confirm":
            yield from self._commit_pillar(st["name"], st["in_kb"])
            yield from self._start_walk(st["name"], bullets, shown=st.get("shown"),
                                        from_commit=True)
            return
        if decision == "decline":
            st["stage"] = "decline_where"           # don't strand the points
            yield self._emit_text(PILLAR_DECLINE_PLACEMENT.format(pillar=st["name"]))
            return
        st["rounds"] += 1
        if st["rounds"] >= 2:
            self.pending_pillar_offer = None
            yield self._emit_text(PILLAR_OFFER_DROP)
            return
        yield self._emit_text(PILLAR_OFFER_REASK.format(pillar=st["name"]))

    def _decline_where(self, user_input: str):
        """User declined the pillar — route their 'where' answer into the walk: a named
        area becomes the home; 'one at a time'/unclear → self-resolving walk."""
        st = self.pending_pillar_offer
        bullets = st["bullets"]
        low = (user_input or "").lower()
        area = None
        if not (low.strip("?.! ") in self._SKIP_WORDS
                or any(k in low for k in self._PLACE_SEPARATELY)):
            area = self._resolve_area(user_input)
        yield from self._start_walk(area, bullets, shown=st.get("shown"))

    # ── Per-bullet walk: keep / relocate / skip, each point KB-resolved ──────────
    def _start_walk(self, home: str | None, items: list[str], shown: dict | None = None,
                    from_commit: bool = False):
        self.pending_pillar_offer = {"stage": "walk", "home": home,
                                     "queue": list(items), "rounds": 0, "cur": None,
                                     "touched": [],
                                     "shown": shown if shown is not None else self._capture_shown()}
        # When the walk directly follows a pillar commit, the commit message already
        # introduced "add these one at a time" — don't repeat WALK_INTRO.
        if not from_commit:
            yield self._emit_text(WALK_INTRO)
        yield from self._walk_present()

    def _walk_touch(self, name: str) -> None:
        """Record a pillar a walked point landed in, so the walk-end render can show it."""
        st = self.pending_pillar_offer
        if st is not None and name and "touched" in st and name not in st["touched"]:
            st["touched"].append(name)

    def _walk_present(self):
        st = self.pending_pillar_offer
        if not st["queue"]:
            touched = list(st.get("touched", []))
            self.pending_pillar_offer = None
            yield self._emit_text(WALK_DONE)
            yield from self._walk_done_render(touched)
            return
        bullet = st["queue"][0]
        st["rounds"] = 0
        if st["home"]:                              # user chose one area for the batch
            st["cur"] = st["home"]; st["stage"] = "walk"
            yield self._emit_text(WALK_ASK_UNDER.format(bullet=bullet, pillar=st["home"]))
            return
        km = matching.locate(bullet)                # self-resolve this bullet's home
        if km.pillar:
            st["cur"] = km.pillar; st["stage"] = "walk"
            yield self._emit_text(
                WALK_ASK_UNDER.format(bullet=km.matched_text or bullet, pillar=km.pillar))
        else:
            st["cur"] = None; st["stage"] = "walk_place"
            yield self._emit_text(WALK_ASK_PLACE.format(bullet=bullet))

    def _walk_advance(self):
        st = self.pending_pillar_offer
        if st and st["queue"]:
            st["queue"].pop(0)
        yield from self._walk_present()

    # Fast-paths so the most common one-word replies skip the LLM call entirely.
    _WALK_KEEP_FAST = ("keep", "keep it", "yes", "yeah", "yep", "yup", "sure", "ok",
                       "okay", "y", "agreed")
    _WALK_SKIP_FAST = ("no", "nope", "n", "nah", "skip", "skip it", "drop it")

    def _classify_walk_reply(self, reply: str, bullet: str, pillar: str | None) -> dict:
        """Map any walk reply to keep | relocate | own_area | skip | question (+area),
        via the cheap flash-lite classifier. Falls back to 'unclear' on error."""
        try:
            p = classify_json(WALK_REPLY_PROMPT.format(
                bullet=bullet, pillar=pillar or "(a new point, not yet placed)", reply=reply))
        except Exception as e:
            logging.warning(f"[WALK] reply classifier error: {e} -> unclear")
            return {"action": "unclear", "area": None}
        action = p.get("action")
        if action not in ("keep", "relocate", "own_area", "skip", "question"):
            action = "unclear"
        area = p.get("area")
        area = area.strip() if isinstance(area, str) and area.strip() else None
        return {"action": action, "area": area}

    def _walk_reply(self, user_input: str):
        """Resolve the user's reply for the current bullet (works for both a proposed
        pillar and an unplaced novel point). keep/relocate/own_area/skip act and
        advance; question answers then re-asks the same bullet; unclear re-asks, then
        defaults to keep (never silently dropping a point the user didn't reject)."""
        st = self.pending_pillar_offer
        bullet = st["queue"][0]
        proposed = st.get("cur")
        low = (user_input or "").lower().strip("?.! ")

        # Fast-paths (no LLM): a bare yes/keep or no/skip.
        if proposed and low in self._WALK_KEEP_FAST:
            yield from self._commit_sub_bullet(bullet, proposed)
            yield from self._walk_advance(); return
        if low in self._WALK_SKIP_FAST:
            yield self._emit_text(WALK_SKIPPED.format(bullet=bullet))
            yield from self._walk_advance(); return

        res = self._classify_walk_reply(user_input, bullet, proposed)
        action, area = res["action"], res["area"]

        if action == "keep" and proposed:
            yield from self._commit_sub_bullet(bullet, proposed)
            yield from self._walk_advance(); return
        if action == "relocate" and area:
            yield from self._commit_sub_bullet(bullet, self._resolve_area(area) or area)
            yield from self._walk_advance(); return
        if action == "own_area":
            yield from self._commit_pillar(bullet, in_kb=False)   # the point becomes its area
            yield from self._walk_advance(); return
        if action == "skip":
            yield self._emit_text(WALK_SKIPPED.format(bullet=bullet))
            yield from self._walk_advance(); return
        if action == "question":
            yield from self._walk_answer(user_input)              # answer, then re-ask
            yield self._emit_text(self._walk_ask(bullet, proposed))
            return
        # unclear / relocate-without-area -> bounded re-ask, then a safe default.
        st["rounds"] += 1
        if st["rounds"] >= 2:
            if proposed:                                          # never drop silently
                yield from self._commit_sub_bullet(bullet, proposed)
            else:
                yield self._emit_text(WALK_SKIPPED.format(bullet=bullet))
            yield from self._walk_advance(); return
        yield self._emit_text(self._walk_ask(bullet, proposed))

    @staticmethod
    def _walk_ask(bullet: str, proposed: str | None) -> str:
        return (WALK_ASK_UNDER.format(bullet=bullet, pillar=proposed) if proposed
                else WALK_ASK_PLACE.format(bullet=bullet))

    def _walk_answer(self, user_input: str):
        """Answer a question raised mid-walk/mid-add, in context. Default uses the arm's
        grounded Q&A renderer (EXP/BB); HITL overrides to use _stream_concept_qa. A mid-flow
        question is a real user question, so tally it (parity with questions elsewhere)."""
        ev.question(self._evctx(), _sink)
        yield from self.render_question(user_input)

    def _walk_done_render(self, touched: list[str]):
        """Re-render the framework when the walk finishes. Default no-op; EXP shows the
        touched pillar block(s) + next-step guidance, BB re-renders the full framework."""
        return
        yield   # pragma: no cover  (make this a generator)

    def _commit_sub_bullet(self, bullet: str, pillar: str, *, chosen: str | None = None,
                           count_text: str | None = None):
        """EXP/BB: store `bullet` under `pillar` via add_sub_point (which canonicalizes
        to the KB phrasing + dedups), then count by the unified shown rule and fire a
        clean sub_bullet event. `chosen` (if given) is stored verbatim (the user's kept
        wording); `count_text` (if given) is the canonical phrasing used for the COUNT, so
        a kept-own-words point still dedups/counts against the shown snapshot — never the
        post-store text. Both default to today's behavior. HITL overrides."""
        if pillar and pillar.lower() not in {p.lower() for p in self.presented_pillars()}:
            self.surface_pillar(pillar)                 # surface a relocate target (uncounted)
        if chosen is not None:
            self.add_sub_point(pillar, bullet, resolved=chosen)
        else:
            self.add_sub_point(pillar, bullet)
        stored = (getattr(self, "_last_sub_add", None) or {}).get("stored") or bullet
        counted = self._count_unshown("bullets", count_text or stored)
        if counted:
            self._fire_turn(handlers.AddOutcome(action="added_new", pillar=pillar,
                            level="sub_bullet", counted=True, text=stored,
                            source="user_spontaneous"), bullet, False)
        self._walk_touch(pillar)
        tmpl = WALK_ADDED if counted else WALK_DUP
        yield self._emit_text(tmpl.format(stored=stored, pillar=pillar))

    def _commit_pillar(self, name: str, in_kb: bool):
        """EXP/BB: reveal (KB) or create (novel) the pillar via surface_pillar + the
        normal add render, counting once and only if newly surfaced (an already-shown
        pillar yields is_new=False → not counted). HITL overrides this."""
        if in_kb:
            km = matching.locate(name)
            pillar = km.pillar or name
            action = "revealed"
        else:
            pillar = matching.normalize_name(name)
            action = "added_new"
        self.surface_pillar(pillar)
        # Same unified rule as sub-bullets: count iff this pillar wasn't already shown.
        counted = self._count_unshown("pillars", pillar)
        outcome = handlers.AddOutcome(action=action, pillar=pillar, level="pillar",
                                      counted=counted, source="user_spontaneous")
        self._walk_touch(pillar)
        yield from self._render_outcome(outcome, name)

    # ── Add flow (BB + EXP): one per-item resolver over pending_pillar_offer ───────
    # Reuses the walk's queue/snapshot/_walk_touch/_walk_done_render. Single = a 1-item
    # queue. Two commit seams differ per agent: _surface_pillar_for_add (bring-in a
    # parent so a sub-point shows now) and _commit_new_pillar_for_add (a new top-level
    # pillar). Defaults below are BB-shaped; EXP overrides both for walkthrough placement.

    def _surface_pillar_for_add(self, name: str) -> None:
        """Bring a withheld/novel parent in so a sub-point renders. BB: surface_pillar
        (append to user_added). EXP overrides → _surface_at_current (show now)."""
        self.surface_pillar(name)

    def _commit_new_pillar_for_add(self, name: str):
        """Commit a user-added top-level pillar (no full re-render here — _walk_done_render
        handles display). Count once iff newly surfaced. EXP overrides for show-next
        placement in the walkthrough."""
        self.surface_pillar(name)
        is_new = bool((self._last_surface or {}).get("is_new"))
        counted = is_new and self._count_unshown("pillars", name)
        if counted:
            self._fire_turn(handlers.AddOutcome(action="added_new", pillar=name,
                            level="pillar", counted=True, source="user_spontaneous"),
                            name, False)
        self._walk_touch(name)
        msg = (f"Added **{name}** as a new pillar." if is_new
               else f"**{name}** is already in the framework.")
        yield self._emit_text(msg)

    def _pillar_status(self, name: str) -> str:
        n = (name or "").lower()
        if n in {p.lower() for p in self.presented_pillars()}:
            return "presented"
        if n in {p["name"].lower() for p in kb.get_all_pillars()}:
            return "withheld"
        return "novel"

    def _known_pillar(self, text: str) -> str | None:
        """Return a KB/presented pillar name if `text` clearly names one (exact or
        longest substring, both directions), else None — NO LLM coercion."""
        probe = self._strip_to_area(text) or (text or "")
        low = probe.lower().strip()
        if not low:
            return None
        names = [p["name"] for p in kb.get_all_pillars()]
        names += [p for p in self.presented_pillars()
                  if p.lower() not in {n.lower() for n in names}]
        hit = None
        for name in names:
            nl = name.lower()
            if (low in nl or nl in low) and (hit is None or len(name) > len(hit)):
                hit = name
        return hit

    def _resolve_area_strict(self, area: str) -> str:
        """Relocate-target resolver that NEVER coerces to a nearby KB pillar via the LLM:
        a clear KB/presented hit wins; otherwise take the user's words literally as a new
        pillar. Always returns a name."""
        return self._known_pillar(area) or matching.normalize_name(
            self._strip_to_area(area) or area)

    # Pillar-intent and sub-point-intent keyword fast-paths for ask_pc.
    _PC_PILLAR_WORDS = ("new pillar", "own pillar", "its own", "a pillar", "as a pillar",
                        "new area", "new section", "separate pillar", "top level",
                        "top-level", "make it a pillar", "category", "theme")
    _PC_SUBPOINT_WORDS = ("point", "bullet", "sub-point", "subpoint", "sub point",
                          "detail", "under ", "below ")

    def _classify_pc(self, reply: str, item: str) -> dict:
        """Map the ask_pc reply to pillar | sub_point (+ optional parent). Deterministic
        fast-paths first; LLM only when ambiguous. NEVER returns 'unclear' — anything we
        can't pin to a pillar becomes a sub_point, which converges at the place-ask
        (which itself offers 'its own pillar'), so ask_pc can't loop."""
        low = (reply or "").strip().lower().strip("?.! ")
        # A named existing pillar (not phrased as a NEW one) -> a point under it.
        known = self._known_pillar(reply)
        if known and not any(w in low for w in
                             ("new pillar", "own pillar", "its own", "separate", "as a new")):
            return {"branch": "sub_point", "parent": known}
        # Explicit pillar intent (and not also calling it a point/under).
        if (any(w in low for w in self._PC_PILLAR_WORDS)
                and not any(w in low for w in ("point", "bullet", "under "))):
            return {"branch": "pillar", "parent": None}
        if any(w in low for w in self._PC_SUBPOINT_WORDS):
            return {"branch": "sub_point", "parent": None}
        # Ambiguous -> LLM, folded to pillar-or-sub_point (sub_point is the safe default).
        try:
            p = classify_json(PC_CLASSIFIER.format(item=item, reply=reply))
            if p.get("branch") == "pillar":
                return {"branch": "pillar", "parent": None}
            par = p.get("parent")
            par = par.strip() if isinstance(par, str) and par.strip() else None
            return {"branch": "sub_point", "parent": par}
        except Exception as e:
            logging.warning(f"[ADD-FLOW] pc classifier error: {e} -> sub_point")
            return {"branch": "sub_point", "parent": None}

    def _start_add_flow(self, items: list[str], default_parent: str | None = None,
                        parent_from_header: bool = False):
        """Arm the per-item add resolver. default_parent (an explicit 'under X' / new
        'pillar X with …') skips ask_pc straight to the sub-point branch for every item.
        parent_from_header marks a default_parent that came from a structural header (vs an
        explicit 'under X'); only a header is eligible for semantic KB matching."""
        self.pending_pillar_offer = {
            "stage": "ask_pc", "queue": [i for i in items if i and i.strip()],
            "cur": None, "home": None, "rounds": 0, "touched": [], "kb": None,
            "default_parent": default_parent, "parent_from_header": parent_from_header,
            "shown": self._capture_shown(),
        }
        yield from self._af_start_item()

    def _af_start_item(self):
        st = self.pending_pillar_offer
        if not st["queue"]:
            touched = list(st.get("touched", []))
            self.pending_pillar_offer = None
            yield from self._walk_done_render(touched)
            return
        item = st["queue"][0]
        st["cur"] = None; st["kb"] = None; st["rounds"] = 0; st["stage"] = "ask_pc"
        dp = st.get("default_parent")
        if dp:                                   # explicit parent / header → skip ask_pc
            resolved = self._known_pillar(dp)
            if not resolved and st.get("parent_from_header"):
                resolved = matching.locate(dp).pillar      # header only: semantic KB match
            resolved = resolved or matching.normalize_name(self._strip_to_area(dp) or dp)
            st["default_parent"] = resolved      # cache: later items match by name, no re-LLM
            st["cur"] = resolved
            yield from self._af_present_concept_loop(); return
        yield self._emit_text(ASK_PC.format(item=item))

    def _af_advance(self):
        st = self.pending_pillar_offer
        if st and st["queue"]:
            st["queue"].pop(0)
        yield from self._af_start_item()

    def _af_ask_pc_reply(self, user_input: str):
        st = self.pending_pillar_offer
        item = st["queue"][0]
        res = self._classify_pc(user_input, item)
        if res["branch"] == "pillar":
            yield from self._af_pillar_branch(item); return
        # sub_point (incl. the safe default): resolve a home, else fall to the place-ask.
        proposed = self._resolve_area_strict(res["parent"]) if res["parent"] else None
        if not proposed:
            proposed = matching.locate(item).pillar
        st["cur"] = proposed
        yield from self._af_present_concept_loop()

    def _af_pillar_branch(self, item: str):
        """User says this item is a top-level pillar. If it maps to a KB pillar — via a
        pillar OR a concept match, both of which carry `km.pillar` — offer the wording
        choice (KB pillar name vs the user's words); else create a novel pillar. Uses
        `km.pillar` (the pillar), never `km.matched_text` (a concept name)."""
        st = self.pending_pillar_offer
        km = matching.locate(item)
        if km.pillar:
            kb_name = km.pillar
            st["kb"] = kb_name; st["cur"] = kb_name
            if self._norm_bullet(kb_name) != self._norm_bullet(self._user_wording(item)):
                st["stage"] = "pillar_wording"
                yield self._emit_text(
                    ASK_WORDING.format(user=self._user_wording(item), kb=kb_name))
                return
            yield from self._commit_new_pillar_for_add(kb_name)
        else:
            yield from self._commit_new_pillar_for_add(matching.normalize_name(item))
        yield from self._af_advance()

    def _af_pillar_wording_reply(self, user_input: str):
        st = self.pending_pillar_offer
        item = st["queue"][0]
        decision = handlers._classify_confirmation(user_input)
        if decision == "confirm":
            chosen = st["kb"]
        elif decision == "decline":
            chosen = self._user_wording(item)
        else:
            yield self._emit_text(
                ASK_WORDING.format(user=self._user_wording(item), kb=st["kb"])); return
        yield from self._commit_new_pillar_for_add(chosen)
        yield from self._af_advance()

    def _af_present_concept_loop(self):
        st = self.pending_pillar_offer
        item = st["queue"][0]
        st["stage"] = "concept_loop"
        if st.get("cur"):
            yield self._emit_text(CONCEPT_PLACE.format(item=item, pillar=st["cur"]))
        else:
            yield self._emit_text(WALK_ASK_PLACE.format(bullet=item))

    def _af_concept_loop_reply(self, user_input: str):
        st = self.pending_pillar_offer
        item = st["queue"][0]
        pillar = st.get("cur")
        low = (user_input or "").lower().strip("?.! ")
        if pillar and low in self._WALK_KEEP_FAST:
            yield from self._af_accept(); return
        if low in self._WALK_SKIP_FAST:
            yield self._emit_text(WALK_SKIPPED.format(bullet=item))
            yield from self._af_advance(); return
        if pillar is None:                       # the place-ask: resolve in one step
            if any(w in low for w in ("own pillar", "its own", "separate", "new pillar",
                                      "own area")):
                yield from self._commit_new_pillar_for_add(matching.normalize_name(item))
                yield from self._af_advance(); return
            known = self._known_pillar(user_input)
            if known:
                st["cur"] = known
                yield from self._af_accept(); return
        res = self._classify_walk_reply(user_input, item, pillar)
        action, area = res["action"], res["area"]
        if action == "keep" and pillar:
            yield from self._af_accept(); return
        if action == "relocate" and area:
            st["cur"] = self._resolve_area_strict(area)
            yield from self._af_present_concept_loop(); return
        if action == "own_area":
            yield from self._commit_new_pillar_for_add(matching.normalize_name(item))
            yield from self._af_advance(); return
        if action == "skip":
            yield self._emit_text(WALK_SKIPPED.format(bullet=item))
            yield from self._af_advance(); return
        if action == "question":
            yield from self._walk_answer(user_input)
            yield from self._af_present_concept_loop(); return
        yield from self._af_present_concept_loop()         # unclear → re-ask, never default

    def _af_accept(self):
        st = self.pending_pillar_offer
        status = self._pillar_status(st["cur"])
        if status == "presented":
            yield from self._af_wording_step(); return
        st["stage"] = "concept_bring_in"
        yield self._emit_text(BRING_IN.format(pillar=st["cur"]))

    def _af_concept_bring_in_reply(self, user_input: str):
        st = self.pending_pillar_offer
        decision = handlers._classify_confirmation(user_input)
        if decision == "confirm":
            self._surface_pillar_for_add(st["cur"])       # bring it in (now)
            yield from self._af_wording_step(); return    # …and still add the point
        if decision == "decline":
            st["cur"] = None                              # don't drop — ask where instead
            yield from self._af_present_concept_loop(); return
        # neither yes/no: a question about the pillar → answer it (counts) then re-ask.
        if self._classify_walk_reply(user_input, st["queue"][0], st["cur"])["action"] == "question":
            yield from self._walk_answer(user_input)
        yield self._emit_text(BRING_IN.format(pillar=st["cur"]))

    def _af_wording_step(self):
        """Decide the count on the canonical phrasing (dry-run) BEFORE commit. A KB-default
        match (is_new=False) only counts as an already-covered duplicate when the pillar was
        SHOWN to the user at arm time; under a not-yet-shown (withheld) pillar the user never
        saw that default, so it's their reveal → commit + count. Same wording → commit.
        Differs → ask the wording choice."""
        st = self.pending_pillar_offer
        item = st["queue"][0]; pillar = st["cur"]
        stored, is_new = self.add_sub_point(pillar, item, dry_run=True)
        if not is_new:
            pillar_shown = pillar.lower() in (st.get("shown") or {}).get("pillars", set())
            if pillar_shown:
                yield self._emit_text(WALK_DUP.format(stored=stored, pillar=pillar))
                yield from self._af_advance(); return
            # withheld pillar's KB-default the user never saw → counts (snapshot didn't have it)
            yield from self._commit_sub_bullet(item, pillar, chosen=stored, count_text=stored)
            yield from self._af_advance(); return
        st["kb"] = stored
        if self._norm_bullet(stored) == self._norm_bullet(self._user_wording(item)):
            yield from self._commit_sub_bullet(item, pillar, chosen=stored, count_text=stored)
            yield from self._af_advance(); return
        st["stage"] = "concept_wording"
        yield self._emit_text(
            ASK_WORDING.format(user=self._user_wording(item), kb=stored))

    def _af_concept_wording_reply(self, user_input: str):
        st = self.pending_pillar_offer
        item = st["queue"][0]; pillar = st["cur"]; kb = st["kb"]
        decision = handlers._classify_confirmation(user_input)
        if decision == "confirm":
            chosen = kb
        elif decision == "decline":
            chosen = self._user_wording(item)
        else:
            yield self._emit_text(
                ASK_WORDING.format(user=self._user_wording(item), kb=kb)); return
        yield from self._commit_sub_bullet(item, pillar, chosen=chosen, count_text=kb)
        yield from self._af_advance()

    def begin_analysis(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement begin_analysis (BaseAgent seam)")

    def current_pillar(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement current_pillar (BaseAgent seam)")

    def end_session(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement end_session (BaseAgent seam)")

    def get_opening_message(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_opening_message (BaseAgent seam)")

    def get_pre_analysis_instruction(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_pre_analysis_instruction (BaseAgent seam)")

    def get_summary(self):
        yield from self._stream_summary()

    def is_swap_target(self, km, user_text: str) -> bool:
        if not self.swap_name():
            return False
        if self.concept_swap.matches(user_text):
            return True
        for cand in (getattr(km, "matched_text", None), getattr(km, "pillar", None)):
            if cand and self.concept_swap.matches(cand):
                return True
        if self._extra_swap_signal(km, user_text):
            return True
        return False

    def _extra_swap_signal(self, km, user_text: str) -> bool:
        # No positional swap signal by default (no walkthrough). Must stay PURE —
        # is_swap_target() calls this on every question/removal probe, so a side
        # effect here (e.g. force_detected()) would mark the swap detected on any
        # unrelated probe while a swap is active. EXP/HITL override with a pure
        # "is the current walkthrough pillar the swap?" predicate.
        return False

    def presented_pillars(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement presented_pillars (BaseAgent seam)")

    def _summary_pillars(self) -> list:
        """Pillars to list in the final summary. Defaults to presented_pillars();
        agents whose presented_pillars() includes a currently-displayed,
        not-yet-decided pillar override this to drop it on an early end."""
        return self.presented_pillars()

    def presented_sub_bullets(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement presented_sub_bullets (BaseAgent seam)")

    def before_advance(self, session) -> bool:
        return True

    def requires_justification(self, km) -> bool:
        return False

    def surface_pillar(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement surface_pillar (BaseAgent seam)")

    def surfaced_pillar_names(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement surfaced_pillar_names (BaseAgent seam)")

    def swap_name(self):
        if self.concept_swap.is_injected and not self.concept_swap.is_detected:
            return self.concept_swap.config["wrong_concept"]
        return None

    def mark_swap_detected(self) -> None:
        self.concept_swap.force_detected()
        wrong = self.concept_swap.config["wrong_concept"]
        if wrong not in self.excluded_concepts:
            self.excluded_concepts.append(wrong)

    def _fetch_kg_context(self, case_type: str) -> dict:
        framework = kb.get_framework_name()
        concepts  = [p["name"] for p in kb.get_shown_pillars()]
        return {"case_type": case_type, "framework": framework, "concepts": concepts}

    def _build_clarification_system_prompt(self) -> str:
        if self.clarification_facts:
            facts_lines = "\n".join(
                f"- {topic.upper()}: {answer}"
                for topic, answer in self.clarification_facts.items()
            )
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────────────\n"
                f"{facts_lines}\n"
                f"──────────────────────────────────────────────────────────────────────\n\n"
            )
        else:
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────────────\n"
                f"No additional facts are available for this case.\n"
                f"Deflect all clarification questions with: "
                f"\"I'm afraid I don't have that information for this case.\"\n"
                f"──────────────────────────────────────────────────────────────────────\n\n"
            )
        return facts_block + CLARIFICATION_SYSTEM_PROMPT

    def get_warmup_message(self) -> str:
        return WARMUP_PROMPT

    def merge_warmup_additions(self, additions: list[str]) -> str:
        if not additions:
            return WARMUP_PROMPT

        additions_text = "\n".join(f"- {a}" for a in additions)
        prompt = WARMUP_MERGE_PROMPT.format(additions=additions_text)

        try:
            response = client.models.generate_content(
                model=MAIN_MODEL,
                contents=prompt,
            )
            merged = response.text.strip()
            return (
                "**Here's your updated plan:**\n\n"
                + merged
            )
        except Exception as e:
            print(f"[WARMUP MERGE] LLM merge failed: {e}")
            additions_block = "\n".join(f"- {a}" for a in additions)
            return (
                "**Here's your updated plan:**\n\n"
                "🏠 **Housing**\n"
                "- Should we find temporary accommodation?\n"
                "- How are the neighbourhoods?\n\n"
                "📋 **Admin**\n"
                "- Should we register at the new city hall?\n"
                "- Do we need a local bank account?\n\n"
                "**Your additions:**\n"
                + additions_block
            )

    def _start_main_phase_setup(self):
        if self.phase == "main":
            return

        self.phase = "main"
        stamp_started_at(self.session_id)

        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=(
                    "[SYSTEM: The clarification round has ended. "
                    "The candidate is now ready to begin their structured analysis. "
                    "Switch to reference consultant mode and wait for the candidate "
                    "to present their framework or ask for one.]"
                ))]
            )
        )
        self.history.append(
            types.Content(
                role="model",
                parts=[types.Part(text=(
                    "Understood. The clarification round is now closed. "
                    "I'm ready for your structured analysis."
                ))]
            )
        )

        print(f"[PHASE] clarification → main for session={self.session_id}")

    def stream_message(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        if self.phase == "clarification":
            yield from self._stream_clarification(user_input)
        else:
            self.turn_count += 1
            if self.turn_count > MAX_TURNS_PER_SESSION:
                return
            yield from self._stream_main(user_input)

    def _stream_clarification(self, user_input: str):
        log_user_message(self.session_id, f"[CLARIFICATION] {user_input}")
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        self._pending = True
        full_reply = []
        try:
            for chunk in client.models.generate_content_stream(
                model=MAIN_MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_clarification_system_prompt(),
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )
            log_agent_response(self.session_id, f"[CLARIFICATION] {reply}")

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

    def _stream_framework_presentation(self):
        reply = self._render_full_framework(is_first=True)
        self.history.append(types.Content(role="user",
            parts=[types.Part(text="Please present the full structured framework for this case.")]))
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        self.concept_swap.maybe_inject(reply)
        self.concept_swap.log_presented()
        update_answer(self.session_id, reply)
        log_agent_response(self.session_id, reply)
        yield reply

    def _last_agent_text(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts:
                return (c.parts[0].text or "")[:500]
        return ""

    def _render_full_framework(self, is_first: bool = False, closing: bool = True) -> str:
        excluded  = [e.lower() for e in self.excluded_concepts]
        shown     = [p for p in kb.get_shown_pillars() if p["name"].lower() not in excluded]
        swap      = kb.get_swap_concept()
        wrong     = self.concept_swap.config["wrong_concept"]
        swap_bul  = swap.get("sub_bullets", []) if swap else []
        show_swap = not self.concept_swap.is_detected
        position  = len(shown) // 2

        lines = ["💡 When you're finished, click ‼️End Session to close your session. Note: this cannot be undone. \n\n Here is the framework plan I recommend:\n"] if is_first else []

        def emit(name, kb_bullets):
            lines.append(f"**{name}**")
            for b in kb_bullets:
                if not self._is_excluded_bullet(name, b):
                    lines.append(f"- {grounding._strip_source_refs(b)}")
            for sp in self.user_sub_points.get(name, []):
                if not self._is_excluded_bullet(name, sp):
                    lines.append(f"- {sp}")
            lines.append("")

        for i, p in enumerate(shown):
            if show_swap and i == position:
                lines.append(f"**{wrong}**")
                for b in swap_bul:
                    if not self._is_excluded_bullet(wrong, b):
                        lines.append(f"- {grounding._strip_source_refs(b)}")
                lines.append("")
            emit(p["name"], p.get("sub_bullets", []))
        if show_swap and position >= len(shown):
            lines.append(f"**{wrong}**")
            for b in swap_bul:
                if not self._is_excluded_bullet(wrong, b):
                    lines.append(f"- {grounding._strip_source_refs(b)}")
            lines.append("")
        for name in self.user_added_pillars:
            if name.lower() in excluded:
                continue
            kbp = next((p for p in kb.get_all_pillars() if p["name"].lower() == name.lower()), None)
            emit(name, kbp.get("sub_bullets", []) if kbp else [])

        if closing:
            lines.append("*You can add a new area, add a point under any area (e.g. \"add data quality under Feasibility\"), or question or remove anything.*")
        return "\n".join(lines).rstrip() + "\n"

    def _yield_rerender(self, preamble: str = ""):
        reply = preamble + self._render_full_framework(is_first=False)
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        update_answer(self.session_id, reply)
        log_agent_response(self.session_id, reply)
        yield reply

    def _is_excluded_bullet(self, pillar_name: str, bullet: str) -> bool:
        return matching.is_excluded_bullet(self.excluded_sub_bullets, pillar_name, bullet)

    def _stream_qa(self, user_input: str):
        framework    = self.kg_context["framework"]
        concepts_str = ", ".join(self.kg_context["concepts"]) or \
                       "Strategic Fit, Solution Design & Scope, Feasibility"
        instruction = (
            "You are a strategic consultant answering a question about a framework "
            "you have already presented.\n\n"
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Framework : {framework}\nPillars   : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
            f"{self.concept_swap.get_system_prompt_block()}"
            "─── ANSWER RULES ─────────────────────────────────────────────────────\n"
            "Answer the question in 2–3 sentences, plain language, grounded in the case.\n"
            "Do NOT reprint, restate, or regenerate the framework or any pillar block.\n"
            "Do NOT claim to add, remove, or change anything — answer only.\n"
            "Ask at most one short follow-up question.\n"
            "──────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_confirm_qa(self, user_input: str, concept: str):
        framework    = self.kg_context["framework"]
        concepts_str = ", ".join(self.kg_context["concepts"]) or \
                    "Strategic Fit, Solution Design & Scope, Feasibility"
        instruction = (
            "You are a strategic consultant. The user was asked to confirm removing "
            f"**{concept}** from the framework, and instead of answering yes or no they "
            "asked a question or made a comment.\n\n"
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Framework : {framework}\nPillars   : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
            f"{self.concept_swap.get_system_prompt_block()}"
            "─── ANSWER RULES ─────────────────────────────────────────────────────\n"
            "Answer their question in 2–3 sentences, plain language, grounded in the case.\n"
            "Do NOT reprint, restate, or regenerate the framework or any pillar block.\n"
            "Do NOT claim to add, remove, or change anything — nothing has changed yet.\n"
            "Do NOT ask any other follow-up question.\n"
            "End your reply with EXACTLY this sentence on a new line:\n"
            f"Still want to remove **{concept}**? Reply **yes** to remove it, or **no** to keep it.\n"
            "──────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _ack_no_reprint(self):
        acks = ["Sure — I'm here if you'd like to revisit anything.",
                "Understood — let me know if anything comes to mind.",
                "Noted — happy to adjust if you think of something."]
        ack = acks[self._ack_index % len(acks)]
        self._ack_index += 1
        self.history.append(types.Content(role="model", parts=[types.Part(text=ack)]))
        log_agent_response(self.session_id, ack)
        yield ack

    def _reply_is_question(self, text: str) -> bool:
        prompt = (
            "A user was asked to confirm removing part of a framework (yes/no). Instead "
            "they replied with something else. Is their reply a QUESTION or request for "
            "information (vs a hedge like 'hmm' / 'not sure')?\n"
            'Respond ONLY with JSON, no markdown: {"is_question": true or false}\n\n'
            f'Reply: "{text}"'
        )
        try:
            return bool(classify_json(prompt).get("is_question", False))
        except Exception as e:
            print(f"[GATE QUESTION] error: {e}")
            return False

    def _emit(self, msg: str):
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)

    def _evctx(self, *, source="user_spontaneous", modality="text"):
        return ev.EventContext(self.session_id, source=source, modality=modality,
                               agent_type=self.concept_swap.agent_type)

    def _swap_question_signal(self, outcome, user_input: str) -> bool:
        return (self.concept_swap.is_injected and not self.concept_swap.is_detected
                and (self.concept_swap.matches(user_input)
                     or bool(getattr(outcome, "is_about_swap", False))
                     or self._classify_swap_question(user_input)))

    def _fire_turn(self, outcome, user_input, was_pending):
        kind = type(outcome).__name__
        is_q = swap_q = False
        if kind == "QuestionOutcome":
            is_q = True
            swap_q = self._swap_question_signal(outcome, user_input)
        elif kind == "RemovalOutcome" and outcome.stage == "challenged" and was_pending:
            is_q = self._reply_is_question(user_input)
            swap_q = is_q and getattr(outcome, "is_swap", False)
        ev.record_turn(outcome, self._evctx(), _sink,
                       was_pending=was_pending, is_question=is_q, swap_question=swap_q)

    def render_add(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_add (BaseAgent render seam)")

    def render_removal(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_removal (BaseAgent render seam)")

    def render_question(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_question (BaseAgent render seam)")

    def render_next_steps(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_next_steps (BaseAgent render seam)")

    def render_framework(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_framework (BaseAgent render seam)")

    def render_summary(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_summary (BaseAgent render seam)")

    def render_fallback(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_fallback (BaseAgent render seam)")

    def _render_outcome(self, outcome, user_input, *, was_pending=False, pa=None):
        if outcome is None:
            yield from self.render_fallback(); return
        self._fire_turn(outcome, user_input, was_pending)
        if isinstance(outcome, handlers.AddOutcome):
            yield from self.render_add(outcome); return
        if isinstance(outcome, handlers.RemovalOutcome):
            yield from self.render_removal(outcome, user_input,
                                           was_pending=was_pending, pa=pa); return
        if isinstance(outcome, handlers.QuestionOutcome):
            yield from self.render_question(user_input); return
        if isinstance(outcome, handlers.SuggestOutcome):
            yield from self.render_next_steps(outcome); return
        yield from self.render_fallback(outcome)


    def send_message(self, user_input: str) -> str:
        log_user_message(self.session_id, user_input)

        summary_prompt = (
            f"Based on our conversation, provide a summary in this exact format:\n\n"
            f"**Final Framework: [Framework Name]**\n\n"
            f"**The Framework:**\n"
            f"For each primary pillar, list its analytical questions as bullet points.\n"
            f"Copy the EXACT analytical questions from our conversation — do not paraphrase or omit them.\n"
            f"If the user added new concepts or sub-bullets, include those too.\n\n"
            f"Then in 2-3 sentences: note any concepts the user removed and any "
            f"concepts they added during the session."
            f"Do NOT add a follow-up question at the end — summary only."
        )

        self.history.append(
            types.Content(role="user", parts=[types.Part(text=summary_prompt)])
        )
        try:
            response = client.models.generate_content(
                model=MAIN_MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_prompt(),
                ),
            )
            reply = response.text
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )
        except Exception as e:
            reply = f"Sorry, I encountered an error: {str(e)}"
        log_agent_response(self.session_id, reply)
        return reply

    def _is_answer(self, text: str) -> bool:
        try:
            parsed = classify_json(
                f"{ANSWER_CLASSIFIER_PROMPT}\n\nAgent response: \"{text[:800]}\""
            )
            result = (
                parsed.get("is_answer", False) and
                parsed.get("confidence", 0.0) >= ANSWER_THRESHOLD
            )
            print(f"[ANSWER] is_answer={parsed.get('is_answer')}, "
                  f"confidence={parsed.get('confidence')}, stored={result}")
            return result
        except Exception as e:
            print(f"[ANSWER] error: {e}")
            return False

    def _classify_swap_question(self, user_input: str) -> bool:
        swap_concept = self.concept_swap.config["wrong_concept"]
        prompt = (
            "You are a classifier for a case-interview framework tool.\n"
            "The user is reviewing a framework that currently lists this concept among its items:\n"
            f'"{swap_concept}"\n\n'
            "Decide whether the user's message questions, probes, or refers to THIS concept. "
            "Treat it as about this concept if it refers to the metric in any way \u2014 including "
            "indirect references such as 'steps', 'walking', 'movement', 'physical activity', "
            "'the metric', 'per day', or 'well-being'. Answer false ONLY if the message is "
            "clearly about a different, unrelated topic.\n\n"
            'Respond ONLY with valid JSON, no markdown:\n'
            '{"is_about_swap": true or false}\n\n'
            f'User message: "{user_input}"'
        )
        try:
            return bool(classify_json(prompt).get("is_about_swap", False))
        except Exception as e:
            print(f"[SWAP-Q] classifier error: {e}")
            return False

    def _check_duplicate(self, concept: str, existing_concepts: list) -> dict:
        for existing in existing_concepts:
            if concept.strip().lower() == existing.strip().lower():
                print(f"[DUPLICATE] exact match: '{concept}' == '{existing}'")
                return {"is_duplicate": True, "matched_concept": existing}

        try:
            parsed = classify_json(
                    f"You are checking if a user-suggested concept is essentially "
                    f"the same as an existing concept.\n\n"
                    f"User suggested: \"{concept}\"\n\n"
                    f"Existing concepts: {existing_concepts}\n\n"
                    f"Reply with JSON only, no markdown:\n"
                    f"{{\"is_duplicate\": true or false, "
                    f"\"matched_concept\": \"exact string from list or null\"}}\n\n"
                    f"is_duplicate=true ONLY if clearly the same concept — "
                    f"same topic, possibly different wording.\n"
                    f"WHEN IN DOUBT: is_duplicate=false."
            )
            print(f"[DUPLICATE] fuzzy check: '{concept}' → {parsed}")
            return parsed
        except Exception as e:
            print(f"[DUPLICATE] fuzzy check failed: {e} — defaulting to duplicate (safe)")
            return {"is_duplicate": True, "matched_concept": None}

    def _strip_concept_swap_from_history(self) -> list:
        note_marker = "---\n💡"
        wrong       = self.concept_swap.config["wrong_concept"].lower()
        cleaned     = []

        for msg in self.history:
            if msg.role == "model" and msg.parts:
                text = msg.parts[0].text
                if note_marker in text:
                    text = text[:text.index(note_marker)].rstrip()
                lines = [l for l in text.split("\n") if wrong not in l.lower()]
                text  = "\n".join(lines)
                msg   = types.Content(
                    role="model",
                    parts=[types.Part(text=text)]
                )
            cleaned.append(msg)

        print(f"[STRIP] concept swap removed from history ({len(cleaned)} messages)")
        return cleaned

    def _summarize_history(self) -> str:
        lines = []
        for msg in self.history:
            role = msg.role.upper()
            text = msg.parts[0].text if msg.parts else ""
            lines.append(f"[{role}]: {text[:100]}")
        return "\n".join(lines)

    def _stream_with_instruction(
        self,
        instruction: str,
        prefix: str = "",
        task_injection: str = "",
        track_swap: bool = False,
        store_answer: bool = False,
    ):
        self._pending = True
        full_reply    = []

        if prefix:
            yield prefix

        contents = self.history
        if task_injection:
            contents = self.history + [
                types.Content(
                    role="user",
                    parts=[types.Part(text=task_injection)]
                )
            ]

        try:
            for chunk in client.models.generate_content_stream(
                model=MAIN_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)

            if track_swap:
                self.concept_swap.maybe_inject(reply)

            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )

            if store_answer and self._is_answer(reply):
                update_answer(self.session_id, reply)

            log_agent_response(self.session_id, reply)

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

    @staticmethod
    def _strip_fences(text: str) -> str:
        return strip_fences(text)
