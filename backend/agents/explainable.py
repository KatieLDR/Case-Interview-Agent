import logging
import re

from google.genai import types

from backend.agents.base import BaseAgent
from backend.domain import grounding, matching
from backend.interaction import handlers, intents
from backend.knowledge import knowledge_base as kb
from backend.knowledge.cases import get_case, get_clarification_facts
from backend.llm import client, CLASSIFIER_MODEL
from backend.logger import (
    create_session, log_user_message, log_agent_response, log_interruption,
)
from backend.logging import events as ev
from backend.logging.sink import firestore_sink as _sink
from backend.agents.prompts.explainable import (
    SOURCE_NAMES, SINGLE_CONCEPT_PROMPT, CONCEPT_QA_PROMPT,
    SUMMARY_PROMPT, SWAP_CAUGHT_PROMPT, ADVANCE_CLASSIFIER_PROMPT,
    CLARIFY_DOUBT_PROMPT, CLARIFY_RESOLVE_PROMPT, REMOVAL_TARGET_PROMPT,
    ADD_CLASSIFY_PROMPT, ADD_RESOLVE_PROMPT, SUB_BULLET_FORMAT_PROMPT,
)
from backend.agents.prompts.base import ADD_ONE_AT_A_TIME
from backend.tools.concept_swap import ConceptSwap

CASE_TYPE = "AI Implementation"
ADVANCE_THRESHOLD = 0.75
CLARIFY_DOUBT_THRESHOLD = 0.7
_SRC_ENTRY_RE  = re.compile(r"\[([a-z])\]\(([^)]+)\)")   # [a](url) entries
_INLINE_REF_RE = re.compile(r"\[([a-z])\]")              # [a] inline refs in bullets

def _parse_source_line(line: str) -> list[tuple[str, str]]:
    return _SRC_ENTRY_RE.findall(line or "")

def _format_named_sources(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return ""
    parts = [f"{letter} [{SOURCE_NAMES.get(url, url)}]({url})" for letter, url in entries]
    return "Sources: " + " · ".join(parts)

class ExplainableAgent(BaseAgent):
    """
    Explainable agent — stateful concept-by-concept walkthrough.

    Inherits from BaseAgent (Step 6b; method list below may predate later steps):
      - Clarification phase (_stream_clarification)
      - _detect_override(), _is_answer(), _strip_fences()
      - _strip_concept_swap_from_history()
      - send_message(), end_session()
      - KG infrastructure (_fetch_kg_context, _update_kg_if_framework_mentioned)
      - _check_duplicate()

    Overrides:
      - __init__                  : explainable case + walkthrough state variables
      - get_opening_message       : tailored intro
      - get_pre_analysis_instruction : agent-specific instruction
      - begin_analysis            : tree/button flow — streams first concept
      - _build_system_prompt      : minimal fallback (used by inherited send_message)
      - _stream_main              : stateful walkthrough logic
      - _detect_override          : walkthrough-aware override classifier
    """

    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self._init_flow_state()
        self.session_id    = create_session(user_id, agent_type="explainable")
        self.original_case = get_case("explainable")
        self.has_main_contribution = False

        self.clarification_facts = get_clarification_facts("explainable")

        self.concept_swap = ConceptSwap(
            agent_type="explainable",
            session_id=self.session_id
        )

        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        logging.info(f"[KG INIT] case_type={CASE_TYPE}, "
              f"framework={self.kg_context['framework']}, "
              f"concepts={self.kg_context['concepts']}")

        # Walkthrough state
        self.walkthrough_concepts = []
        self.walkthrough_index    = 0
        self.walkthrough_active   = False
        self.walkthrough_done     = False
        self.swap_presented       = False
        self.swap_position        = 0
        self._seen_concepts       = set()
        self.user_added_pillars = []
        self.pending_add          = None

        self.history = [
            types.Content(
                role="user",
                parts=[types.Part(text=self.original_case)]
            ),
            types.Content(
                role="model",
                parts=[types.Part(text=(
                    "I have received the case. We are now in the clarification round. "
                    "Please feel free to ask any questions about the case before you begin."
                ))]
            ),
        ]

    # Opening message
    def get_opening_message(self) -> str:
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
            f"---\n"
            f"Take your time to read it. Feel free to ask any clarifying questions "
            f"before you begin. When you're ready, I'll walk you through the framework "
            f"one concept at a time, you can ask questions at each step.\n\n"
        )

    def get_pre_analysis_instruction(self) -> str:
        return (
            "📖 *After you click the button below, I'll walk you through "
            "each concept one at a time, you can ask questions or suggest "
            "changes at any step.*"
        )

    def begin_analysis(self):
        self._start_main_phase_setup()

        yield (
            "⚠️ Your goal is to build a structured framework plan for the GenAI rollout. "
            "Review each pillar below carefully. The agent is here to help you create your framework based on current industry best practices. "
            "It can only support you when you actively engage, **not just read through it.**"
        )

        yield "⏱️ Your 20-minute session has started. The timer is shown on the left."
        self.walkthrough_concepts = self._build_walkthrough_concepts()
        self.walkthrough_active   = True
        self.walkthrough_index    = 0
        self.swap_presented       = False

        yield (
            "💡 *When you're finished, click **‼️End Session** "
            "to close your session. **Note: this cannot be undone.***\n\n---\n\n"
        )

        yield from self._stream_concept(is_first=True)
        yield ADD_ONE_AT_A_TIME

    # Walkthrough state helpers
    def _build_walkthrough_concepts(self) -> list:
        base     = list(self.kg_context["concepts"])
        wrong    = self.concept_swap.config["wrong_concept"]
        position = min(1, len(base))
        base.insert(position, wrong)
        self.swap_position = position
        logging.info(f"[WALKTHROUGH] built={base}, swap_position={position}, "
                     f"framework={self.kg_context['framework']}")
        return base

    def _current_concept(self) -> str | None:
        excluded_lower = [e.lower() for e in self.excluded_concepts]
        while self.walkthrough_index < len(self.walkthrough_concepts):
            concept = self.walkthrough_concepts[self.walkthrough_index]
            if concept.lower() not in excluded_lower:
                return concept
            logging.debug(f"[WALKTHROUGH] skipping excluded: {concept}")
            self.walkthrough_index += 1
        return None

    def _is_wrong_concept(self, concept: str) -> bool:
        return concept.lower() == self.concept_swap.config["wrong_concept"].lower()

    def _advance_to_next_unseen(self) -> bool:
        excl = [e.lower() for e in self.excluded_concepts]
        n = len(self.walkthrough_concepts)
        for i in range(self.walkthrough_index + 1, n):
            c = self.walkthrough_concepts[i].lower()
            if c not in excl and c not in self._seen_concepts:
                self.walkthrough_index = i
                return True
        for i in range(0, n):
            c = self.walkthrough_concepts[i].lower()
            if c not in excl and c not in self._seen_concepts:
                self.walkthrough_index = i
                return True
        self.walkthrough_index = n
        return False
    
    def _match_key_question(self, item: str, pillar_name: str) -> dict | None:
        text, _score = matching.match_key_question(item, pillar_name)
        if not text:
            return None
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == pillar_name.lower()),
            None
        )
        if pillar is None:
            return None
        raw = next(
            (q for q in pillar.get("key_questions", [])
             if matching._strip_source_refs(q) == text),
            text,
        )
        return {"question": raw, "sources": pillar.get("key_questions_sources", "")}

    def _is_excluded_bullet(self, pillar_name: str, bullet: str) -> bool:
        return matching.is_excluded_bullet(self.excluded_sub_bullets, pillar_name, bullet)

    def _last_agent_message(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts and c.parts[0].text:
                return c.parts[0].text[:600]
        return ""

    def _format_sub_bullet(self, item: str) -> str:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=SUB_BULLET_FORMAT_PROMPT.format(item=item),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            out = self._strip_fences(response.text).strip().strip("-• ").rstrip(".")
            if out:
                logging.info(f"[SUB-POINT FORMAT] '{item}' → '{out}'")
                return out
        except Exception as e:
            logging.warning(f"[SUB-POINT FORMAT] error: {e} — keeping raw")
        return item.strip()

    def _render_bullets_and_sources(self, concept: str) -> tuple[str, str]:
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == concept.lower()),
            None
        )
        bullet_lines = []

        if pillar is None:
            for sp in self.user_sub_points.get(concept, []):
                if not self._is_excluded_bullet(concept, sp):
                    bullet_lines.append(f"- {sp}")
            return "\n".join(bullet_lines), ""

        # Static sources keep their original letters/order
        merged        = _parse_source_line(pillar.get("sub_bullets_sources", ""))
        url_to_letter = {url: letter for letter, url in merged}
        next_ord      = (max(ord(l) for l, _ in merged) + 1) if merged else ord("a")

        # Static bullets — refs untouched; omit any the user removed
        for b in pillar.get("sub_bullets", []):
            if not self._is_excluded_bullet(concept, b):
                bullet_lines.append(f"- {b}")

        # User sub-points — matched refs resolve via key_questions_sources, continue letters
        kq_map = dict(_parse_source_line(pillar.get("key_questions_sources", "")))

        def _repl(m):
            nonlocal next_ord
            url = kq_map.get(m.group(1))
            if not url:
                return "" 
            if url in url_to_letter:
                return f"[{url_to_letter[url]}]"  # dedup by URL
            letter = chr(next_ord); next_ord += 1
            url_to_letter[url] = letter
            merged.append((letter, url))
            return f"[{letter}]"

        for sp in self.user_sub_points.get(concept, []):
            if self._is_excluded_bullet(concept, sp):
                continue
            bullet_lines.append(f"- {_INLINE_REF_RE.sub(_repl, sp)}")

        return "\n".join(bullet_lines), _format_named_sources(merged)

    def _render_pillar_block(self, concept: str) -> str:
        bullets, src = self._render_bullets_and_sources(concept)
        lines = [f"**{concept}**"]
        if bullets:
            lines.append(bullets)
        if src:
            lines.append("")
            lines.append(src)
        return "\n".join(lines)

    def _render_pillar_block_no_sources(self, concept: str) -> str:
        bullets, _ = self._render_bullets_and_sources(concept)
        bullets = _INLINE_REF_RE.sub("", bullets).strip()
        lines = [f"**{concept}**"]
        if bullets:
            lines.append(bullets)
        return "\n".join(lines)
    
    def _concept_grounding(self, concept: str) -> str:
        return grounding.ground_pillar(concept)

    # Main phase — stateful walkthrough router
    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # Gate flag — flips True on the user's FIRST main-phase message
        self.has_main_contribution = True

        # Multi-point safeguard: an open pillar offer takes priority over routing.
        if self.pending_pillar_offer is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            yield from self._resolve_pillar_offer(user_input)
            return

        # Confirm-before-commit: an open add preview takes priority over routing.
        if self.pending_add is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            yield from self._resolve_add_confirm(user_input)
            return

        _cur    = self.current_pillar() or "(none)"
        _pillar = next((p for p in kb.get_all_pillars()
                        if p["name"].lower() == _cur.lower()), None)
        _bul = list(_pillar.get("sub_bullets", [])) if _pillar else []
        _bul += self.user_sub_points.get(_cur, [])
        _bul = [b for b in _bul if not self._is_excluded_bullet(_cur, b)]
        res = intents.classify_intent(
            user_input,
            current_pillar=_cur,
            current_bullets="\n".join(f"- {b}" for b in _bul) or "(none)",
            walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
            last_agent=self._last_agent_message() or "(nothing yet)",
        )
        intent = res.intent
        logging.info(f"[INTENT] {intent} — detail={res.detail!r} parent={res.parent!r}")

        # Multi-point safeguard: ≥2 separable additions in one message. With a
        # candidate pillar -> offer to add it (pending); otherwise -> passive reminder.
        if (res.multi and self.pending is None and self.pending_suggestion is None
                and getattr(self, "pending_placement", None) is None):
            log_user_message(self.session_id, user_input)
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            if res.pillar_name:
                yield from self._offer_pillar(res.pillar_name, res.items)
            else:
                yield from self._safeguard_multi(res.items)
            return

        if (self.pending is None and self.pending_suggestion is None
                and getattr(self, "pending_placement", None) is None
                and self.swap_presented and not self.concept_swap.is_detected
                and intent not in ("add", "remove", "question")):
            if self.concept_swap.check_detection(user_input):       # sets is_detected
                wrong = self.concept_swap.config["wrong_concept"]
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)            # skip it in the walk
                logging.info("[SWAP] caught via semantic backstop")
                log_user_message(self.session_id, user_input)
                self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
                yield from self._stream_swap_caught()
                yield "\n\n"
                if self.current_pillar() is None:
                    yield from self._stream_summary()
                else:
                    yield from self._stream_concept(is_first=False)
                return

        # Confirm-before-commit: a single add is previewed first; commit on "yes".
        if (intent == "add" and not res.multi and self.pending is None
                and self.pending_suggestion is None
                and getattr(self, "pending_placement", None) is None):
            log_user_message(self.session_id, user_input)
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            yield from self._preview_add(res, user_input)
            return

        # Log + append the user turn
        log_user_message(self.session_id, user_input)
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
        was_pending = self.pending is not None
        pa_snapshot = self.pending
        self._last_intent = intent
        outcome = handlers.dispatch(res, self, user_text=user_input)
        yield from self._render_outcome(outcome, user_input,
                                        was_pending=was_pending, pa=pa_snapshot)
    
    # HandlerSession queries
    def _presented_concepts(self) -> list:
        excluded = [e.lower() for e in self.excluded_concepts]
        return [c for c in self.walkthrough_concepts[:self.walkthrough_index + 1]
                if c.lower() not in excluded]

    def presented_pillars(self) -> list:
        return self._presented_concepts()

    def presented_sub_bullets(self) -> dict:
        out = {}
        for name in self._presented_concepts():
            kbp = next((p for p in kb.get_all_pillars()
                        if p["name"].lower() == name.lower()), None)
            bl = []
            if kbp:
                bl += [matching._strip_source_refs(b) for b in kbp.get("sub_bullets", [])
                       if not self._is_excluded_bullet(name, b)]
            bl += [_INLINE_REF_RE.sub("", sp).strip()
                   for sp in self.user_sub_points.get(name, [])
                   if not self._is_excluded_bullet(name, sp)]
            out[name] = bl
        return out

    def surfaced_pillar_names(self) -> set:
        names = {p["name"].lower() for p in kb.get_shown_pillars()}
        names |= {c.lower() for c in self.walkthrough_concepts}
        names |= {n.lower() for n in self.user_added_pillars}
        names |= {e.lower() for e in self.excluded_concepts}
        return names

    def current_pillar(self):
        excluded = [e.lower() for e in self.excluded_concepts]
        idx = self.walkthrough_index
        while idx < len(self.walkthrough_concepts):
            c = self.walkthrough_concepts[idx]
            if c.lower() not in excluded:
                return c
            idx += 1
        return None

    # HandlerSession mutators
    def surface_pillar(self, name: str) -> None:
        in_walk = name.lower() in [c.lower() for c in self.walkthrough_concepts]
        already = name.lower() in [p.lower() for p in self.user_added_pillars]
        if in_walk or already:
            self._last_surface = {"name": name, "is_new": False}
            return
        self.walkthrough_concepts.append(name)
        self.user_added_pillars.append(name)
        self._last_surface = {"name": name, "is_new": True}

    def _surface_at_current(self, name: str) -> None:
        """Bring a withheld/novel pillar in AT the current walkthrough position (vs
        appending to the end). It then sits inside presented_pillars() — so counting
        reads its bullets as shown — and is marked seen so it isn't re-walked at the
        end. The current concept is unchanged."""
        if (name.lower() in [c.lower() for c in self.walkthrough_concepts]
                or name.lower() in [p.lower() for p in self.user_added_pillars]):
            self._last_surface = {"name": name, "is_new": False}
            return
        idx = self.walkthrough_index
        self.walkthrough_concepts.insert(idx, name)
        if idx <= self.swap_position:
            self.swap_position += 1
        self.walkthrough_index = idx + 1
        self.user_added_pillars.append(name)
        self._seen_concepts.add(name.lower())
        self._last_surface = {"name": name, "is_new": True}

    def add_sub_point(self, pillar: str, text: str, *,
                      dry_run: bool = False, resolved: str | None = None):
        if resolved is not None:
            stored = resolved
        else:
            match = self._match_key_question(text, pillar)
            if match:
                stored = match["question"]
            else:
                cb = matching.canonical_add_bullet(text, refs=True)
                stored = cb if cb else self._format_sub_bullet(text)
        kbp = next((p for p in kb.get_all_pillars()
                    if p["name"].lower() == pillar.lower()), None)
        _norm_ref = lambda b: _INLINE_REF_RE.sub("", b).strip().lower()
        static_hit = next((b for b in (kbp.get("sub_bullets", []) if kbp else [])
                           if _norm_ref(b) == _norm_ref(stored)), None)
        already_static = static_hit is not None
        existing = self.user_sub_points.get(pillar, [])
        is_new = (not already_static
                  and stored not in existing
                  and not self._is_excluded_bullet(pillar, stored))
        if dry_run:
            return stored, is_new
        self.user_sub_points.setdefault(pillar, [])
        if is_new:
            self.user_sub_points[pillar].append(stored)
        matched = None
        if not is_new:
            matched = static_hit or next((p for p in self.user_sub_points[pillar]
                                          if p == stored), stored)
            matched = _INLINE_REF_RE.sub("", matched).strip()
        self._last_sub_add = {"pillar": pillar, "stored": stored, "raw": text,
                              "is_new": is_new, "matched": matched}
        return stored, is_new

    # Confirm-before-commit: EXP previews every add and only commits + counts on "yes".
    def _resolve_add_dest(self, text: str, parent: str | None):
        """Read-only: where would this add land? Returns
        (kind, pillar, stored, is_dup, status); status is presented/withheld/novel."""
        presented = {n.lower() for n in self.presented_pillars()}
        kb_names  = {p["name"].lower() for p in kb.get_all_pillars()}

        def _status(name: str) -> str:
            n = (name or "").lower()
            return "presented" if n in presented else ("withheld" if n in kb_names else "novel")

        if parent:
            matched, _ = matching.match_pillar(parent)
            pillar     = matched or matching.normalize_name(parent)
            status     = _status(pillar)
            stored, is_new = self.add_sub_point(pillar, text, dry_run=True)
            return "sub_bullet", pillar, stored, (not is_new and status == "presented"), status
        km = matching.locate(text)
        if km.level == "concept" and km.pillar:
            status = _status(km.pillar)
            stored, is_new = self.add_sub_point(km.pillar, text, dry_run=True)
            return "sub_bullet", km.pillar, stored, (not is_new and status == "presented"), status
        if km.level == "pillar" and km.pillar:
            return "pillar", km.pillar, None, (km.pillar.lower() in presented), _status(km.pillar)
        name = matching.normalize_name(text)
        return "pillar", name, None, (name.lower() in presented), _status(name)

    def _preview_add(self, res, user_input: str):
        text   = res.detail or user_input
        parent = (res.parent if (res.parent and handlers._parent_is_explicit(res.parent, user_input))
                  else None)
        kind, pillar, stored, is_dup, status = self._resolve_add_dest(text, parent)
        if is_dup:
            msg = f"That's already covered under **{pillar}** as *{stored}*."
            self._emit(msg); yield msg; return
        self.pending_add = {"kind": kind, "pillar": pillar, "raw": text,
                            "stored": stored, "status": status}
        if kind == "sub_bullet":
            if status == "presented":
                msg = (f"I'll add *{stored}* under **{pillar}**, reply **yes** to confirm, or "
                       f"tell me a different wording or area.")
            elif status == "withheld":
                msg = (f"It sounds like that belongs under **{pillar}**, want me to bring in "
                       f"**{pillar}** and add *{stored}* there? *(yes, or name a different area)*")
            else:  # novel — suggest, but the user decides wording / where
                msg = (f"I don't see **{pillar}** in the framework yet, I'd add *{stored}* under a "
                       f"new pillar **{pillar}**. Reply **yes**, name an existing area to use "
                       f"instead, or reword.")
        elif status == "withheld":
            msg = (f"**{pillar}** is part of the framework but not shown yet, want me to "
                   f"bring it in? *(yes / no)*")
        else:  # novel pillar
            msg = (f"I don't see **{pillar}** in the framework yet, I'd add it as a new pillar. "
                   f"Reply **yes**, name an existing area instead, or give a different name.")
        self._emit(msg); yield msg

    def _resolve_add_confirm(self, user_input: str):
        pa       = self.pending_add
        decision = handlers._classify_confirmation(user_input)
        if decision == "confirm":
            self.pending_add = None
            counted = self._counts_as_new(pa["pillar"], pa.get("stored") or pa["raw"])
            if pa["kind"] == "pillar":
                self._surface_at_current(pa["pillar"])
                if counted and (self._last_surface or {}).get("is_new"):
                    ev.record(handlers.AddOutcome(action="added_new", pillar=pa["pillar"],
                              level="pillar", counted=True, source="user_spontaneous"),
                              self._evctx(), _sink)
                lead = (f"Brought in **{pa['pillar']}**." if pa.get("status") == "withheld"
                        else f"Noted, I've added **{pa['pillar']}** as a separate area.")
                msg = lead + "\n\n" + self._render_pillar_block(pa["pillar"])
            else:
                if pa.get("status") != "presented":
                    self._surface_at_current(pa["pillar"])
                stored, _ = self.add_sub_point(pa["pillar"], pa["raw"], resolved=pa.get("stored"))
                if counted:
                    ev.record(handlers.AddOutcome(action="added_new", pillar=pa["pillar"],
                              level="sub_bullet", counted=True, text=stored,
                              source="user_spontaneous"), self._evctx(), _sink)
                lead = (f"Brought in **{pa['pillar']}**, " if pa.get("status") == "withheld"
                        else f"Good point, I've added it under **{pa['pillar']}**.")
                msg = lead + " Here's how it looks now:\n\n" + self._render_pillar_block(pa["pillar"])
            self._emit(msg); yield msg
            return
        if decision == "decline":
            self.pending_add = None
            msg = "No problem, I've left it out."
            self._emit(msg); yield msg
            return
        # Anything else is an edit: re-classify the reply as the revised add and re-preview.
        self.pending_add = None
        new_res = intents.classify_intent(
            user_input,
            current_pillar=self.current_pillar() or "(none)",
            current_bullets="(none)",
            walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
            last_agent=self._last_agent_message() or "(nothing yet)",
        )
        yield from self._preview_add(new_res, user_input)

    # swap channel
    def _on_swap_now(self) -> bool:
        cur = self.current_pillar()
        return cur is not None and self._is_wrong_concept(cur)

    def _extra_swap_signal(self, km, user_text: str) -> bool:
        cur = self.current_pillar()
        return bool(cur and self._is_wrong_concept(cur))

    # outcome renderer
    def _swap_question_signal(self, outcome, user_input: str) -> bool:
        return (self.swap_presented and not self.concept_swap.is_detected
                and self._on_swap_now())

    _NEXT_AFFORD = ("\n\n*Add a point here, raise a new area, or question anything \u2014 "
                    "or say \"next\" to move on.*")

    def _walk_done_render(self, touched):
        """After a multi-point walk: show each pillar the points landed in (they were
        surfaced during the walk, so render unconditionally), then the next-step
        guidance affordance."""
        seen = set()
        for p in touched:
            if not p or p.lower() in seen:
                continue
            seen.add(p.lower())
            yield self._emit_text(self._render_pillar_block(p) + "\n\n")
        yield self._emit_text(self._NEXT_AFFORD.strip())

    def render_add(self, o):
        if o.action == "navigated":
            target = o.pillar
            if target and target in self.walkthrough_concepts:
                idx = self.walkthrough_concepts.index(target)
                back = idx <= self.walkthrough_index
                if not back:
                    self.walkthrough_concepts.pop(idx)
                    self.walkthrough_concepts.insert(self.walkthrough_index, target)
                else:
                    self.walkthrough_index = idx
                self.walkthrough_done = False
                # If the user raised a specific point, persist it BEFORE streaming
                # so it renders inline in the pillar block (like HITL), not as a
                # trailing "You raised this" note. Store the already-resolved KB
                # bullet verbatim (no second match); display-only — the pillar
                # reveal already counts the agency.
                cid = getattr(o, "navigate_concept_id", None)
                nb = (matching.concept_bullet(cid, refs=True) if cid else None) \
                    or getattr(o, "navigate_bullet", None)
                added = False
                if nb:
                    already = {matching._strip_source_refs(b).strip().lower()
                               for b in (self.presented_sub_bullets().get(target, []))}
                    nb_key = matching._strip_source_refs(nb).strip().lower()
                    if nb_key not in already:
                        bucket = self.user_sub_points.setdefault(target, [])
                        if not any(matching._strip_source_refs(b).strip().lower() == nb_key
                                   for b in bucket):
                            bucket.append(nb)
                        added = True
                if added:
                    yield f"Good point, I've added it under **{target}**. Here's where we are.\n\n"
                else:
                    lead = "Going back to" if back else "Let's look at"
                    yield f"{lead} **{target}**, here's where we are.\n\n"
                yield from self._stream_concept(is_first=False)
            else:
                ev.question(self._evctx(), _sink)   # revisit -> grounded Q&A (no turn outcome)
                yield from self._stream_concept_qa()
            return

        if o.action == "ask_placement":
            cur = self.current_pillar()
            if cur:
                msg = (f"Where should **{o.text}** go, its own pillar, or a point under "
                       f"**{cur}**? *(Or name another pillar.)*")
            else:
                msg = (f"Where should **{o.text}** go, its own pillar, or a point under one "
                       f"of the existing pillars? *(If under one, which?)*")
            self._emit(msg); yield msg; return

        if o.action == "navigate_offer":
            gist = f" — {o.explanation}" if o.explanation else ""
            presented = {n.lower() for n in self.presented_pillars()}
            if o.matched_text:
                msg = (f"That's already covered under **{o.pillar}** as *{o.matched_text}*"
                       f"{gist} Want to discuss it together with **{o.pillar}**, or put it here in **{cur}**?")
            elif o.pillar and o.pillar.lower() in presented:
                msg = (f"We've already discussed **{o.pillar}** in the framework{gist}. "
                       f"Do you want to revisit it and add anything there?")
            else:
                msg = (f"**{o.pillar}** is a pillar I'll cover{gist} "
                       f"Want to discuss it now, or shall we continue with the current pillar?")
            self._emit(msg); yield msg; return

        if o.action == "duplicate":
            if o.level == "pillar" and o.pillar:
                msg = f"**{o.pillar}** is already part of the framework."
            elif o.pillar and o.matched_text:
                msg = (f"That's already covered under **{o.pillar}** as *{o.matched_text}*. "
                       f"Want to adjust it?")
            elif o.pillar:
                ref = o.matched_text or o.text
                msg = ((f"That's already covered under **{o.pillar}** as *{ref}*."
                        if ref else f"That's already covered under **{o.pillar}**.")
                      )
            else:
                msg = "That's already in the framework."
            self._emit(msg); yield msg; return

        if o.action == "revealed" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                self._seen_concepts.add((o.pillar or "").lower())
                kbp = next((p for p in kb.get_all_pillars()
                            if p["name"].lower() == (o.pillar or "").lower()), None)
                desc = (kbp.get("description", "").strip() if kbp else "")
                parts = [f"Good call, **{o.pillar}** is an important pillar to consider."]
                if desc:
                    parts.append(desc)
                parts.append(self._render_pillar_block(o.pillar))
                parts.append('Want to add a bullet to this pillar, or ready to move on to the next one?')
                msg = "\n\n".join(parts)
            else:
                msg = (f"**{o.pillar}** is already part of the framework, we'll get to it."
                      )
            self._emit(msg); yield msg; return

        if o.action == "added_new" and o.level == "sub_bullet":
            st = self._last_sub_add or {}
            if st.get("is_new"):
                if o.also_covered:
                    gist = f" \u2014 {o.explanation}" if o.explanation else ""
                    also = (f" It also relates to **{o.also_covered}**{gist} "
                            f"Say the word if you'd rather it sit there.")
                else:
                    also = ""
                msg = (f"Good point, I've added it under **{o.pillar}**.{also} "
                       f"Here's how it looks now:\n\n"
                       f"{self._render_pillar_block(o.pillar)}")
            else:
                matched = (self._last_sub_add or {}).get("matched")
                if matched:
                    msg = (f"That looks like it's already covered under **{o.pillar}** as "
                           f"*{matched}*. Want to add it as a separate pillar anyway, or "
                           f"leave it as is?")
                else:
                    msg = (f"That's already noted under **{o.pillar}**.")
            self._emit(msg); yield msg; return

        if o.action == "added_new" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                msg = (f"Noted, I've added **{o.pillar}** as a separate pillar. "
                       f"What points would you like under it? Add them one at a time."
                      )
            else:
                msg = f"**{o.pillar}** is already part of the framework."
            self._emit(msg); yield msg; return

        msg = "Noted."; self._emit(msg); yield msg

    def render_removal(self, o, user_input, *, was_pending=False, pa=None):
        stage = o.stage
        if stage == "confirmed":
            if o.is_swap:
                yield from self._stream_swap_caught()
                yield "\n\n"
                if self.current_pillar() is None:
                    yield from self._stream_summary()
                else:
                    yield from self._stream_concept(is_first=False)
                return
            if pa and pa.type == "remove_sub_bullet":
                msg = (f"Done, I've removed that bullet from **{pa.pillar}**. "
                       f"Here's how it looks now:\n\n"
                       f"{self._render_pillar_block(pa.pillar)}")
                self._emit(msg); yield msg; return
            yield f"Understood, removing **{o.target}** from the framework. Let's continue.\n\n"
            if self.current_pillar() is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)
            return

        if stage == "abandoned":
            if pa and pa.type == "remove_sub_bullet":
                msg = f"No problem, I'll keep that bullet in **{pa.pillar}**."
            else:
                tgt = pa.target if pa else o.target
                msg = f"No problem, I'll keep **{tgt}** in the framework."
            self._emit(msg); yield msg; return

        if stage == "nothing_to_remove":
            if o.suggest_add_alternative:
                msg = (f"**{o.suggest_add_alternative}** isn't in the current framework. "
                       f"Did you mean to *add* it? Reply **yes** to add it.")
            else:
                msg = (f"**{o.target or 'That'}** isn't part of the current framework, "
                       f"so there's nothing to remove there.")
            self._emit(msg); yield msg; return

        if stage == "needs_disambiguation":
            msg = ("Which part would you like to remove, the current pillar, or a "
                   "specific bullet within it? You can name it.\n\n"
                   "*(Or say **never mind** to keep everything as is.)*")
            self._emit(msg); yield msg; return

        if stage == "challenged":
            if was_pending:
                if self._reply_is_question(user_input):
                    yield from self._stream_concept_qa(); return
                msg = f"No rush, reply **yes** to remove **{o.target}**, or **no** to keep it."
                self._emit(msg); yield msg; return
            if self.pending and self.pending.type == "remove_sub_bullet":
                yield from self._stream_sub_bullet_pushback(self.pending.pillar, o.target)
            else:
                yield from self._stream_pushback("concept", o.target)
            return

        msg = f"No rush, reply **yes** to remove **{o.target}**, or **no** to keep it."
        self._emit(msg); yield msg

    def render_question(self, user_input):
        if self.walkthrough_done:
            yield from self._stream_freeform(cs_detected=False)
        else:
            yield from self._stream_concept_qa()

    def render_next_steps(self, o):
        if getattr(o, "revealed", False):
            msg = (f"Good point, I've brought in **{o.suggested_item}**; "
                   f"we'll cover it in the walkthrough.")
            self._emit(msg); yield msg; return
        if not getattr(o, "suggested_item", None):
            msg = ("You've surfaced the main pillars I'd flag, feel free to revisit, add, "
                   "remove, or question any part of the framework.")
            self._emit(msg); yield msg; return
        why = (o.grounding or "").split("\n")[0].strip()
        msg = (f"One pillar we haven't covered yet is **{o.suggested_item}**"
               + (f", {why}" if why else "")
               + ".\n\nIt's worth considering whether it applies to your case. "
                 "Shall I bring it in?")
        self._emit(msg); yield msg

    def render_fallback(self, outcome=None):
        if outcome is None:
            msg = ("You've surfaced the main pillars I'd flag, feel free to revisit, add, "
                   "remove, or question any part of the framework.")
            self._emit(msg); yield msg; return
        if isinstance(outcome, handlers.AdvanceOutcome) and not self.walkthrough_done:
            if self._advance_to_next_unseen():
                yield from self._stream_concept(is_first=False)
            else:
                yield from self._stream_summary()
                invite = ("\n\nThat's the full framework as it stands. Want to revisit any "
                          "area to add, change, or question something? Otherwise, click "
                          "**‼️End Session** above to finish.")
                self._emit(invite); yield invite
            return
        if getattr(self, "_last_intent", "none") == "doubt":
            ev.question(self._evctx(), _sink)
            on_swap = (self.swap_presented and not self.concept_swap.is_detected
                       and self._on_swap_now())
            if on_swap:
                ev.swap_questioned(self._evctx(), _sink)
            if self.walkthrough_done:
                yield from self._stream_freeform(cs_detected=False)
            else:
                yield from self._stream_concept_qa()
            return
        if (self.walkthrough_done
                and isinstance(outcome, handlers.AdvanceOutcome)):
            msg = ("Sounds like you're happy with the framework. When you're ready to finish, "
                   "click **‼️End Session** above. Or if there's still something you'd like to "
                   "**add**, **remove**, or **question**, go ahead.")
            self._emit(msg); yield msg; return
        msg = ("I want to make sure I help with the right thing. You can **add** a bullet, "
               "**remove** something, **question** any part of the framework, ask me to "
               "**suggest** what else to consider, or say **move on** to continue.")
        self._emit(msg); yield msg

    def render_summary(self):
        yield from self._stream_summary()

    def render_framework(self, preamble=""):
        if preamble:
            yield preamble
        if self.current_pillar() is None:
            yield from self._stream_summary()
        else:
            yield from self._stream_concept(is_first=False)

    # Pending resolution + pushback    
    def _resolve_to_concept_name(self, name: str) -> str:
        for c in self.walkthrough_concepts:
            if c.lower() == name.lower():
                return c
        fallback = self._current_concept() or name
        logging.info(f"[RESOLVE] '{name}' not in walkthrough_concepts, fallback: '{fallback}'")
        return fallback

    # Addition placement flow
    def _stream_pushback(self, pending_type: str, detail: str):
        concept = self._current_concept() or "the current concept"

        if pending_type == "concept":
            instruction = (
                "You are a strategic consultant. The user wants to remove "
                "**" + detail + "** from the framework. "
                "Push back using counterfactual reasoning.\n\n"
                "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
                "1. One sentence: If we remove " + detail + ", then [consequence].\n"
                "2. One sentence grounding it in the case context.\n"
                "3. End with: Would you still like to remove it?\n\n"
                "─── RULES ──────────────────────────────────────────────────────────────────\n"
                "- Consulting reasoning only\n"
                "- Soft tone — consequence as information, not command\n"
                "- Do NOT refuse the removal\n"
                "- Do NOT present any other concept\n"
                "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
                "Concept questioned: **" + detail + "**\n"
                "Current concept: **" + concept + "**\n"
                "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
                "─────────────────────────────────────────────────────────────────────\n"
            )
        else:
            instruction = (
                "You are a strategic consultant. The user wants to switch to "
                "**" + detail + "** framework. "
                "Push back using counterfactual reasoning.\n\n"
                "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
                "1. One sentence: If we switch to " + detail + ", we would [consequence].\n"
                "2. One sentence: why current framework fits this case.\n"
                "3. End with: Would you still like to switch?\n\n"
                "─── RULES ──────────────────────────────────────────────────────────────────\n"
                "- Consulting reasoning only\n"
                "- Soft tone — consequence as information, not command\n"
                "- Do NOT refuse the switch\n"
                "- Do NOT present any concept yet\n"
                "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
                "Proposed: **" + detail + "**\n"
                "Current: **" + self.kg_context["framework"] + "**\n"
                "Case: " + self.kg_context["case_type"] + "\n"
                "─────────────────────────────────────────────────────────────────────\n"
            )

        yield from self._stream_with_instruction(instruction=instruction)

    # Sub-bullet removal: reasoned pushback + 3-way confirm
    def _stream_sub_bullet_pushback(self, pillar: str, bullet: str):
        concept = self._current_concept() or pillar
        instruction = (
            "You are a strategic consultant. The user wants to remove ONE specific "
            "point from the **" + pillar + "** pillar (not the whole pillar). "
            "Push back briefly using counterfactual reasoning.\n\n"
            "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
            "1. One sentence: If we drop this point, then [specific consequence].\n"
            "2. End with exactly: Would you still like to remove it?\n\n"
            "─── RULES ──────────────────────────────────────────────────────────────────\n"
            "- Consulting reasoning only; soft tone — consequence as information.\n"
            "- This is about ONE point, NOT the whole pillar — do not threaten to remove the pillar.\n"
            "- Do NOT present any other concept.\n"
            "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
            "Point to remove: \"" + bullet + "\"\n"
            "Pillar: **" + pillar + "** | Current concept: **" + concept + "**\n"
            "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
            "─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_concept(self, is_first: bool):
        concept = self._current_concept()
        if concept is None:
            yield from self._stream_summary()
            return
        self._seen_concepts.add(concept.lower())
        is_wrong   = self._is_wrong_concept(concept)
        swap_block = self.concept_swap.get_system_prompt_block() if is_wrong else ""

        if not is_wrong:
            # Look up pillar by name directly (walkthrough uses pillar names)
            pillar = next(
                (p for p in kb.get_all_pillars() if p["name"].lower() == concept.lower()),
                None
            )
            if pillar is None:
                # Fall back to concept-level lookup for user-added concepts
                pillar = kb.get_pillar_for_concept_name(concept)
            if pillar is None:
                # User-added concept not in KB — present plainly (no provenance note).
                prefix = "**" + concept + "**\n"
            else:
                description = pillar.get("description", "")
                bullet_lines, named_sources = self._render_bullets_and_sources(concept)
                sources_line  = f"\n\n{named_sources}" if named_sources else ""

                prefix = (
                    f"**{concept}**\n\n"
                    f"{description}\n\n"
                    f"{bullet_lines}"
                    f"{sources_line}\n\n"
                    f"*Would you like to add, remove, or question anything here? If you are happy with this pillar, shall we move on to the next one? Feel free to raise anything you think is important to consider.*"
                )
        else:
            swap    = kb.get_swap_concept()
            bullets = swap.get("sub_bullets", []) if swap else []
            bullet_lines = "\n".join(f"- {_INLINE_REF_RE.sub('', b).strip()}" for b in bullets)
            prefix = (
                f"**{concept}**\n\n"
                f"{bullet_lines}\n\n"
                f"*Would you like to add, remove, or question anything here? If you are happy with this pillar, shall we move on to the next pillar? Feel free to raise anything you think is important to consider.*"
            )

        if is_first:
            prefix = "Here is the first pillar I recommend. Do you want to include it in your framework plan?\n\n" + prefix

        # Yield static text, append to history
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=f"[Present concept: {concept}]")])
        )
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=prefix)])
        )
        log_agent_response(self.session_id, prefix)

        if is_wrong:
            self.swap_presented = True
            if not self.concept_swap.is_injected:
                self.concept_swap.maybe_inject(prefix)
                self.concept_swap.log_presented()
                logging.info(f"[SWAP] concept presented at position={self.swap_position}")

        yield prefix

    def _stream_concept_qa(self, just_added: str | None = None):
        current = self._current_concept()
        concept = current or "the current concept"
        on_swap = (self.swap_presented
                   and not self.concept_swap.is_detected
                   and current is not None
                   and self._is_wrong_concept(current))

        closing = (
            "End with exactly:\n"
            "*I can see why you'd question this shall we include it or move on without it?*"
            if on_swap else
            "End with exactly:\n"
            "*Would you like to add, remove, or question anything here? If you are happy with this pillar, shall we move on to the next one? Feel free to raise anything you think is important to consider.*"
        )

        qa_prompt = CONCEPT_QA_PROMPT.format(on_swap=on_swap)

        grounding = self._concept_grounding(concept)
        grounding_block = ""
        if grounding:
            grounding_block = (
                "─── KNOWN POINTS FOR THIS CONCEPT (ground your answer here) ──────────\n"
                + grounding + "\n"
                "─── GROUNDING RULE ──────────────────────────────────────────────────\n"
                "Base your answer on the KNOWN POINTS above and the case. You may explain\n"
                "or apply them to this case, but do NOT introduce regulations, statistics,\n"
                "named sources, or framework concepts that are not among the known points.\n"
                "Do NOT output bracketed letter markers like [a].\n"
                "──────────────────────────────────────────────────────────────────────\n"
            )

        added_note = ""
        if just_added:
            added_note = (
                "Good idea, I'll add **" + just_added
                + "** after we finish **" + concept + "**.\n\n"
            )

        instruction = (
            qa_prompt + "\n\n"
            "─── CLOSING INSTRUCTION ──────────────────────────────────────────────\n"
            + closing + "\n"
            + grounding_block
            + "─── CONTEXT ──────────────────────────────────────────────────────────\n"
            "Current concept: **" + concept + "**\n"
            "On swap concept: " + str(on_swap) + "\n"
            "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
            "Framework concepts (in order): " + ", ".join(
                c for c in self.walkthrough_concepts
                if just_added is None or c.lower() != just_added.lower()
            ) + "\n"
            "──────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction, prefix=added_note)

    def _stream_swap_caught(self):
        wrong = self.concept_swap.config["wrong_concept"]
        msg = f"Understood, we'll set **{wrong}** aside and continue."
        self._emit(msg)
        yield msg

    def _stream_freeform(self, cs_detected: bool):
        concepts_str = " → ".join(self.kg_context["concepts"])
        instruction  = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
            f"You are a strategic consultant. The user has seen the full framework. "
            f"Answer their question concisely in plain language — no jargon, no "
            f"mention of technical systems. Ask ONE follow-up question after answering."
        )
        _ = cs_detected
        yield from self._stream_with_instruction(instruction=instruction)

    # Session + system prompt
    def end_session(self) -> None:
        from backend.logger import end_session as _end_session
        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection('sessions').document(self.session_id).update({
                'concept_swap_detected': self.concept_swap.is_detected,
                'swap_detected_at_end':  self.concept_swap.is_detected,
            })
            logging.info(f'[END SESSION] stamped session={self.session_id}, '
                         f'swap_detected={self.concept_swap.is_detected}')
        except Exception as e:
            logging.warning(f'[END SESSION] Firestore stamp failed: {e}')
        _end_session(self.session_id)

    def _build_system_prompt(self) -> str:
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"
        return (
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n\n"
            f"You are a strategic consultant. Answer concisely in plain language. "
            f"No jargon. No mention of databases or technical systems."
        )
