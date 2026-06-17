from google.genai import types

from backend.agents.base import BaseAgent
from backend.domain import grounding, matching
from backend.interaction import handlers, intents
from backend.knowledge import knowledge_base as kb
from backend.knowledge.cases import get_case, get_clarification_facts
from backend.llm import (
    client, CLASSIFIER_MODEL,
)
from backend.logger import (
    create_session, end_session,log_user_message,
    log_agent_response, log_interruption, update_answer,
)
from backend.logging import events as ev
from backend.logging.sink import firestore_sink as _sink
from backend.agents.prompts.black_box import (
    SYSTEM_PROMPT, REMOVAL_TARGET_PROMPT, SUB_BULLET_FORMAT_PROMPT,
)
from backend.agents.prompts.base import ADD_ONE_AT_A_TIME
from backend.tools.concept_swap import ConceptSwap

from dotenv import load_dotenv

CASE_TYPE = "AI Implementation"

_CANCEL_PHRASES = {
    "never mind", "nevermind", "cancel", "stop", "forget it", "forget about it",
    "no", "nope", "no thanks", "nothing", "none", "leave it", "leave it alone",
    "keep it", "keep everything", "actually no", "skip", "skip it", "dont", "don't",
}

load_dotenv()

class BlackBoxAgent(BaseAgent):
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self._init_flow_state()
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self.has_main_contribution = False
        self.user_added_pillars = []
        self._ack_index         = 0
        self.clarification_facts = get_clarification_facts("black_box")

        self.concept_swap = ConceptSwap(
            agent_type="black_box",
            session_id=self.session_id
        )

        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        print(f"[KB INIT] case_type={CASE_TYPE}, "
              f"framework={self.kg_context['framework']}, "
              f"concepts={self.kg_context['concepts']}")

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

    # KG helpers
    def _build_system_prompt(self) -> str:
        framework    = self.kg_context["framework"]
        concepts_str = ", ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else \
                       "Strategic Fit, Solution Design & Scope, Feasibility"

        framework_block = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Case Type : {self.kg_context['case_type']}\n"
            f"Framework : {framework}\n"
            f"Pillars   : {concepts_str}\n"
            f"CRITICAL: Use pillars as PRIMARY BUCKET HEADERS.\n"
            f"Generate exactly 2 analytical questions directly under each pillar — no sub-headers.\n"
            f"Do NOT add any pillar not listed here UNLESS the user explicitly requests it.\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
        )

        swap_block = self.concept_swap.get_system_prompt_block()
        return framework_block + swap_block + SYSTEM_PROMPT

    # Opening message
    def get_opening_message(self) -> str:
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
        )

    def get_pre_analysis_instruction(self) -> str:
        return (
            "📖 *After you click the button below, read each concept carefully — "
            "add any ideas or questions that come to mind as we go.*"
        )

    # Phase transition
    def begin_analysis(self):
        self._start_main_phase_setup()

        yield (
            "⚠️ Your goal is to build a structured plan for this case. "
            "Review each factor below, share your thoughts, and you **should not only read it** but also add or remove anything you think is missing."
        )

        yield "⏱️ Your 20-minute session has started. The timer is shown on the left."

        yield from self._stream_framework_presentation()
        yield ADD_ONE_AT_A_TIME

        yield (
            "\n\n---\n\n"
            "📖 *Now it's your turn — do you have any questions about the framework "
            "presented? Add, remove, or update anything as you like. Once you've shared "
            "your first thoughts, an **‼️End Session** button will appear so you can "
            "finish whenever you're ready.*"
        )

    # Main phase streaming
    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        self.has_main_contribution = True
        log_user_message(self.session_id, user_input)
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        # Multi-point safeguard: an open pillar offer takes priority over routing.
        if self.pending_pillar_offer is not None:
            yield from self._resolve_pillar_offer(user_input)
            return

        res = intents.classify_intent(
            user_input,
            current_pillar="(none)",
            current_bullets="(none)",
            walkthrough_pillars=", ".join(self.kg_context["concepts"])
                or "Strategic Fit, Solution Design & Scope, Feasibility",
            last_agent=self._last_agent_text() or "(nothing yet)",
        )
        intent = res.intent

        # Multi-point safeguard: ≥2 separable additions. With a candidate pillar ->
        # offer to add it (pending); otherwise -> passive reminder. (User turn already
        # logged above.)
        if (res.multi and self.pending is None and self.pending_suggestion is None
                and getattr(self, "pending_placement", None) is None):
            if res.pillar_name:
                yield from self._offer_pillar(res.pillar_name, res.items)
            else:
                yield from self._safeguard_multi(res.items)
            return

        if (self.pending is None and self.pending_suggestion is None
                and getattr(self, "pending_placement", None) is None
                and self.concept_swap.is_injected and intent not in ("add", "remove", "question")):
            if self.concept_swap.check_detection(user_input):
                yield from self._yield_rerender("Understood — I've taken that out.\n\n")
                return

        was_pending = self.pending is not None
        pa_snapshot = self.pending
        outcome = handlers.dispatch(res, self, user_text=user_input)
        yield from self._render_outcome(outcome, user_input,
                                        was_pending=was_pending, pa=pa_snapshot)

    # Deterministic render
    def _last_agent_message(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts and c.parts[0].text:
                return c.parts[0].text[:600]
        return ""

    def _store_sub_point(self, pillar: str, item: str, modality: str = "text"):
        pillar  = self._normalize_pillar(pillar)
        matched, _ = matching.match_key_question(item, pillar)
        if not matched:
            matched = matching.canonical_add_bullet(item, refs=False)
        stored  = matched if matched else self._format_sub_bullet(item)

        kbp = next((p for p in kb.get_all_pillars() if p["name"].lower() == pillar.lower()), None)
        if any(stored.lower() == grounding._strip_source_refs(b).lower()
               for b in (kbp.get("sub_bullets", []) if kbp else [])):
            return stored, False

        for other, pts in self.user_sub_points.items():
            if other.lower() != pillar.lower():
                for s in list(pts):
                    if s.lower() == stored.lower():
                        pts.remove(s)
                        print(f"[MOVE] '{stored}' {other} -> {pillar}")
        existing = self.user_sub_points.setdefault(pillar, [])
        if any(s.lower() == stored.lower() for s in existing):
            return stored, False
        existing.append(stored)

        return stored, True

    def _normalize_pillar(self, name: str) -> str:
        for p in kb.get_all_pillars():
            if p["name"].lower() == name.lower():
                return p["name"]
        for p in self.user_added_pillars:
            if p.lower() == name.lower():
                return p
        return name

    def _format_sub_bullet(self, item: str) -> str:
        try:
            r = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=SUB_BULLET_FORMAT_PROMPT.format(item=item),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            out = grounding._strip_source_refs(self._strip_fences(r.text)).strip().strip("-• ").rstrip(".")
            if out:
                return out
        except Exception as e:
            print(f"[SUB-POINT FORMAT] error: {e}")
        return item.strip()

    # HandlerSession queries
    def presented_pillars(self) -> list:
        excluded = [e.lower() for e in self.excluded_concepts]
        names = [p["name"] for p in kb.get_shown_pillars()
                 if p["name"].lower() not in excluded]
        if self.concept_swap.is_injected and not self.concept_swap.is_detected:
            names.append(self.concept_swap.config["wrong_concept"])
        for name in self.user_added_pillars:
            if name.lower() not in excluded and name not in names:
                names.append(name)
        return names

    def presented_sub_bullets(self) -> dict:
        out = {}
        excluded = [e.lower() for e in self.excluded_concepts]
        def collect(name, kb_bullets):
            bl = [grounding._strip_source_refs(b) for b in kb_bullets
                  if not self._is_excluded_bullet(name, b)]
            bl += [sp for sp in self.user_sub_points.get(name, [])
                   if not self._is_excluded_bullet(name, sp)]
            out.setdefault(name, [])
            out[name] += bl
        for p in kb.get_shown_pillars():
            if p["name"].lower() in excluded:
                continue
            collect(p["name"], p.get("sub_bullets", []))
        swap = kb.get_swap_concept()
        if self.concept_swap.is_injected and not self.concept_swap.is_detected and swap:
            collect(self.concept_swap.config["wrong_concept"], swap.get("sub_bullets", []))
        for name in self.user_added_pillars:
            if name.lower() in excluded:
                continue
            kbp = next((p for p in kb.get_all_pillars()
                        if p["name"].lower() == name.lower()), None)
            collect(name, kbp.get("sub_bullets", []) if kbp else [])
        return out

    def surfaced_pillar_names(self) -> set:
        names = {p["name"].lower() for p in kb.get_shown_pillars()}
        names |= {n.lower() for n in self.user_added_pillars}
        names |= {e.lower() for e in self.excluded_concepts}
        return names

    def current_pillar(self):
        return None

    # HandlerSession mutators
    def surface_pillar(self, name: str) -> None:
        shown = name.lower() in [p["name"].lower() for p in kb.get_shown_pillars()]
        already = name.lower() in [p.lower() for p in self.user_added_pillars]
        if shown or already:
            self._last_surface = {"name": name, "is_new": False}
            return
        self.user_added_pillars.append(name)
        self._last_surface = {"name": name, "is_new": True}

    def add_sub_point(self, pillar: str, text: str) -> None:
        stored, is_new = self._store_sub_point(pillar, text, "text")
        self._last_sub_add = {"pillar": pillar, "stored": stored, "raw": text, "is_new": is_new}

    # Outcome renderer
    def render_question(self, user_input):
        yield from self._stream_qa(user_input)

    def render_framework(self, preamble=""):
        yield from self._yield_rerender(preamble)

    def render_summary(self):
        yield from self._stream_summary()

    def render_fallback(self, outcome=None):
        if outcome is None:
            msg = ("You've surfaced the main areas I'd flag — feel free to add, "
                   "remove, or question any part of what's here.")
            self._emit(msg); yield msg; return
        yield from self._ack_no_reprint()

    def _walk_done_render(self, touched):
        """After a multi-point walk: re-render the full framework (BB shows all)."""
        yield from self._yield_rerender("")

    def render_add(self, o):
        if o.action == "ask_placement":
            msg = (f"Should **{o.text}** be its own area, or a point under one of the "
                   f"existing areas? *(If under one, which?)*")
            self._emit(msg); yield msg; return

        if o.action == "navigate_offer":
            pp = self.pending_placement or {}
            reveal_withheld = bool(pp.get("reveal_on_accept"))   # sub-point of a WITHHELD pillar
            self.pending_placement = None
            gist = f" {o.explanation}" if o.explanation else ""
            nb = getattr(o, "navigate_bullet", None)
            stored_new = False
            if o.level == "sub_bullet" and o.pillar and nb:
                # A withheld pillar isn't on screen yet — surface it so the point renders
                # (and so dedup sees its KB bullets).
                if reveal_withheld:
                    self.surface_pillar(o.pillar)
                already = {grounding._strip_source_refs(b).strip().lower()
                           for b in self.presented_sub_bullets().get(o.pillar, [])}
                if grounding._strip_source_refs(nb).strip().lower() not in already:
                    self.add_sub_point(o.pillar, nb)
                    stored_new = bool((self._last_sub_add or {}).get("is_new"))
                # Count the contribution once (the shared offer left it uncounted). A
                # withheld pillar surfaced via a sub-point ALWAYS counts — the reveal is
                # the contribution, even when the concept's bullet is one the revealed
                # pillar shows statically (so stored_new is False), mirroring EXP's
                # reveal-on-accept. Otherwise count only a genuinely-new stored point.
                # BB resolves at the offer, so the handler never counts these (no
                # 'unreached' in BB's all-shown view).
                if not o.counted and (reveal_withheld or stored_new):
                    ev.record(handlers.AddOutcome(
                        action="added_new", pillar=o.pillar, level="sub_bullet",
                        counted=True, text=nb, source="user_spontaneous"),
                        self._evctx(), _sink)
            if stored_new:
                pre = f"Added under **{o.pillar}**.{gist}\n\n"
            elif reveal_withheld:
                # just brought a withheld pillar into the framework via the sub-point
                pre = f"Good point — I've brought in **{o.pillar}**.{gist}\n\n"
            elif o.level == "concept" and o.matched_text:
                pre = (f"That's already covered under **{o.pillar}** as "
                       f"*{o.matched_text}*.{gist}\n\n")
            else:
                pre = f"**{o.pillar}** is already in the framework.{gist}\n\n"
            yield from self._yield_rerender(pre); return

        if o.action == "duplicate":
            if o.level == "pillar" and o.pillar:
                pre = f"**{o.pillar}** is already in the framework.\n\n"
            elif o.pillar and o.matched_text:
                pre = (f"That's already covered under **{o.pillar}** as "
                       f"*{o.matched_text}*. Want to adjust it?\n\n")
            elif o.pillar:
                pre = f"That's already covered under **{o.pillar}**.\n\n"
            else:
                pre = "That's already in the framework.\n\n"
            yield from self._yield_rerender(pre); return

        if o.action == "navigated":
            yield from self._ack_no_reprint(); return

        if o.action == "revealed" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                yield from self._yield_rerender(f"Added **{o.pillar}** as a new area.\n\n")
            else:
                yield from self._yield_rerender(f"**{o.pillar}** is already in the framework.\n\n")
            return

        if o.action == "added_new" and o.level == "sub_bullet":
            st = self._last_sub_add or {}
            if st.get("is_new"):
                if o.also_covered:
                    gist = f" \u2014 {o.explanation}" if o.explanation else ""
                    also = (f" It also relates to **{o.also_covered}**{gist} "
                            f"Say the word if you'd rather it sit there.")
                else:
                    also = ""
                yield from self._yield_rerender(f"Added under **{o.pillar}**.{also}\n\n")
            else:
                stored = (self._last_sub_add or {}).get("stored")
                pre = (f"That's already covered under **{o.pillar}** as *{stored}*.\n\n"
                       if stored else f"That's already under **{o.pillar}**.\n\n")
                yield from self._yield_rerender(pre)
            return

        if o.action == "added_new" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                yield from self._yield_rerender(
                    f"Added **{o.pillar}** as a new area. What points would you "
                    f"like under it? Add them one at a time.\n\n")
            else:
                yield from self._yield_rerender(f"**{o.pillar}** is already in the framework.\n\n")
            return

        yield from self._ack_no_reprint()

    def render_removal(self, o, user_input, *, was_pending=False, pa=None):
        stage = o.stage
        if stage == "confirmed":
            if o.is_swap:
                wrong = pa.target if pa else o.target
                yield from self._yield_rerender(f"Done — I've removed **{wrong}**.\n\n")
                return
            if pa and pa.type == "remove_sub_bullet":
                yield from self._yield_rerender(
                    f"Done — I've removed that point from **{pa.pillar}**.\n\n")
                return
            yield from self._yield_rerender(f"Done — I've removed **{o.target}**.\n\n")
            return

        if stage == "abandoned":
            if pa and pa.type == "remove_sub_bullet":
                msg = f"No problem — I'll keep that point in **{pa.pillar}**."
            else:
                msg = f"No problem — I'll keep **{o.target}**."
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
            options = self.presented_pillars()
            opt = ("\n\nCurrently in the framework: "
                   + ", ".join(f"**{n}**" for n in options) + ".") if options else ""
            msg = ("Which part would you like to remove? You can name the pillar or the point."
                   + opt + "\n\n*(Or say **never mind** to keep everything as is.)*")
            self._emit(msg); yield msg; return

        if stage == "challenged":
            if was_pending:
                if self._reply_is_question(user_input):
                    # §3.6 question (+ swap_questioned W9) already fired by _fire_turn.
                    yield from self._stream_confirm_qa(user_input, o.target); return
                msg = f"No rush — reply **yes** to remove **{o.target}**, or **no** to keep it."
                self._emit(msg); yield msg; return
            if self.pending and self.pending.type == "remove_sub_bullet":
                msg = (f"Are you sure you want to remove this point from "
                       f"**{self.pending.pillar}**?\n\n*\"{o.target}\"*\n\n"
                       f"*Reply **yes** to confirm, or **no** to keep it.*")
            else:
                msg = (f"Are you sure you want to remove **{o.target}**?\n\n"
                       f"*Reply **yes** to confirm, or **no** to keep it.*")
            self._emit(msg); yield msg; return

        msg = f"No rush — reply **yes** to remove **{o.target}**, or **no** to keep it."
        self._emit(msg); yield msg

    def render_next_steps(self, o):
        if getattr(o, "revealed", False):
            yield from self._yield_rerender(f"Good point — I've included **{o.suggested_item}**.\n\n")
            return
        if not getattr(o, "suggested_item", None):
            msg = ("You've surfaced the main areas I'd flag — feel free to add, "
                   "remove, or question any part of what's here.")
            self._emit(msg); yield msg; return
        why = (o.grounding or "").split("\n")[0].strip()
        msg = (f"One area we haven't covered yet is **{o.suggested_item}**"
               + (f" — {why}" if why else "")
               + "\n\nIt's worth considering whether it applies to your case.")
        self._emit(msg); yield msg

    def _stream_summary(self):
        summary = "**Final Framework Summary**\n\n" + \
                  self._render_full_framework(is_first=False, closing=False)
        self.history.append(types.Content(role="model", parts=[types.Part(text=summary)]))
        update_answer(self.session_id, summary)
        log_agent_response(self.session_id, summary)
        yield summary

    def get_summary(self):
        yield from self._stream_summary()

    # Session control
    def end_session(self) -> None:
        final_framework = ""
        fallback        = ""

        for msg in reversed(self.history):
            if msg.role != "model" or not msg.parts:
                continue
            text = msg.parts[0].text
            if not fallback:
                fallback = text
            if self._is_answer(text):
                final_framework = text
                print(f"[END SESSION] found final framework ({len(text)} chars)")
                break

        if not final_framework:
            final_framework = fallback
            print("[END SESSION] no structured framework — using last model message")

        detected_at_end = self.concept_swap.is_detected
        if self.concept_swap.is_detected:
            print("[END SESSION] concept swap already detected during chat")
        else:
            print("[END SESSION] concept swap not detected during session")

        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection("sessions").document(self.session_id).update({
                "final_framework":       final_framework[:1000],
                "concept_swap_detected": self.concept_swap.is_detected,
                "swap_detected_at_end":  detected_at_end,
            })
            print(f"[END SESSION] Firestore stamped for session={self.session_id}")
        except Exception as e:
            print(f"[END SESSION] Firestore stamp failed: {e}")

        end_session(self.session_id)
    