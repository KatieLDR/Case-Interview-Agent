import logging
import re

from google.genai import types

from backend.agents.base import BaseAgent
from backend.domain import grounding, matching
from backend.interaction import handlers as h
from backend.interaction import intents
from backend.knowledge import knowledge_base as kb
from backend.knowledge.cases import get_case, get_clarification_facts
from backend.llm import client, CLASSIFIER_MODEL, classify_json
from backend.logger import (
    create_session, log_user_message, log_agent_response, log_interruption,
)
from backend.logging import events as ev
from backend.logging.sink import firestore_sink as _sink
from backend.agents.prompts.hitl import (
    PROACTIVE_PROMPTS, JUSTIFICATION_ACKS, SUB_BULLET_FORMAT_PROMPT,
    HITL_CLARIFICATION_SYSTEM_PROMPT, HITL_MAIN_SYSTEM_PROMPT,
)
from backend.tools.concept_swap import ConceptSwap

CASE_TYPE = "AI Implementation"

_BUTTON_ACTION_INTENTS = frozenset(
    intents.intent_for_button(b) for b in intents.BUTTON_INTENT
) - {None}
_HITL_BUTTON_LABELS = {
    "advance": "**✅ Include this pillar** or **❌ Exclude this pillar**",
    "remove":  "**❌ Exclude this pillar** or **➖ Remove a bullet**",
    "add":     "**➕ Add a bullet**",
    "revisit": "**↩️ Add point to a past pillar**",
}
_DEICTIC_PARENTS = frozenset({
    "this", "it", "here", "this one", "this concept", "this area",
    "this pillar", "this section", "current", "current concept",
    "the current concept", "the current pillar",
})

def _is_deictic_parent(name) -> bool:
    if not name:
        return False
    return re.sub(r"\s+", " ", str(name).strip().lower()).strip("?.!, ") in _DEICTIC_PARENTS

class HITLAgent(BaseAgent):
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self._init_flow_state()
        self.session_id    = create_session(user_id, agent_type="hitl")
        self.original_case = get_case("hitl")
        self.clarification_facts = get_clarification_facts("hitl")

        self.concept_swap = ConceptSwap(
            agent_type="hitl",
            session_id=self.session_id
        )

        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        logging.info(
            f"[KG INIT] case_type={CASE_TYPE}, "
            f"framework={self.kg_context['framework']}, "
            f"concepts={self.kg_context['concepts']}"
        )

        self.walkthrough_concepts = []
        self.walkthrough_index    = 0
        self.walkthrough_active   = False
        self.walkthrough_done     = False
        self.approved_concepts    = []
        self.swap_presented       = False
        self.swap_position        = 0

        self.concept_blocks = {}

        self.justification_pillars = set()
        self._justified_concepts   = set()

        # Interaction-mode flags
        self.awaiting_user_suggestion  = False
        self.awaiting_justification    = False
        self.justification_for         = None
        self.holding_after_justification = False
        self.awaiting_sub_point        = False
        self.awaiting_revisit_add      = False
        self.awaiting_pillar_name      = False
        self.revisit_target            = None
        self.post_walkthrough_revisit  = False
        self.prompt_index              = 0
        self.ack_index                 = 0
        self.user_contributed_concepts = set()
        self.navigated_pillars         = set()

        # Conversation history
        self.history = [
            types.Content(
                role="user",
                parts=[types.Part(text=self.original_case)]
            ),
            types.Content(
                role="model",
                parts=[types.Part(text=(
                    "I have received the case. We are now in the clarification round. "
                    "Feel free to ask any questions about the case before we begin."
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
            f"about the case before you begin.\n\n"
            f"When you're ready, I'll walk you through the framework one pillar at a "
            f"time. For each pillar, you could ask me a question before deciding or "
            f"**include** or **exclude** the pillar in your framework plan, **add** a bullet you come up with, or **remove** a bullet you think doesn't belong to the pillar.\n\n"
        )

    def get_pre_analysis_instruction(self) -> str:
        return (
            "📖 *After you click the button below, I'll walk you through each pillar "
            "one at a time. Use button to **Include**, **Exclude**, **➕ Add**, or **➖ Remove** to make your decision on each pillar. "
            "If you have any questions, feel free to ask first.*"
        )

    # Phase transition
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
            f"✅ **Let's begin.**\n\n"
            f"I'll walk you through the framework one pillar at a time. "
            f"For each one, use the buttons to **include** or **exclude** from current framework, "
            f"**➕ add** your own bullet (one at a time) or **➖ remove** any bullet I suggested, or just type a question first.\n\n"
            f"---\n\n"
        )
        yield from self._stream_concept(is_first=True)

    # Walkthrough state helpers
    def _build_walkthrough_concepts(self) -> list:
        base     = list(self.kg_context["concepts"])
        wrong    = self.concept_swap.config["wrong_concept"]
        position = min(1, len(base))
        base.insert(position, wrong)
        self.swap_position = position

        shown = [p["name"] for p in kb.get_shown_pillars()]
        self.justification_pillars = set(shown)

        logging.info(
            f"[WALKTHROUGH] built={base}, swap_position={position}, "
            f"justification_pillars={sorted(self.justification_pillars)}, "
            f"framework={self.kg_context['framework']}"
        )
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

    def _walkthrough_complete_message(self, reopened: bool = False):
        # Re-render the full framework (like EXP), then invite freeform discussion.
        # _stream_summary sets walkthrough_done = True, which routes subsequent
        # messages to _stream_post_walkthrough (revisit/Q&A) and hides the
        # per-concept buttons. `reopened` is set when returning from a
        # post-walkthrough pillar revisit rather than first completion.
        if reopened:
            yield "✅ Updated. Here's the framework as it stands now:\n\n"
        else:
            yield "✅ We've covered all the pillars. Here's the framework as it stands:\n\n"
        yield from self._stream_summary()
        invite = (
            "\n\nThat's the full framework as it stands — want to revisit any pillar to "
            "add, change, or question something? Just mention the pillar name. Otherwise, click **‼️End Session** "
            "to finish. **Note: this cannot be undone**.\n\n"
        )
        self.history.append(types.Content(role="model", parts=[types.Part(text=invite)]))
        log_agent_response(self.session_id, invite)
        yield invite

    def _get_proactive_prompt(self) -> str:
        prompt = PROACTIVE_PROMPTS[self.prompt_index % len(PROACTIVE_PROMPTS)]
        self.prompt_index += 1
        return prompt

    def _locate_concept(self, name: str) -> int | None:
        for i, c in enumerate(self.walkthrough_concepts):
            if c.lower() == name.lower():
                return i
        return None

    def _normalize_pillar(self, name: str) -> str:
        for c in self.walkthrough_concepts:
            if c.lower() == name.lower():
                return c
        for p in kb.get_all_pillars():
            if p["name"].lower() == name.lower():
                return p["name"]
        return name

    @staticmethod
    def _is_substantive_justification(text: str) -> bool:
        t = text.strip()
        words = [w for w in t.split() if any(ch.isalpha() for ch in w)]
        return len(words) >= 3 and len(t) >= 12

    # LLM classifiers
    def _check_duplicate_proactive(self, concept: str) -> dict:
        all_concepts = list(self.walkthrough_concepts)
        try:
            result = classify_json(
                    f"You are checking if a user-suggested concept matches any concept "
                    f"in an existing list.\n\n"
                    f"User suggested: \"{concept}\"\n\n"
                    f"Existing concepts: {all_concepts}\n\n"
                    f"Reply with JSON only, no markdown, no explanation:\n"
                    f"{{\"is_duplicate\": true or false, "
                    f"\"matched_concept\": \"exact string from list or null\"}}\n\n"
                    f"is_duplicate=true ONLY if the user suggestion is clearly the same "
                    f"concept as one in the list — same topic, possibly different wording.\n"
                    f"matched_concept must be the exact string from the list.\n"
                    f"WHEN IN DOUBT: is_duplicate=false."
            )
            logging.info(f"[DUPLICATE] check result: {result}")
            return result
        except Exception as e:
            logging.warning(f"[DUPLICATE] check failed: {e} — defaulting to not duplicate")
            return {"is_duplicate": False, "matched_concept": None}

    def _format_sub_bullet(self, item: str) -> str:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=SUB_BULLET_FORMAT_PROMPT.format(item=item),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            out = grounding._strip_source_refs(self._strip_fences(response.text)).strip().strip("-• ").rstrip(".")
            if out:
                logging.info(f"[SUB-POINT FORMAT] '{item}' → '{out}'")
                return out
        except Exception as e:
            logging.warning(f"[SUB-POINT FORMAT] error: {e} — keeping raw")
        return item.strip()

    def _store_sub_point(self, pillar: str, item: str, modality: str = "text",
                         source: str = "user_spontaneous",
                         *, resolved: str | None = None) -> tuple[str, bool]:
        pillar  = self._normalize_pillar(pillar)
        if resolved is not None:
            # Caller already resolved the KB phrasing (e.g. the sub-point navigate
            # path, which matched a concept up front). Use it verbatim so the
            # stored bullet matches what the caller announced — re-resolving here
            # can drift to a different concept/key-question.
            stored = resolved
        else:
            matched, _ = matching.match_key_question(item, pillar)
            if not matched:
                matched = matching.canonical_add_bullet(item, refs=False)
            stored  = matched if matched else self._format_sub_bullet(item)
        # Dedup against what the pillar already shows: the streamed block (if the
        # pillar has been presented) AND its canonical KB sub-bullets, so a
        # KB-matched bullet for a not-yet-streamed pillar isn't duplicated on render.
        block = self.concept_blocks.get(pillar, "")
        shown_lines = {l.strip().lstrip("-• ").strip().lower()
                       for l in block.splitlines() if l.strip()}
        kb_pillar = next((p for p in kb.get_all_pillars()
                          if p["name"].lower() == pillar.lower()), None)
        if kb_pillar:
            shown_lines.update(grounding._strip_source_refs(b).strip().lower()
                               for b in kb_pillar.get("sub_bullets", []))
        if stored.lower() in shown_lines:
            return stored, False
        existing = self.user_sub_points.setdefault(pillar, [])
        if any(s.lower() == stored.lower() for s in existing):
            logging.info(f"[SUB-POINT] duplicate skipped: '{stored}' under '{pillar}'")
            return stored, False
        existing.append(stored)
        ev.record(h.AddOutcome(action="added_new", pillar=pillar, level="sub_bullet",
                               counted=True, text=stored, source=source),
                  self._evctx(source=source, modality=modality), _sink)
        logging.info(f"[SUB-POINT] '{item}' → '{stored}' under '{pillar}'")
        return stored, True

    # Main phase — stateful walkthrough router
    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        if self.pending is not None:
            log_user_message(self.session_id, user_input)
            pa = self.pending
            if pa.requires_justification and pa.justification is None:
                if h._classify_confirmation(user_input) == "decline":
                    outcome = h.resolve_pending(self, user_input, decision="decline")
                else:
                    outcome = h.resolve_pending(self, user_input, decision="confirm",
                                                justification=user_input)
            else:
                outcome = h.resolve_pending(self, user_input)
            yield from self.render_removal(outcome, pa=pa)
            return

        if self.awaiting_sub_point:
            concept = self._current_concept() or "this concept"
            log_user_message(self.session_id, f"[SUB-POINT ADD] {user_input}")
            if self._is_multi_entry(user_input, concept):
                msg = (f"Let's add one bullet at a time, which single bullet would you like "
                       f"under **{concept}**? (Click **✅ Done adding** when finished.)")
                self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
                log_agent_response(self.session_id, msg)
                yield msg
                return
            stored, is_new = self._store_sub_point(concept, user_input, modality="button")

            block  = self.concept_blocks.get(concept, "").strip()
            points = self.user_sub_points.get(concept, [])
            rerender = f"**{concept}**"
            if block:
                rerender += f"\n{block}"
            if points:
                rerender += "\n" + "\n".join(f"- {p}" for p in points)

            lead = (
                f"Added under **{concept}**. Here's how it looks now:"
                if is_new else
                f"That's already covered under **{concept}** as *{stored}*. Here's how it looks now:"
            )
            yield (
                f"{lead}\n\n"
                f"{rerender}\n\n"
                f"Anything else to add under **{concept}**? "
                f"When you're done (or have a question) click ✅ Done to exit editing mode. "
            )
            return

        if self.awaiting_revisit_add:
            target = self.revisit_target or self._current_concept() or "this concept"
            log_user_message(self.session_id, f"[REVISIT ADD] {user_input}")
            if self._is_multi_entry(user_input, target):
                msg = (f"Let's add one bullet at a time, which single bullet would you like "
                       f"to add to **{target}**? (Click **✅ Done adding** when finished.)")
                self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
                log_agent_response(self.session_id, msg)
                yield msg
                return
            stored, is_new = self._store_sub_point(target, user_input, modality="button")
            lead = (f"Added to **{target}**." if is_new
                    else f"That's already covered under **{target}** as *{stored}*.")
            yield (
                f"{lead}\n\n"
                f"Anything else to add to **{target}**? "
                f"Click **✅ Done adding** when you're finished."
            )
            return

        if self.awaiting_justification:
            if not h.is_meaningful_justification(user_input):
                log_user_message(self.session_id, f"[JUSTIFICATION:retry] {user_input}")
                yield "Could you say a bit more about your reasoning? A sentence is plenty.\n\n"
                return

            log_user_message(self.session_id, f"[JUSTIFICATION:{self.justification_for}] {user_input}")
            logging.info(f"[JUSTIFICATION] collected for={self.justification_for}: '{user_input}'")
            self.awaiting_justification = False
            self.justification_for      = None
            cur = self._current_concept()
            if cur is not None:
                self._justified_concepts.add(cur)
                self.walkthrough_index += 1

            yield from self._stream_justification_ack()
            return

        if self.awaiting_pillar_name:
            self.awaiting_pillar_name = False
            log_user_message(self.session_id, f"[PROACTIVE PILLAR] {user_input}")
            pres = intents.classify_intent(
                user_input,
                current_pillar=self._current_concept() or "(none)",
                current_bullets=self._ctx_bullets(),
                walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
                last_agent=self._last_agent_text() or "(nothing yet)",
            )
            concept = pres.detail if (pres.intent == "add" and pres.detail) else None
            if concept is None:
                concept = self._mentioned_pillar(user_input)
            if pres.multi or concept is None:
                # Keep it one pillar at a time — re-prompt for a single name.
                self.awaiting_pillar_name = True
                msg = ("Let's add one pillar at a time, which single pillar would "
                       "you like to add?")
                self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
                log_agent_response(self.session_id, msg)
                yield msg
                return
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            yield from self._go_to_pillar(concept)
            return

        if self.awaiting_user_suggestion:
            # Proactive prompt is button-driven: only the buttons act. A genuine
            # question is answered (Q&A) and the prompt re-shows; anything else is
            # nudged back to the buttons so the next step stays an explicit choice.
            log_user_message(self.session_id, f"[PROACTIVE RESPONSE] {user_input}")
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            pres = intents.classify_intent(
                user_input,
                current_pillar=self._current_concept() or "(none)",
                current_bullets=self._ctx_bullets(),
                walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
                last_agent=self._last_agent_text() or "(nothing yet)",
            )
            if pres.intent in ("question", "doubt") and not pres.multi:
                self.awaiting_user_suggestion = False
                ev.question(self._evctx(modality="text"), _sink)
                yield from self._stream_concept_qa()
                yield from self._stream_proactive_prompt()
                return
            # Keep awaiting_user_suggestion True so the proactive buttons re-render.
            msg = ("In this mode the next step is a button — use **➕ Add my own pillar** "
                   "or **💡 Ask agent suggestion** below (or **‼️ End Session**).")
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg
            return

        just_added_concept = None
        res = intents.classify_intent(
            user_input,
            current_pillar=self._current_concept() or "(none)",
            current_bullets=self._ctx_bullets(),
            walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
            last_agent=self._last_agent_text() or "(nothing yet)",
        )
        intent = res.intent

        if self.pending_suggestion is not None:
            _ps = self.pending_suggestion
            _accepting = (h._accepts_offer(user_input)
                          or h._add_accepts_suggestion(_ps, intent, res.detail,
                                                       res.parent, user_input))
            self.pending_suggestion = None
            if _accepting and _ps.get("origin") == "agent_suggest":
                log_user_message(self.session_id, user_input)
                self.history.append(
                    types.Content(role="user", parts=[types.Part(text=user_input)]))
                yield from self._reveal_suggested_pillar(_ps["item"])
                return

        # Multi-point: HITL is button-driven — nudge to add one at a time via buttons.
        if res.multi:
            log_user_message(self.session_id, user_input)
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            yield from self._nudge_multi()
            return

        cs_detected = False
        swap_live   = self.swap_presented and not self.concept_swap.is_detected

        if swap_live and intent not in ("add", "remove", "question"):
            cs_detected = self.concept_swap.check_detection(user_input)
            if cs_detected:
                wrong = self.concept_swap.config["wrong_concept"]
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)
                self.walkthrough_index += 1
                logging.info(f"[SWAP] caught via text — index→{self.walkthrough_index}")

        if (self.walkthrough_active
                and self.walkthrough_index > self.swap_position
                and not self.swap_presented):
            logging.error(f"[INVARIANT] rewinding to swap_position={self.swap_position}")
            self.walkthrough_index = self.swap_position

        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        logging.info(
            f"[ROUTE] active={self.walkthrough_active}, done={self.walkthrough_done}, "
            f"swap_presented={self.swap_presented}, index={self.walkthrough_index}, "
            f"intent={intent}, cs_detected={cs_detected}"
        )

        if not self.walkthrough_active:
            self.walkthrough_concepts     = self._build_walkthrough_concepts()
            self.walkthrough_active       = True
            self.walkthrough_index        = 0
            self.swap_presented           = False
            yield from self._stream_concept(is_first=True)

        elif self.walkthrough_done:
            yield from self._stream_post_walkthrough(user_input, res)

        elif cs_detected:
            yield from self._stream_swap_caught()
            yield "\n\n"
            yield from self._continue_or_finish()

        elif intent in _BUTTON_ACTION_INTENTS:
            yield from self._nudge_to_button(intent)

        elif intent == "ask_agent_to_suggest":
            yield from self._handle_suggest(user_input)

        else:
            current  = self._current_concept()
            _on_swap = (swap_live and current is not None
                        and self._is_wrong_concept(current))
            ev.question(self._evctx(modality="text"), _sink)
            if _on_swap:
                ev.swap_questioned(self._evctx(modality="text"), _sink)
            yield from self._stream_concept_qa(just_added=just_added_concept)

    # Proactive prompt + justification
    def _ctx_bullets(self) -> str:
        cur   = self._current_concept() or ""
        block = (self.concept_blocks.get(cur, "") or "").strip()
        pts   = self.user_sub_points.get(cur, [])
        lines = ([block] if block else []) + [f"- {p}" for p in pts]
        return "\n".join(lines) or "(none)"

    def _handle_suggest(self, user_input: str):
        if self._current_concept() is not None and not self.walkthrough_done:
            ev.record(h.AdvanceOutcome(passive=False, elicited=True),
                      self._evctx(modality="text"), _sink)
            self.walkthrough_index += 1
            if self._current_concept() is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_concept(is_first=False)
            return
        msg = ("You've surfaced the main pillars I'd flag. Use the buttons to add, "
               "remove, or revisit any part of the framework.")
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _nudge_to_button(self, intent: str):
        label = _HITL_BUTTON_LABELS.get(intent, "the buttons below")
        msg = ("In this mode each change is made with a button, so it stays explicit. "
               f"You can do that with {label} below — go ahead and click when you're ready.")
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _is_multi_entry(self, text: str, pillar: str) -> bool:
        """True when the user typed several points at once (so we gate to one at a
        time). Cheap structural check first; only consult the classifier when there's
        a separator hint, so a plain single point costs no extra call."""
        _, items = intents._structural_list(text)
        if len(items) > 1:
            return True
        if not re.search(r"[,;\n]|\band\b", text or ""):
            return False
        res = intents.classify_intent(
            text,
            current_pillar=pillar,
            current_bullets=self._ctx_bullets(),
            walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
            last_agent=self._last_agent_text() or "(nothing yet)",
        )
        return bool(res.multi)

    def _nudge_multi(self):
        """A pasted multi-point message: HITL is button-driven, so add points one at a
        time with the ➕ Add button rather than running EXP/BB's freetext walk."""
        msg = ("Looks like a few bullets at once — in this mode we add them **one at a "
               "time**. **➕ Add bullet to consider in this pillar** adds under the pillar "
               "we're on now; to add elsewhere, jump there first or use **➕ Add my own "
               "pillar** at the next prompt.")
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _suggest_next(self):
        """Present the next, not-yet-shown concept as the agent's suggestion WITHOUT
        advancing again — the index already points at it (the preceding decision
        advanced it before this proactive prompt). Log the elicited advance, since the
        user deferred the choice to the agent."""
        logging.info("[PROACTIVE] ask_agent_to_suggest -> present current (no advance)")
        ev.record(h.AdvanceOutcome(passive=False, elicited=True),
                  self._evctx(modality="text"), _sink)
        if self._current_concept() is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_concept(is_first=False)

    def _go_to_pillar(self, concept: str):
        """Navigate the walkthrough to the pillar the user named: reorder an
        already-listed pillar to the front (counting it once as a pillar add), report
        an already-covered pillar in place, or surface a new/withheld one. Shared by
        the freetext proactive path and the ➕ Add my own pillar button."""
        dup = self._check_duplicate_proactive(concept)

        if dup["is_duplicate"] and dup["matched_concept"]:
            target           = dup["matched_concept"]
            matched_withheld = None
        else:
            matched_withheld, _ = matching.match_pillar(concept)
            target           = matched_withheld or concept

        idx = self._locate_concept(target)
        if idx is not None:
            if idx >= self.walkthrough_index:
                if idx != self.walkthrough_index:
                    self.walkthrough_concepts.pop(idx)
                    self.walkthrough_concepts.insert(self.walkthrough_index, target)
                if target.lower() not in self.navigated_pillars:
                    self.navigated_pillars.add(target.lower())
                    ev.record(h.AddOutcome(action="added_new", pillar=target,
                                           level="pillar", counted=True,
                                           source="user_elicited"),
                              self._evctx(source="user_elicited", modality="text"), _sink)
                logging.info(f"[PROACTIVE] navigate to '{target}' (was idx={idx})")
                gist    = matching.pillar_gist(target)
                why_txt = f" {gist}" if gist else ""
                yield f"Sure — let's look at **{target}** now.{why_txt}\n\n"
                yield from self._stream_concept(is_first=False)
            else:
                logging.info(f"[PROACTIVE] '{target}' already covered (idx={idx})")
                yield (
                    f"We've already covered **{target}** — it's in your "
                    f"framework.\n\n"
                )
                yield from self._stream_proactive_prompt()
        else:
            if matched_withheld:
                logging.info(f"[PROACTIVE] new pillar → withheld pillar '{matched_withheld}'")
                yield (
                    f"Good call, **{matched_withheld}** is an important pillar. "
                    f"Let's include it.\n\n"
                )
            else:
                logging.info(f"[PROACTIVE] new user concept: '{target}'")
                yield f"Great suggestion — let's explore **{target}** now.\n\n"
            yield from self._stream_user_contributed_concept(target)

    def on_proactive_add_pillar(self):
        """➕ Add my own pillar (proactive prompt): ask for ONE pillar name, then
        navigate so the per-concept buttons target the pillar the user raised."""
        self.awaiting_user_suggestion = False
        self.awaiting_pillar_name = True
        msg = "Which pillar would you like to add? Tell me one at a time."
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def on_proactive_suggest(self):
        """💡 Ask agent suggestion (proactive prompt): present the agent's next pick."""
        self.awaiting_user_suggestion = False
        yield from self._suggest_next()

    def _stream_proactive_prompt(self):
        self.holding_after_justification = False
        self.awaiting_user_suggestion = True
        prompt = self._get_proactive_prompt()
        logging.info(f"[PROACTIVE] prompt_index={self.prompt_index - 1}")
        yield prompt

    def _stream_justification_prompt(self, for_decision: str, concept: str = None):
        self.awaiting_justification = True
        self.justification_for      = for_decision
        concept_name = concept or self._current_concept() or "this concept"

        if for_decision == "accept":
            yield (
                f"Before we move on — what makes **{concept_name}** essential for this case? "
                f"What would we risk missing if we excluded it?"
            )
        else:
            yield (
                f"You've chosen to leave out **{concept_name}**. "
                f"What's your reasoning for excluding it?"
            )

    def _stream_justification_ack(self):
        ack = JUSTIFICATION_ACKS[self.ack_index % len(JUSTIFICATION_ACKS)]
        self.ack_index += 1
        yield ack + "\n\n"
        yield from self._continue_or_finish()

    def _stream_justification_hold_ack(self, concept: str):
        ack = JUSTIFICATION_ACKS[self.ack_index % len(JUSTIFICATION_ACKS)]
        self.ack_index += 1
        yield ack + "\n\n"
        yield (f"Anything else you'd add to **{concept}**? "
               f"Say **move on** when you're ready for the next pillar.")

    def _stream_user_contributed_concept(self, concept: str):
        self.walkthrough_concepts.insert(self.walkthrough_index, concept)
        if self.walkthrough_index <= self.swap_position:
            self.swap_position += 1
            logging.info(f"[USER CONCEPT] swap_position shifted to {self.swap_position}")
        ev.record(h.AddOutcome(action="added_new", pillar=concept, level="pillar",
                               counted=True, source="user_elicited"),
                  self._evctx(source="user_elicited", modality="text"), _sink)
        logging.info(f"[USER CONCEPT] inserted at index={self.walkthrough_index}: '{concept}'")

        yield from self._stream_concept(is_first=False)

    def _reveal_suggested_pillar(self, concept: str):
        self.walkthrough_concepts.insert(self.walkthrough_index, concept)
        if self.walkthrough_index <= self.swap_position:
            self.swap_position += 1
            logging.info(f"[REVEAL] swap_position shifted to {self.swap_position}")
        ev.record(h.SuggestOutcome(level="pillar", suggested_item=concept,
                                   grounding=(grounding.ground_pillar(concept) or None),
                                   accepted=True, revealed=True),
                  self._evctx(modality="text"), _sink)
        logging.info(f"[REVEAL] agent-suggested pillar accepted: '{concept}'")
        yield from self._stream_concept(is_first=False)

    # Streaming sub-methods
    def _add_sub_point(self, matched_concept: str, sub_point: str):
        if _is_deictic_parent(matched_concept):
            current = self._current_concept()
            if current:
                logging.info(f"[PROACTIVE] deictic parent {matched_concept!r} -> {current!r}")
                matched_concept = current
        pillar = self._normalize_pillar(matched_concept)
        stored, is_new = self._store_sub_point(pillar, sub_point, source="user_elicited")
        if is_new:
            yield f"Got it! Adding that under **{pillar}**:\n- {stored}\n\n"
        else:
            yield f"That's already covered under **{pillar}** as *{stored}*.\n\n"

    def _stream_concept(self, is_first: bool):
        concept = self._current_concept()
        if concept is None:
            yield from self._walkthrough_complete_message()
            return

        is_wrong = self._is_wrong_concept(concept)

        pillar = None
        if not is_wrong:
            pillar = next(
                (p for p in kb.get_all_pillars()
                 if p["name"].lower() == concept.lower()),
                None
            )

        if is_wrong or pillar is not None:
            if is_wrong:
                swap    = kb.get_swap_concept()
                bullets = swap.get("sub_bullets", []) if swap else []
            else:
                bullets = pillar.get("sub_bullets", [])

            bullets = [grounding._strip_source_refs(b) for b in bullets]
            bullet_lines = "\n".join(f"- {b}" for b in bullets)
            self.concept_blocks[concept] = bullet_lines

            display = bullet_lines
            added   = self.user_sub_points.get(concept, [])
            if added:
                display += "\n" + "\n".join(f"- {p}" for p in added)

            prefix = f"**{concept}**\n{display}"
            if is_first:
                prefix = "💡 When you're finished, click ‼️End Session to close your session. Note: this cannot be undone. \n\n Here is the first pillar I recommend. Do you want to include it in your framework plan?\n\n" + prefix

            self.history.append(
                types.Content(role="user",
                              parts=[types.Part(text=f"[Present concept: {concept}]")])
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
                    logging.info(f"[SWAP] presented at position={self.swap_position}")

            logging.info(f"[CONCEPT BLOCK] static stored for '{concept}'")
            yield prefix
            return

        heading = "Here is the first pillar I recommend. Do you want to include it in your framework plan?\n\n" if is_first else ""
        heading += f"**{concept}**\n"
        yield heading

        instruction = (
            f"{self._build_system_prompt()}\n\n"
            f"─── CONCEPT TO PRESENT NOW ───────────────────────────────────\n"
            f"Concept: **{concept}**\n"
            f"This is a user-added area, not part of the standard framework.\n"
            f"Output ONLY 2 sub-bullets for this concept, specific to this case.\n"
            f"Do NOT repeat the concept name.\n"
            f"Do NOT add a closing question — buttons handle advancement.\n"
            f"Do NOT include any sources or citations.\n"
            f"Maximum 2 sub-bullets.\n"
            f"─────────────────────────────────────────────────────────────\n"
        )
        task_injection = (
            f"[Output only 2 sub-bullets for **{concept}**. "
            f"No concept name, no closing question, no sources.]"
        )

        full_reply = []
        for token in self._stream_with_instruction(
            instruction    = instruction,
            prefix         = "",
            task_injection = task_injection,
            track_swap     = False,
            store_answer   = False,
        ):
            full_reply.append(token)
            yield token

        self.concept_blocks[concept] = "".join(full_reply).strip()
        logging.info(f"[CONCEPT BLOCK] LLM-generated stored for '{concept}'")

    def _stream_concept_qa(self, just_added: str | None = None):
        current = self._current_concept()
        concept = current or "the current concept"
        on_swap = (
            self.swap_presented
            and not self.concept_swap.is_detected
            and current is not None
            and self._is_wrong_concept(current)
        )

        added_note = ""
        grounding = self._concept_grounding(concept)
        grounding_block = ""
        if grounding:
            grounding_block = (
                f"─── KNOWN POINTS FOR THIS CONCEPT (ground your answer here) ───\n"
                f"{grounding}\n"
                f"─── GROUNDING RULE ───────────────────────────────────────────\n"
                f"Base your answer on the KNOWN POINTS above and the case. Do NOT\n"
                f"introduce regulations, statistics, or framework concepts that are\n"
                f"not among the known points. Do NOT output bracketed letter markers.\n"
                f"─────────────────────────────────────────────────────────────\n\n"
            )
        if just_added:
            added_note = (
                f"Good idea — I'll add **{just_added}** after we finish "
                f"**{concept}**.\n\n"
            )

        instruction = (
            f"{self._build_system_prompt()}\n\n"
            f"─── CURRENT CONCEPT ──────────────────────────────────────────\n"
            f"Concept: **{concept}**\n"
            f"On swap concept: {on_swap}\n"
            f"Framework: {self.kg_context['framework']} | "
            f"Case: {self.kg_context['case_type']}\n"
            f"Framework concepts (in order): "
            f"{', '.join(c for c in self.walkthrough_concepts if just_added is None or c.lower() != just_added.lower())}\n"
            f"─────────────────────────────────────────────────────────────\n\n"
            f"─────────────────────────────────────────────────────────────\n\n"
            f"{grounding_block}"
            f"─── RULES ────────────────────────────────────────────────────\n"
            f"Answer in 2–3 sentences. Plain language only.\n"
            f"Do NOT end with a question — buttons handle advancement.\n"
            f"Do NOT re-present the concept block.\n"
            f"Do NOT present or describe any other concept.\n"
            f"Do NOT claim to add, remove, keep, or skip anything.\n"
            f"Do NOT suggest whether the candidate should approve or reject.\n"
            f"─────────────────────────────────────────────────────────────\n"
        )

        yield from self._stream_with_instruction(
            instruction = instruction,
            prefix      = added_note,
        )
        if self.should_show_buttons():
            yield (
                "\n\n*Would you like to **include** or **exclude** this, "
                "or **➕ add** a bullet? Just use the buttons below.*"
            )
    def _concept_grounding(self, concept: str) -> str:
        return grounding.ground_pillar(concept)

    def _stream_swap_caught(self):
        wrong       = self.concept_swap.config["wrong_concept"]
        wrong_fw    = self.concept_swap.config["wrong_framework"]
        active_fw   = self.kg_context["framework"]
        active_case = self.kg_context["case_type"]

        instruction = (
            f"{self._build_system_prompt()}\n\n"
            f"─── CONTEXT ──────────────────────────────────────────────────\n"
            f"Wrong concept flagged: **{wrong}**\n"
            f"It belongs to: {wrong_fw} (a different type of analysis)\n"
            f"This case is: {active_case} — {active_fw}\n"
            f"─────────────────────────────────────────────────────────────\n\n"
            f"─── RULES ────────────────────────────────────────────────────\n"
            f"Respond in 2–3 sentences:\n"
            f"1. Acknowledge the catch warmly\n"
            f"2. Explain briefly why it belongs to a different analysis\n"
            f"3. One closing sentence confirming you are moving on\n"
            f"- Do NOT end with a question\n"
            f"- Do NOT preview the next concept\n"
            f"─────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_freeform(self):
        concepts_str = " → ".join(self.kg_context["concepts"])
        instruction  = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────\n\n"
            f"You are a strategic thinking partner. The user has seen the full "
            f"framework. Answer their question concisely in plain language. "
            f"Ask ONE follow-up question after answering."
        )
        yield from self._stream_with_instruction(instruction=instruction)

    # Post-walkthrough revisit
    _EDIT_INTENTS = frozenset({"add", "revisit", "remove"})

    def _mentioned_pillar(self, text: str) -> str | None:
        """Deterministic scan: return the longest non-excluded walkthrough
        concept whose name appears verbatim in `text`, else None."""
        if not text:
            return None
        norm = re.sub(r"\s+", " ", text.lower())
        excluded = [e.lower() for e in self.excluded_concepts]
        best = None
        for c in self.walkthrough_concepts:
            if c.lower() in excluded:
                continue
            if c.lower() in norm and (best is None or len(c) > len(best)):
                best = c
        return best

    def _resolve_known_pillar(self, name: str) -> str | None:
        """Resolve a free-text mention to an existing, non-excluded walkthrough
        pillar. Returns the exact concept string, or None if it isn't a known
        navigable pillar. Tries a cheap verbatim scan first, then the proactive
        block's LLM resolution (_check_duplicate_proactive → match_pillar)."""
        if not name:
            return None
        direct = self._mentioned_pillar(name)
        if direct:
            return direct
        dup = self._check_duplicate_proactive(name)
        if dup["is_duplicate"] and dup["matched_concept"]:
            target = dup["matched_concept"]
        else:
            matched_withheld, _ = matching.match_pillar(name)
            target = matched_withheld or name
        idx = self._locate_concept(target)
        if idx is None:
            return None
        concept = self.walkthrough_concepts[idx]
        if concept.lower() in [e.lower() for e in self.excluded_concepts]:
            return None
        return concept

    def _locate_subpoint(self, text: str) -> tuple[str, str | None, str | None, str | None] | None:
        """Resolve free text to a *sub-point* of a pillar, mirroring EXP/BB
        (which resolve sub-bullet granularity via matching.locate in the shared
        outcome router). Returns (pillar, matched_point, why, bullet) when the
        text names a point that belongs to a non-excluded KB pillar — whether
        that pillar is already in the walkthrough or still withheld (a withheld
        pillar is simply one not yet in walkthrough_concepts). `bullet` is the
        canonical KB phrasing for the matched concept, so the caller stores
        exactly the concept it announced (no second, possibly-divergent match).
        The caller surfaces/navigates the pillar and counts the contribution at
        sub_bullet level in both cases. A verbatim pillar mention is left to the
        pillar-level resolvers."""
        if not text or self._mentioned_pillar(text):
            return None
        km = matching.locate(text)
        if km.level != "concept" or not km.pillar:
            return None
        if km.pillar.lower() in [e.lower() for e in self.excluded_concepts]:
            return None
        bullet = matching.concept_bullet(km.concept_id, refs=False) if km.concept_id else None
        return (self._normalize_pillar(km.pillar), km.matched_text,
                matching.pillar_gist(km.pillar) or None, bullet)

    def _navigate_for_subpoint(self, pillar: str, point: str | None, why: str | None,
                               bullet: str | None = None, *, user_text: str):
        """A mention matched a sub-point of `pillar` during the walkthrough.
        Tell the user where it lives and why, add their point as a visible
        sub-bullet under the pillar, then move the walkthrough to that pillar so
        the per-concept buttons show. A pillar already in the walkthrough is
        re-ordered to the front (like the pillar-navigate path); a withheld
        pillar not yet in the list is inserted at the current position. A pillar
        already covered keeps the point and is reported in place without
        rewinding.

        The point is stored via `_store_sub_point`, which uses KB phrasing when
        the text matches a known concept/key-question and otherwise rephrases the
        user's wording to house style. Whether the pillar was already in the
        framework or surfaced from withheld, the contribution is counted at
        sub_bullet level — never as a pillar add."""
        idx     = self._locate_concept(pillar)
        pt      = f" *{point}*" if point else ""
        why_txt = f" — {why}" if why else ""
        if idx is not None and idx < self.walkthrough_index:
            logging.info(f"[PROACTIVE] sub-point of already-covered '{pillar}'")
            self._store_sub_point(pillar, user_text, source="user_elicited", resolved=bullet)
            yield (f"That bullet{pt} is part of **{pillar}**{why_txt} "
                   f"We've already covered **{pillar}**, I've added it there.\n\n")
            yield from self._stream_proactive_prompt()
            return
        if idx is None:
            # Withheld pillar (not yet in the walkthrough): bring it into the
            # framework at the current position. The swap concept's tracked
            # position shifts with the insert, as in the other reveal paths.
            self.walkthrough_concepts.insert(self.walkthrough_index, pillar)
            if self.walkthrough_index <= self.swap_position:
                self.swap_position += 1
            logging.info(f"[PROACTIVE] withheld sub-point -> surface '{pillar}'")
        elif idx != self.walkthrough_index:
            self.walkthrough_concepts.pop(idx)
            self.walkthrough_concepts.insert(self.walkthrough_index, pillar)
        self.navigated_pillars.add(pillar.lower())
        self._store_sub_point(pillar, user_text, source="user_elicited", resolved=bullet)
        logging.info(f"[PROACTIVE] sub-point '{point}' -> navigate to '{pillar}'")
        yield (f"That bullet{pt} belongs under **{pillar}**{why_txt} "
               f"Let's look at **{pillar}**.\n\n")
        yield from self._stream_concept(is_first=False)

    def _stream_post_walkthrough(self, user_input: str, res):
        """After the walkthrough, free text is pillar-revisit-centric:
        - an edit/navigate intent (add/revisit/remove) naming a known pillar, or
          a question/doubt that names a known pillar, re-opens that pillar with
          the normal walkthrough buttons (questions are also answered);
        - anything that doesn't resolve to a known pillar asks the user to name
          which pillar they mean."""
        is_question = res.intent in ("question", "doubt")
        if res.intent in self._EDIT_INTENTS:
            target = None
            for cand in (res.parent, res.detail):
                target = self._resolve_known_pillar(cand)
                if target:
                    break
        else:
            # questions and everything else carry no reliable detail — resolve
            # the pillar from the raw message.
            target = self._resolve_known_pillar(user_input)

        if target:
            yield from self._revisit_pillar_full(target, answer_question=is_question)
            return

        # Sub-bullet granularity (mirrors EXP/BB): the mention may name a point
        # that belongs to a known pillar. Re-open that pillar (buttons show),
        # prefacing with where the point lives and why.
        sub = self._locate_subpoint(user_input)
        if sub is not None:
            pillar, point, why, bullet = sub
            pt      = f" *{point}*" if point else ""
            why_txt = f" — {why}" if why else ""
            if self._locate_concept(pillar) is None:
                # Withheld pillar surfaced post-walkthrough: add it to the
                # framework so _revisit_pillar_full can navigate to it.
                self.walkthrough_concepts.append(pillar)
            self._store_sub_point(pillar, user_input, source="user_elicited", resolved=bullet)
            yield f"That bullet{pt} belongs under **{pillar}**{why_txt}\n\n"
            yield from self._revisit_pillar_full(pillar, answer_question=is_question)
            return

        yield from self._ask_which_pillar()

    def _ask_which_pillar(self):
        """No pillar could be resolved post-walkthrough — ask the user to name
        one of the pillars on the framework (or end the session)."""
        pillars = self.presented_pillars()
        if not pillars:
            yield from self._stream_freeform()
            return
        listing = ", ".join(f"**{p}**" for p in pillars)
        msg = (
            f"Which pillar would you like to revisit? Just name one of: {listing}. "
            f"Otherwise, click **‼️End Session** to finish."
        )
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _revisit_pillar_full(self, target: str, *, answer_question: bool = False):
        """Go back to an already-covered pillar and re-present it with the
        normal walkthrough buttons. Logs the same navigation event as the
        in-walkthrough proactive path, guarded by navigated_pillars. When
        `answer_question` is set, also answer the user's question in context."""
        idx = self._locate_concept(target)
        if idx is None:
            yield from self._ask_which_pillar()
            return
        self.walkthrough_index        = idx
        self.walkthrough_done         = False
        self.walkthrough_active       = True
        self.post_walkthrough_revisit = True
        # No add_pillar event: revisiting navigates to a pillar that is already
        # in the framework — it is not a new contribution.
        logging.info(f"[REVISIT] post-walkthrough go-back to '{target}' "
                     f"(idx={idx}, answer_question={answer_question})")
        yield f"Sure — let's revisit **{target}**.\n\n"
        yield from self._stream_concept(is_first=False)
        if answer_question:
            ev.question(self._evctx(modality="text"), _sink)
            if (self.swap_presented and not self.concept_swap.is_detected
                    and self._is_wrong_concept(target)):
                ev.swap_questioned(self._evctx(modality="text"), _sink)
            yield "\n\n"
            yield from self._stream_concept_qa()

    def _finish_revisit(self):
        """Finish a post-walkthrough revisit: jump the index past the end so
        _current_concept() is None again, then re-render the summary."""
        self.post_walkthrough_revisit = False
        self.walkthrough_index = len(self.walkthrough_concepts)
        logging.info("[REVISIT] finished — returning to summary/freeform")
        yield from self._walkthrough_complete_message(reopened=True)

    def _continue_or_finish(self):
        """Shared 'advance to next concept / complete' tail. During a
        post-walkthrough revisit, return to the summary instead of advancing."""
        if self.post_walkthrough_revisit:
            yield from self._finish_revisit()
            return
        nxt = self._current_concept()
        if nxt is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_proactive_prompt()

    # System prompts
    def _build_system_prompt(self) -> str:
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"

        kg_block = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────\n"
            f"Case Type : {self.kg_context['case_type']}\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"(These are the correct ordered concepts. Ground your answer here.)\n"
            f"──────────────────────────────────────────────────────────────\n\n"
        )

        swap_block = self.concept_swap.get_system_prompt_block()
        return kg_block + swap_block + HITL_MAIN_SYSTEM_PROMPT

    def _build_clarification_system_prompt(self) -> str:
        if self.clarification_facts:
            facts_lines = "\n".join(
                f"- {topic.upper()}: {answer}"
                for topic, answer in self.clarification_facts.items()
            )
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────\n"
                f"{facts_lines}\n"
                f"──────────────────────────────────────────────────────────────\n\n"
            )
        else:
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────\n"
                f"No additional facts are available for this case.\n"
                f"Deflect all clarification questions with: "
                f"\"I'm afraid I don't have that information for this case.\"\n"
                f"──────────────────────────────────────────────────────────────\n\n"
            )
        return facts_block + HITL_CLARIFICATION_SYSTEM_PROMPT

    # Button handlers
    def on_approve_concept(self):
        concept = self._current_concept()
        if concept is None:
            return

        if concept not in self.approved_concepts:
            self.approved_concepts.append(concept)

        logging.info(f"[APPROVE] concept='{concept}', index={self.walkthrough_index}")

        if not self.before_advance(self):
            yield from self._stream_justification_prompt("accept", concept=concept)
            return

        self.walkthrough_index += 1
        yield from self._continue_or_finish()

    def on_reject_concept(self):
        concept = self._current_concept()
        if concept is None:
            return
        is_swap = self._is_wrong_concept(concept)
        req = (concept in self.justification_pillars) or is_swap
        self.pending = h.PendingAction(
            type="remove_pillar", target=concept, level="pillar",
            is_swap=is_swap, requires_justification=req)
        logging.info(f"[REJECT] parked pending concept='{concept}' (swap={is_swap}, req_just={req})")
        if req:
            yield (
                f"Before we exclude **{concept}** — what's your reasoning for leaving it "
                f"out? A sentence is plenty."
            )
        else:
            yield (
                f"Are you sure you want to exclude **{concept}**?\n\n"
                f"*Use the buttons below to confirm.*"
            )

    def on_confirm_reject(self):
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="confirm")
        yield from self.render_removal(outcome, pa=pa)

    def on_cancel_reject(self):
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="decline")
        yield from self.render_removal(outcome, pa=pa)

    def render_removal(self, o, user_input=None, *, was_pending=False, pa=None):
        ev.record(o, self._evctx(modality="button"), _sink)
        stage = o.stage

        if stage == "needs_justification":
            yield "Could you say a bit more about your reasoning? A sentence is plenty.\n\n"
            return

        if stage == "confirmed":
            if pa.is_swap:
                logging.info(f"[SWAP] detected via Reject button — concept='{pa.target}'")
                self.walkthrough_index += 1
                yield f"Got it! Removing **{pa.target}** from the framework.\n\n"
                yield from self._after_removal_continue(pa.target, was_reject=True)
                return
            if pa.type == "remove_sub_bullet":
                self._apply_sub_bullet_removal(pa.pillar, pa.target)
                logging.info(f"[REMOVE POINT CONFIRMED] '{pa.target}' from '{pa.pillar}'")
                yield f"Done! Removed that bullet from **{pa.pillar}**.\n\n"
                return
            logging.info(f"[REJECT CONFIRMED] concept='{pa.target}'")
            self.walkthrough_index += 1
            yield f"Got it! Removing **{pa.target}** from the framework.\n\n"
            yield from self._after_removal_continue(pa.target, was_reject=True)
            return

        if stage == "abandoned":
            if pa.type == "remove_sub_bullet":
                logging.info(f"[REMOVE POINT CANCELLED] keeping point under '{pa.pillar}'")
                yield f"No problem — keeping that bullet in **{pa.pillar}**.\n\n"
                return
            if pa.target and pa.target not in self.approved_concepts:
                self.approved_concepts.append(pa.target)
            logging.info(f"[REJECT CANCELLED] concept='{pa.target}' kept in framework")
            if not self.before_advance(self):
                yield from self._stream_justification_prompt("accept", concept=pa.target)
                return
            self.walkthrough_index += 1
            yield f"Keeping **{pa.target}** — let's continue.\n\n"
            yield from self._after_removal_continue(pa.target, was_reject=False)
            return

        yield "Reply **yes** to remove, or **no** to keep it.\n\n"

    def _after_removal_continue(self, concept, *, was_reject):
        yield from self._continue_or_finish()

    def _apply_sub_bullet_removal(self, concept: str, bullet: str):
        target = bullet.strip().lstrip("-• ").strip().lower()
        block = self.concept_blocks.get(concept, "")
        if block:
            self.concept_blocks[concept] = "\n".join(
                l for l in block.splitlines()
                if l.strip().lstrip("-• ").strip().lower() != target
            )
        pts = self.user_sub_points.get(concept, [])
        self.user_sub_points[concept] = [p for p in pts if p.strip().lower() != target]

    def render_summary(self):
        yield from self._stream_summary()

    def render_framework(self, preamble=""):
        if preamble:
            yield preamble
        if self._current_concept() is None:
            yield from self._stream_summary()
        else:
            yield from self._stream_concept(is_first=False)

    def render_add(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders adds via its button/stream flow (➕ Add → _store_sub_point / "
            "re-render in _stream_main), not the outcome router (6h-3, decision (i)).")

    def render_question(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders questions via _stream_concept_qa in its button/stream flow, "
            "not the outcome router (6h-3, decision (i)).")

    def render_next_steps(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders suggestions via _handle_suggest in its button/stream flow, "
            "not the outcome router (6h-3, decision (i)).")

    def render_fallback(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders advance/affordance via _stream_proactive_prompt / "
            "_nudge_to_button in its button/stream flow, not the outcome router "
            "(6h-3, decision (i)).")

    def on_add_to_concept(self):
        concept = self._current_concept()
        if concept is None:
            return
        self.awaiting_sub_point = True
        logging.info(f"[ADD] add-mode opened for concept='{concept}'")
        yield (
            f"What bullet would you like to add under **{concept}**? "
            f"You can add as many as you like, just type each one, and "
            f"click **✅ Done adding** when you're finished."
        )

    def on_done_adding(self):
        self.awaiting_sub_point   = False
        self.awaiting_revisit_add = False
        self.revisit_target       = None
        concept = self._current_concept() or "this concept"
        logging.info(f"[ADD] add-mode closed for concept='{concept}'")
        yield f"Got it. Back to **{concept}**, do you want to include it or exclude it from current plan?\n\n Feel free to discuss your concerns regarding the pillar or bullet .\n\n"

    def on_remove_point(self, bullet: str):
        concept = self._current_concept()
        if concept is None or not bullet:
            return
        self.pending = h.PendingAction(
            type="remove_sub_bullet", target=bullet, level="concept",
            pillar=concept, requires_justification=False)
        logging.info(f"[REMOVE POINT] parked: '{bullet}' under '{concept}'")
        yield (
            f"Remove this bullet from **{concept}**?\n\n- {bullet}\n\n"
            f"*Use the buttons below to confirm.*"
        )

    def on_confirm_remove_point(self):
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="confirm")
        yield from self.render_removal(outcome, pa=pa)

    def on_cancel_remove_point(self):
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="decline")
        yield from self.render_removal(outcome, pa=pa)

    def on_revisit_pillar(self, pillar: str):
        if not pillar:
            return
        self.revisit_target       = self._normalize_pillar(pillar)
        self.awaiting_revisit_add = True
        logging.info(f"[REVISIT] add-mode opened for past pillar='{self.revisit_target}'")
        yield (
            f"Sure! What bullet would you like to add to **{self.revisit_target}**? "
            f"Type it, and click **✅ Done adding** when you're finished."
        )

    def _extra_swap_signal(self, km, user_text: str) -> bool:
        cur = self.current_pillar()
        return bool(cur and self._is_wrong_concept(cur))

    def before_advance(self, session) -> bool:
        cur = self._current_concept()
        if cur is None:
            return True
        if self._is_wrong_concept(cur):
            return cur in self._justified_concepts
        if cur not in self.justification_pillars:
            return True
        return cur in self._justified_concepts

    def requires_justification(self, km) -> bool:
        name = getattr(km, "pillar", None)
        if not name:
            return False
        if self._is_wrong_concept(name):
            return True
        return name in self.justification_pillars

    def current_pillar(self):
        excluded = [e.lower() for e in self.excluded_concepts]
        idx = self.walkthrough_index
        while idx < len(self.walkthrough_concepts):
            c = self.walkthrough_concepts[idx]
            if c.lower() not in excluded:
                return c
            idx += 1
        return None

    def presented_pillars(self) -> list:
        excluded = [e.lower() for e in self.excluded_concepts]
        return [c for c in self.walkthrough_concepts[:self.walkthrough_index + 1]
                if c.lower() not in excluded]

    def _summary_pillars(self) -> list:
        # presented_pillars() includes the current concept (the +1) so in-flight
        # listings show the pillar the user is looking at. The final summary must
        # not: walkthrough_index points at the *undecided* current pillar (a
        # decision is what advances the index), so on an early End Session that
        # pillar hasn't been walked through. Drop it; after completion (or a
        # finished revisit) _current_concept() is None and the full set stands.
        current = self._current_concept()
        pillars = self.presented_pillars()
        if current is None:
            return pillars
        return [c for c in pillars if c.lower() != current.lower()]

    def presented_sub_bullets(self) -> dict:
        out = {}
        for name in self.presented_pillars():
            bl = []
            for line in self.concept_blocks.get(name, "").splitlines():
                t = grounding._strip_source_refs(line.strip().lstrip("-• ").strip())
                if t and not matching.is_excluded_bullet(self.excluded_sub_bullets, name, t):
                    bl.append(t)
            for sp in self.user_sub_points.get(name, []):
                t = grounding._strip_source_refs(sp).strip()
                if t and not matching.is_excluded_bullet(self.excluded_sub_bullets, name, t):
                    bl.append(t)
            out[name] = bl
        return out

    def surfaced_pillar_names(self) -> set:
        names = {p["name"].lower() for p in kb.get_shown_pillars()}
        names |= {c.lower() for c in self.walkthrough_concepts}
        names |= {c.lower() for c in self.user_contributed_concepts}
        names |= {e.lower() for e in self.excluded_concepts}
        return names

    def surface_pillar(self, name: str) -> None:
        if name.lower() in [c.lower() for c in self.walkthrough_concepts]:
            self._last_surface = {"name": name, "is_new": False}
            return
        self.walkthrough_concepts.insert(self.walkthrough_index, name)
        self.user_contributed_concepts.add(name)
        self._last_surface = {"name": name, "is_new": True}

    def add_sub_point(self, pillar: str, text: str) -> None:
        stored, is_new = self._store_sub_point(pillar, text)
        self._last_sub_add = {"pillar": self._normalize_pillar(pillar),
                              "stored": stored, "raw": text, "is_new": is_new}

    # UI state queries
    def should_show_buttons(self) -> bool:
        return (
            self.phase == "main"
            and self.walkthrough_active
            and not self.walkthrough_done
            and self._current_concept() is not None
            and self.pending is None
            and not self.awaiting_user_suggestion
            and not self.awaiting_justification
            and not self.awaiting_sub_point
            and not self.awaiting_revisit_add
            and not self.awaiting_pillar_name
            and (self._current_concept() not in self.user_contributed_concepts)
        )

    def should_show_confirmation_buttons(self) -> bool:
        p = self.pending
        return (p is not None and p.type == "remove_pillar"
                and not (p.requires_justification and p.justification is None))

    def should_show_remove_point_confirmation(self) -> bool:
        p = self.pending
        return p is not None and p.type == "remove_sub_bullet"

    def removable_bullets(self) -> list[str]:
        concept = self._current_concept()
        if concept is None:
            return []
        out, seen = [], set()
        for src in (self.concept_blocks.get(concept, "").splitlines(),
                    [f"- {p}" for p in self.user_sub_points.get(concept, [])]):
            for l in src:
                t = l.strip().lstrip("-• ").strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    out.append(t)
        return out

    def past_pillars(self) -> list[str]:
        wrong = self.concept_swap.config["wrong_concept"].lower()
        excluded = [e.lower() for e in self.excluded_concepts]
        current = (self._current_concept() or "").lower()
        out, seen = [], set()
        for c in self.walkthrough_concepts[:self.walkthrough_index]:
            cl = c.lower()
            if cl in excluded or cl == wrong or cl == current or cl in seen:
                continue
            seen.add(cl)
            out.append(c)
        return out

    # Summary + session
    def end_session(self) -> None:
        from backend.logger import end_session as _end_session

        if self.approved_concepts:
            final_framework = f"Approved concepts: {', '.join(self.approved_concepts)}"
        else:
            final_framework = "No concepts explicitly approved."

        logging.info(
            f"[END SESSION] approved={self.approved_concepts}, "
            f"rejected={self.excluded_concepts}, "
            f"swap_detected={self.concept_swap.is_detected}"
        )

        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection("sessions").document(self.session_id).update({
                "final_framework":       final_framework,
                "concept_swap_detected": self.concept_swap.is_detected,
                "swap_detected_at_end":  self.concept_swap.is_detected,
                "concepts_approved":     self.approved_concepts,
                "concepts_rejected":     self.excluded_concepts,
            })
            logging.info(f"[END SESSION] Firestore stamped for session={self.session_id}")
        except Exception as e:
            logging.warning(f"[END SESSION] Firestore stamp failed: {e}")

        _end_session(self.session_id)
