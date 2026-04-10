import uuid
import inspect
import chainlit as cl
from backend.black_box_agent import BlackBoxAgent
from backend.coach_agent import CoachAgent
from backend.explainable_agent import ExplainableAgent
from backend.hitl_agent import HITLAgent


# ── Session startup ────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("agent", None)
    cl.user_session.set("agent_type", None)
    cl.user_session.set("ended", False)

    await cl.Message(
        content=(
            f"👋 Welcome to the **Case Interview Assistant**!\n\n"
            f"Please select an agent to get started:"
        ),
        actions=[
            cl.Action(
                name="select_agent_1",
                label="🤖 Agent 1",
                description="Select Agent 1",
                payload={}
            ),
            cl.Action(
                name="select_agent_2",
                label="🤖 Agent 2",
                description="Select Agent 2",
                payload={}
            ),
            cl.Action(
                name="select_agent_3",
                label="🤖 Agent 3",
                description="Select Agent 3",
                payload={}
            ),
        ]
    ).send()


# ── Agent selection callbacks ──────────────────────────────────────────────
@cl.action_callback("select_agent_1")
async def on_select_agent_1(action: cl.Action):
    await _init_agent("black_box")

@cl.action_callback("select_agent_2")
async def on_select_agent_2(action: cl.Action):
    await _init_agent("explainable")

@cl.action_callback("select_agent_3")
async def on_select_agent_3(action: cl.Action):
    await _init_agent("hitl")


# ── Shared agent initialisation ────────────────────────────────────────────
async def _init_agent(agent_type: str):
    if cl.user_session.get("agent") is not None:
        await cl.Message(
            content="⚠️ You have already selected an agent for this session."
        ).send()
        return

    user_id = cl.user_session.get("user_id")

    if agent_type == "black_box":
        agent = BlackBoxAgent(user_id=user_id)
        intro = (
            f"✅ **Agent 1 selected!**\n\n"
            f"🪪 **Your Session ID:** `{agent.session_id}`\n"
            f"📝 Note this ID for future reference or support.\n\n"
            f"---\n"
        )
    elif agent_type == "explainable":
        agent = ExplainableAgent(user_id=user_id)
        intro = (
            f"✅ **Agent 2 selected!**\n\n"
            f"🪪 **Your Session ID:** `{agent.session_id}`\n"
            f"📝 Note this ID for future reference or support.\n\n"
            f"---\n"
        )
    elif agent_type == "hitl":
        agent = HITLAgent(user_id=user_id)
        intro = (
            f"✅ **Agent 3 selected!**\n\n"
            f"🪪 **Your Session ID:** `{agent.session_id}`\n"
            f"📝 Note this ID for future reference or support.\n\n"
            f"---\n"
        )

    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_type", agent_type)

    await cl.Message(content=intro).send()

    await cl.Message(
        content=agent.get_opening_message(),
        actions=[
            cl.Action(
                name="start_main_phase",
                label="✅ I'm Ready — Let's Start",
                description="End clarification and begin your structured analysis",
                payload={}
            ),
        ]
    ).send()


# ── "I'm Ready" button — transitions clarification → main ─────────────────
@cl.action_callback("start_main_phase")
async def on_start_main_phase(action: cl.Action):
    agent  = cl.user_session.get("agent")
    ended  = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(content="⚠️ No agent selected yet.").send()
        return

    if ended:
        await cl.Message(content="⚠️ This session has already ended.").send()
        return

    result = agent.start_main_phase()

    msg = cl.Message(content="")
    await msg.send()

    if inspect.isgenerator(result):
        for token in result:
            await msg.stream_token(token)
    else:
        await msg.stream_token(result)

    await msg.update()

    await _attach_buttons(agent)


# ── Stop button handler ────────────────────────────────────────────────────
@cl.on_stop
async def on_stop():
    agent = cl.user_session.get("agent")
    if agent is not None:
        from backend.logger import log_interruption
        log_interruption(agent.session_id, context="user_clicked_stop")


# ── Incoming messages ──────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    agent  = cl.user_session.get("agent")
    ended  = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(
            content="⚠️ Please select an agent first by clicking above."
        ).send()
        return

    if ended:
        await cl.Message(
            content="⚠️ This session has already ended. Please refresh to start a new one."
        ).send()
        return

    if message.content.strip().lower() in ("/end", "/summary"):
        await _send_summary()
        return

    # Stream response token by token
    msg = cl.Message(content="")
    await msg.send()
    for token in agent.stream_message(message.content):
        await msg.stream_token(token)
    await msg.update()

    if not cl.user_session.get("ended", False):
        await _attach_buttons(agent)


# ── HITL — Approve concept ─────────────────────────────────────────────────
@cl.action_callback("approve_concept")
async def on_approve_concept(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None or ended:
        return

    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_approve_concept():
        await msg.stream_token(token)
    await msg.update()

    await _attach_buttons(agent)


# ── HITL — Reject concept (triggers pushback) ──────────────────────────────
@cl.action_callback("reject_concept")
async def on_reject_concept(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None or ended:
        return

    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_reject_concept():
        await msg.stream_token(token)
    await msg.update()

    await _attach_buttons(agent)


# ── HITL — Confirm reject (commit exclusion) ───────────────────────────────
@cl.action_callback("confirm_reject")
async def on_confirm_reject(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None or ended:
        return

    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_confirm_reject():
        await msg.stream_token(token)
    await msg.update()

    await _attach_buttons(agent)


# ── HITL — Cancel reject (keep concept) ───────────────────────────────────
@cl.action_callback("cancel_reject")
async def on_cancel_reject(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None or ended:
        return

    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_cancel_reject():
        await msg.stream_token(token)
    await msg.update()

    await _attach_buttons(agent)


# ── Shared button attachment logic ─────────────────────────────────────────
async def _attach_buttons(agent):
    """
    Attach the correct buttons based on agent type and current state.
    Called after every streaming response completes.

    BlackBox/Explainable: I'm Ready (clarification) or Summary (main)
    HITL clarification:   I'm Ready
    HITL main — concept:  Include / Skip
    HITL main — pending:  Keep it / Yes skip it
    HITL main — done:     Summary
    All agents ended:     Nothing

    Change log: 2026-04-09 — extracted from on_message for reuse across
    all streaming callbacks.
    """
    if cl.user_session.get("ended", False):
        return

    # ── HITL-specific button logic ─────────────────────────────────────
    if isinstance(agent, HITLAgent):
        if agent.phase == "clarification":
            await cl.Message(
                content="",
                actions=[
                    cl.Action(
                        name="start_main_phase",
                        label="✅ I'm Ready — Let's Start",
                        description="End clarification and begin your structured analysis",
                        payload={}
                    ),
                ]
            ).send()

        elif agent.should_show_confirmation_buttons():
            # Pending reject — show confirm/cancel
            # "Keep it" first — default action after pushback is to reconsider
            await cl.Message(
                content="",
                actions=[
                    cl.Action(
                        name="cancel_reject",
                        label="↩️ Keep it",
                        description="Keep this concept in the framework",
                        payload={}
                    ),
                    cl.Action(
                        name="confirm_reject",
                        label="✅ Yes, skip it",
                        description="Confirm removal of this concept",
                        payload={}
                    ),
                ]
            ).send()

        elif agent.should_show_buttons():
            # Active concept awaiting decision
            await cl.Message(
                content="",
                actions=[
                    cl.Action(
                        name="approve_concept",
                        label="✅ Include",
                        description="Include this concept in the framework",
                        payload={}
                    ),
                    cl.Action(
                        name="reject_concept",
                        label="❌ Skip",
                        description="Skip this concept",
                        payload={}
                    ),
                ]
            ).send()

        else:
            # Walkthrough done — show summary
            # Only show if walkthrough has actually started
            # (not during Q1/Q2 clarification step)
            if agent.walkthrough_active or agent.walkthrough_done:
                await cl.Message(
                    content="",
                    actions=[
                        cl.Action(
                            name="get_summary",
                            label="📊 Get Summary & End Session",
                            description="Get your session summary and end the session",
                            payload={}
                        ),
                    ]
                ).send()
            # else: Q1/Q2 phase — no buttons shown
            
        return

    # ── BlackBox / Explainable button logic ───────────────────────────
    if hasattr(agent, "phase") and agent.phase == "clarification":
        await cl.Message(
            content="",
            actions=[
                cl.Action(
                    name="start_main_phase",
                    label="✅ I'm Ready — Let's Start",
                    description="End clarification and begin your structured analysis",
                    payload={}
                ),
            ]
        ).send()
    else:
        await cl.Message(
            content="",
            actions=[
                cl.Action(
                    name="get_summary",
                    label="📊 Get Summary & End Session",
                    description="Get your session summary and end the session",
                    payload={}
                ),
            ]
        ).send()


# ── Get summary button ─────────────────────────────────────────────────────
@cl.action_callback("get_summary")
async def on_get_summary(action: cl.Action):
    await _send_summary()


# ── End session button ─────────────────────────────────────────────────────
@cl.action_callback("end_session")
async def on_end_session(action: cl.Action):
    await _close_session()


# ── Shared summary logic ───────────────────────────────────────────────────
async def _send_summary():
    agent      = cl.user_session.get("agent")
    agent_type = cl.user_session.get("agent_type")
    ended      = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(content="⚠️ No agent selected yet.").send()
        return

    if ended:
        await cl.Message(content="⚠️ Session already ended.").send()
        return

    # ── HITL — stream summary directly from walkthrough state ─────────
    # Does not use send_message() — history scan unreliable for HITL.
    # Summary derived from approved_concepts list instead.
    # Change log: 2026-04-09
    if agent_type == "hitl":
        cl.user_session.set("ended", True)
        agent.end_session()

        msg = cl.Message(content="📊 **Your Session Summary:**\n\n")
        await msg.send()
        for token in agent.get_summary():
            await msg.stream_token(token)
        await msg.update()

        await cl.Message(
            content=(
                f"---\n"
                f"✅ **Session Ended**\n"
                f"🪪 **Session ID:** `{agent.session_id}`\n"
                f"*Keep this ID for your records. "
                f"Refresh the page to start a new session.*"
            )
        ).send()
        return

    # ── BlackBox / Explainable — use send_message() ───────────────────
    if agent_type == "black_box":
        prompt = (
            "Please summarise the cases we explored in this session, "
            "the key frameworks used, and any notable follow-up questions asked."
        )
    else:
        prompt = (
            "Please provide a comprehensive performance summary covering: "
            "overall structure, hypothesis-driven thinking, depth of analysis, "
            "communication clarity, and 2-3 specific areas for improvement."
        )

    async with cl.Step(name="Generating your summary..."):
        summary = agent.send_message(prompt)

    cl.user_session.set("ended", True)
    agent.end_session()

    await cl.Message(
        content=(
            f"📊 **Your Session Summary:**\n\n{summary}\n\n"
            f"---\n"
            f"✅ **Session Ended**\n"
            f"🪪 **Session ID:** `{agent.session_id}`\n"
            f"*Keep this ID for your records. Refresh the page to start a new session.*"
        )
    ).send()


# ── Shared session closing logic ───────────────────────────────────────────
async def _close_session():
    agent  = cl.user_session.get("agent")
    ended  = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(content="⚠️ No agent selected yet.").send()
        return

    if ended:
        await cl.Message(content="⚠️ Session already ended.").send()
        return

    cl.user_session.set("ended", True)

    await cl.Message(
        content=(
            f"---\n"
            f"✅ **Session Ended**\n\n"
            f"🪪 **Session ID:** `{agent.session_id}`\n"
            f"*Keep this ID for your records. Refresh the page to start a new session.*"
        )
    ).send()

    agent.end_session()