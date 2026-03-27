import uuid
import chainlit as cl
from backend.black_box_agent import BlackBoxAgent
from backend.coach_agent import CoachAgent

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
        ]
    ).send()


# ── Agent selection callbacks ──────────────────────────────────────────────
@cl.action_callback("select_agent_1")
async def on_select_agent_1(action: cl.Action):
    await _init_agent("black_box")


@cl.action_callback("select_agent_2")
async def on_select_agent_2(action: cl.Action):
    await _init_agent("coach")


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
    else:
        agent = CoachAgent(user_id=user_id)
        intro = (
            f"✅ **Agent 2 selected!**\n\n"
            f"🪪 **Your Session ID:** `{agent.session_id}`\n"
            f"📝 Note this ID for future reference or support.\n\n"
            f"---\n"
        )

    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_type", agent_type)

    await cl.Message(content=intro).send()

    # Present case + clarification prompt
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

    confirmation = agent.start_main_phase()

    await cl.Message(
        content=confirmation,
        actions=[
            cl.Action(
                name="get_summary",
                label="📊 Get Summary & End Session",
                description="Get your session summary and end the session",
                payload={}
            ),
        ]
    ).send()


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
            content="⚠️ Please select an agent first by clicking **Agent 1** or **Agent 2** above."
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

    # Reattach buttons based on current phase
    if not cl.user_session.get("ended", False):
        if hasattr(agent, "phase") and agent.phase == "clarification":
            # Still in clarification — keep "I'm Ready" button visible
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
            # Main phase — show summary button
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