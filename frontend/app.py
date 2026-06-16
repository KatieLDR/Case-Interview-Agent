import uuid
import chainlit as cl
import asyncio

from backend.agents.base import MAX_TURNS_PER_SESSION
from backend.agents.black_box import BlackBoxAgent
from backend.agents.explainable import ExplainableAgent
from backend.agents.hitl import HITLAgent

# Security caps
MAX_INPUT_CHARS = 6000


# Session startup
@cl.on_chat_start
async def on_chat_start():
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("agent", None)
    cl.user_session.set("agent_type", None)
    cl.user_session.set("ended", False)

    await cl.Message(
        content=(
            "Welcome to the **Problem-Solving Assistant**!\n\n"
            "Before we begin, please read the guide by clicking the **Readme** button "
            "in the top-right corner.\n\n"
            "When you're ready, click the button below."
        ),
        actions=[
            cl.Action(
                name="readme_confirmed",
                label="I've read it ✅",
                description="Confirm you have read the guide",
                payload={}
            ),
        ]
    ).send()


@cl.action_callback("readme_confirmed")
async def on_readme_confirmed(action: cl.Action):
    await cl.Message(
        content="Great! Please select an agent to get started:",
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


# Agent selection callbacks
@cl.action_callback("select_agent_1")
async def on_select_agent_1(action: cl.Action):
    await _init_agent("black_box")


@cl.action_callback("select_agent_2")
async def on_select_agent_2(action: cl.Action):
    await _init_agent("explainable")


@cl.action_callback("select_agent_3")
async def on_select_agent_3(action: cl.Action):
    await _init_agent("hitl")


# Shared agent initialisation
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
    await cl.Message(content=agent.get_warmup_message()).send()


@cl.action_callback("lets_go")
async def on_lets_go(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(content="⚠️ No agent selected yet.").send()
        return

    if ended:
        await cl.Message(content="⚠️ This session has already ended.").send()
        return

    await cl.Message(
    content=(
        "You are about to read a short business case. "
        "Take a moment to read it carefully. "
        "We will build a structured plan together afterwards."
        )
    ).send()

    await asyncio.sleep(3)
    await cl.Message(content=agent.get_opening_message()).send()

    await asyncio.sleep(10)
    await cl.Message(
        content=(
            "💬 **Before we begin:** Feel free to ask any questions about the case, "
            "I'm here to help clarify. When you're ready to build your answer, "
            "click the I'm Ready ✅ button below to start."
        ),
        actions=[
            cl.Action(
                name="start_main_phase",
                label="I'm Ready ✅",
                description="End clarification and begin your structured analysis",
                payload={}
            ),
        ]
    ).send()


@cl.action_callback("done_warmup")
async def on_done_warmup(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(content="⚠️ No agent selected yet.").send()
        return

    if ended:
        await cl.Message(content="⚠️ This session has already ended.").send()
        return

    if agent.phase != "warmup":
        return

    warmup_messages = cl.user_session.get("warmup_messages", [])
    if not warmup_messages:
        await cl.Message(
            content=(
                "It looks like you haven't typed anything yet! "
                "Give it a try — there are no right or wrong answers. 😊"
            ),
            actions=[
                cl.Action(
                    name="done_warmup",
                    label="✅ Done",
                    description="Finish the warm-up exercise",
                    payload={}
                ),
            ]
        ).send()
        return

    # Retrieve last merged plan from session
    merged_plan = cl.user_session.get("warmup_merged_plan", "")

    from backend.logger import log_warmup_response
    log_warmup_response(agent.session_id, merged_plan)

    agent.phase = "clarification"
    print(f"[WARMUP] final plan logged, phase → clarification "
          f"for session={agent.session_id}")

    await cl.Message(
        content=(
            f"✅ **Great work!**\n\n"
            f"Here's your final plan:\n\n"
            f"{merged_plan}\n\n"
            f"---\n\n"
            f"You've just practiced breaking down a problem into structured areas, "
            f"that's exactly what you'll do next.\n\n"
            f"Click **Let's go! 🚀** below to start the real case."
        ),
        actions=[
            cl.Action(
                name="lets_go",
                label="Let's go! 🚀",
                description="Move on to the main task",
                payload={}
            ),
        ]
    ).send()


# "I'm Ready" button — shows instruction + Got it button
@cl.action_callback("start_main_phase")
async def on_start_main_phase(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(content="⚠️ No agent selected yet.").send()
        return

    if ended:
        await cl.Message(content="⚠️ This session has already ended.").send()
        return

    # Show per-agent instruction
    await cl.Message(content=agent.get_pre_analysis_instruction()).send()

    # Show "Got it" button
    await cl.Message(
        content="",
        actions=[
            cl.Action(
                name="begin_analysis",
                label="Got it, show me the full analysis ✅",
                description="Begin the structured analysis",
                payload={}
            ),
        ]
    ).send()


# "Got it" button — triggers begin_analysis(), stamps started_at
@cl.action_callback("begin_analysis")
async def on_begin_analysis(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)

    if agent is None:
        await cl.Message(content="⚠️ No agent selected yet.").send()
        return

    if ended:
        await cl.Message(content="⚠️ This session has already ended.").send()
        return

    stream = agent.begin_analysis()

    # First yield — goal instruction
    goal_msg = next(stream)
    await cl.Message(content=goal_msg).send()

    await asyncio.sleep(8)

    # Second yield is always the timer message — send as separate bubble
    timer_msg = next(stream)
    await cl.Message(content=timer_msg).send()

    await asyncio.sleep(2)


    # Rest streams normally
    msg = cl.Message(content="")
    await msg.send()
    for token in stream:
        await msg.stream_token(token)
    await msg.update()
    await _attach_buttons(agent)


# Stop button handler
@cl.on_stop
async def on_stop():
    agent = cl.user_session.get("agent")
    if agent is not None:
        from backend.logger import log_interruption
        log_interruption(agent.session_id, context="user_clicked_stop")


# Incoming messages
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

    if len(message.content) > MAX_INPUT_CHARS:
        await cl.Message(
            content=f"⚠️ Your message is too long ({len(message.content)} characters). "
                    f"Please keep messages under {MAX_INPUT_CHARS} characters."
        ).send()
        return

    if message.content.strip().lower() in ("/end", "/summary"):
        await _send_summary()
        return

    # Warmup phase
    if hasattr(agent, "phase") and agent.phase == "warmup":
        warmup_messages = cl.user_session.get("warmup_messages", [])
        warmup_messages.append(message.content)
        cl.user_session.set("warmup_messages", warmup_messages)

        merged = agent.merge_warmup_additions(warmup_messages)
        cl.user_session.set("warmup_merged_plan", merged)

        await cl.Message(
            content=(
                f"Got it!\n\n"
                f"{merged}\n\n"
                f"---\n\n"
                f"Anything else to add, or is there anything you'd remove or change? "
                f"When you want to finish the practice, click the **✅ Done** button below."
            ),
            actions=[
                cl.Action(
                    name="done_warmup",
                    label="✅ Done",
                    description="Finish the warm-up exercise",
                    payload={}
                ),
            ]
        ).send()
        return

    msg = cl.Message(content="")
    await msg.send()
    for token in agent.stream_message(message.content):
        await msg.stream_token(token)
    await msg.update()

    if agent.turn_count >= MAX_TURNS_PER_SESSION:
        await cl.Message(
            content="⏱️ **You've reached the session limit.**\n\nThank you for your time! "
                    "I'll now generate your session summary..."
        ).send()
        await _send_summary()
        return

    if not cl.user_session.get("ended", False):
        await _attach_buttons(agent)


# HITL — Approve concept
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


# HITL — Reject concept (triggers pushback)
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


@cl.action_callback("add_to_concept")
async def on_add_to_concept(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_add_to_concept():
        await msg.stream_token(token)
    await msg.update()
    await _attach_buttons(agent)


@cl.action_callback("done_adding")
async def on_done_adding(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_done_adding():
        await msg.stream_token(token)
    await msg.update()
    await _attach_buttons(agent)


# HITL — ➖ Remove a point: open picker
@cl.action_callback("remove_point_open")
async def on_remove_point_open(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    bullets = agent.removable_bullets()
    if not bullets:
        await cl.Message(content="There are no points to remove right now.").send()
        await _attach_buttons(agent)
        return
    bullet_actions = [
        cl.Action(
            name="remove_point_item",
            label=b[:80],
            description=b,
            payload={"bullet": b},
        )
        for b in bullets
    ]
    await cl.Message(
        content="Which point would you like to remove?",
        actions=bullet_actions,
    ).send()


# HITL — ➖ Remove a point: item selected
@cl.action_callback("remove_point_item")
async def on_remove_point_item(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    bullet = action.payload.get("bullet", "")
    if not bullet:
        return
    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_remove_point(bullet):
        await msg.stream_token(token)
    await msg.update()
    await _attach_buttons(agent)


# HITL — ➖ Remove a point: confirm
@cl.action_callback("confirm_remove_point")
async def on_confirm_remove_point(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_confirm_remove_point():
        await msg.stream_token(token)
    await msg.update()
    await _attach_buttons(agent)


# HITL — ➖ Remove a point: cancel
@cl.action_callback("cancel_remove_point")
async def on_cancel_remove_point(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_cancel_remove_point():
        await msg.stream_token(token)
    await msg.update()
    await _attach_buttons(agent)


# HITL — ↩️ Revisit past pillar: open picker
@cl.action_callback("revisit_pillar_open")
async def on_revisit_pillar_open(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    past = agent.past_pillars()
    if not past:
        await cl.Message(content="No past pillars to revisit yet.").send()
        await _attach_buttons(agent)
        return
    pillar_actions = [
        cl.Action(
            name="revisit_pillar_item",
            label=p,
            description=p,
            payload={"pillar": p},
        )
        for p in past
    ]
    await cl.Message(
        content="Which past pillar would you like to add a point to?",
        actions=pillar_actions,
    ).send()


# HITL — ↩️ Revisit past pillar: pillar selected
@cl.action_callback("revisit_pillar_item")
async def on_revisit_pillar_item(action: cl.Action):
    agent = cl.user_session.get("agent")
    ended = cl.user_session.get("ended", False)
    if agent is None or ended:
        return
    pillar = action.payload.get("pillar", "")
    if not pillar:
        return
    msg = cl.Message(content="")
    await msg.send()
    for token in agent.on_revisit_pillar(pillar):
        await msg.stream_token(token)
    await msg.update()
    await _attach_buttons(agent)


# HITL — Confirm reject (commit exclusion)
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


# HITL — Cancel reject (keep concept)
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


# Shared button attachment logic
async def _attach_buttons(agent):
    if hasattr(agent, "phase") and agent.phase == "warmup":
        return

    if cl.user_session.get("ended", False):
        return

    # HITL-specific button logic
    if isinstance(agent, HITLAgent):
        if agent.phase == "clarification":
            await cl.Message(
                content="",
                actions=[
                    cl.Action(
                        name="start_main_phase",
                        label="I'm Ready ✅",
                        description="End clarification and begin your structured analysis",
                        payload={}
                    ),
                ]
            ).send()

        elif agent.should_show_remove_point_confirmation():
            await cl.Message(
                content="",
                actions=[
                    cl.Action(
                        name="cancel_remove_point",
                        label="↩️ Keep it",
                        description="Keep this point",
                        payload={}
                    ),
                    cl.Action(
                        name="confirm_remove_point",
                        label="✅ Yes, remove it",
                        description="Confirm removal of this point",
                        payload={}
                    ),
                ]
            ).send()

        elif agent.should_show_confirmation_buttons():
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

        elif agent.awaiting_sub_point or agent.awaiting_revisit_add:
            await cl.Message(
                content="",
                actions=[
                    cl.Action(name="done_adding", label="✅ Done adding",
                              description="Finish adding points to this concept", payload={}),
                ]
            ).send()

        elif agent.should_show_buttons():
            actions = [
                cl.Action(name="approve_concept", label="✅ Include",
                          description="Include this concept in the framework", payload={}),
                cl.Action(name="reject_concept", label="❌ Skip",
                          description="Skip this concept", payload={}),
                cl.Action(name="add_to_concept", label="➕ Add point to consider in this pillar",
                          description="Add your own point under this concept", payload={}),
                cl.Action(name="remove_point_open", label="➖ Remove a point in this pillar",
                          description="Remove one of the existing points", payload={}),
            ]
            if agent.past_pillars():
                actions.append(
                    cl.Action(name="revisit_pillar_open", label="↩️ Add point to a past pillar",
                              description="Go back and add a point to a concept you already decided on",
                              payload={})
                )
            await cl.Message(content="", actions=actions).send()

        else:
            if agent.walkthrough_active or agent.walkthrough_done:
                await cl.Message(
                    content="",
                    actions=[
                        cl.Action(
                            name="get_summary",
                            label="‼️End Session",
                            description="End your session",
                            payload={}
                        ),
                    ]
                ).send()

        return

    # BlackBox / Explainable button logic
    if hasattr(agent, "phase") and agent.phase == "clarification":
        await cl.Message(
            content="",
            actions=[
                cl.Action(
                    name="start_main_phase",
                    label="I'm Ready ✅",
                    description="End clarification and begin your structured analysis",
                    payload={}
                ),
            ]
        ).send()

    else:
            # Gate End Session until the user has made a main-phase contribution
            if not getattr(agent, "has_main_contribution", True):
                return

            end_desc = "End your session"
            if type(agent).__name__ == "BlackBoxAgent":
                end_desc = "End your session — this cannot be undone"

            await cl.Message(
                content="",
                actions=[
                    cl.Action(
                        name="get_summary",
                        label="‼️End Session",
                        description=end_desc,
                        payload={}
                    ),
                ]
            ).send()


# Get summary button
@cl.action_callback("get_summary")
async def on_get_summary(action: cl.Action):
    await _send_summary()


# End session button
@cl.action_callback("end_session")
async def on_end_session(action: cl.Action):
    await _close_session()


# Shared summary logic
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

    # HITL — stream summary directly from walkthrough state
    if agent_type in ("hitl", "explainable", "black_box"):
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


# Shared session closing logic
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
            f"✅ **Session Ended**\n"
            f"🪪 **Session ID:** `{agent.session_id}`\n"
            f"*Keep this ID for your records. Refresh the page to start a new session.*"
        )
    ).send()

    agent.end_session()