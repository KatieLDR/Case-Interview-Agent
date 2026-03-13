## ── Case Bank ──────────────────────────────────────────────────────────────
## Each agent has a fixed case assigned to it.
## To change a case, update the text here — no other files need to change.

CASES = {
    "black_box": """
Leading food delivery companies like Uber Eats and DoorDash are amassing an 
unprecedented amount of data about consumer preferences for restaurant takeout 
meals. Among the many uses for this data, it's also powering an entirely new 
type of virtual restaurant: ghost restaurants.

Ghost restaurants have no storefront. While they operate a kitchen and cook 
dishes, their only "shingle," so to speak, is hung online, within an app like 
Uber Eats or DoorDash. They employ no front of the house — no waiters, hosts, 
or busboys — and sell their dishes online only.

One of the leading food delivery companies has engaged your team to help build 
an approach to identifying opportunities for potential virtual restaurants, 
which they can encourage restaurateurs to begin and operate.
""".strip(),

    "coach": """
Your client wants to open a burger store in Taipei City. The target is to 
expand exposure and earn revenue as quickly as possible. The client also 
expects to break even on the initial investment within the first year 
after opening.

What aspects will you analyze to make sure the client's burger store 
breaks even in the first year?
""".strip(),
}


def get_case(agent_type: str) -> str:
    """Return the case assigned to the given agent type."""
    return CASES.get(agent_type, "No case available for this agent type.")