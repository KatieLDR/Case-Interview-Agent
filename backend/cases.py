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

    "backup_case_1":"""
A large, multinational CPG company is considering a large investment into packaging robots, which could be put to work in three key operational facilities in the United States. If they move forward, this would represent the largest investment in packaging automation by 2X and the senior staff is divided over the investment. A competitor recently invested in similar technology nine months ago and has suffered an embarrassing packaging related recall (e.g., incorrectly sealed packages led to items to spoiling on store shelves). The three facilities in question are located in small, mid-western cities and the client's facilities employ roughly 25% of the labor force in each respective city.

Your Partner has asked that you take the lead on drawing up a framework to reach a decision. Which key issues would you highlight?""".strip(),

    "backup_case_2": """
You and your team are advising the Strategy team for one of the largest soft-rock mining companies in the world (Mining Co.). The mine primarily extracts potash (a fertilizer) and some smaller quantities of various other ores. Located in a remote region that requires you to fly-in to a mining site, you're keen to quickly identify opportunities to help your client before your next flight home.

New deposits of the minerals Silica Sand and Bentonite have been discovered within the area the mining company owns the resources rights to. The CEO and Strategy team are interested in knowing whether it would be profitable to mine these new resources and some of the potential risks associated with moving forward.""".strip()
}


def get_case(agent_type: str) -> str:
    """Return the case assigned to the given agent type."""
    return CASES.get(agent_type, "No case available for this agent type.")