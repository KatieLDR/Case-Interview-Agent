## ── Case Bank ──────────────────────────────────────────────────────────────
## Each agent has a fixed case assigned to it.
## To change a case, update the text here — no other files need to change.
##
## clarification_facts: dict of {question_pattern: answer}
##   - Agent answers ONLY from this sheet during the clarification phase
##   - Agent infers from the sheet when questions are related but not exact
##   - Agent deflects with "I don't have that information" when out of scope
##   - Leave empty dict {} as placeholder until facts are ready

CASES = {
    # ── Active experiment cases ────────────────────────────────────────────

    "black_box": {
        "prompt": """A San Francisco-based, medical devices firm has asked your team to help determine a strategy for developing and marketing products for the emerging Asian market, particularly China. Which key issues would you look into when helping set an entry strategy for this medical device firm?""",

        "clarification_facts": {
            "goal": (
                "The client's primary objective is to increase revenue by tapping into "
                "the growing demand for medical devices in China. They also see this as "
                "an opportunity to diversify their market base and reduce their dependence "
                "on the US market."
            ),
            "competition": (
                "The medical devices market in China is growing rapidly, with a market size "
                "of approximately $78 billion in 2020 and expected to grow at a CAGR of 15% "
                "over the next five years. Major players include both local companies like "
                "Mindray and international firms like Medtronic and Johnson & Johnson."
            ),
            "resources": (
                "The client has a distribution partnership with a Japanese firm, which could "
                "potentially be leveraged to facilitate entry into the Chinese market. However, "
                "this would require further negotiation and agreement."
            ),
            "regulatory": (
                "China has stringent regulatory requirements for medical devices. The China "
                "National Medical Products Administration (NMPA) requires foreign manufacturers "
                "to obtain a Medical Device Registration Certificate. This involves clinical "
                "trials and a comprehensive review of the product's safety and effectiveness."
            ),
            "budget": (
                "The client has set aside $20 million for this expansion. This budget includes "
                "costs for research and development, marketing, distribution, and regulatory "
                "compliance. However, they are open to revising this budget based on the "
                "potential return on investment."
            ),
            "success cases": (
                "The medical devices market in China has seen successful entries by foreign "
                "companies such as Medtronic and Johnson & Johnson."
            ),
            "unique value": (
                "The client specializes in orthopedic devices, particularly hip and knee "
                "replacements. These devices are in demand in the Asian market, with China's "
                "aging population driving a significant portion of this demand."
            ),
            "purchasing decisions": (
                "China has a rapidly growing middle class with increasing purchasing power. "
                "Purchasing decisions for medical devices in China are often influenced by "
                "price, quality, brand reputation, and recommendations from healthcare "
                "professionals. Additionally, relationships and trust play a significant "
                "role in business dealings in China."
            ),
            "localization": (
                "The client has not specified whether products would need localization "
                "for the Asian market."
            ),
            "market structure": (
                "The medical devices market in China is growing rapidly, with a market size "
                "of approximately $78 billion in 2020 and expected to grow at a CAGR of 15% "
                "over the next five years. Major players include both local companies like "
                "Mindray and international firms like Medtronic and Johnson & Johnson."
            ),
            "manufacturing": (
                "The client has a robust production capacity, with two manufacturing facilities "
                "in the US. They also have a strong distribution network in North America and "
                "Europe. However, they would need to establish a distribution network in Asia "
                "to meet potential demand."
            ),
            "geopolitical risks": (
                "The client is aware of the political and geopolitical risks associated with "
                "entering the Chinese market. They understand that factors such as trade "
                "tensions, regulatory changes, and government policies can impact their "
                "market entry strategy."
            ),
        },
    },

    "coach": {
        "prompt": """Your client wants to open a burger store in Taipei City. The target is to \
expand exposure and earn revenue as quickly as possible. The client also \
expects to break even on the initial investment within the first year \
after opening.

What aspects will you analyze to make sure the client's burger store \
breaks even in the first year?""",

        # TODO: add clarification facts for coach case
        "clarification_facts": {},
    },

    "explainable": {
        "prompt": """You and your team are advising the Strategy team for one of the largest \
soft-rock mining companies in the world (Mining Co.). The mine primarily \
extracts potash (a fertilizer) and some smaller quantities of various other \
ores. Located in a remote region that requires you to fly-in to a mining \
site, you're keen to quickly identify opportunities to help your client \
before your next flight home.

New deposits of the minerals Silica Sand and Bentonite have been discovered \
within the area the mining company owns the resources rights to. The CEO and \
Strategy team are interested in knowing whether it would be profitable to mine \
these new resources and some of the potential risks associated with moving \
forward.""",

        # TODO: add clarification facts for explainable case
        "clarification_facts": {},
    },

    "hitl": {
        "prompt": """A large, multinational CPG company is considering a large investment into \
packaging robots, which could be put to work in three key operational \
facilities in the United States. If they move forward, this would represent \
the largest investment in packaging automation by 2X and the senior staff is \
divided over the investment. A competitor recently invested in similar \
technology nine months ago and has suffered an embarrassing packaging related \
recall (e.g., incorrectly sealed packages led to items to spoiling on store \
shelves). The three facilities in question are located in small, mid-western \
cities and the client's facilities employ roughly 25% of the labor force in \
each respective city.

Your Partner has asked that you take the lead on drawing up a framework to \
reach a decision. Which key issues would you highlight?""",

        # TODO: add clarification facts for hitl case
        "clarification_facts": {},
    },

    # ── Backup / unused cases ──────────────────────────────────────────────

    "backup_ghost_restaurant": {
        "prompt": """Leading food delivery companies like Uber Eats and DoorDash are amassing an \
unprecedented amount of data about consumer preferences for restaurant takeout \
meals. Among the many uses for this data, it's also powering an entirely new \
type of virtual restaurant: ghost restaurants.

Ghost restaurants have no storefront. While they operate a kitchen and cook \
dishes, their only "shingle," so to speak, is hung online, within an app like \
Uber Eats or DoorDash. They employ no front of the house — no waiters, hosts, \
or busboys — and sell their dishes online only.

One of the leading food delivery companies has engaged your team to help build \
an approach to identifying opportunities for potential virtual restaurants, \
which they can encourage restaurateurs to begin and operate.""",

        "clarification_facts": {},
    },
}


# ── Public accessors ───────────────────────────────────────────────────────

def get_case(agent_type: str) -> str:
    """Return the case prompt string for the given agent type.
    Backward compatible — always returns a plain string."""
    entry = CASES.get(agent_type)
    if entry is None:
        return "No case available for this agent type."
    # Support both old plain-string format and new dict format
    if isinstance(entry, dict):
        return entry["prompt"]
    return entry


def get_clarification_facts(agent_type: str) -> dict:
    """Return the clarification facts dict for the given agent type.
    Returns empty dict if not yet defined."""
    entry = CASES.get(agent_type)
    if isinstance(entry, dict):
        return entry.get("clarification_facts", {})
    return {}