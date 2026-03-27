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

        "clarification_facts": {
            "objective": (
                "The client's primary objective is to increase profits while diversifying "
                "their portfolio. They are also interested in sustainable mining practices "
                "to minimize environmental impact and maintain their social license to operate."
            ),
            "market_demand": (
                "The global market for Silica Sand is approximately 350 million metric tons "
                "per year, growing at 5.5% annually. Bentonite demand is smaller at around "
                "20 million metric tons per year, growing at a similar rate. Key end uses "
                "include construction, glass manufacturing, and drilling fluids."
            ),
            "pricing": (
                "The current selling price for Silica Sand is around $70 per ton and $90 per "
                "ton for Bentonite. Prices are expected to increase at 3% annually due to "
                "growing demand and limited supply. Historically, prices have been relatively "
                "stable with minor fluctuations."
            ),
            "costs": (
                "The cost per ton for Silica Sand is approximately $50, including equipment, "
                "labor, and transportation. Bentonite is slightly higher at $60 per ton due "
                "to its more complex extraction process. Transportation to end markets adds "
                "approximately $20 per ton for Silica Sand and $25 per ton for Bentonite."
            ),
            "deposits": (
                "The Silica Sand deposit is estimated at 50 million metric tons with a lifespan "
                "of approximately 10 years at current demand. The Bentonite deposit is around "
                "10 million metric tons with a lifespan of approximately 30 years. Both deposits "
                "are relatively accessible — close to the surface with moderate terrain complexity. "
                "Quality and purity are comparable to market standards."
            ),
            "competition": (
                "The Silica Sand market is fragmented, with multiple suppliers worldwide. "
                "The Bentonite market is more concentrated, with a few key players dominating. "
                "Specific competitor names are not information I have available for this case."
            ),
            "infrastructure": (
                "The company already has existing transport routes that can be utilized. "
                "Transportation costs to end markets are approximately $20-25 per ton. "
                "Distance to key customers or ports is not information I have available."
            ),
            "operations": (
                "The company has some capacity to mine these resources with current equipment "
                "and labor, but would need to invest in additional equipment and training to "
                "fully exploit the deposits. The estimated investment required is around "
                "$10 million. The fly-in remote setup increases labor costs due to transportation "
                "and accommodation needs, and may create challenges in recruiting and retaining "
                "skilled workers."
            ),
            "regulatory": (
                "Regulatory requirements are stringent, with strict environmental and safety "
                "standards. Potential environmental impacts include water pollution and habitat "
                "destruction, which could lead to fines and sanctions if not properly managed. "
                "Specific permits, approvals, and reclamation obligations apply but detailed "
                "specifics are not available for this case."
            ),
            "risks": (
                "Key risks include commodity price swings — if demand weakens, oversupply could "
                "reduce selling prices and compress margins. Execution risks include the $10 million "
                "capex requirement, remote location labor challenges, and environmental compliance. "
                "The client is also exposed to cost overruns given the complex extraction process "
                "for Bentonite."
            ),
            "timeline": (
                "The client is looking to make a decision within the next 3 months and aims to "
                "start production within 6 months if the project is deemed feasible."
            ),
            "sustainability": (
                "The client is interested in sustainable mining practices to minimize environmental "
                "impact and maintain their social license to operate. Community and indigenous "
                "stakeholder considerations are part of their overall objective."
            ),
            "scope": (
                "The client is interested in knowing whether it would be profitable to mine both "
                "Silica Sand and Bentonite. They are open to mining both minerals but also want "
                "to understand the potential risks of moving forward."
            ),
        },
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