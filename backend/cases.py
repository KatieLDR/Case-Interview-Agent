# cases.py
# Last updated: 2026-05-08
# Change log:
#   2026-05-08 — All three agents migrated to coffee shop case (same case across conditions).
#                Previous cases moved to backup section below.
#                Coffee shop case adapted from Mining Co. profitability case.
#                Clarification facts researched and sourced (see coffee_shop_case_v1.md).

# ── Coffee Shop Case Prompt ────────────────────────────────────────────────────
# Shared across all three agents (BlackBox, Explainable, HITL).

_COFFEE_SHOP_PROMPT = """\
Your friend Anna owns a small coffee shop in Munich that has been profitable \
for the past 3 years. She roasts her own coffee beans and uses her own \
packaged beans in the shop. She is now considering launching these packaged \
beans to sell in supermarkets across Germany, but is unsure whether it's a \
good idea financially.

Anna has asked you to help her think it through. Build a structured approach \
(framework) that covers the key areas she should analyse before making a \
decision, focusing on whether this would be profitable and what risks exist. \
For each key area, identify the most important factors — questions you might \
want to deep dive into — just like you practised in the warm-up exercise.\
"""

# ── Coffee Shop Clarification Facts ───────────────────────────────────────────
# Shared across all three agents.
# Sources documented in coffee_shop_case_v1.md.

_COFFEE_SHOP_FACTS = {
    "objective": (
        "Anna's primary objective is to grow her revenue by diversifying into a new sales "
        "channel, while keeping the café as her core business. She is not looking to pivot "
        "away from the café — the supermarket launch should complement what she has built. "
        "Anna also cares about ethical sourcing and quality, which she sees as central to "
        "her brand identity."
    ),
    "market_demand": (
        "The German coffee market was valued at approximately US$ 6 billion in 2024 and is "
        "expected to grow at around 5% annually, reaching US$ 9 billion by 2033. Roasted "
        "coffee (ground and whole bean) is the largest segment, accounting for over 50% of "
        "revenue. The whole bean segment in particular is seeing strong growth, driven by "
        "home brewing trends and rising interest in specialty and premium coffee. Over 80% "
        "of coffee purchases are made in supermarkets and discount shops, making retail the "
        "dominant channel. Germans consume over 160 litres of coffee per person per year — "
        "more than any other beverage. "
        "[Sources: Renub Research https://www.renub.com/germany-coffee-market-p.php; "
        "Statista https://www.statista.com/outlook/cmo/hot-drinks/coffee/germany; "
        "German Coffee Association https://www.kaffeeverband.de/en]"
    ),
    "pricing": (
        "Specialty whole bean coffee from independent German roasters typically sells at "
        "€14–22 per 250g (e.g. The Barn Berlin: €14.70/250g; Munich specialty shops: "
        "~€22/250g). This is where Anna's café brand would likely position, compared to "
        "mainstream supermarket coffee at €5–10 per 500g. Prices are expected to continue "
        "rising, driven by sustained supply pressures from major producers like Brazil and "
        "Vietnam. "
        "[Sources: The Barn Berlin https://thebarn.de; "
        "iamexpat.de https://www.iamexpat.de/expat-info/germany-news/germany-expecting-coffee-price-shock-2025]"
    ),
    "costs": (
        "Launching into supermarkets requires upfront investment in additional roasting and "
        "packaging capacity, estimated at €10,000–50,000. Ongoing production and distribution "
        "costs are higher per unit than café sales due to packaging compliance and logistics. "
        "German supermarkets typically retain around 30–35% of the retail price as their "
        "margin, significantly reducing Anna's net revenue per bag. "
        "[Source: Statista German retail gross margin "
        "https://www.statista.com/statistics/500766/retail-gross-profit-margin-germany/]"
    ),
    "competition": (
        "The German supermarket coffee market is moderately concentrated, dominated by "
        "established brands such as Tchibo, Dallmayr, and Jacobs Douwe Egberts, alongside "
        "strong private labels from retailers like ALDI and Lidl. "
        "[Sources: Mordor Intelligence https://www.mordorintelligence.com/industry-reports/germany-coffee-market; "
        "comunicaffe.com https://www.comunicaffe.com/germany-sales-of-whole-beans-are-set-to-outpace-ground-coffee-sales-this-year-in-the-home-market/]"
    ),
    "distribution": (
        "Anna has no existing relationships with supermarket buyers or distributors, and "
        "would likely need to work through a food distributor to access retail shelf space. "
        "Additional logistics costs — including warehousing and last-mile delivery to stores "
        "— would add to the per-unit cost compared to selling directly from the café."
    ),
    "operations": (
        "Anna currently roasts her own beans in small batches for the café, with limited "
        "spare capacity. Scaling to supermarket volumes would require additional roasting "
        "shifts or equipment upgrades, and potentially new staff for roasting, packaging, "
        "and quality control. Managing both the café and a retail product line simultaneously "
        "adds operational complexity."
    ),
    "regulatory": (
        "Selling packaged food in Germany requires compliance with EU food labelling rules "
        "and the German Packaging Act (VerpackG), which mandates packaging registration and "
        "recycling obligations. The EU Deforestation Regulation (EUDR), effective December "
        "2025, adds supply chain traceability requirements for coffee. "
        "[Sources: VerpackG https://www.sustainable-markets.com/german-packaging-act-verpackg-a-2025-guide-to-international-packaging-laws-part-1/; "
        "EUDR https://www.cbi.eu/market-information/coffee/germany/market-entry]"
    ),
    "risks": (
        "Key risks include cannibalization of existing café sales, brand dilution if a "
        "mass-market supermarket presence undermines Anna's premium café positioning, and "
        "the possibility that consumer demand for her product on supermarket shelves is "
        "lower than expected."
    ),
    "timeline": (
        "Anna is looking to make a decision within the next 3 months, with a realistic "
        "target of reaching supermarket shelves within 12–18 months if she decides to "
        "proceed."
    ),
    "sustainability": (
        "Anna is committed to ethical sourcing and sustainable practices as part of her "
        "brand identity. Organic or Fairtrade certification could strengthen her supermarket "
        "positioning but adds cost and compliance overhead."
    ),
    "scope": (
        "Anna wants to understand whether launching packaged whole beans in German "
        "supermarkets would be profitable, and what risks she should consider before "
        "making a decision."
    ),
}


# ── Active Cases ───────────────────────────────────────────────────────────────

CASES = {

    "black_box": {
        "prompt": _COFFEE_SHOP_PROMPT,
        "clarification_facts": _COFFEE_SHOP_FACTS,
    },

    "explainable": {
        "prompt": _COFFEE_SHOP_PROMPT,
        "clarification_facts": _COFFEE_SHOP_FACTS,
    },

    "hitl": {
        "prompt": _COFFEE_SHOP_PROMPT,
        "clarification_facts": _COFFEE_SHOP_FACTS,
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


    # ── Backup / unused cases ──────────────────────────────────────────────────
    # Preserved for reference. Not used by any active agent.

    "backup_blackbox_china": {
        # Original BlackBox case — SF medical devices entering China
        "prompt": """Your client is a San Francisco-based medical device company that has been \
operating for 10 years and has a successful product line of innovative \
wearable health monitors. They are considering expanding their business \
into China.

They have asked your team to evaluate the feasibility and potential \
profitability of this expansion. What key areas would you analyze to \
provide a comprehensive assessment of whether they should enter the \
Chinese market?""",
        "clarification_facts": {
            "objective": (
                "The client's primary objective is to significantly increase their market share "
                "and revenue by tapping into the large and growing Chinese healthcare market. "
                "They also aim to diversify their market presence beyond the US."
            ),
            "market_size": (
                "The Chinese medical device market is one of the largest in the world, valued "
                "at approximately $100 billion USD in 2023 and expected to grow at a CAGR of "
                "around 8-10% over the next five years."
            ),
            "competition": (
                "The Chinese medical device market is highly competitive, with both domestic "
                "companies (e.g., Mindray, Neusoft Medical) and international players (e.g., "
                "Philips, GE Healthcare, Siemens Healthineers) vying for market share."
            ),
            "regulatory": (
                "China has stringent regulatory requirements for medical devices managed by the "
                "National Medical Products Administration (NMPA). The approval process can take "
                "2-5 years and requires extensive clinical trials and documentation."
            ),
            "distribution": (
                "Distribution in China often requires local partnerships or joint ventures due "
                "to regulatory and cultural complexities. E-commerce platforms like Alibaba "
                "Health and JD Health are also important channels."
            ),
            "pricing": (
                "Pricing strategies in China need to account for local purchasing power, "
                "government reimbursement policies, and intense price competition from "
                "domestic manufacturers."
            ),
            "intellectual_property": (
                "IP protection in China remains a concern for foreign companies. Robust IP "
                "strategies including patents, trademarks, and trade secrets are essential "
                "before market entry."
            ),
            "geopolitical_risks": (
                "The client is aware of the political and geopolitical risks associated with "
                "entering the Chinese market. They understand that factors such as trade "
                "tensions, regulatory changes, and government policies can impact their "
                "market entry strategy."
            ),
        },
    },

    "backup_explainable_mining": {
        # Original Explainable case — Mining Co. profitability
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
                "The Silica Sand deposit is estimated at 50 million metric tons with a "
                "lifespan of approximately 10 years at current demand. The Bentonite deposit "
                "is around 10 million metric tons with a lifespan of approximately 30 years. "
                "Both deposits are relatively accessible — close to the surface with moderate "
                "terrain complexity."
            ),
            "competition": (
                "The Silica Sand market is fragmented, with multiple suppliers worldwide. "
                "The Bentonite market is more concentrated, with a few key players dominating."
            ),
            "infrastructure": (
                "The company already has existing transport routes that can be utilized. "
                "Transportation costs to end markets are approximately $20–25 per ton."
            ),
            "operations": (
                "The company has some capacity to mine these resources with current equipment "
                "and labor, but would need to invest in additional equipment and training. "
                "The estimated investment required is around $10 million. The fly-in remote "
                "setup increases labor costs."
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
                "reduce selling prices and compress margins. Execution risks include the $10 "
                "million capex requirement, remote location labor challenges, and environmental "
                "compliance. The client is also exposed to cost overruns given the complex "
                "extraction process for Bentonite."
            ),
            "timeline": (
                "The client is looking to make a decision within the next 3 months and aims to "
                "start production within 6 months if the project is deemed feasible."
            ),
            "sustainability": (
                "The client is interested in sustainable mining practices to minimize "
                "environmental impact and maintain their social license to operate. Community "
                "and indigenous stakeholder considerations are part of their overall objective."
            ),
            "scope": (
                "The client is interested in knowing whether it would be profitable to mine "
                "both Silica Sand and Bentonite. They are open to mining both minerals but "
                "also want to understand the potential risks of moving forward."
            ),
        },
    },

    "backup_hitl_ma": {
        # Original HITL case — Rainbow & Gumdrops M&A
        "prompt": """A global foods maker and marketer, which is based in the US, is \
contemplating a strategic acquisition of a smaller British confectionery \
company to bolster its presence in emerging markets and establish a \
footprint in additional businesses, such as gum and candy. They've hired \
our team to help them evaluate this potential acquisition and help the \
senior leadership make a decision. Which key pieces of analysis would you \
look at to guide them?""",
        "clarification_facts": {
            "objective": (
                "The client's primary objective is to diversify their product portfolio and "
                "establish a stronger presence in the confectionery market. They also see "
                "potential in leveraging the British company's established brand and "
                "distribution channels to enter new markets, particularly in Asia and Africa."
            ),
            "target_company": (
                "The British confectionery company specializes in gum and candy products and "
                "has a strong presence in the UK and some European markets. They have a "
                "diverse product line and a loyal customer base, particularly among younger "
                "consumers."
            ),
            "financials": (
                "The British confectionery company has annual revenues of approximately "
                "$500 million, with an EBITDA margin of around 15%. They have been growing "
                "at a rate of 5% annually over the past three years."
            ),
            "synergies": (
                "Potential synergies include cost savings from combined manufacturing and "
                "distribution networks, as well as revenue synergies from cross-selling "
                "products in new markets."
            ),
            "integration": (
                "The client has experience in integrating acquired companies, having "
                "successfully integrated several smaller firms, but also had a couple of "
                "acquisitions that did not meet expectations due to cultural clashes and "
                "integration challenges."
            ),
            "client_capabilities": (
                "The client currently has a small presence in the confectionery market, "
                "primarily through a line of chocolate bars that account for about 5% of "
                "their total sales. They do not currently have strong on-the-ground "
                "capability in Asia or Africa."
            ),
            "deal_structure": (
                "The client is considering a full acquisition of the British confectionery "
                "company as part of their strategic growth plan, looking to fully integrate "
                "the company into their operations."
            ),
            "regulatory": (
                "The regulatory environment in the UK is relatively straightforward. However, "
                "the emerging markets the client is looking to enter have more complex and "
                "unpredictable regulatory environments, with potential foreign ownership "
                "restrictions and antitrust concerns."
            ),
            "target_willingness": (
                "The British confectionery company has not publicly disclosed their "
                "willingness to sell at this time."
            ),
            "competing_bidders": (
                "The confectionery company does not currently have any known competing "
                "bidders for acquisition."
            ),
            "ip_patents": (
                "The British confectionery company does not have any significant intellectual "
                "property, proprietary recipes, or patents that would be considered a key "
                "asset in this potential acquisition."
            ),
            "exit_option": (
                "The client has not specifically mentioned any plans for a clear exit option "
                "such as divestment or spin-off for this potential acquisition."
            ),
            "financing": (
                "The client is considering financing options to minimise the impact on their "
                "cash flow."
            ),
        },
    },

    "backup_ghost_restaurant": {
        # Original ghost restaurant case — unused
        "prompt": """Leading food delivery companies like Uber Eats and DoorDash are amassing \
an unprecedented amount of data about consumer preferences for restaurant \
takeout meals. Among the many uses for this data, it's also powering an \
entirely new type of virtual restaurant: ghost restaurants.

Ghost restaurants have no storefront. While they operate a kitchen and cook \
dishes, their only "shingle," so to speak, is hung online, within an app \
like Uber Eats or DoorDash.

Your client is considering launching a ghost restaurant. What would you \
advise them to consider before making this decision?""",
        "clarification_facts": {},
    },
}


# ── Public accessors ───────────────────────────────────────────────────────

def get_case(agent_type: str) -> str:
    """Return the case prompt string for the given agent type."""
    entry = CASES.get(agent_type)
    if entry is None:
        return "No case available for this agent type."
    return entry["prompt"]


def get_clarification_facts(agent_type: str) -> dict:
    """Return the clarification facts dict for the given agent type."""
    entry = CASES.get(agent_type)
    if isinstance(entry, dict):
        return entry.get("clarification_facts", {})
    return {}