"""
CDD Assistant — AI-Powered Customer Due Diligence Tool
Automatically researches customers via OpenSanctions and web search,
producing a structured CDD report ready for analyst review.
"""

import os

import requests
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000
MAX_TOOL_CALLS = 25  # cap to prevent runaway loops

# OpenSanctions free public API — no account needed
OPENSANCTIONS_URL = "https://api.opensanctions.org/search/default"

SYSTEM_PROMPT = """\
You are an expert financial crime compliance assistant specialising in Customer Due Diligence (CDD).

You have three research tools: sanctions_screen, web_search, and fetch_webpage.

STEP 1 — ALWAYS call sanctions_screen first. This queries the OpenSanctions database, which \
aggregates 100+ global sanctions lists (OFAC SDN, UN, EU, HMT, DFAT, SECO, and more) plus \
PEP data and enforcement actions — the same underlying data used by professional screening \
platforms. Run it for the entity name, and separately for any key individuals (directors, UBOs).

STEP 2 — Supplement with web_search. Run targeted searches to find:
- Company registration, incorporation details, directors, and registered address
- Beneficial ownership structure and UBO identities
- Adverse media: fraud, money laundering, bribery, regulatory actions, criminal investigations
- Business background, sector, clients, and revenues
- Corroboration or further detail on any screening matches found
- Country-specific risk intelligence (FATF grey/black list status, corruption indices)

STEP 3 — For the most relevant URLs, call fetch_webpage to read the full page content.

Run AT LEAST 12 total tool calls before writing the report.

────────────────────────────────────────────────────────────────────────
REPORT REQUIREMENTS — every section must be specific and evidence-based.
────────────────────────────────────────────────────────────────────────

**1. Customer Identification**
List every piece of identifying information found: full legal name, registration number, \
registered address, incorporation date, jurisdiction, directors/officers, company type. \
Flag each missing KYC field and state exactly why it is required.

**2. Identity Verification**
State which specific documents are required for this customer type, which have been provided, \
and which are outstanding. Name exact documents (e.g. certificate of incorporation, \
register of directors, certified passport copy, utility bill dated within 3 months).

**3. Beneficial Ownership**
Name every identified beneficial owner with ownership percentage if known. If the structure \
is opaque, describe exactly what was found and what is missing. Flag nominee arrangements, \
layered holding structures, or offshore entities with specifics.

**4. Business Relationship Purpose**
State the exact business activity, expected transaction types, stated volumes \
(monthly/annual), and declared source of funds or wealth. Flag any inconsistency.

**5. Risk Assessment**
Score EACH factor with supporting evidence — never leave one unaddressed:
- Geographic risk (jurisdiction of incorporation, of operations, of UBOs — note FATF status)
- Sector/industry risk
- PEP exposure (direct or by association)
- Sanctions exposure
- Ownership structure complexity
- Adverse media
Conclude with preliminary overall rating (Low / Medium / High) with a clear rationale. \
Mark as preliminary, requiring analyst validation.

**6. Sanctions and PEP Screening**
Lead with the OpenSanctions result. For each match: name on list, which specific list(s) \
it appears on, listing reason, listing date, and entity type. If no matches: state which \
lists were covered by the search. Then list any supplementary web searches run for \
specific lists (e.g. direct OFAC search, HMT list check).

**7. Adverse Media**
List each relevant article with: headline, publication, date, URL, and one-sentence summary. \
Distinguish confirmed facts from allegations. If none found after thorough searching, state \
that explicitly alongside the queries used.

**8. Enhanced Due Diligence Triggers**
List each EDD trigger present (e.g. offshore jurisdiction, PEP connection, sanctions match, \
complex ownership, adverse media, FATF high-risk country, cash-intensive sector). For each, \
specify the exact additional documentation or steps required. If no triggers, confirm explicitly.

**9. CDD Case Narrative**
Write at least four paragraphs:
(1) Who the customer is and what they do.
(2) Key risk factors identified.
(3) How each risk was assessed or mitigated by the research.
(4) Recommended decision with justification — Approve / Decline / Escalate.
Label: ⚠️ DRAFT — Pending Analyst Review.

**10. Ongoing Monitoring Notes**
State a specific review frequency with justification. List at least three specific \
transaction types or behavioural patterns to flag for this customer.

GENERAL RULES:
- Every factual claim must cite a source URL or state "source: OpenSanctions" / "provided by analyst".
- Never write "further investigation may be needed" without specifying exactly what.
- Never fabricate data or infer ownership without evidence.
- Use professional language appropriate for a compliance case file.\
"""

TOOLS = [
    {
        "name": "sanctions_screen",
        "description": (
            "Screen a name against the OpenSanctions database — 100+ global sanctions lists "
            "(OFAC SDN, UN, EU, HMT, DFAT, SECO, and more) plus PEP lists and enforcement "
            "actions. No account needed. ALWAYS call this first for any name being reviewed. "
            "Call once for the entity, then separately for each known director or UBO."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full name of the person or company to screen.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the internet for information relevant to a CDD review. "
            "Use to supplement sanctions screening with registry data, adverse media, "
            "and ownership details. Each call should focus on one specific aspect. "
            "After getting results, use fetch_webpage on the most relevant URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific search query, e.g. 'Acme Holdings BVI company registry', "
                        "'John Smith director company', 'Meridian Trading fraud 2024'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch the full text content of a specific webpage. Use this after web_search "
            "to read full articles, company registry entries, or news reports. "
            "Much more detailed than search snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the webpage to fetch and read.",
                }
            },
            "required": ["url"],
        },
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_client() -> Anthropic:
    """Initialise the Anthropic client, checking st.secrets then environment variables."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not found.**\n\n"
            "**On Streamlit Cloud:** go to Settings → Secrets and add:\n"
            "```\nANTHROPIC_API_KEY = \"your-key\"\n```\n"
            "**Running locally:** create `.env` and add:\n"
            "```\nANTHROPIC_API_KEY=your-key\n```"
        )
        st.stop()
    return Anthropic(api_key=api_key)


def sanctions_screen(name: str) -> str:
    """
    Screen a name against the OpenSanctions database.
    Covers OFAC SDN, UN, EU, HMT, DFAT, SECO, 100+ other lists, PEPs, and enforcement actions.
    Free public API — no account needed.
    """
    try:
        resp = requests.get(
            OPENSANCTIONS_URL,
            params={"q": name, "limit": 20},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        total = data.get("total", {}).get("value", 0)

        if not results:
            return (
                f"OpenSanctions screening for '{name}': No matches found.\n"
                f"Lists checked include: OFAC SDN, OFAC Non-SDN Consolidated, "
                f"UN Security Council, EU Financial Sanctions, HMT UK, DFAT Australia, "
                f"SECO Switzerland, OSFI Canada, and 100+ additional global lists, "
                f"plus PEP and enforcement action datasets."
            )

        lines = [
            f"OpenSanctions — {total} potential match(es) for '{name}':\n"
            f"(Covers OFAC, UN, EU, HMT, DFAT, SECO, 100+ lists + PEP data)\n"
        ]
        for r in results[:12]:
            props = r.get("properties", {})
            datasets = r.get("datasets", [])
            schema = r.get("schema", "Unknown")
            score = round(r.get("score", 0), 2)

            aliases = props.get("alias", [])
            birth_dates = props.get("birthDate", [])
            nationality = props.get("nationality", [])
            countries = props.get("country", [])
            topics = props.get("topics", [])
            # Sanctions-specific fields
            listing_date = props.get("listingDate", [])
            reason = props.get("reason", []) or props.get("notes", [])

            entry = (
                f"  Name       : {r.get('caption', 'N/A')}\n"
                f"  Type       : {schema}\n"
                f"  Match score: {score}\n"
                f"  Lists      : {', '.join(datasets[:10])}\n"
            )
            if topics:
                entry += f"  Topics     : {', '.join(topics)}\n"
            if aliases:
                entry += f"  AKAs       : {', '.join(aliases[:5])}\n"
            if birth_dates:
                entry += f"  DOB        : {', '.join(birth_dates)}\n"
            if nationality:
                entry += f"  Nationality: {', '.join(nationality)}\n"
            if countries:
                entry += f"  Countries  : {', '.join(countries)}\n"
            if listing_date:
                entry += f"  Listed     : {', '.join(listing_date)}\n"
            if reason:
                entry += f"  Reason     : {' '.join(reason)[:300]}\n"
            lines.append(entry)

        if total > 12:
            lines.append(f"  ... {total - 12} more result(s). Consider narrowing the name.")

        return "\n".join(lines)

    except requests.HTTPError as exc:
        return f"OpenSanctions API error (HTTP {exc.response.status_code}): {exc.response.text[:300]}"
    except Exception as exc:
        return f"OpenSanctions screening error: {exc}"


def run_web_search(query: str, max_results: int = 8) -> str:
    """Execute a DuckDuckGo search and return formatted results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found for this query."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
                f"Snippet: {r.get('body', 'N/A')}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Search error: {exc}"


def fetch_webpage(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract the main text content from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CDD-Research/1.0)"}
        resp = requests.get(url, timeout=12, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)
        return text[:max_chars] + ("\n\n[Page truncated]" if len(text) > max_chars else "")
    except Exception as exc:
        return f"Could not fetch page: {exc}"


def run_cdd_research(client: Anthropic, subject: str, context: str, status_box):
    """
    Agentic loop: Claude screens via OpenSanctions, searches the web, and reads pages,
    then synthesises results into a full CDD report.
    Returns (report_text, list_of_actions).
    """
    user_message = f"Subject to review: {subject}"
    if context.strip():
        user_message += f"\n\nAdditional context provided by analyst:\n{context.strip()}"

    messages = [{"role": "user", "content": user_message}]
    tool_call_count = 0
    actions_done = []

    with status_box:
        progress = st.empty()
        log = st.empty()

        while tool_call_count <= MAX_TOOL_CALLS:
            progress.info("Analysing…")

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                progress.success("Research complete — report ready.")
                report = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return report, actions_done

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_call_count += 1

                    if block.name == "sanctions_screen":
                        name = block.input["name"]
                        actions_done.append(f"🛡️ Sanctions: {name}")
                        progress.info(f"Sanctions & PEP screening: *{name}*")
                        result = sanctions_screen(name)

                    elif block.name == "web_search":
                        query = block.input["query"]
                        actions_done.append(f"🔍 {query}")
                        progress.info(f"Searching: *{query}*")
                        result = run_web_search(query)

                    elif block.name == "fetch_webpage":
                        url = block.input["url"]
                        actions_done.append(f"📄 {url}")
                        progress.info(f"Reading: *{url[:80]}*")
                        result = fetch_webpage(url)

                    else:
                        result = "Unknown tool."

                    log.markdown(
                        "**Research activity:**\n"
                        + "\n".join(f"- {a}" for a in actions_done[-12:])
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                break

        report = next(
            (b.text for b in response.content if b.type == "text"),
            "Research reached the activity limit. Please review findings above.",
        )
        return report, actions_done


def init_session_state():
    for key, default in [
        ("report", None),
        ("searches", []),
        ("follow_ups", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="CDD Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    client = get_client()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚖️ CDD Assistant")
        st.caption("AI-Assisted Customer Due Diligence")
        st.divider()

        st.info(
            "🛡️ **Sanctions & PEP screening**\n\n"
            "Powered by [OpenSanctions](https://opensanctions.org) — "
            "covers OFAC, UN, EU, HMT, DFAT, SECO, and 100+ other global lists, "
            "plus PEP and enforcement action data. No account required."
        )

        st.divider()

        with st.expander("📖 How to use", expanded=False):
            st.markdown(
                """
                1. Type any **name or company name** in the search box
                2. Optionally add context (jurisdiction, relationship type, etc.)
                3. Click **Run CDD Review**
                4. The AI automatically screens against 100+ sanctions and PEP
                   lists, searches the web, and reads relevant pages
                5. Review the structured report
                6. Use the chat below for follow-up questions
                7. Download the report for your case file
                ---
                ⚠️ All outputs are drafts. Final decisions remain
                the analyst's responsibility.
                """
            )

        st.divider()

        if st.session_state.report:
            if st.button("🗑️ New Review", use_container_width=True, type="secondary"):
                st.session_state.report = None
                st.session_state.searches = []
                st.session_state.follow_ups = []
                st.rerun()

            st.download_button(
                label="💾 Download Report",
                data=st.session_state.report,
                file_name="cdd_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.divider()
        st.warning(
            "⚠️ **Analyst Review Required**\n\n"
            "AI outputs are **draft only**. A qualified analyst must review "
            "and approve all findings before any onboarding decision is made."
        )

    # ── Main panel ─────────────────────────────────────────────────────────────
    st.markdown("## Customer Due Diligence Review")

    # ── Input form ─────────────────────────────────────────────────────────────
    if not st.session_state.report:
        with st.form("cdd_form"):
            subject = st.text_input(
                "Name or company name",
                placeholder="e.g.  Meridian Trading Ltd     or     John Smith",
            )

            with st.expander("Optional: additional context"):
                extra_context = st.text_area(
                    "Additional context",
                    placeholder=(
                        "Jurisdiction: British Virgin Islands\n"
                        "Director: John Smith (British national)\n"
                        "Relationship purpose: Trade finance\n"
                        "Expected monthly volume: USD 500,000\n"
                        "Source of funds: Export revenues"
                    ),
                    height=120,
                    label_visibility="collapsed",
                )

            submitted = st.form_submit_button(
                "🔍 Run CDD Review", use_container_width=True, type="primary"
            )

        if submitted and subject.strip():
            status_box = st.container()
            report, searches = run_cdd_research(
                client, subject.strip(), extra_context, status_box
            )
            st.session_state.report = report
            st.session_state.searches = searches
            st.rerun()
        elif submitted:
            st.warning("Please enter a name or company name before running the review.")

    # ── Report display ─────────────────────────────────────────────────────────
    if st.session_state.report:
        if st.session_state.searches:
            screen_count = sum(1 for a in st.session_state.searches if a.startswith("🛡️"))
            web_count = sum(1 for a in st.session_state.searches if a.startswith("🔍"))
            page_count = sum(1 for a in st.session_state.searches if a.startswith("📄"))
            with st.expander(
                f"Research performed: {screen_count} sanctions screening · "
                f"{web_count} web searches · {page_count} pages read",
                expanded=False,
            ):
                for a in st.session_state.searches:
                    st.markdown(f"- {a}")

        st.divider()
        st.markdown(st.session_state.report)
        st.divider()

        # ── Follow-up chat ─────────────────────────────────────────────────────
        st.markdown("### 💬 Follow-up Questions")
        st.caption(
            "Ask the assistant to clarify, expand on, or investigate any part of the report."
        )

        for msg in st.session_state.follow_ups:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a follow-up question…"):
            st.session_state.follow_ups.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            context_msgs = [
                {
                    "role": "user",
                    "content": f"The following CDD report was generated:\n\n{st.session_state.report}",
                },
                {
                    "role": "assistant",
                    "content": "Understood. I have reviewed the CDD report. What follow-up questions do you have?",
                },
            ] + st.session_state.follow_ups

            with st.chat_message("assistant"):
                box = st.empty()
                full_response = ""
                try:
                    with client.messages.stream(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        system=SYSTEM_PROMPT,
                        messages=context_msgs,
                    ) as stream:
                        for chunk in stream.text_stream:
                            full_response += chunk
                            box.markdown(full_response + "▌")
                    box.markdown(full_response)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.session_state.follow_ups.pop()
                    st.stop()

            st.session_state.follow_ups.append(
                {"role": "assistant", "content": full_response}
            )


if __name__ == "__main__":
    main()
