"""
CDD Assistant — AI-Powered Customer Due Diligence Tool
Automatically researches customers via Dow Jones Risk & Compliance and
web search, producing a structured CDD report ready for analyst review.
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

# ── Dow Jones R&C config ────────────────────────────────────────────────────────
DJ_TOKEN_URL = "https://accounts.dowjones.com/oauth2/v1/token"
DJ_SCREEN_URL = "https://api.dowjones.com/risk/v1/profiles/search"

# Categories to request from the DJ R&C API
DJ_CATEGORIES = [
    "b_peps",   # Politically Exposed Persons
    "b_soe",    # State-Owned Enterprises
    "e_sl",     # Sanctions lists (all)
    "b_am",     # Adverse media
    "b_oel",    # Other enforcement lists
]

SYSTEM_PROMPT = """\
You are an expert financial crime compliance assistant specialising in Customer Due Diligence (CDD).

You have three research tools: dow_jones_screen, web_search, and fetch_webpage.

STEP 1 — ALWAYS run dow_jones_screen first. This is the primary professional screening \
database covering PEP lists, global sanctions lists (OFAC SDN, UN, EU, HMT and 1,000+ others), \
state-owned enterprises, and adverse media. Review every match carefully and note the exact \
categories, listing reasons, and dates returned.

STEP 2 — Supplement with web_search. Even if dow_jones_screen returns results, run targeted \
web searches to find:
- Company registration and ownership structure (Companies House, national registries)
- Any adverse media not captured in the DJ database (recent news, local publications)
- Beneficial ownership and UBO details
- Business background, clients, revenue, sector
- Corroboration or further detail on any DJ matches found

STEP 3 — For the most relevant URLs from your searches, call fetch_webpage to read full \
page content rather than relying on snippets.

Run AT LEAST 10 total tool calls before writing the report.

────────────────────────────────────────────────────────────────────────
REPORT REQUIREMENTS — every section must be specific and evidence-based.
────────────────────────────────────────────────────────────────────────

**1. Customer Identification**
List every piece of identifying information found: full legal name, registration number, \
registered address, incorporation date, jurisdiction, directors/officers, company type, \
LEI if available. Flag each KYC field that is missing and state exactly why it is required.

**2. Identity Verification**
State which documents are required for this customer type, which have been provided, \
and which are outstanding. Name specific documents (e.g. certificate of incorporation, \
register of directors, utility bill dated within 3 months, passport copy).

**3. Beneficial Ownership**
Name every identified beneficial owner with percentage stake if known. If the structure \
is opaque, describe exactly what was found and what remains unknown. Flag nominee \
arrangements, layered holding structures, or offshore entities with specifics.

**4. Business Relationship Purpose**
State the exact business activity, expected transaction types, stated volumes \
(monthly/annual), and declared source of funds or wealth. Flag any inconsistency \
with the customer's known profile or jurisdiction.

**5. Risk Assessment**
Score EACH of the following with supporting evidence — never leave one unaddressed:
- Geographic risk (jurisdiction of incorporation, of operations, of UBOs)
- Sector/industry risk
- PEP exposure (direct or by association)
- Sanctions exposure
- Ownership structure complexity
- Adverse media
Conclude with a preliminary overall rating (Low / Medium / High) with clear rationale. \
Mark as preliminary, requiring analyst validation.

**6. Sanctions and PEP Screening**
Lead with the Dow Jones R&C screening result. State the exact result for each category: \
"DJ R&C — [category]: [number of matches / no matches]. [Detail of any match]."
Then confirm which additional lists were searched manually via web. For any match, provide: \
full name on list, listing authority, reason for listing, date listed, and DJ profile ID.

**7. Adverse Media**
List each relevant article or report with: headline, publication, date, URL, and \
one-sentence summary. Distinguish confirmed facts from allegations. If no adverse \
media was found after thorough searching, state that explicitly alongside the queries used.

**8. Enhanced Due Diligence Triggers**
List each EDD trigger present (e.g. offshore jurisdiction, PEP connection, sanctions \
match, complex ownership, adverse media, high-risk sector). For each trigger, specify \
the exact additional documentation or steps required. If no EDD triggers exist, confirm.

**9. CDD Case Narrative**
Write at least four paragraphs:
(1) Who the customer is and what they do.
(2) Key risk factors identified.
(3) How each risk was assessed or mitigated by the research.
(4) Recommended decision with justification — Approve / Decline / Escalate for further review.
Label: ⚠️ DRAFT — Pending Analyst Review.

**10. Ongoing Monitoring Notes**
State a specific review frequency (e.g. 6-monthly) with justification. List at least \
three specific transaction types or behavioural patterns to flag for this customer.

GENERAL RULES:
- Every factual claim must cite its source URL or state "source: Dow Jones R&C" or "provided by analyst".
- If dow_jones_screen returns a "not configured" message, note this prominently in Section 6 \
  and confirm that web-based screening was used as an alternative.
- Never fabricate data or infer ownership without evidence.
- Never write "further investigation may be needed" without specifying exactly what.
- Use professional language appropriate for a compliance case file.\
"""

TOOLS = [
    {
        "name": "dow_jones_screen",
        "description": (
            "Screen a name against the Dow Jones Risk & Compliance database. "
            "This is the primary professional screening tool — ALWAYS call this first. "
            "It covers 1,000+ global sanctions lists (OFAC, UN, EU, HMT and more), "
            "PEP lists, state-owned enterprises, and curated adverse media. "
            "Call once for the entity name and, if relevant, once more for key individuals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full name of the person or company to screen.",
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["PERSON", "COMPANY", "ALL"],
                    "description": (
                        "Type of entity. Use PERSON for individuals, COMPANY for businesses, "
                        "ALL if unsure."
                    ),
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the internet for information relevant to a CDD review. "
            "Use this to supplement Dow Jones screening with registry data, recent news, "
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


def _dj_bearer_token() -> str | None:
    """Obtain a DJ OAuth2 bearer token from client credentials, if configured."""
    client_id = st.secrets.get("DOWJONES_CLIENT_ID") or os.getenv("DOWJONES_CLIENT_ID")
    client_secret = st.secrets.get("DOWJONES_CLIENT_SECRET") or os.getenv("DOWJONES_CLIENT_SECRET")
    if not (client_id and client_secret):
        return None
    try:
        resp = requests.post(
            DJ_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception:
        return None


def dow_jones_screen(name: str, entity_type: str = "ALL") -> str:
    """Screen a name against the Dow Jones Risk & Compliance database."""
    # Resolve credentials: prefer direct API key, then OAuth2
    api_key = st.secrets.get("DOWJONES_API_KEY") or os.getenv("DOWJONES_API_KEY")
    token = api_key or _dj_bearer_token()

    if not token:
        return (
            "⚠️ Dow Jones Risk & Compliance: Not configured.\n"
            "To enable professional screening add to app secrets:\n"
            "  DOWJONES_API_KEY = \"your-key\"  (API key auth)\n"
            "  — or —\n"
            "  DOWJONES_CLIENT_ID = \"...\"\n"
            "  DOWJONES_CLIENT_SECRET = \"...\"\n"
            "Web-based sanctions and PEP searching will be used as an alternative."
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "data": {
            "attributes": {
                "filter_search_string": name,
                "filter_entity_type": entity_type,
                "search_type": "CONTAINS",
                "page_size": 25,
                "filter_content_category": DJ_CATEGORIES,
            }
        }
    }

    try:
        resp = requests.post(DJ_SCREEN_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        profiles = data.get("data", [])
        meta = data.get("meta", {})
        total = meta.get("total_count", len(profiles))

        if not profiles:
            return (
                f"Dow Jones R&C screening for '{name}' ({entity_type}): "
                f"No matches found across PEP, sanctions (OFAC/UN/EU/HMT/others), "
                f"SOE, and adverse media categories."
            )

        lines = [
            f"Dow Jones R&C — {total} potential match(es) for '{name}' ({entity_type}):\n"
        ]
        for p in profiles[:15]:
            attrs = p.get("attributes", {})
            categories = attrs.get("categories", [])
            score = attrs.get("score", "N/A")
            primary = attrs.get("primary_name", "N/A")
            ptype = attrs.get("entity_type", "N/A")
            profile_id = p.get("id", "N/A")
            also_known = attrs.get("also_known_as", [])

            lines.append(
                f"  Profile ID : {profile_id}\n"
                f"  Name       : {primary}\n"
                f"  Type       : {ptype}\n"
                f"  Categories : {', '.join(categories) if categories else 'N/A'}\n"
                f"  Match score: {score}\n"
                + (f"  AKAs       : {', '.join(also_known[:5])}\n" if also_known else "")
            )

        if total > 15:
            lines.append(f"  ... and {total - 15} more. Refine search for full list.")

        return "\n".join(lines)

    except requests.HTTPError as exc:
        code = exc.response.status_code
        body = exc.response.text[:400]
        return f"Dow Jones API error (HTTP {code}): {body}"
    except Exception as exc:
        return f"Dow Jones API error: {exc}"


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
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        return text[:max_chars] + ("\n\n[Page truncated]" if len(text) > max_chars else "")
    except Exception as exc:
        return f"Could not fetch page: {exc}"


def run_cdd_research(client: Anthropic, subject: str, context: str, status_box):
    """
    Agentic loop: Claude screens via DJ R&C, searches the web, and reads pages,
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

                    if block.name == "dow_jones_screen":
                        name = block.input["name"]
                        etype = block.input.get("entity_type", "ALL")
                        actions_done.append(f"🛡️ DJ R&C: {name} ({etype})")
                        progress.info(f"Dow Jones screening: *{name}*")
                        result = dow_jones_screen(name, etype)

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


def dj_configured() -> bool:
    """Return True if Dow Jones credentials are available."""
    return bool(
        st.secrets.get("DOWJONES_API_KEY")
        or os.getenv("DOWJONES_API_KEY")
        or (
            (st.secrets.get("DOWJONES_CLIENT_ID") or os.getenv("DOWJONES_CLIENT_ID"))
            and (st.secrets.get("DOWJONES_CLIENT_SECRET") or os.getenv("DOWJONES_CLIENT_SECRET"))
        )
    )


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

        # DJ R&C status indicator
        if dj_configured():
            st.success("🛡️ Dow Jones R&C connected")
        else:
            st.warning(
                "🛡️ **Dow Jones R&C not configured**\n\n"
                "Add credentials to Secrets to enable professional screening:\n"
                "```\nDOWJONES_API_KEY = \"your-key\"\n```\n"
                "or\n"
                "```\nDOWJONES_CLIENT_ID = \"...\"\n"
                "DOWJONES_CLIENT_SECRET = \"...\"\n```\n"
                "Web-based screening will be used instead."
            )

        st.divider()

        with st.expander("📖 How to use", expanded=False):
            st.markdown(
                """
                1. Type any **name or company name** in the search box
                2. Optionally add context (jurisdiction, relationship type, etc.)
                3. Click **Run CDD Review**
                4. The AI runs Dow Jones R&C screening, web searches,
                   and reads relevant pages automatically
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
                placeholder="e.g. Meridian Trading Ltd  or  John Smith",
                label_visibility="visible",
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
            dj_count = sum(1 for a in st.session_state.searches if a.startswith("🛡️"))
            web_count = sum(1 for a in st.session_state.searches if a.startswith("🔍"))
            page_count = sum(1 for a in st.session_state.searches if a.startswith("📄"))
            with st.expander(
                f"Research performed: {dj_count} DJ screening · "
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
