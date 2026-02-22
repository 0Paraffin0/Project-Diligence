"""
CDD Assistant — AI-Powered Customer Due Diligence Tool
Automatically researches customers via web search and produces
a structured CDD report ready for analyst review.
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
MAX_TOOL_CALLS = 20  # cap to prevent runaway loops

SYSTEM_PROMPT = """\
You are an expert financial crime compliance assistant specialising in Customer Due Diligence (CDD).

You have two research tools: web_search and fetch_webpage. Use both extensively before writing \
the report. Snippets alone are rarely enough — after finding a relevant URL, use fetch_webpage \
to read the full content of the page.

RESEARCH REQUIREMENTS — run AT LEAST 10 tool calls covering:
1. Company/individual registration: "[name] company registration [country]", "[name] registered address director"
2. Sanctions — search each list separately:
   - "[name] OFAC SDN sanctions"
   - "[name] UN consolidated sanctions list"
   - "[name] EU sanctions list"
   - "[name] HMT UK financial sanctions"
3. PEP status: "[name] politically exposed person government official"
4. Adverse media: "[name] fraud", "[name] money laundering", "[name] bribery corruption", \
"[name] criminal investigation regulatory action"
5. Ownership structure: "[name] beneficial owner shareholder UBO"
6. General background: "[name] business activities revenue clients"

After each search, fetch the full content of the 1–2 most relevant URLs using fetch_webpage.

REPORT REQUIREMENTS — every section must be specific and evidence-based:

**1. Customer Identification**
List every piece of identifying information found: full legal name, registration number, \
registered address, incorporation date, jurisdiction, directors, company type. \
Explicitly flag each missing KYC field and why it is required.

**2. Identity Verification**
State exactly which documents are needed for this customer type, which have been provided, \
and which are outstanding. Do not be vague — name the specific documents.

**3. Beneficial Ownership**
Name every identified beneficial owner with ownership percentage if known. If the structure \
is opaque, describe exactly what is known and what is missing. Flag nominee arrangements, \
layered structures, or offshore holding companies with specifics.

**4. Business Relationship Purpose**
State the exact business activity, specific transaction types expected, stated monthly/annual \
volumes, and declared source of funds or wealth. Flag any inconsistency with the customer profile.

**5. Risk Assessment**
Score EACH of the following factors with supporting evidence — never leave one unaddressed:
- Geographic risk (jurisdiction of incorporation, operation, and UBOs)
- Sector/industry risk
- PEP exposure
- Sanctions exposure
- Ownership complexity
- Adverse media
Conclude with preliminary overall rating (Low / Medium / High) with a clear rationale. \
State this is preliminary and requires analyst validation.

**6. Sanctions and PEP Screening**
State explicitly which lists were checked. For each: "Searched [list] — [result]." \
If a potential match is found, provide the full name, listing reason, and date listed. \
Do not summarise — be exact.

**7. Adverse Media**
List each relevant article or report with: headline, publication, date, URL, and a \
one-sentence summary. Distinguish confirmed findings from allegations. If no adverse \
media was found after thorough searching, state that explicitly with the queries used.

**8. Enhanced Due Diligence Triggers**
List each trigger present (e.g. offshore jurisdiction, complex ownership, adverse media hit, \
PEP connection). For each trigger, specify the exact additional documentation or steps required. \
If no EDD triggers are present, state this explicitly.

**9. CDD Case Narrative**
Write at least four paragraphs: (1) who the customer is, (2) key risk factors found, \
(3) how each risk was assessed or mitigated, (4) recommended decision with justification \
(Approve / Decline / Escalate for further review). Label as DRAFT pending analyst review.

**10. Ongoing Monitoring Notes**
State a specific review frequency (e.g. 6-monthly, annual) with justification. List at \
least three specific transaction types or behavioural patterns to flag for this customer.

GENERAL RULES:
- Every factual claim must cite a source URL or state "provided by analyst".
- Never write "further investigation may be required" without specifying exactly what.
- Never fabricate data or infer ownership without evidence.
- Use professional language appropriate for a compliance case file.
- Label the entire output: ⚠️ DRAFT — Pending Analyst Review.\
"""

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the internet for information relevant to a CDD review. "
            "Run multiple specific searches covering sanctions, adverse media, company "
            "registries, PEP status, and background. Each call should focus on one aspect. "
            "After getting results, use fetch_webpage on the most relevant URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific search query, e.g. 'Acme Holdings BVI company registry', "
                        "'John Smith OFAC sanctions', 'Meridian Trading fraud 2023'"
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
            "to read full articles, company registry entries, sanctions list pages, or news "
            "reports. Much more detailed than search snippets."
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
            "**On Streamlit Cloud:** go to your app's Settings → Secrets and add:\n"
            "```\nANTHROPIC_API_KEY = \"your-api-key-here\"\n```\n"
            "**Running locally:** create a `.env` file next to `app.py` and add:\n"
            "```\nANTHROPIC_API_KEY=your-api-key-here\n```"
        )
        st.stop()
    return Anthropic(api_key=api_key)


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
        # Collapse excessive blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        return text[:max_chars] + ("\n\n[Page truncated]" if len(text) > max_chars else "")
    except Exception as exc:
        return f"Could not fetch page: {exc}"


def run_cdd_research(client: Anthropic, customer_details: str, status_box):
    """
    Agentic loop: Claude decides what to search and fetch, we execute,
    Claude synthesises results into a CDD report.
    Returns (report_text, list_of_actions).
    """
    messages = [{"role": "user", "content": customer_details}]
    tool_call_count = 0
    actions_done = []

    with status_box:
        progress = st.empty()
        log = st.empty()

        while tool_call_count <= MAX_TOOL_CALLS:
            progress.info("Analysing findings…")

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

                    if block.name == "web_search":
                        query = block.input["query"]
                        actions_done.append(f"🔍 {query}")
                        progress.info(f"Searching: *{query}*")
                        result = run_web_search(query)

                    elif block.name == "fetch_webpage":
                        url = block.input["url"]
                        actions_done.append(f"📄 {url}")
                        progress.info(f"Reading page: *{url[:80]}*")
                        result = fetch_webpage(url)

                    else:
                        result = "Unknown tool."

                    log.markdown(
                        "**Research activity:**\n"
                        + "\n".join(f"- {a}" for a in actions_done[-10:])
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                break

        # Fallback if loop hits the cap
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

        with st.expander("📖 How to use", expanded=True):
            st.markdown(
                """
                1. Enter the customer name and any known details
                2. Click **Run CDD Review** — the AI automatically searches
                   the internet across all key risk areas
                3. Review the generated report
                4. Use the chat at the bottom for follow-up questions
                5. Download the report for your case file
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

    # ── Input form (shown until a report exists) ───────────────────────────────
    if not st.session_state.report:
        st.markdown(
            "Enter the customer's name and any known details. "
            "The AI will search the internet and produce a full CDD report automatically."
        )

        with st.form("cdd_form"):
            customer_input = st.text_area(
                "Customer details",
                placeholder=(
                    "Example:\n"
                    "Company: Meridian Trading Ltd\n"
                    "Incorporated: British Virgin Islands\n"
                    "Director: John Smith (British national)\n"
                    "Relationship purpose: Trade finance\n"
                    "Expected monthly volume: USD 500,000"
                ),
                height=160,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "🔍 Run CDD Review", use_container_width=True, type="primary"
            )

        if submitted and customer_input.strip():
            status_box = st.container()
            report, searches = run_cdd_research(client, customer_input, status_box)
            st.session_state.report = report
            st.session_state.searches = searches
            st.rerun()
        elif submitted:
            st.warning("Please enter some customer details before running the review.")

    # ── Report display ─────────────────────────────────────────────────────────
    if st.session_state.report:
        if st.session_state.searches:
            with st.expander(
                f"🔍 {len(st.session_state.searches)} research actions performed",
                expanded=False,
            ):
                for q in st.session_state.searches:
                    st.markdown(f"- {q}")

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

            # Give the model the report as context for follow-ups
            context = [
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
                        thinking={"type": "adaptive"},
                        messages=context,
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
