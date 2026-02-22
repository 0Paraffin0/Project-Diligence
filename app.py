"""
CDD Assistant — AI-Powered Customer Due Diligence Tool
Automatically researches customers via web search and produces
a structured CDD report ready for analyst review.
"""

import os

import streamlit as st
from anthropic import Anthropic
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000
MAX_SEARCHES = 10  # cap to prevent runaway loops

SYSTEM_PROMPT = """\
You are an expert financial crime compliance assistant specialising in Customer Due Diligence (CDD).

You have access to a web_search tool. When given a customer to review, proactively search the \
internet to gather information across all relevant CDD risk areas before writing your report.

Run multiple targeted searches — at minimum cover:
- Company or individual registration and ownership structure
- Sanctions exposure (OFAC, UN, EU, HMT)
- PEP (Politically Exposed Person) status
- Adverse media: fraud, money laundering, bribery, regulatory actions
- General background and business activities

After completing your research, produce a structured report with all 10 sections:

1. Customer Identification — summarise known identifying details; flag missing KYC fields
2. Identity Verification — note what has been verified and what is outstanding
3. Beneficial Ownership — map owners above 25% threshold; flag complex or offshore structures
4. Business Relationship Purpose — summarise stated purpose, expected volumes, source of funds
5. Risk Assessment — preliminary rating (Low / Medium / High) with evidence; do not finalise
6. Sanctions and PEP Screening — flag any list matches with supporting detail
7. Adverse Media — summarise findings; distinguish confirmed facts from unverified allegations
8. Enhanced Due Diligence Triggers — recommend EDD steps if high-risk indicators are present
9. CDD Case Narrative — draft summary and recommended decision (Approve / Decline / Escalate)
10. Ongoing Monitoring Notes — recommended review frequency and behaviours to monitor

Behavioural guidelines:
- Cite the source URL for every finding.
- Clearly state when a search returned no relevant results.
- Never fabricate data or infer ownership without evidence.
- Flag anything unusual or potentially high-risk.
- Use plain, professional language suitable for a compliance case file.
- Label the full report as a DRAFT pending analyst review.
- Always remind the analyst that final decisions are their responsibility.\
"""

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the internet for information relevant to a CDD review. "
            "Run multiple specific searches to cover sanctions lists, adverse media, "
            "corporate registry data, PEP status, and general background. "
            "Each call should focus on one specific aspect."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific search query, e.g. 'Acme Holdings BVI company registry', "
                        "'John Smith OFAC sanctions', 'Meridian Trading fraud news 2024'"
                    ),
                }
            },
            "required": ["query"],
        },
    }
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


def run_web_search(query: str, max_results: int = 5) -> str:
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


def run_cdd_research(client: Anthropic, customer_details: str, status_box):
    """
    Agentic loop: Claude decides what to search, we execute,
    Claude synthesises results into a CDD report.
    Returns (report_text, list_of_queries).
    """
    messages = [{"role": "user", "content": customer_details}]
    search_count = 0
    searches_done = []

    with status_box:
        progress = st.empty()
        log = st.empty()

        while search_count <= MAX_SEARCHES:
            progress.info("Thinking about what to research…")

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
                return report, searches_done

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use" and block.name == "web_search":
                        query = block.input["query"]
                        search_count += 1
                        searches_done.append(query)
                        progress.info(f"Searching ({search_count}): *{query}*")
                        log.markdown(
                            "**Searches so far:**\n"
                            + "\n".join(f"- {q}" for q in searches_done)
                        )
                        result = run_web_search(query)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                break

        # Fallback if loop exits without end_turn
        report = next(
            (b.text for b in response.content if b.type == "text"),
            "Research reached the search limit. Please review findings above.",
        )
        return report, searches_done


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
                f"🔍 {len(st.session_state.searches)} web searches performed",
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
