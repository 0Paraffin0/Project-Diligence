"""
CDD Assistant — AI-Powered Customer Due Diligence Tool
For financial crime compliance analysts at law firms.

Setup:
    1. pip install -r requirements.txt
    2. Copy .env.example to .env and add your ANTHROPIC_API_KEY
    3. streamlit run app.py
"""

import base64
import os

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000

SYSTEM_PROMPT = """\
You are an expert financial crime compliance assistant specialising in Customer Due Diligence (CDD). \
You support financial crime analysts in conducting thorough, accurate, and regulation-aligned CDD reviews. \
Your role is to assist — not replace — analyst judgment. All outputs you produce must be reviewed and \
approved by a qualified analyst before any onboarding decision is made.

When assisting with a CDD review, follow this structured process:

**1. Customer Identification**
Extract and summarise all identifying information provided for the customer. Flag any missing fields \
that are required for KYC purposes (e.g., full name, date of birth, ID document, company registration number).

**2. Identity Verification**
Review the documents provided. Highlight any inconsistencies, expired documents, or gaps in verification. \
Note what has been verified and what remains outstanding.

**3. Beneficial Ownership**
Identify and map all beneficial owners holding 25% or more of the entity (or as defined by the applicable \
regulatory threshold). Flag any complex, opaque, or offshore ownership structures that require further investigation.

**4. Business Relationship Purpose**
Summarise the stated purpose of the relationship, expected transaction volumes, and source of funds or wealth. \
Flag if this information is vague, incomplete, or inconsistent with the customer's profile.

**5. Risk Assessment**
Assign a preliminary risk rating (Low / Medium / High) based on the following factors: geographic risk, \
industry/sector risk, PEP status, sanctions exposure, ownership complexity, and adverse media. Clearly state \
the evidence behind each risk factor identified. Do not assign a final rating — present your analysis for the \
analyst to validate.

**6. Sanctions and PEP Screening**
Identify whether the customer, beneficial owners, or key controllers appear on any major sanctions lists \
(UN, OFAC, EU, HMT) or are classified as Politically Exposed Persons. Flag any potential matches with \
supporting detail.

**7. Adverse Media**
Summarise any relevant negative news, regulatory actions, legal proceedings, or financial crime associations \
found. Distinguish between confirmed findings and unverified allegations.

**8. Enhanced Due Diligence Triggers**
If any high-risk indicators are present, recommend that Enhanced Due Diligence be conducted and specify \
what additional steps or documentation should be obtained.

**9. CDD Case Narrative**
Draft a structured case narrative that summarises: who the customer is, what risk factors were identified, \
how each was assessed or mitigated, and a recommended decision (Approve / Decline / Escalate for further \
review). Clearly label this as a draft pending analyst review.

**10. Ongoing Monitoring Notes**
Based on the risk profile, recommend a review frequency and note any specific transaction types or \
behaviours that should be monitored going forward.

Behavioural guidelines:
- Always cite the source of your findings (e.g., document provided, public registry, screening result).
- If information is insufficient to complete a section, clearly state what is missing and why it matters.
- Never fabricate data, infer ownership without evidence, or present assumptions as facts.
- Flag anything unusual, inconsistent, or potentially high-risk rather than smoothing it over.
- Use plain, professional language suitable for a compliance case file.
- Always remind the analyst that your output is a draft and that final decisions are their responsibility.\
"""


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


def init_session_state():
    """Initialise Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def encode_file(uploaded_file) -> tuple[str, str]:
    """Read an uploaded file and return (base64_encoded_string, mime_type)."""
    raw = uploaded_file.read()
    return base64.standard_b64encode(raw).decode("utf-8"), uploaded_file.type


def build_api_content(text: str, attachments: list) -> list | str:
    """
    Build the content field for an API message.
    Returns a plain string when there are no attachments,
    or a list of typed content blocks when attachments are present.
    """
    if not attachments:
        return text

    blocks = []
    for name, b64_data, mime in attachments:
        if mime == "application/pdf":
            blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64_data,
                },
                "title": name,
            })
        elif mime.startswith("image/"):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": b64_data,
                },
            })

    blocks.append({"type": "text", "text": text})
    return blocks


def render_message_content(content):
    """Display a stored message's content in the Streamlit UI."""
    if isinstance(content, str):
        st.markdown(content)
        return

    for block in content:
        btype = block.get("type")
        if btype == "text":
            st.markdown(block["text"])
        elif btype == "document":
            st.caption(f"📄 *{block.get('title', 'Document')}* attached")
        elif btype == "image":
            st.caption("🖼️ Image attached")


def export_conversation() -> str:
    """Serialise the conversation history as plain text for download."""
    parts = []
    for msg in st.session_state.messages:
        label = "ANALYST" if msg["role"] == "user" else "CDD ASSISTANT"
        content = msg["content"]
        if isinstance(content, str):
            body = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                else:
                    text_parts.append(
                        f"[{block.get('title', block.get('type', 'attachment'))}]"
                    )
            body = "\n".join(text_parts)
        else:
            body = str(content)
        parts.append(f"{'─' * 60}\n{label}\n{'─' * 60}\n{body}")
    return "\n\n".join(parts)


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

        with st.expander("📖 How to use this tool", expanded=True):
            st.markdown(
                """
                **Step 1 — Describe the customer**
                In the chat, tell the assistant who the customer is:
                - Individual or company?
                - Country of incorporation / residence
                - Nature of the business relationship
                - Source of funds or wealth (if known)

                **Step 2 — Attach documents** *(optional)*
                Upload passports, corporate certificates, ownership charts,
                or other KYC documents below. They will be sent with your
                next message.

                **Step 3 — Review AI output**
                The assistant works through each CDD step. Ask follow-up
                questions or request a full case narrative at any time.

                **Step 4 — Export**
                Download the full conversation as a text file to include
                in your case file.

                ---
                ⚠️ All outputs are drafts. Final decisions remain
                the analyst's responsibility.
                """
            )

        st.divider()

        st.markdown("**📎 Attach Documents**")
        uploaded_files = st.file_uploader(
            "Upload KYC documents",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help=(
                "Supported formats: PDF, PNG, JPG. "
                "Documents are sent with your next message."
            ),
            label_visibility="collapsed",
        )
        if uploaded_files:
            for f in uploaded_files:
                st.caption(f"✅ {f.name}")

        st.divider()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🗑️ New Case", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()
        with btn_col2:
            if st.session_state.messages:
                st.download_button(
                    label="💾 Export",
                    data=export_conversation(),
                    file_name="cdd_review.txt",
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
    st.markdown(
        "Describe the customer and attach any relevant documents. "
        "The assistant will work through each CDD step and flag areas of concern."
    )

    if not st.session_state.messages:
        st.info(
            "💡 **Example prompt to get started:**\n\n"
            "*\"Please begin a CDD review for a new client: Meridian Trading Ltd, "
            "incorporated in the British Virgin Islands. The relationship purpose is "
            "trade finance, with an expected monthly transaction volume of USD 500,000. "
            "The director is John Smith (British national). I've attached the certificate "
            "of incorporation and Mr Smith's passport.\"*"
        )

    # Render existing conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message_content(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Describe the customer or ask a follow-up question…"):

        # Encode any attached files
        attachments = []
        if uploaded_files:
            for f in uploaded_files:
                f.seek(0)
                b64, mime = encode_file(f)
                attachments.append((f.name, b64, mime))

        user_content = build_api_content(prompt, attachments)

        # Store and display the user message
        st.session_state.messages.append({"role": "user", "content": user_content})
        with st.chat_message("user"):
            render_message_content(user_content)

        # Stream Claude's response
        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""

            try:
                with client.messages.stream(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    thinking={"type": "adaptive"},
                    messages=st.session_state.messages,
                ) as stream:
                    for text_chunk in stream.text_stream:
                        full_response += text_chunk
                        response_box.markdown(full_response + "▌")

                response_box.markdown(full_response)

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                # Remove the incomplete user message to keep conversation valid
                st.session_state.messages.pop()
                st.stop()

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
