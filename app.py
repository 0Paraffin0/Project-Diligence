"""
CDD Assistant — AI-Powered Customer Due Diligence Tool
Researches via OpenSanctions + web, scores risk across four components,
and produces an 8-section structured report ready for analyst review.
"""

import json
import os
import re
import time
from datetime import datetime, timedelta

import requests
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000
MAX_TOOL_CALLS = 25
OPENSANCTIONS_URL = "https://api.opensanctions.org/search/default"

# ── Risk scoring utilities ─────────────────────────────────────────────────────

RISK_WEIGHTS = {
    "customer_risk":        0.40,
    "matter_risk":          0.35,
    "jurisdiction_risk":    0.20,
    "delivery_channel_risk":0.05,
}
CUSTOMER_SF_WEIGHTS = {
    "sanctions_exposure":    0.30,
    "pep_exposure":          0.25,
    "adverse_media":         0.20,
    "ownership_complexity":  0.15,
    "identity_verification": 0.10,
}
MATTER_SF_WEIGHTS = {
    "matter_type":           0.40,
    "source_of_funds":       0.40,
    "transaction_modifier":  0.20,
}


def score_level(s: int) -> str:
    if s <= 25: return "Low"
    if s <= 50: return "Medium"
    if s <= 75: return "High"
    return "Critical"


def score_color(s: int) -> str:
    if s <= 25: return "#2e7d32"
    if s <= 50: return "#e6a817"
    if s <= 75: return "#e07b30"
    return "#c62828"


def score_bg(s: int) -> str:
    if s <= 25: return "#e8f5e9"
    if s <= 50: return "#fffde7"
    if s <= 75: return "#fff3e0"
    return "#ffebee"


def calc_weighted_overall(components: dict) -> int:
    cr = components.get("customer_risk", {}).get("score", 0)
    mr = components.get("matter_risk", {}).get("score", 0)
    jr = components.get("jurisdiction_risk", 0)
    dr = components.get("delivery_channel_risk", 0)
    return round(cr * 0.40 + mr * 0.35 + jr * 0.20 + dr * 0.05)


def calc_weighted_customer(sf: dict) -> int:
    return round(sum(sf.get(k, 0) * w for k, w in CUSTOMER_SF_WEIGHTS.items()))


def calc_weighted_matter(sf: dict) -> int:
    return round(sum(sf.get(k, 0) * w for k, w in MATTER_SF_WEIGHTS.items()))


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert financial crime compliance analyst specialising in Customer Due Diligence (CDD).

═══ RESEARCH WORKFLOW ═══════════════════════════════════════════════════════════

STEP 1 — sanctions_screen: Call for the entity name, then separately for each director/UBO found.
STEP 2 — web_search: Run targeted searches for company registry, directors, UBOs, adverse media,
         source of funds, business activity, and country risk (FATF status, corruption indices).
STEP 3 — fetch_webpage: Read full content of the most relevant URLs found.
Run AT LEAST 12 tool calls total.

═══ SCORING GUIDANCE (0 = lowest risk, 100 = highest risk) ═════════════════════

identity_verification : 0=fully verified  25=minor gaps  50=significant gaps  75=major gaps  100=absent/unverifiable
pep_exposure          : 0=none  25=remote/historical  50=associate  75=close family/former  100=direct current PEP
sanctions_exposure    : 0=clear  30=name similarity only  70=unconfirmed match  90=strong match  100=confirmed hit
adverse_media         : 0=none  25=minor historical  50=significant reports  75=serious allegations  100=convicted
ownership_complexity  : 0=simple/clear  25=one holding co  50=multi-layer traceable  75=offshore unclear  100=opaque
matter_type           : 0=simple retail  25=standard biz  50=trade finance/investment  75=real estate/HV  100=crypto/shell
source_of_funds       : 0=verified payroll  25=documented biz revenue  50=some gaps  75=unverified claim  100=unexplained/refused
transaction_modifier  : 0=none  25=above-average volume  50=cash element  75=structuring indicators  100=clear obfuscation
jurisdiction_risk     : 0=UK/US/EU/AU/CA/JP  25=low-risk  50=standard  75=FATF grey list  100=FATF black/sanctioned
delivery_channel_risk : 0=face-to-face  25=video-verified  50=digital  75=third-party introduced  100=fully anonymous

confidence_score: your confidence in the accuracy of the findings (0–100).
Lower confidence if key information could not be verified or was unavailable.

═══ REQUIRED OUTPUT FORMAT ══════════════════════════════════════════════════════

After completing all research, output EXACTLY the following block and nothing else.
All string values must be valid JSON (escape quotes, no trailing commas).

<CDD_REPORT_JSON>
{
  "subject_name": "string",
  "review_date": "YYYY-MM-DD",
  "executive_summary": "2-3 sentences summarising key findings and overall risk determination.",

  "risk_scoring": {
    "overall_risk_score": 0,
    "overall_risk_level": "Low|Medium|High|Critical",
    "confidence_score": 0,
    "components": {
      "customer_risk": {
        "score": 0,
        "sub_factors": {
          "identity_verification": 0,
          "pep_exposure": 0,
          "sanctions_exposure": 0,
          "adverse_media": 0,
          "ownership_complexity": 0
        }
      },
      "matter_risk": {
        "score": 0,
        "sub_factors": {
          "matter_type": 0,
          "source_of_funds": 0,
          "transaction_modifier": 0
        }
      },
      "jurisdiction_risk": 0,
      "delivery_channel_risk": 0
    },
    "escalation_flags": [],
    "auto_override": false,
    "recommended_action": "string"
  },

  "risk_categories": {
    "identity": {
      "score": 0,
      "status": "string",
      "findings": "string",
      "verified_fields": [],
      "missing_fields": []
    },
    "sanctions": {
      "score": 0,
      "status": "string",
      "findings": "string",
      "matches": [
        {"name": "string", "lists": [], "match_score": 0, "reason": "string", "listing_date": "string"}
      ],
      "lists_checked": []
    },
    "pep": {
      "score": 0,
      "status": "string",
      "findings": "string",
      "connections": [
        {"name": "string", "role": "string", "relationship": "string", "jurisdiction": "string"}
      ]
    },
    "adverse_media": {
      "score": 0,
      "status": "string",
      "findings": "string",
      "articles": [
        {"headline": "string", "publication": "string", "date": "string", "url": "string", "summary": "string", "severity": "Low|Medium|High"}
      ]
    },
    "geography": {
      "score": 0,
      "status": "string",
      "findings": "string",
      "jurisdictions": [
        {"name": "string", "role": "Incorporation|Operations|UBO Nationality", "fatf_status": "string", "risk_rating": "Low|Medium|High|Critical"}
      ]
    },
    "ownership": {
      "score": 0,
      "status": "string",
      "findings": "string",
      "ubos": [
        {"name": "string", "ownership_pct": "string", "nationality": "string", "verified": false}
      ],
      "structure_description": "string"
    }
  },

  "customer_identification": {
    "legal_name": "string",
    "trading_name": null,
    "registration_number": null,
    "incorporation_date": null,
    "jurisdiction": "string",
    "registered_address": null,
    "directors": [],
    "company_type": null,
    "business_activity": "string"
  },

  "source_references": [
    {"type": "OpenSanctions|Web Search|Webpage|Analyst Provided", "query": "string", "title": "string", "url": null, "relevant_finding": "string"}
  ],

  "matching_logic": "string explaining how name matches were evaluated and why each was confirmed or dismissed",

  "recommended_action": {
    "action": "Approve|Approve with Conditions|Escalate to Enhanced Due Diligence|Decline|Refer to MLRO",
    "rationale": "string",
    "conditions": [],
    "edd_requirements": []
  },

  "ongoing_monitoring": {
    "review_frequency": "string",
    "transaction_flags": [],
    "next_review_date": "string"
  },

  "analyst_narrative": "3-4 paragraph draft: (1) who is the customer and what they do, (2) key risk factors identified, (3) how each risk was assessed or mitigated, (4) recommended decision with justification. End with: DRAFT — Pending Analyst Review."
}
</CDD_REPORT_JSON>
"""

# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "sanctions_screen",
        "description": (
            "Screen a name against the OpenSanctions database — 100+ global sanctions lists "
            "(OFAC SDN, UN, EU, HMT, DFAT, SECO, and more) plus PEP lists and enforcement "
            "actions. ALWAYS call this first for any name being reviewed. "
            "Call once for the entity, then separately for each known director or UBO."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name of the person or company to screen."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the internet for information relevant to a CDD review. "
            "Use for company registry data, adverse media, ownership, and country risk. "
            "Each call should target one specific aspect. Follow up with fetch_webpage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Specific search query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_webpage",
        "description": "Fetch the full text of a specific URL. Use after web_search to read articles, registry entries, or news reports.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch."}
            },
            "required": ["url"],
        },
    },
]

# ── Tool execution ─────────────────────────────────────────────────────────────

def get_client() -> Anthropic:
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not found.**\n\n"
            "Add it to `.env` locally or Streamlit Cloud Secrets."
        )
        st.stop()
    return Anthropic(api_key=api_key)


def sanctions_screen(name: str) -> str:
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
                f"OpenSanctions — No matches for '{name}'.\n"
                f"Lists checked: OFAC SDN, OFAC Non-SDN Consolidated, UN Security Council, "
                f"EU Financial Sanctions, HMT UK, DFAT Australia, SECO Switzerland, "
                f"OSFI Canada, and 100+ additional global lists plus PEP datasets."
            )

        lines = [f"OpenSanctions — {total} result(s) for '{name}':\n"]
        for r in results[:12]:
            props = r.get("properties", {})
            datasets = r.get("datasets", [])
            entry = (
                f"  Name    : {r.get('caption', 'N/A')}\n"
                f"  Type    : {r.get('schema', '?')}\n"
                f"  Score   : {round(r.get('score', 0), 2)}\n"
                f"  Lists   : {', '.join(datasets[:10])}\n"
            )
            for field, label in [
                ("topics", "Topics"), ("alias", "AKAs"), ("birthDate", "DOB"),
                ("nationality", "Nationality"), ("country", "Countries"),
                ("listingDate", "Listed"), ("reason", "Reason"),
            ]:
                vals = props.get(field, [])
                if vals:
                    entry += f"  {label:<8}: {', '.join(str(v) for v in vals[:5])[:200]}\n"
            lines.append(entry)

        if total > 12:
            lines.append(f"  … {total - 12} more result(s). Narrow the name if needed.")
        return "\n".join(lines)

    except requests.HTTPError as exc:
        return f"OpenSanctions API error {exc.response.status_code}: {exc.response.text[:300]}"
    except Exception as exc:
        return f"OpenSanctions error: {exc}"


def run_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        if not results:
            return "No results."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Search error: {exc}"


def fetch_webpage(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CDD-Research/1.0)"}
        resp = requests.get(url, timeout=12, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = "\n".join(
            ln for ln in soup.get_text(separator="\n", strip=True).splitlines() if ln.strip()
        )
        return text[:4000] + ("\n\n[Page truncated]" if len(text) > 4000 else "")
    except Exception as exc:
        return f"Could not fetch page: {exc}"


# ── Agentic research loop ──────────────────────────────────────────────────────

def run_cdd_research(client: Anthropic, subject: str, context: str, status_box):
    """
    Runs the agentic loop. Returns (parsed_report: dict|None, raw_text: str, audit_log: list).
    """
    user_msg = f"Subject to review: {subject}"
    if context.strip():
        user_msg += f"\n\nAdditional analyst context:\n{context.strip()}"

    messages = [{"role": "user", "content": user_msg}]
    audit_log = []
    actions = []
    session_start = datetime.now()

    with status_box:
        progress = st.empty()
        log_box  = st.empty()

        while len(audit_log) <= MAX_TOOL_CALLS:
            progress.info("Researching…")

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                progress.success("Research complete — building report…")
                raw = next((b.text for b in response.content if b.type == "text"), "")
                parsed = parse_report_json(raw)
                if parsed is not None:
                    return parsed, raw, audit_log, session_start

                # Claude wrote a narrative instead of JSON — ask again explicitly
                progress.info("Formatting structured report…")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your research is complete. Now produce the structured report.\n"
                        "Output ONLY the JSON block — start with <CDD_REPORT_JSON> "
                        "and end with </CDD_REPORT_JSON>. No other text before or after."
                    ),
                })
                json_resp = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    # No tools — we only want text output here
                )
                raw2 = next((b.text for b in json_resp.content if b.type == "text"), raw)
                parsed2 = parse_report_json(raw2)
                return parsed2, raw2, audit_log, session_start

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    t_start = time.time()

                    if block.name == "sanctions_screen":
                        name = block.input["name"]
                        actions.append(f"🛡️ Sanctions: {name}")
                        progress.info(f"Sanctions & PEP screening: *{name}*")
                        result = sanctions_screen(name)

                    elif block.name == "web_search":
                        query = block.input["query"]
                        actions.append(f"🔍 {query}")
                        progress.info(f"Searching: *{query}*")
                        result = run_web_search(query)

                    elif block.name == "fetch_webpage":
                        url = block.input["url"]
                        actions.append(f"📄 {url[:70]}")
                        progress.info(f"Reading: *{url[:80]}*")
                        result = fetch_webpage(url)

                    else:
                        result = "Unknown tool."

                    duration_ms = round((time.time() - t_start) * 1000)
                    audit_log.append({
                        "timestamp":   datetime.now().strftime("%H:%M:%S"),
                        "tool":        block.name,
                        "input":       block.input,
                        "result_preview": result[:200].replace("\n", " "),
                        "duration_ms": duration_ms,
                    })

                    log_box.markdown(
                        "**Research activity:**\n" +
                        "\n".join(f"- {a}" for a in actions[-14:])
                    )
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": results})
            else:
                break

        raw = next((b.text for b in response.content if b.type == "text"), "")
        parsed = parse_report_json(raw)
        if parsed is None:
            # Attempt JSON extraction even from a non-end_turn response
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": (
                    "Your research is complete. Now produce the structured report.\n"
                    "Output ONLY the JSON block — start with <CDD_REPORT_JSON> "
                    "and end with </CDD_REPORT_JSON>. No other text before or after."
                ),
            })
            json_resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            raw = next((b.text for b in json_resp.content if b.type == "text"), raw)
            parsed = parse_report_json(raw)
        return parsed, raw, audit_log, session_start


# ── JSON extraction / parsing ──────────────────────────────────────────────────

def _repair_and_parse(blob: str) -> dict | None:
    """Try progressively more aggressive repairs before giving up."""
    blob = blob.strip()
    # Strip markdown code fences that Claude sometimes wraps JSON in
    blob = re.sub(r"^```(?:json)?\s*", "", blob)
    blob = re.sub(r"\s*```$", "", blob)
    blob = blob.strip()

    # Pass 1 — raw
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass

    # Pass 2 — remove trailing commas before } or ]
    blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
    try:
        return json.loads(blob2)
    except json.JSONDecodeError:
        pass

    # Pass 3 — also collapse literal \n / \t that sometimes appear unescaped
    blob3 = re.sub(r"(?<!\\)\n", " ", blob2)
    try:
        return json.loads(blob3)
    except json.JSONDecodeError:
        pass

    return None


def parse_report_json(raw: str) -> dict | None:
    # Strategy 1: look for the expected XML tags
    match = re.search(r"<CDD_REPORT_JSON>(.*?)</CDD_REPORT_JSON>", raw, re.DOTALL)
    if match:
        result = _repair_and_parse(match.group(1))
        if result is not None:
            return result

    # Strategy 2: Claude sometimes forgets the tags but still outputs valid JSON —
    # find the outermost { … } block that contains "subject_name"
    match2 = re.search(r'\{[^{}]*"subject_name".*\}', raw, re.DOTALL)
    if match2:
        result = _repair_and_parse(match2.group(0))
        if result is not None:
            return result

    return None


# ── CSS ────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    .score-card {
        border-radius: 10px; padding: 18px 12px; text-align: center;
        margin: 4px; border-width: 0 0 0 5px; border-style: solid;
    }
    .score-num  { font-size: 42px; font-weight: 800; line-height: 1; }
    .score-lvl  { font-size: 14px; font-weight: 700; margin-top: 2px; }
    .score-lbl  { font-size: 11px; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }
    .bar-row    { margin-bottom: 10px; }
    .bar-label  { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 3px; }
    .bar-track  { background: #eee; border-radius: 4px; height: 9px; }
    .bar-fill   { height: 9px; border-radius: 4px; }
    .flag-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600; margin: 2px;
        background: #ffebee; color: #c62828; border: 1px solid #c62828;
    }
    .action-box {
        border-radius: 10px; padding: 20px 24px;
        border-left: 6px solid; margin: 8px 0;
    }
    .source-row { font-size: 12px; padding: 6px 0; border-bottom: 1px solid #f0f0f0; }
    .audit-row  {
        font-size: 11px; border-left: 3px solid #e0e0e0;
        padding: 4px 0 4px 10px; margin: 3px 0; color: #444;
    }
    .cat-header { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
    .match-card {
        background: #fff3e0; border-left: 4px solid #e07b30;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0; font-size: 13px;
    }
    .article-card {
        border-left: 4px solid #bbb; padding: 8px 12px;
        margin: 6px 0; font-size: 13px; background: #fafafa; border-radius: 4px;
    }
    .article-high { border-color: #c62828; background: #ffebee; }
    .article-med  { border-color: #e07b30; background: #fff3e0; }
    .section-divider { border: none; border-top: 2px solid #f0f0f0; margin: 32px 0 20px; }
    </style>
    """, unsafe_allow_html=True)


# ── Render helpers ─────────────────────────────────────────────────────────────

def score_card_html(score: int, label: str) -> str:
    c = score_color(score); bg = score_bg(score); lvl = score_level(score)
    return (
        f'<div class="score-card" style="background:{bg}; border-color:{c};">'
        f'<div class="score-num" style="color:{c};">{score}</div>'
        f'<div class="score-lvl" style="color:{c};">{lvl}</div>'
        f'<div class="score-lbl">{label}</div>'
        f'</div>'
    )


def bar_html(label: str, score: int, weight_pct: str = "") -> str:
    c = score_color(score); lvl = score_level(score)
    right = f'{score} — {lvl}' + (f' · weight {weight_pct}' if weight_pct else '')
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label"><span>{label}</span>'
        f'<span style="color:{c};font-weight:600;">{right}</span></div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{score}%;background:{c};"></div></div>'
        f'</div>'
    )


def action_color(action: str) -> str:
    a = action.lower()
    if "decline" in a or "mlro" in a: return "#c62828"
    if "enhanced" in a or "escalate" in a: return "#e07b30"
    if "condition" in a: return "#e6a817"
    return "#2e7d32"


# ── Section renderers ──────────────────────────────────────────────────────────

def render_executive_summary(d: dict):
    rs = d.get("risk_scoring", {})
    overall  = rs.get("overall_risk_score", 0)
    conf     = rs.get("confidence_score", 0)
    comps    = rs.get("components", {})
    cr_score = comps.get("customer_risk", {}).get("score", 0)
    mr_score = comps.get("matter_risk",   {}).get("score", 0)
    jr_score = comps.get("jurisdiction_risk", 0)
    dr_score = comps.get("delivery_channel_risk", 0)
    flags    = rs.get("escalation_flags", [])
    rec_act  = rs.get("recommended_action", "")

    st.markdown("## 1 · Executive Risk Summary")
    st.markdown(f'<p style="font-size:15px;color:#444;">{d.get("executive_summary","")}</p>',
                unsafe_allow_html=True)

    # Score cards
    cols = st.columns(5)
    labels = ["Overall Risk", "Customer Risk", "Matter Risk", "Jurisdiction", "Delivery Channel"]
    scores = [overall, cr_score, mr_score, jr_score, dr_score]
    for col, lbl, sc in zip(cols, labels, scores):
        col.markdown(score_card_html(sc, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence
    conf_c = score_color(100 - conf)  # high confidence = green (inverted)
    st.markdown(
        f'<div style="font-size:13px;color:#555;">Confidence in findings: '
        f'<strong style="color:{conf_c};">{conf} / 100</strong></div>',
        unsafe_allow_html=True
    )

    # Escalation flags
    if flags:
        badges = "".join(f'<span class="flag-badge">⚠ {f}</span>' for f in flags)
        st.markdown(f'<div style="margin:10px 0;">{badges}</div>', unsafe_allow_html=True)

    # Recommended action (top-level)
    if rec_act:
        ac = action_color(rec_act)
        st.markdown(
            f'<div style="background:{score_bg(overall)};border-left:5px solid {ac};'
            f'padding:12px 16px;border-radius:8px;font-weight:700;font-size:15px;color:{ac};">'
            f'➜ {rec_act}</div>',
            unsafe_allow_html=True,
        )


def render_risk_category_panels(d: dict):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 2 · Risk Category Panels")
    cats = d.get("risk_categories", {})

    tab_labels = ["🪪 Identity", "🛡️ Sanctions", "👤 PEP",
                  "📰 Adverse Media", "🌍 Geography", "🏗️ Ownership"]
    tabs = st.tabs(tab_labels)

    # ── Identity ──
    with tabs[0]:
        c = cats.get("identity", {})
        sc = c.get("score", 0)
        st.markdown(bar_html("Identity Verification Risk", sc), unsafe_allow_html=True)
        st.markdown(f"**Status:** {c.get('status','—')}")
        st.markdown(c.get("findings", "No findings recorded."))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Verified fields**")
            for f in c.get("verified_fields", []) or ["None identified"]:
                st.markdown(f"- ✅ {f}")
        with col2:
            st.markdown("**Missing / outstanding**")
            for f in c.get("missing_fields", []) or ["None identified"]:
                st.markdown(f"- ❌ {f}")

    # ── Sanctions ──
    with tabs[1]:
        c = cats.get("sanctions", {})
        sc = c.get("score", 0)
        st.markdown(bar_html("Sanctions Exposure", sc), unsafe_allow_html=True)
        st.markdown(f"**Status:** {c.get('status','—')}")
        st.markdown(c.get("findings", ""))
        matches = c.get("matches", [])
        if matches:
            st.markdown("**Matches found:**")
            for m in matches:
                mc = score_color(m.get("match_score", 0))
                st.markdown(
                    f'<div class="match-card">'
                    f'<strong>{m.get("name","")}</strong> &nbsp;'
                    f'<span style="color:{mc};font-weight:700;">Match score: {m.get("match_score",0)}</span><br>'
                    f'Lists: {", ".join(m.get("lists",[]))}<br>'
                    f'Reason: {m.get("reason","—")}<br>'
                    f'Listed: {m.get("listing_date","—")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No sanctions matches found.")
        lists = c.get("lists_checked", [])
        if lists:
            with st.expander("Lists checked"):
                st.markdown(", ".join(lists))

    # ── PEP ──
    with tabs[2]:
        c = cats.get("pep", {})
        sc = c.get("score", 0)
        st.markdown(bar_html("PEP Exposure", sc), unsafe_allow_html=True)
        st.markdown(f"**Status:** {c.get('status','—')}")
        st.markdown(c.get("findings", ""))
        conns = c.get("connections", [])
        if conns:
            st.markdown("**PEP connections:**")
            for cn in conns:
                st.markdown(
                    f"- **{cn.get('name','')}** — {cn.get('role','')} "
                    f"({cn.get('relationship','')}, {cn.get('jurisdiction','')})"
                )
        else:
            st.success("No PEP connections identified.")

    # ── Adverse Media ──
    with tabs[3]:
        c = cats.get("adverse_media", {})
        sc = c.get("score", 0)
        st.markdown(bar_html("Adverse Media", sc), unsafe_allow_html=True)
        st.markdown(f"**Status:** {c.get('status','—')}")
        st.markdown(c.get("findings", ""))
        articles = c.get("articles", [])
        if articles:
            for a in articles:
                sev = a.get("severity", "Low")
                cls = "article-high" if sev == "High" else ("article-med" if sev == "Medium" else "article-card")
                url = a.get("url") or ""
                link = f'<a href="{url}" target="_blank">{a.get("headline","")}</a>' if url else a.get("headline","")
                st.markdown(
                    f'<div class="article-card {cls}">'
                    f'<strong>{link}</strong><br>'
                    f'<span style="font-size:11px;color:#666;">'
                    f'{a.get("publication","")}&nbsp;·&nbsp;{a.get("date","")}&nbsp;·&nbsp;Severity: {sev}</span><br>'
                    f'{a.get("summary","")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No significant adverse media found.")

    # ── Geography ──
    with tabs[4]:
        c = cats.get("geography", {})
        sc = c.get("score", 0)
        st.markdown(bar_html("Jurisdiction Risk", sc), unsafe_allow_html=True)
        st.markdown(f"**Status:** {c.get('status','—')}")
        st.markdown(c.get("findings", ""))
        jurisdictions = c.get("jurisdictions", [])
        if jurisdictions:
            rows = []
            for j in jurisdictions:
                r = j.get("risk_rating", "?")
                rc = score_color({"Low":15,"Medium":40,"High":65,"Critical":90}.get(r, 50))
                rows.append(
                    f'<tr><td>{j.get("name","")}</td><td>{j.get("role","")}</td>'
                    f'<td>{j.get("fatf_status","—")}</td>'
                    f'<td style="color:{rc};font-weight:700;">{r}</td></tr>'
                )
                st.markdown(
                    '<table style="width:100%;font-size:13px;">'
                    '<thead><tr><th>Jurisdiction</th><th>Role</th><th>FATF Status</th><th>Risk</th></tr></thead>'
                    '<tbody>' + "".join(rows) + '</tbody></table>',
                    unsafe_allow_html=True,
                )

    # ── Ownership ──
    with tabs[5]:
        c = cats.get("ownership", {})
        sc = c.get("score", 0)
        st.markdown(bar_html("Ownership Complexity", sc), unsafe_allow_html=True)
        st.markdown(f"**Status:** {c.get('status','—')}")
        st.markdown(c.get("findings", ""))
        st.markdown(f"**Structure:** {c.get('structure_description','—')}")
        ubos = c.get("ubos", [])
        if ubos:
            st.markdown("**Identified UBOs / Beneficial Owners:**")
            for u in ubos:
                verified = "✅ Verified" if u.get("verified") else "❌ Unverified"
                st.markdown(
                    f"- **{u.get('name','')}** — {u.get('ownership_pct','?')}% "
                    f"· {u.get('nationality','?')} · {verified}"
                )
        else:
            st.warning("No UBOs identified.")


def render_risk_scoring(d: dict):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 3 · Transparent Risk Scoring")
    rs = d.get("risk_scoring", {})
    comps = rs.get("components", {})

    cr = comps.get("customer_risk", {})
    mr = comps.get("matter_risk", {})
    jr = comps.get("jurisdiction_risk", 0)
    dr = comps.get("delivery_channel_risk", 0)
    cr_sf = cr.get("sub_factors", {})
    mr_sf = mr.get("sub_factors", {})

    # Re-calculate scores from sub-factors for transparency
    cr_calc = calc_weighted_customer(cr_sf)
    mr_calc = calc_weighted_matter(mr_sf)
    overall_calc = calc_weighted_overall(comps)
    overall_ai   = rs.get("overall_risk_score", 0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Customer Risk Sub-factors")
        sf_labels = {
            "sanctions_exposure":    ("Sanctions Exposure",    "30%"),
            "pep_exposure":          ("PEP Exposure",          "25%"),
            "adverse_media":         ("Adverse Media",         "20%"),
            "ownership_complexity":  ("Ownership Complexity",  "15%"),
            "identity_verification": ("Identity Verification", "10%"),
        }
        html = ""
        for key, (lbl, wgt) in sf_labels.items():
            html += bar_html(lbl, cr_sf.get(key, 0), wgt)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(
            f"**AI score:** {cr.get('score',0)} &nbsp;|&nbsp; "
            f"**Formula:** {cr_calc} (weighted avg of sub-factors)",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Matter Risk Sub-factors")
        mf_labels = {
            "matter_type":          ("Matter / Relationship Type", "40%"),
            "source_of_funds":      ("Source of Funds",            "40%"),
            "transaction_modifier": ("Transaction Modifier",       "20%"),
        }
        html = ""
        for key, (lbl, wgt) in mf_labels.items():
            html += bar_html(lbl, mr_sf.get(key, 0), wgt)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(
            f"**AI score:** {mr.get('score',0)} &nbsp;|&nbsp; "
            f"**Formula:** {mr_calc} (weighted avg of sub-factors)",
            unsafe_allow_html=True,
        )

    st.markdown("#### Component Weights → Overall Score")
    cr_s = cr.get('score', 0); mr_s = mr.get('score', 0)
    formula = (
        f"({cr_s} × 40%) + ({mr_s} × 35%) + ({jr} × 20%) + ({dr} × 5%) = **{overall_calc}**"
    )
    st.markdown(
        bar_html("Jurisdiction Risk", jr, "20%") +
        bar_html("Delivery Channel Risk", dr, "5%"),
        unsafe_allow_html=True,
    )
    st.markdown(f"**Formula:** {formula}")
    if abs(overall_calc - overall_ai) > 5:
        st.info(
            f"Note: AI overall score is **{overall_ai}** vs formula-derived **{overall_calc}**. "
            f"The AI may have applied qualitative adjustments."
        )


def render_source_references(d: dict):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 4 · Source References")
    refs = d.get("source_references", [])
    ml   = d.get("matching_logic", "")

    if ml:
        with st.expander("Matching logic & name disambiguation"):
            st.markdown(ml)

    if not refs:
        st.info("No source references recorded.")
        return

    type_icons = {
        "OpenSanctions": "🛡️", "Web Search": "🔍",
        "Webpage": "📄", "Analyst Provided": "📋",
    }
    for r in refs:
        icon = type_icons.get(r.get("type", ""), "•")
        url = r.get("url")
        title = r.get("title") or r.get("query") or "—"
        link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
        st.markdown(
            f'<div class="source-row">'
            f'{icon} <strong>{r.get("type","")}</strong> &nbsp;·&nbsp; {link}<br>'
            f'<span style="color:#666;font-size:11px;">{r.get("relevant_finding","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_recommended_action(d: dict):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 5 · Recommended Action")
    ra = d.get("recommended_action", {})
    action = ra.get("action", "—")
    ac = action_color(action)
    bg = score_bg({"Approve": 15, "Approve with Conditions": 42, "Decline": 90,
                   "Escalate to Enhanced Due Diligence": 68, "Refer to MLRO": 80}.get(action, 60))

    st.markdown(
        f'<div class="action-box" style="background:{bg};border-color:{ac};">'
        f'<div style="font-size:20px;font-weight:800;color:{ac};">➜ {action}</div>'
        f'<div style="margin-top:10px;font-size:14px;">{ra.get("rationale","")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        conds = ra.get("conditions", [])
        if conds:
            st.markdown("**Conditions**")
            for c in conds:
                st.markdown(f"- {c}")
    with col2:
        edd = ra.get("edd_requirements", [])
        if edd:
            st.markdown("**EDD Requirements**")
            for e in edd:
                st.markdown(f"- {e}")

    mon = d.get("ongoing_monitoring", {})
    if mon:
        st.markdown("**Ongoing Monitoring**")
        st.markdown(f"Review frequency: **{mon.get('review_frequency','—')}** · Next review: **{mon.get('next_review_date','—')}**")
        flags = mon.get("transaction_flags", [])
        if flags:
            st.markdown("Transaction flags to watch:")
            for f in flags:
                st.markdown(f"- 🚩 {f}")


def render_analyst_narrative(d: dict):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 6 · Analyst Narrative  *(editable)*")
    st.caption("Pre-populated from AI draft. Edit as required before saving to case file.")

    draft = d.get("analyst_narrative", "")
    # Initialise editable narrative in session state once per report
    if "narrative_text" not in st.session_state or st.session_state.get("narrative_subject") != d.get("subject_name"):
        st.session_state.narrative_text = draft
        st.session_state.narrative_subject = d.get("subject_name")

    narrative = st.text_area(
        "Case narrative",
        value=st.session_state.narrative_text,
        height=280,
        key="narrative_editor",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("💾 Save", type="primary", use_container_width=True):
            st.session_state.narrative_text = narrative
            st.success("Saved.")
    with col2:
        if narrative != draft:
            st.caption("⚠️ Unsaved changes")

    return narrative


def render_customer_id(d: dict):
    ci = d.get("customer_identification", {})
    if not ci:
        return
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## Customer Identification")
    cols = st.columns(3)
    fields = [
        ("Legal name",          ci.get("legal_name")),
        ("Trading name",        ci.get("trading_name")),
        ("Reg. number",         ci.get("registration_number")),
        ("Jurisdiction",        ci.get("jurisdiction")),
        ("Incorporation date",  ci.get("incorporation_date")),
        ("Company type",        ci.get("company_type")),
        ("Registered address",  ci.get("registered_address")),
        ("Business activity",   ci.get("business_activity")),
    ]
    for i, (lbl, val) in enumerate(fields):
        cols[i % 3].markdown(f"**{lbl}**  \n{val or '—'}")
    dirs = ci.get("directors", [])
    if dirs:
        st.markdown(f"**Directors / Officers:** {', '.join(dirs)}")


def render_audit_log(audit_log: list, session_start: datetime, subject: str):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 7 · Full Audit Log")
    elapsed = (datetime.now() - session_start).seconds
    tool_counts = {}
    for e in audit_log:
        tool_counts[e["tool"]] = tool_counts.get(e["tool"], 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total tool calls", len(audit_log))
    col2.metric("Sanctions screens", tool_counts.get("sanctions_screen", 0))
    col3.metric("Web searches",      tool_counts.get("web_search", 0))
    col4.metric("Pages read",        tool_counts.get("fetch_webpage", 0))

    st.caption(
        f"Session started: {session_start.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"Elapsed: {elapsed}s · Model: {MODEL}"
    )

    tool_icons = {"sanctions_screen": "🛡️", "web_search": "🔍", "fetch_webpage": "📄"}
    with st.expander("View full audit log", expanded=False):
        for i, entry in enumerate(audit_log, 1):
            icon = tool_icons.get(entry["tool"], "•")
            inp  = next(iter(entry["input"].values()), "") if entry["input"] else ""
            st.markdown(
                f'<div class="audit-row">'
                f'<strong>#{i}</strong> {icon} <code>{entry["tool"]}</code> '
                f'· <em>{entry["timestamp"]}</em> · {entry["duration_ms"]}ms<br>'
                f'Input: <code>{str(inp)[:120]}</code><br>'
                f'Result: {entry["result_preview"][:150]}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Download full audit log as JSON
    st.download_button(
        "⬇️ Download audit log (JSON)",
        data=json.dumps({
            "subject": subject,
            "session_start": session_start.isoformat(),
            "model": MODEL,
            "entries": audit_log,
        }, indent=2),
        file_name="cdd_audit_log.json",
        mime="application/json",
    )


def build_download_report(d: dict, narrative: str, audit_log: list) -> str:
    rs = d.get("risk_scoring", {})
    comps = rs.get("components", {})
    lines = [
        f"CDD REPORT — {d.get('subject_name', '?')}",
        f"Review date: {d.get('review_date', '?')}",
        f"Generated by CDD Assistant · Model: {MODEL}",
        "=" * 70,
        "",
        "EXECUTIVE SUMMARY",
        d.get("executive_summary", ""),
        "",
        "RISK SCORES",
        f"  Overall            : {rs.get('overall_risk_score',0)} / 100  ({rs.get('overall_risk_level','')})",
        f"  Confidence         : {rs.get('confidence_score',0)} / 100",
        f"  Customer Risk      : {comps.get('customer_risk',{}).get('score',0)}",
        f"  Matter Risk        : {comps.get('matter_risk',{}).get('score',0)}",
        f"  Jurisdiction Risk  : {comps.get('jurisdiction_risk',0)}",
        f"  Delivery Channel   : {comps.get('delivery_channel_risk',0)}",
        f"  Escalation flags   : {', '.join(rs.get('escalation_flags',[]))}",
        "",
        "RECOMMENDED ACTION",
        f"  {d.get('recommended_action',{}).get('action','—')}",
        f"  {d.get('recommended_action',{}).get('rationale','')}",
        "",
        "ANALYST NARRATIVE",
        narrative,
        "",
        "SOURCE REFERENCES",
    ]
    for r in d.get("source_references", []):
        lines.append(f"  [{r.get('type','')}] {r.get('title') or r.get('query','?')} — {r.get('url','')}")
    lines += [
        "",
        "AUDIT LOG",
        f"  Total tool calls: {len(audit_log)}",
        f"  Sanctions screens: {sum(1 for e in audit_log if e['tool']=='sanctions_screen')}",
        f"  Web searches: {sum(1 for e in audit_log if e['tool']=='web_search')}",
        f"  Pages read: {sum(1 for e in audit_log if e['tool']=='fetch_webpage')}",
        "",
        "⚠️ DRAFT — Pending Analyst Review. Not for external distribution.",
    ]
    return "\n".join(lines)


def init_session():
    for k, v in [
        ("report_data", None), ("raw_text", ""), ("audit_log", []),
        ("session_start", None), ("follow_ups", []),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="CDD Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_session()
    client = get_client()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚖️ CDD Assistant")
        st.caption("AI-Assisted Customer Due Diligence")
        st.divider()

        st.info(
            "🛡️ **Sanctions & PEP screening**\n\n"
            "Powered by [OpenSanctions](https://opensanctions.org) — "
            "OFAC, UN, EU, HMT, DFAT, SECO, 100+ lists + PEPs. "
            "No account required."
        )
        st.divider()

        with st.expander("📖 How to use"):
            st.markdown(
                "1. Enter a name or company below\n"
                "2. Add context (jurisdiction, directors, etc.)\n"
                "3. Click **Run CDD Review**\n"
                "4. Review the 7-section scored report\n"
                "5. Edit the analyst narrative\n"
                "6. Download the report and audit log\n\n"
                "⚠️ All outputs are drafts. Analyst review required."
            )

        with st.expander("📊 Scoring methodology"):
            st.markdown(
                "**Weights**\n"
                "- Customer Risk: 40%\n"
                "- Matter Risk: 35%\n"
                "- Jurisdiction Risk: 20%\n"
                "- Delivery Channel: 5%\n\n"
                "**Customer sub-factors**\n"
                "- Sanctions: 30%\n"
                "- PEP: 25%\n"
                "- Adverse Media: 20%\n"
                "- Ownership: 15%\n"
                "- Identity: 10%\n\n"
                "Scores: 0=lowest risk, 100=highest risk\n\n"
                "**Levels**\n"
                "🟢 0–25 Low · 🟡 26–50 Medium\n"
                "🟠 51–75 High · 🔴 76–100 Critical"
            )

        st.divider()
        if st.session_state.report_data:
            if st.button("🗑️ New Review", use_container_width=True, type="secondary"):
                for k in ["report_data", "raw_text", "audit_log", "session_start",
                          "follow_ups", "narrative_text", "narrative_subject"]:
                    st.session_state.pop(k, None)
                st.rerun()

        st.warning(
            "⚠️ **Analyst Review Required**\n\n"
            "AI outputs are draft only. A qualified analyst must review "
            "all findings before any onboarding decision."
        )

    # ── Input ──────────────────────────────────────────────────────────────────
    if not st.session_state.report_data:
        st.markdown("## Customer Due Diligence Review")
        with st.form("cdd_form"):
            subject = st.text_input(
                "Name or company name",
                placeholder="e.g.  Meridian Trading Ltd   or   John Smith",
            )
            with st.expander("Optional: additional context"):
                extra = st.text_area(
                    "Context",
                    placeholder=(
                        "Jurisdiction: British Virgin Islands\n"
                        "Directors: John Smith (British national)\n"
                        "Relationship purpose: Trade finance\n"
                        "Expected monthly volume: USD 500,000\n"
                        "Source of funds: Export revenues\n"
                        "Delivery channel: Remote / digital onboarding"
                    ),
                    height=130,
                    label_visibility="collapsed",
                )
            submitted = st.form_submit_button(
                "🔍 Run CDD Review", use_container_width=True, type="primary"
            )

        if submitted and subject.strip():
            status_box = st.container()
            data, raw, audit_log, t_start = run_cdd_research(
                client, subject.strip(), extra, status_box
            )
            if data is None:
                st.error(
                    "The AI did not return a structured JSON report. "
                    "Raw output is shown below — try re-running."
                )
                st.text(raw)
            else:
                st.session_state.report_data  = data
                st.session_state.raw_text     = raw
                st.session_state.audit_log    = audit_log
                st.session_state.session_start = t_start
                st.rerun()
        elif submitted:
            st.warning("Please enter a name or company name.")
        return

    # ── Report ─────────────────────────────────────────────────────────────────
    d   = st.session_state.report_data
    log = st.session_state.audit_log
    t0  = st.session_state.session_start or datetime.now()
    subject = d.get("subject_name", "")

    render_executive_summary(d)
    render_risk_category_panels(d)
    render_risk_scoring(d)
    render_source_references(d)
    render_recommended_action(d)
    render_customer_id(d)
    narrative = render_analyst_narrative(d)
    render_audit_log(log, t0, subject)

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download report (TXT)",
            data=build_download_report(d, narrative, log),
            file_name=f"cdd_{subject.replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "💾 Download full JSON",
            data=json.dumps(d, indent=2),
            file_name=f"cdd_{subject.replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Follow-up chat ─────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 8 · Follow-up Questions")
    st.caption("Ask the assistant to clarify, expand, or investigate further.")

    for msg in st.session_state.follow_ups:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a follow-up question…"):
        st.session_state.follow_ups.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ctx = [
            {"role": "user",      "content": f"CDD report JSON:\n\n{st.session_state.raw_text}"},
            {"role": "assistant", "content": "Understood. I have the full CDD report in context."},
        ] + st.session_state.follow_ups

        with st.chat_message("assistant"):
            box = st.empty(); full = ""
            try:
                with client.messages.stream(
                    model=MODEL, max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT, messages=ctx,
                ) as stream:
                    for chunk in stream.text_stream:
                        full += chunk; box.markdown(full + "▌")
                box.markdown(full)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.session_state.follow_ups.pop()
                return

        st.session_state.follow_ups.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
