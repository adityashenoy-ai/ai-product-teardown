# app.py
import streamlit as st
import json, re, time
from openai import OpenAI

st.set_page_config(page_title="AI Product Teardown Engine — Compare", layout="wide")
st.title("🔎 AI Product Teardown Engine — Compare Mode")
st.caption("Reverse-engineer any product. Compare two teardowns side-by-side. Industry-aware prompt templates included.")

# -------------------------------
# OpenAI client init (new SDK)
# -------------------------------
client = None
if "OPENAI_API_KEY" in st.secrets:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error("Failed to initialize OpenAI client: " + str(e))
else:
    st.warning("No OPENAI_API_KEY found in Streamlit Secrets. You can still use demo mode outputs but LLM memos will be disabled.")

# -------------------------------
# Industry fine-tuned prompt templates
# -------------------------------
INDUSTRY_TEMPLATES = {
    "General / Consumer": "Focus on consumer acquisition, retention hooks, network effects, pricing psychology, and UX simplicity.",
    "FinTech": "Focus on regulatory constraints, payments & flows, trust signals, onboarding (KYC), monetization via interchange/fees, fraud & compliance concerns, and merchant/partner flows.",
    "Marketplace": "Focus on two-sided network dynamics (supply/demand), liquidity, take rates, onboarding incentives, quality controls, and marketplace matching algorithms.",
    "SaaS / B2B": "Focus on buyer personas, sales motions (self-serve vs enterprise), onboarding/activation for users/teams, trial/enterprise pricing, retention via ROI, and product-led growth experiments.",
    "EdTech": "Focus on learning outcomes, curriculum design, engagement loops, teacher/platform dynamics, certification, and measurement of learning retention.",
    "HealthTech": "Focus on trust & compliance (HIPAA-like), clinician workflows, patient onboarding, safety-critical UX, integrations with EMR, and monetization models."
}

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("Teardown Settings")
    industry = st.selectbox("Industry template", list(INDUSTRY_TEMPLATES.keys()), index=0)
    depth = st.selectbox("Analysis depth", ["Quick (bullets)", "Standard (detailed)", "Deep (comprehensive)"], index=1)
    include_user_flow = st.checkbox("Include user flow & microcopy", value=True)
    include_metrics = st.checkbox("Include KPIs & measurement plan", value=True)
    include_templates = st.checkbox("Include templates (PRD, experiment briefs)", value=True)
    model = st.selectbox("LLM Model", ["gpt-4o-mini","gpt-4o"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 0.9, 0.2, step=0.1)
    run_button = st.button("Generate / Refresh Teardowns")
    st.markdown("---")
    st.info("Add OPENAI_API_KEY in Streamlit Secrets to enable LLM calls. If missing, demo outputs will be shown.")

# -------------------------------
# Inputs for two products
# -------------------------------
st.markdown("## Inputs — Product A & Product B")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Product A")
    app_a = st.text_input("Name / URL / short description (A)", value="Google Pay")
    explicit_a = st.text_area("Optional: paste product key features (A)", height=80)

with col2:
    st.subheader("Product B")
    app_b = st.text_input("Name / URL / short description (B)", value="PhonePe")
    explicit_b = st.text_area("Optional: paste product key features (B)", height=80)

# -------------------------------
# Prompt builders & LLM helpers
# -------------------------------
def depth_to_instruction(d):
    return {
        "Quick (bullets)": "Produce concise, high-impact bullet points (1-4 bullets per section).",
        "Standard (detailed)": "Provide structured analysis with 4-8 bullets per section and short rationale sentences.",
        "Deep (comprehensive)": "Provide an in-depth multi-paragraph analysis using frameworks and examples for each section."
    }.get(d, "")

def build_teardown_prompt(product_text, industry_key, depth, include_user_flow, include_metrics, include_templates):
    """
    Returns a prompt instructing the LLM to output EXACTLY one JSON object with:
    { strategy, growth_loops, engagement_mechanics, kpis, ux_teardown, swot, opportunities, one_pager }
    """
    industry_hint = INDUSTRY_TEMPLATES.get(industry_key, "")
    explicit_instruction = depth_to_instruction(depth)
    include_user_flow_flag = "yes" if include_user_flow else "no"
    include_metrics_flag = "yes" if include_metrics else "no"
    include_templates_flag = "yes" if include_templates else "no"

    prompt = f"""
You are an expert Product Manager & Growth strategist.

Product description:
\"\"\"{product_text}\"\"\"

Context: {industry_hint}
Depth instruction: {explicit_instruction}

Produce exactly ONE JSON object (no surrounding text) with the following keys:
- strategy: an array of 3-6 concise strings describing positioning, target segments, monetization levers.
- growth_loops: an array of 3-6 strings describing primary acquisition & virality loops with estimated impact percentages where reasonable.
- engagement_mechanics: an array of 4-8 strings describing activation, retention hooks, notifications, onboarding steps.
- kpis: an object with keys: north_star, leading_indicators (array), dashboard_analytics (array of metric names).
- ux_teardown: array of observations about UX/flows, friction points, and microcopy suggestions (include sample microcopy if INCLUDE_USER_FLOW=yes).
- swot: object with keys: strengths (array), weaknesses (array), opportunities (array), threats (array).
- opportunities: array of short product/experiment ideas prioritized (short/medium/long-term).
- one_pager: a short markdown string (3-6 sentences) summarizing the product thesis and top recommendations.

REQUIREMENTS:
- Return valid JSON only (no commentary).
- If INCLUDE_USER_FLOW is {include_user_flow_flag} then include 1 short suggested user flow (3-6 steps) inside ux_teardown items.
- If INCLUDE_METRICS is {include_metrics_flag} then include realistic KPI names and 1 example target (e.g., conversion 3% -> 4%).
- If INCLUDE_TEMPLATES is {include_templates_flag} then include one experiment idea in each of growth_loops and engagement_mechanics.

Be concrete and action-oriented.
"""
    return prompt

def call_llm(prompt, model="gpt-4o-mini", temperature=0.2, tries=2):
    if client is None:
        return None
    last_exc = None
    for attempt in range(tries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature
            )
            # attempt to extract assistant message
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)
    st.error(f"LLM call failed after retries: {last_exc}")
    return None

# robust JSON extraction
def extract_json(text):
    if not text:
        return None
    # try fenced JSON
    m = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        # try to find first { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                # try minor fixes: remove trailing commas
                candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(candidate2)
                except Exception:
                    return None
    return None

# -------------------------------
# Function: generate teardown (with fallback demo)
# -------------------------------
def generate_teardown(product_text, industry_key, depth, include_user_flow, include_metrics, include_templates, model, temperature):
    prompt = build_teardown_prompt(product_text, industry_key, depth, include_user_flow, include_metrics, include_templates)
    raw = call_llm(prompt, model=model, temperature=temperature, tries=3)
    parsed = extract_json(raw) if raw else None
    if parsed is None:
        # fallback demo minimal output (useful if no API key)
        demo = {
            "strategy": [
                f"Position as fast, low-friction {industry_key.lower()} product",
                "Target mass market and power users",
                "Monetize via freemium and transactional fees"
            ],
            "growth_loops": [
                "Referral loop: user invites friend -> both get credits (expected +1-3% uplift)",
                "Merchant incentives: onboarding merchants drives supply"
            ],
            "engagement_mechanics": [
                "Onboarding checklist with progress bar",
                "Daily digest push for key actions"
            ],
            "kpis": {
                "north_star": "DAU -> Paid conversion",
                "leading_indicators": ["activation_rate", "7d_retention", "week1_cohort_conversion"],
                "dashboard_analytics": ["signup_rate", "activation_rate", "revenue_per_user"]
            },
            "ux_teardown": [
                "Clear CTA on home screen -> sample microcopy: 'Pay in 2 taps'",
                "Suggested user flow: ['Install', 'Create account', 'Onboard payment method', 'Complete first transaction']" if include_user_flow else ""
            ],
            "swot": {
                "strengths": ["Strong UX", "Partnerships"],
                "weaknesses": ["Customer acquisition cost"],
                "opportunities": ["New monetization"],
                "threats": ["Regulatory risk"]
            },
            "opportunities": [
                "Test 1: incentivized referral for transactions",
                "Test 2: merchant onboarding pilot"
            ],
            "one_pager": f"{product_text} — Quick product thesis and 3-line exec summary."
        }
        return demo, raw
    return parsed, raw

# -------------------------------
# Generate teardowns when requested
# -------------------------------
teardown_a = None; raw_a = None
teardown_b = None; raw_b = None

if run_button:
    if not app_a.strip() or not app_b.strip():
        st.error("Enter both Product A and Product B (name/URL/short description).")
    else:
        with st.spinner("Generating teardown for Product A..."):
            product_text_a = app_a.strip() + ("\n\n" + explicit_a.strip() if explicit_a.strip() else "")
            teardown_a, raw_a = generate_teardown(product_text_a, industry, depth, include_user_flow, include_metrics, include_templates, model, temperature)
        with st.spinner("Generating teardown for Product B..."):
            product_text_b = app_b.strip() + ("\n\n" + explicit_b.strip() if explicit_b.strip() else "")
            teardown_b, raw_b = generate_teardown(product_text_b, industry, depth, include_user_flow, include_metrics, include_templates, model, temperature)

        st.success("Teardowns generated (or demo outputs provided). Scroll to compare.")

# -------------------------------
# Display side-by-side comparison
# -------------------------------
st.markdown("## Compare Teardowns")
if teardown_a is None or teardown_b is None:
    st.info("Click **Generate / Refresh Teardowns** to produce teardowns for both products. If OPENAI_API_KEY is missing you'll get demo outputs.")
else:
    # Two-column side-by-side panes
    left, right = st.columns(2)
    with left:
        st.markdown(f"### Product A — **{app_a.strip()}**")
        st.download_button("Download A (JSON)", data=json.dumps(teardown_a, indent=2, ensure_ascii=False), file_name=f"teardown_A_{app_a.strip().replace(' ','_')}.json", mime="application/json")
        st.download_button("Download A (MD)", data=markdown_from_teardown(teardown_a, app_a.strip()), file_name=f"teardown_A_{app_a.strip().replace(' ','_')}.md", mime="text/markdown")
        st.markdown("#### One-page summary")
        st.write(teardown_a.get("one_pager") or "")
        st.markdown("#### Strategy")
        for s in teardown_a.get("strategy", []):
            st.markdown(f"- {s}")
        st.markdown("#### Growth Loops")
        for g in teardown_a.get("growth_loops", teardown_a.get("growthLoops", [])):
            st.markdown(f"- {g}")
        st.markdown("#### Key KPIs")
        st.json(teardown_a.get("kpis", {}))

    with right:
        st.markdown(f"### Product B — **{app_b.strip()}**")
        st.download_button("Download B (JSON)", data=json.dumps(teardown_b, indent=2, ensure_ascii=False), file_name=f"teardown_B_{app_b.strip().replace(' ','_')}.json", mime="application/json")
        st.download_button("Download B (MD)", data=markdown_from_teardown(teardown_b, app_b.strip()), file_name=f"teardown_B_{app_b.strip().replace(' ','_')}.md", mime="text/markdown")
        st.markdown("#### One-page summary")
        st.write(teardown_b.get("one_pager") or "")
        st.markdown("#### Strategy")
        for s in teardown_b.get("strategy", []):
            st.markdown(f"- {s}")
        st.markdown("#### Growth Loops")
        for g in teardown_b.get("growth_loops", teardown_b.get("growthLoops", [])):
            st.markdown(f"- {g}")
        st.markdown("#### Key KPIs")
        st.json(teardown_b.get("kpis", {}))

    # Comparison highlights: simple diff-like comparison for top items
    st.markdown("---")
    st.header("Quick Comparison Highlights")
    comp_cols = st.columns(3)
    # North-star comparison
    try:
        ns_a = teardown_a.get("kpis", {}).get("north_star", "—")
        ns_b = teardown_b.get("kpis", {}).get("north_star", "—")
    except Exception:
        ns_a = ns_b = "—"
    comp_cols[0].metric("North-star (A)", ns_a, delta=None)
    comp_cols[1].metric("North-star (B)", ns_b, delta=None)
    # Strategy length (proxy)
    comp_cols[2].write("Strategy breadth")
    comp_cols[2].write(f"A: {len(teardown_a.get('strategy',[]))} items  |  B: {len(teardown_b.get('strategy',[]))} items")

    # Side-by-side table for SWOT strengths
    st.markdown("### SWOT — Strengths (side-by-side)")
    strengths_a = teardown_a.get("swot", {}).get("strengths", []) if teardown_a.get("swot") else []
    strengths_b = teardown_b.get("swot", {}).get("strengths", []) if teardown_b.get("swot") else []
    maxlen = max(len(strengths_a), len(strengths_b))
    rows = []
    for i in range(maxlen):
        a = strengths_a[i] if i < len(strengths_a) else ""
        b = strengths_b[i] if i < len(strengths_b) else ""
        rows.append({"A": a, "B": b})
    st.table(rows)

    # Opportunity differences
    st.markdown("### Opportunity Ideas (A vs B)")
    opp_a = teardown_a.get("opportunities", [])
    opp_b = teardown_b.get("opportunities", [])
    left_o, right_o = st.columns(2)
    with left_o:
        st.subheader(f"A — {app_a.strip()}")
        for i, o in enumerate(opp_a[:8]):
            st.markdown(f"{i+1}. {o}")
    with right_o:
        st.subheader(f"B — {app_b.strip()}")
        for i, o in enumerate(opp_b[:8]):
            st.markdown(f"{i+1}. {o}")

# -------------------------------
# Utilities: markdown generator
# -------------------------------
def markdown_from_teardown(td, title):
    md = [f"# Product Teardown — {title}\n"]
    md.append("## One-pager\n")
    md.append(td.get("one_pager","") + "\n\n")
    md.append("## Strategy\n")
    for s in td.get("strategy",[]):
        md.append(f"- {s}\n")
    md.append("\n## Growth Loops\n")
    for g in td.get("growth_loops", td.get("growthLoops", [])):
        md.append(f"- {g}\n")
    md.append("\n## Engagement Mechanics\n")
    for e in td.get("engagement_mechanics", td.get("engagement", [])):
        md.append(f"- {e}\n")
    md.append("\n## KPIs\n")
    md.append("```json\n")
    md.append(json.dumps(td.get("kpis",{}), indent=2, ensure_ascii=False))
    md.append("\n```\n")
    md.append("\n## UX Tear-down\n")
    for u in td.get("ux_teardown", td.get("ux", [])):
        md.append(f"- {u}\n")
    md.append("\n## SWOT\n")
    md.append("```json\n")
    md.append(json.dumps(td.get("swot",{}), indent=2, ensure_ascii=False))
    md.append("\n```\n")
    md.append("\n## Opportunities\n")
    for o in td.get("opportunities",[]):
        md.append(f"- {o}\n")
    return "\n".join(md)

# -------------------------------
# Examples / Quick demo
# -------------------------------
st.markdown("---")
st.markdown("#### Tip & Examples")
st.markdown("Try examples: `Google Pay` vs `PhonePe`, `Duolingo` vs `Babbel`, `Notion` vs `Coda`.")
st.caption("For best results paste a short list of product features or the app's one-line positioning into the description fields.")

