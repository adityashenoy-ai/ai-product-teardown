# app.py
import streamlit as st
import json
import time
from openai import OpenAI

st.set_page_config(page_title="AI Product Teardown Engine", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🔎 AI Product Teardown Engine")
st.caption("Enter an app name or URL and get a product teardown: strategy, growth loops, UX, KPIs, SWOT, and opportunity map.")

# --------- OpenAI Client ----------
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("No OpenAI key found in Streamlit Secrets. Add OPENAI_API_KEY to enable LLM-powered teardowns.")
client = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("Failed to initialize OpenAI client: " + str(e))

# --------- Sidebar Controls ----------
with st.sidebar:
    st.header("Teardown Settings")
    app_input = st.text_input("Product name / URL / short description", value="Google Pay")
    depth = st.selectbox("Analysis depth", ["Quick (short bullets)", "Standard (detailed)", "Deep (comprehensive)"], index=1)
    include_user_flow = st.checkbox("Include suggested user flow & microcopy", value=True)
    include_metrics = st.checkbox("Include KPIs and measurement plan", value=True)
    include_templates = st.checkbox("Include playbook templates (PRD, experiment ideas)", value=True)
    model = st.selectbox("LLM Model", ["gpt-4o-mini","gpt-4o"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 0.9, 0.2, step=0.1)
    run_button = st.button("Generate Teardown")

# --------- Prompt builders ----------
def build_teardown_prompt(app_text, depth, include_user_flow, include_metrics, include_templates):
    """
    Returns a strict prompt that instructs the LLM to return JSON with keys:
    strategy, growth_loops, engagement_mechanics, kpis, ux_teardown, swot, opportunities, one_pager
    """
    depth_map = {
        "Quick (short bullets)": "concise bullet points (1-3 bullets per section)",
        "Standard (detailed)": "detailed analysis with examples and 5-7 bullets per section",
        "Deep (comprehensive)": "comprehensive multi-paragraph analysis, include frameworks and examples"
    }
    depth_instructions = depth_map.get(depth, "detailed analysis")
    sections = [
        {"id":"strategy","title":"Product Strategy & Positioning","desc":"market, value prop, target segments, pricing levers"},
        {"id":"growth_loops","title":"Growth Loops & Acquisition Strategy","desc":"main growth loops, channels, virality mechanics"},
        {"id":"engagement","title":"Engagement Mechanics & Retention","desc":"hooks, habit loops, onboarding, notifications"},
        {"id":"kpis","title":"KPIs & Measurement Plan","desc":"north-star, leading indicators, dashboard metrics"},
        {"id":"ux","title":"Design & UX Tear-Down","desc":"key flows, friction, microcopy, UI suggestions"},
        {"id":"swot","title":"SWOT & Competitive Positioning","desc":"strengths, weaknesses, opportunities, threats"},
        {"id":"opps","title":"Opportunity Map & Roadmap Ideas","desc":"short/medium/long term features and experiments"},
        {"id":"one_pager","title":"One-Page Executive Summary","desc":"single page summary for execs"},
    ]
    # Build JSON-return instruction
    prompt = f"""You are an expert Product Manager, Growth Lead, and UX Strategist.
Given the following product description: \"\"\"{app_text}\"\"\", produce a structured PRODUCT TEARDOWN.

**INSTRUCTIONS**
- Return EXACTLY one JSON object (no explanatory text) with the following keys: strategy, growth_loops, engagement_mechanics, kpis, ux_teardown, swot, opportunities, one_pager.
- Each key's value should be either an array of strings or an object with named fields as appropriate.
- Follow the depth instruction: {depth_instructions}.
- For numeric or scoring recommendations (e.g., expected conversion uplift), provide reasoned ballpark percentages when possible.
- Keep output actionable and concrete; include at least one experiment idea per major section.

Include user-flow & microcopy only if requested.

**INCLUDE_USER_FLOW**: {str(include_user_flow)}
**INCLUDE_METRICS**: {str(include_metrics)}
**INCLUDE_TEMPLATES**: {str(include_templates)}

Now produce the JSON output.
"""
    return prompt

# --------- Robust LLM call ----------
def call_llm(prompt, model="gpt-4o-mini", temperature=0.2, tries=2, wait=1.0):
    if client is None:
        return None
    last = None
    for i in range(tries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature
            )
            # Try to get message content
            content = None
            try:
                content = resp.choices[0].message.content
            except Exception:
                content = str(resp)
            return content
        except Exception as e:
            last = e
            time.sleep(wait * (i+1))
    st.error(f"LLM call failed: {last}")
    return None

# --------- JSON extraction helper ----------
import re, json
def extract_json(text):
    if not text:
        return None
    # Try fenced json first
    m = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        # attempt to extract curly block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return None

# --------- Main generation flow ----------
if run_button:
    if not app_input.strip():
        st.error("Please enter a product name, URL or short description.")
    else:
        with st.spinner("Running product teardown..."):
            prompt = build_teardown_prompt(app_input.strip(), depth, include_user_flow, include_metrics, include_templates)
            raw = call_llm(prompt, model=model, temperature=temperature, tries=3)
            parsed = extract_json(raw) if raw else None

        if parsed is None:
            st.error("Failed to parse structured output from the model. Raw output shown below.")
            st.code(raw[:5000] if raw else "No model output")
        else:
            # Display results across tabs and allow downloads
            st.success("Teardown ready — explore the tabs below.")
            tabs = st.tabs(["Executive One-Pager","Strategy","Growth Loops","Engagement","KPIs","UX Tear-Down","SWOT","Opportunity Map","Downloads"])
            # Executive one-pager
            with tabs[0]:
                st.header("One Page Executive Summary")
                one = parsed.get("one_pager") or parsed.get("onePager") or parsed.get("onePage") or ""
                if isinstance(one, (list, dict)):
                    st.markdown(json.dumps(one, indent=2, ensure_ascii=False))
                else:
                    st.markdown(one)

            # Strategy
            with tabs[1]:
                st.header("Product Strategy & Positioning")
                strat = parsed.get("strategy", {})
                if isinstance(strat, list):
                    for b in strat:
                        st.markdown(f"- {b}")
                else:
                    st.json(strat)

            # Growth Loops
            with tabs[2]:
                st.header("Growth Loops & Acquisition")
                gl = parsed.get("growth_loops", parsed.get("growthLoops", []))
                if isinstance(gl, list):
                    for g in gl:
                        st.markdown(f"- {g}")
                else:
                    st.json(gl)

            # Engagement
            with tabs[3]:
                st.header("Engagement Mechanics & Retention")
                em = parsed.get("engagement_mechanics", parsed.get("engagement", []))
                if isinstance(em, list):
                    for e in em:
                        st.markdown(f"- {e}")
                else:
                    st.json(em)

            # KPIs
            with tabs[4]:
                st.header("KPIs & Measurement Plan")
                kp = parsed.get("kpis", {})
                st.json(kp)

            # UX
            with tabs[5]:
                st.header("Design & UX Tear-Down")
                ux = parsed.get("ux_teardown", parsed.get("ux", {}))
                if isinstance(ux, list):
                    for u in ux:
                        st.markdown(f"- {u}")
                else:
                    st.json(ux)

            # SWOT
            with tabs[6]:
                st.header("SWOT")
                sw = parsed.get("swot", {})
                st.json(sw)

            # Opportunities
            with tabs[7]:
                st.header("Opportunity Map & Roadmap Ideas")
                opp = parsed.get("opportunities", parsed.get("opps", []))
                if isinstance(opp, list):
                    for o in opp:
                        st.markdown(f"- {o}")
                else:
                    st.json(opp)

            # Downloads
            with tabs[8]:
                st.header("Downloads & Templates")
                # JSON download
                st.download_button("Download teardown (JSON)", data=json.dumps(parsed, indent=2, ensure_ascii=False), file_name=f"teardown_{app_input.strip().replace(' ','_')}.json", mime="application/json")
                # Markdown generation (pretty)
                md_lines = []
                md_lines.append(f"# Product Teardown — {app_input.strip()}\n")
                md_lines.append("## One-pager\n")
                if isinstance(one, str):
                    md_lines.append(one + "\n")
                else:
                    md_lines.append(json.dumps(one, indent=2, ensure_ascii=False) + "\n")
                md_lines.append("## Strategy\n")
                md_lines.append("\n".join([f"- {s}" for s in (strat if isinstance(strat,list) else [str(strat)])]) + "\n")
                md_lines.append("## Growth Loops\n")
                md_lines.append("\n".join([f"- {s}" for s in (gl if isinstance(gl,list) else [str(gl)])]) + "\n")
                md_lines.append("## Engagement\n")
                md_lines.append("\n".join([f"- {s}" for s in (em if isinstance(em,list) else [str(em)])]) + "\n")
                md_lines.append("## KPIs\n")
                md_lines.append("```json\n" + json.dumps(kp, indent=2, ensure_ascii=False) + "\n```\n")
                md_lines.append("## UX Tear-down\n")
                md_lines.append("\n".join([f"- {s}" for s in (ux if isinstance(ux,list) else [str(ux)])]) + "\n")
                md_lines.append("## SWOT\n")
                md_lines.append("```json\n" + json.dumps(sw, indent=2, ensure_ascii=False) + "\n```\n")
                md_lines.append("## Opportunities\n")
                md_lines.append("\n".join([f"- {s}" for s in (opp if isinstance(opp,list) else [str(opp)])]) + "\n")

                md = "\n".join(md_lines)
                st.download_button("Download teardown (Markdown)", data=md, file_name=f"teardown_{app_input.strip().replace(' ','_')}.md", mime="text/markdown")
                if include_templates:
                    st.markdown("### Templates")
                    st.markdown("- PRD template (link or copy)") 
                    st.markdown("- Experiment A/B brief")
                    st.markdown("- Copy for onboarding flow")

# --------- Helpful examples ----------
st.markdown("---")
st.markdown("**Example prompts:** `Google Pay`, `Cred`, `Zomato`, `Duolingo`, `Notion`, `Flipkart Seller App`")
st.caption("Tip: For more accurate teardowns, paste a short description or the product's key features/target users.")
