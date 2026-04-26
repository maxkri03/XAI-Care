"""
XAI-Care – Explainable AI for medical triage.

Streamlit app that uses llmSHAP to show which symptom words
drove an LLM's diagnostic assessment.

Run with:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, EmbeddingCosineSimilarity
from llmSHAP.llm import OpenAIInterface
from llmSHAP.attribution_methods.coalition_sampler import CounterfactualSampler

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI-Care",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .diagnosis-box {
        background: #1a2332;
        border-left: 4px solid #4CAF50;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .word-chip {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        margin: 3px 4px;
        font-size: 1.05rem;
        font-weight: 500;
        line-height: 2.2;
    }
    .impact-label {
        font-size: 0.65rem;
        vertical-align: super;
        margin-left: 2px;
        font-weight: 700;
    }
    .section-header {
        color: #aab4c4;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def preprocess_input(
    age: int,
    sex: str,
    conditions: str,
    symptoms: str,
    model_name: str,
    api_key: str,
) -> dict:
    """
    Build the feature dict that llmSHAP will attribute over.

    Structured fields become individual sentence-valued keys.
    The free-text symptom description is split by the LLM into one key per
    distinct symptom (multi-word phrases, not single tokens).
    """
    from openai import OpenAI
    import json

    client = OpenAI(api_key=api_key)

    data = {
        "age": f"The patient is {age} years old.",
        "sex": f"The patient is {sex.lower()}.",
    }
    if conditions.strip() and conditions.strip().lower() not in ("none", "-", "n/a"):
        data["conditions"] = f"Previous conditions: {conditions}."

    extraction_prompt = (
        "Analyze the text below and extract two categories of information:\n\n"
        "1. SYMPTOMS: physical symptoms or complaints the patient is experiencing.\n"
        "2. ADDITIONAL: other clinically relevant context that is NOT a symptom — "
        "such as recent travel, exposure to sick contacts, occupation, living conditions, "
        "recent procedures, or similar epidemiological/contextual clues.\n\n"
        "Return a JSON object with exactly two keys:\n"
        "  \"symptoms\": an array of strings, each a complete sentence starting with "
        "'The patient has' or 'The patient is experiencing', one symptom per string.\n"
        "  \"additional\": an array of strings, each a complete sentence starting with "
        "'The patient' describing one contextual fact. Use an empty array if none.\n\n"
        "Respond in the same language as the input. "
        "Return ONLY valid JSON – no markdown, no explanation.\n\n"
        f"Text: {symptoms}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.0,
        max_tokens=400,
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]

    extracted = json.loads(raw.strip())
    symptom_list = extracted.get("symptoms", [])
    additional_list = extracted.get("additional", [])

    for i, symptom in enumerate(symptom_list, start=1):
        data[f"symptom_{i}"] = str(symptom).strip()
    for i, additional in enumerate(additional_list, start=1):
        data[f"additional_{i}"] = str(additional).strip()

    return data


def score_to_color(score: float, max_score: float) -> str:
    """Map a SHAP score to a background-color CSS string.
    Positive  → red tones  (contributes to diagnosis)
    Negative  → blue tones (suppresses diagnosis)
    Near zero → neutral gray
    """
    if max_score == 0:
        return "rgba(80, 80, 80, 0.3)"

    normalized = score / max_score  # −1 … +1

    if normalized > 0:
        # White → red
        intensity = min(normalized, 1.0)
        r = int(255)
        g = int(255 * (1 - intensity * 0.85))
        b = int(255 * (1 - intensity * 0.85))
        alpha = 0.25 + 0.65 * intensity
        return f"rgba({r},{g},{b},{alpha:.2f})"
    elif normalized < 0:
        # White → blue
        intensity = min(abs(normalized), 1.0)
        r = int(255 * (1 - intensity * 0.85))
        g = int(255 * (1 - intensity * 0.85))
        b = int(255)
        alpha = 0.25 + 0.65 * intensity
        return f"rgba({r},{g},{b},{alpha:.2f})"
    else:
        return "rgba(80, 80, 80, 0.3)"


def render_highlighted_text(attribution: dict) -> str:
    """Build HTML with color-coded feature chips + impact percentages."""
    feature_items = {
        k: v for k, v in attribution.items()
        if k != "__context__" and v.get("value", "").strip()
    }

    if not feature_items:
        return "<em>No features to display.</em>"

    scores = [abs(item["score"]) for item in feature_items.values()]
    max_score = max(scores) if scores else 1.0

    html_parts = []
    for key, item in feature_items.items():
        value = item.get("value", "")
        score = item.get("score", 0.0)
        pct   = abs(score) * 100

        bg_color    = score_to_color(score, max_score)
        text_color  = "#ffffff" if abs(score / max_score) > 0.3 else "#cccccc"
        label_color = "#ff9999" if score > 0 else "#9999ff" if score < 0 else "#888888"
        key_label   = key.replace("_", " ").upper()

        html_parts.append(
            f'<span class="word-chip" style="background:{bg_color};color:{text_color};" title="{key}">'
            f'<span style="font-size:0.6rem;opacity:0.7;display:block;line-height:1.2;">{key_label}</span>'
            f'{value}'
            f'<span class="impact-label" style="color:{label_color};">'
            f'{pct:.0f}%'
            f'</span>'
            f'</span>'
        )

    return " ".join(html_parts)


def run_analysis(
    age: int,
    sex: str,
    conditions: str,
    symptoms: str,
    model_name: str,
    api_key: str,
) -> tuple:
    """Run llmSHAP and return (diagnosis, attribution_dict, preprocessed_data)."""

    os.environ["OPENAI_API_KEY"] = api_key

    # Pre-process: LLM splits symptom text into meaningful sentence-keys
    data = preprocess_input(age, sex, conditions, symptoms, model_name, api_key)

    handler = DataHandler(data)  # all keys are variable – no permanent keys

    sampler = CounterfactualSampler()

    system_prompt = (
        "You are a medical diagnostic assistant at an emergency department. "
        "You receive patient information as a list of facts (some may be absent). "
        "Give a concise diagnostic assessment in 2–3 sentences based only on what is present. "
        "End with: 'Primary assessment: [diagnosis]'."
    )

    model = OpenAIInterface(model_name=model_name, max_tokens=300, temperature=0.0)
    codec = BasicPromptCodec(system=system_prompt)

    value_function = EmbeddingCosineSimilarity(
        api_url_endpoint="https://api.openai.com/v1",
    )

    attribution_result = ShapleyAttribution(
        model=model,
        data_handler=handler,
        prompt_codec=codec,
        sampler=sampler,
        use_cache=True,
        num_threads=16,
        verbose=False,
        value_function=value_function,
    ).attribution()

    return attribution_result.output, attribution_result.attribution, data


model_name = "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY", "")


# ── Header ─────────────────────────────────────────────────────────────────────
header_col1, header_col2 = st.columns([1, 12])
with header_col1:
    st.image("Bild1.png", width=70)
with header_col2:
    st.title("XAI-Care — Explainable AI Diagnostic Decisions")

st.markdown(
    "Enter the patient's data and symptoms below. The AI will give a diagnosis "
    "**and** show exactly which features drove that conclusion."
)
st.markdown("---")

# ── Patient Data (visible, not in sidebar) ─────────────────────────────────────
st.markdown("<p class='section-header'>Patient Data</p>", unsafe_allow_html=True)

row1_col1, row1_col2 = st.columns([1, 1])
with row1_col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=65)
with row1_col2:
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])

conditions = st.text_area(
    "Previous conditions",
    value="",
    height=80,
    placeholder="e.g. hypertension, type 2 diabetes",
)

st.markdown("---")

# ── Symptoms + Color legend ────────────────────────────────────────────────────
col_input, col_legend = st.columns([3, 1])

with col_input:
    symptoms = st.text_area(
        "Symptom description",
        value="",
        height=220,
        placeholder="Describe symptoms in plain text (English or Swedish)...",
    )

with col_legend:
    st.markdown("<p class='section-header'>Color legend</p>", unsafe_allow_html=True)
    st.markdown(
        '<span class="word-chip" style="background:rgba(255,50,50,0.7);color:white;">strong push</span>'
        '<br>'
        '<span class="word-chip" style="background:rgba(255,180,180,0.5);color:#333;">mild push</span>'
        '<br>'
        '<span class="word-chip" style="background:rgba(80,80,80,0.3);color:#ccc;">neutral</span>'
        '<br>'
        '<span class="word-chip" style="background:rgba(180,180,255,0.5);color:#333;">mild suppress</span>'
        '<br>'
        '<span class="word-chip" style="background:rgba(50,50,255,0.7);color:white;">strong suppress</span>',
        unsafe_allow_html=True,
    )

# ── Analyze button ─────────────────────────────────────────────────────────────
analyze_btn = st.button("🔬 Analyze symptoms", type="primary", use_container_width=True)

# ── Results ────────────────────────────────────────────────────────────────────
if analyze_btn:
    if not api_key:
        st.error("Please provide an OpenAI API key in the sidebar or set OPENAI_API_KEY in a .env file.")
        st.stop()
    if not symptoms.strip():
        st.warning("Please enter symptom text.")
        st.stop()

    with st.spinner("Step 1/2 — Extracting features from symptom description…"):
        try:
            diagnosis, attribution, preprocessed = run_analysis(
                age=age,
                sex=sex,
                conditions=conditions,
                symptoms=symptoms,
                model_name=model_name,
                api_key=api_key,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # ── Extracted features preview ──
    with st.expander("Extracted features (pre-processed input to llmSHAP)", expanded=True):
        for key, value in preprocessed.items():
            st.markdown(f"**{key}:** {value}")

    # ── Diagnosis box ──
    st.markdown("### Diagnosis")
    st.markdown(
        f'<div class="diagnosis-box">{diagnosis}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Feature impact visualization ──
    st.markdown("### Feature Impact (XAI)")
    st.caption(
        "Each feature's color and % show how much it pushed the AI toward (🔴) "
        "or away from (🔵) its diagnosis."
    )

    highlighted_html = render_highlighted_text(attribution)
    st.markdown(
        f'<div style="line-height:3.2;font-size:1.0rem;padding:12px;'
        f'background:#0e1117;border-radius:8px;">{highlighted_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Bar chart ──
    st.markdown("### Top Contributing Features")

    feature_items = {
        k: v for k, v in attribution.items()
        if k != "__context__" and v.get("value", "").strip()
    }

    if feature_items:
        import plotly.graph_objects as go

        sorted_items = sorted(
            feature_items.items(),
            key=lambda x: abs(x[1]["score"]),
            reverse=True,
        )

        # Key as y-label; full sentence as hover text
        labels  = [item[0].replace("_", " ") for item in sorted_items]
        hover   = [item[1]["value"] for item in sorted_items]
        scores  = [item[1]["score"] * 100 for item in sorted_items]
        colors  = ["#ff4444" if s > 0 else "#4444ff" for s in scores]

        fig = go.Figure(go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{abs(s):.1f}%" for s in scores],
            textposition="outside",
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Impact (%)",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#ffffff",
            height=max(300, len(labels) * 50),
            margin=dict(l=10, r=60, t=20, b=40),
            xaxis=dict(gridcolor="#2a3040"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw data expander ──
    with st.expander("Raw attribution data (debug)"):
        st.json(attribution)

