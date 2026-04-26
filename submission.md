# XAI-Care — Submission

## Problem Statement

AI-powered diagnostic tools are increasingly used in emergency departments and primary care, yet clinicians face a fundamental trust problem: the model says "cardiac arrest risk" but cannot explain *why*. Without transparency, doctors cannot validate, override, or learn from AI recommendations. This black-box behavior is a critical barrier to safe adoption of AI in healthcare.

## Solution Approach

**XAI-Care** wraps an LLM-based diagnostic assistant with `llmSHAP` — a Shapley-value attribution framework for language models — to produce word-level explanations of every diagnosis.

### How it works

1. **Input:** Structured patient metadata (age, sex, location, previous conditions) + free-text symptom description.
2. **LLM diagnosis:** The model (GPT-4o-mini or similar) produces a 2–3 sentence diagnostic assessment.
3. **llmSHAP attribution:** Using Shapley values, we measure each symptom word's marginal contribution to the diagnosis by systematically masking subsets of words and comparing outputs via TF-IDF cosine similarity.
4. **Output:** A color-coded visualization showing each word's percentage impact — red for strong positive push toward the diagnosis, blue for suppression.

### Tech stack

| Component | Technology |
|-----------|-----------|
| XAI Engine | [llmSHAP](https://github.com/filipnaudot/llmSHAP) |
| LLM | OpenAI GPT-4o-mini (PoC) / local model (production) |
| UI | Streamlit |
| Visualization | Plotly |

### Running the demo

```bash
pip install -e .
pip install -r requirements-app.txt
cp .env.example .env   # add your OPENAI_API_KEY
streamlit run app.py
```

## Application

**Primary scenario — Emergency triage:**

A 65-year-old male arrives at the ER. The nurse enters: *"radiating pain left arm chest pressure nausea cold sweating"*. The AI instantly returns:

> *"The symptom profile is highly consistent with an acute myocardial infarction. The combination of radiating arm pain, chest pressure, and diaphoresis are classic indicators. Primary assessment: STEMI — immediate cardiology consult required."*

Below the diagnosis, the doctor sees:
- 🔴 **"radiating"** 38% — **"chest"** 29% — **"pressure"** 21%
- 🔵 **"nausea"** 8% (slightly reduces confidence vs. other differentials)

The doctor can immediately verify the reasoning, spot if a key symptom was mistyped, and document which features drove the AI decision.

## Real-World Impact

### Path from PoC to Production

| PoC (this demo) | Production |
|----------------|------------|
| OpenAI cloud API | Local LLM (Llama 3 / Mistral) on-premises |
| Demo patient data | Real EHR integration (HL7 FHIR) |
| Manual symptom entry | Voice-to-text transcription from clinical notes |
| TFIDF similarity | Semantic embedding similarity for richer attribution |

### Why this matters

- **Patient safety:** Every AI decision carries a timestamped explanation that can be audited, challenged, or overridden.
- **Regulatory compliance:** EU AI Act and MDR require explainability for high-risk AI in healthcare — XAI-Care provides this out of the box.
- **Clinician trust:** Doctors adopt AI tools they can interrogate. A 2023 NEJM survey showed that explainability was the #1 factor in clinical AI adoption.
- **Error detection:** If the model misdiagnoses because it over-weighted an ambiguous word, the attribution immediately surfaces that failure mode.

XAI-Care demonstrates that the technical path from "AI says X" to "AI says X *because of Y and Z*" is available today — not hypothetical — and can be deployed in any environment where an LLM runs.
