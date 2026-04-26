# XAI-Care — Pitch Deck

---

## Slide 1: Title

# XAI-Care
### Making AI Diagnostic Decisions Explainable

*Powered by llmSHAP*

---

## Slide 2: The Problem

# Doctors Can't Trust What They Can't Understand

- AI is entering clinical decision support at rapid pace
- But today's LLMs are **black boxes** — they give answers, not reasons
- In high-stakes medicine, "trust me" is not acceptable
- Clinicians need to know **why** the AI thinks what it thinks

> "An AI says the patient is having a heart attack. The doctor doesn't know why. Should she trust it?"

---

## Slide 3: The Status Quo is Dangerous

# Without Explainability

| Scenario | Outcome |
|---|---|
| AI is right, no explanation | Doctor may override — patient harmed |
| AI is wrong, no explanation | No way to catch the error |
| Audit required | No trail to follow |
| Legal/compliance review | Black box fails EU AI Act requirements |

**The problem isn't the AI. It's the silence around its decisions.**

---

## Slide 4: Our Solution

# XAI-Care: Explainability Built Into the Diagnosis

- Patient symptoms go in → AI diagnosis comes out
- **llmSHAP runs on the output** and attributes each word's contribution to the decision
- The result: a color-coded explanation, in plain language, visible in seconds

> "Radiating pain in left arm" — **42% contribution**
> "Pressure in the chest" — **31% contribution**

The doctor sees not just *what* the AI decided, but *why*.

---

## Slide 5: Live Demo

# See It Work

*(Run demo here)*

1. Enter patient profile + symptom description
2. AI produces diagnosis
3. llmSHAP highlights which words drove the decision — with percentages

**Before XAI-Care:** A black box verdict.
**After XAI-Care:** A transparent, auditable decision the doctor can act on.

---

## Slide 6: How It Works

# The Architecture

```
Patient Input (age, history, symptoms)
        ↓
      LLM
(diagnosis + reasoning)
        ↓
    llmSHAP
(word-level attribution)
        ↓
  Color-coded UI
(% influence per token)
```

- llmSHAP is **model-agnostic** — works with any LLM
- Open source, auditable, reproducible
- Built on established SHAP methodology from ML research

---

## Slide 7: Real-World Path

# From PoC to Production

| Today (Hackathon PoC) | Tomorrow (Hospital Deployment) |
|---|---|
| Cloud LLM via API | Local LLM trained on hospital EHR data |
| Synthetic patient data | Real patient data, GDPR-compliant |
| Streamlit web app | Integration with existing clinical systems |
| Demo environment | Audit logs, role-based access, compliance |

**The architecture doesn't change. We just swap the backend.**

llmSHAP is model-agnostic — the explainability layer works regardless of which LLM is underneath.

---

## Slide 8: Why Now

# The Regulatory Window is Open — and Closing

- **EU AI Act (2024):** High-risk AI systems (including clinical decision support) must be explainable and auditable
- **GDPR:** Patients have the right to explanation for automated decisions
- **Clinical trust gap:** Adoption of AI in hospitals is blocked by lack of transparency, not lack of capability

XAI-Care doesn't fight regulation — it's built for it.

---

## Slide 9: Impact

# Who Benefits

**Clinicians**
- Faster, more confident triage decisions
- Ability to override AI with documented reasoning

**Patients**
- Right to understand decisions made about their care
- Reduced risk from unchecked AI errors

**Hospitals**
- Compliance with EU AI Act out of the box
- Auditable AI trails for liability and review

**Healthcare systems**
- Faster, safer AI adoption at scale

---

## Slide 10: The Ask

# What Comes Next

We didn't come here with a pitch deck and a promise.
**We came with a working system.**

> The next step: a pilot partnership with an emergency department or clinical AI team.

XAI-Care shows that explainable AI in healthcare is not a future problem — it's a solved architecture waiting to be deployed.

---

## Slide 11: Team + Stack

# Built This Weekend

**Tech stack:**
- Python + llmSHAP (open source)
- Claude / GPT as LLM backend (swappable)
- Streamlit for the UI

**Team:** [Your names here]

**Repo:** [GitHub link]

---
