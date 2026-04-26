# XAI-Care — Explainable AI for Medical Diagnostics

A PoC that wraps an LLM diagnostic assistant with [llmSHAP](https://github.com/filipnaudot/llmSHAP) to show *which* symptoms drove the AI's conclusion — and by how much.

## Quick Start

```bash
pip install -e .
pip install -r requirements-app.txt
cp .env.example .env   # add OPENAI_API_KEY
streamlit run app.py
```

## How It Works

1. Patient data is structured into sentence-valued features
2. Each feature is ablated (removed) and the model's output is compared to the full output via **cosine embedding similarity**
3. The similarity score (not a Shapley value) indicates how much the output changed when that feature was removed


## Stack

| | |
|---|---|
| XAI | [llmSHAP](https://github.com/filipnaudot/llmSHAP) |
| LLM | GPT-4o-mini (PoC) / local model (production) |
| UI | Streamlit + Plotly |

## License

MIT
