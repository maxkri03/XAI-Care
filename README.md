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

1. Patient data + free-text symptoms are structured into sentence-valued features by an LLM.
2. `llmSHAP` computes Shapley values — masking feature subsets and comparing outputs via embedding similarity.
3. Results are displayed with each feature's percentage impact on the final diagnosis.

## Stack

| | |
|---|---|
| XAI | [llmSHAP](https://github.com/filipnaudot/llmSHAP) |
| LLM | GPT-4o-mini (PoC) / local model (production) |
| UI | Streamlit + Plotly |

## License

MIT
