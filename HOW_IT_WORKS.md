# How XAI-Care produces feature importance

End-to-end walkthrough of `app.py` — what happens from the moment a user clicks **Analyze symptoms** until the colour-coded chips and bar chart appear. Includes every LLM call, every math step, and the data shape between stages.

---

## The high-level pipeline

```
[ UI form ]
    │  age, sex, conditions, free-text symptoms
    ▼
[ STAGE 1 — Feature extraction (LLM #1) ]    preprocess_input()
    │  dict { "age": "...", "sex": "...", "symptom_1": "...", ... }
    ▼
[ STAGE 2 — Shapley attribution (LLM #2 + embeddings) ]   ShapleyAttribution.attribution()
    │  attribution dict { feature_key: { value, score } }
    │  diagnosis text  (the "grand coalition" generation)
    ▼
[ STAGE 3 — Score normalisation ]    _normalized_result()
    │  scores rescaled so Σ|score| = 1
    ▼
[ STAGE 4 — Rendering ]
    ├─ score_to_color()           → red/blue gradient chips
    ├─ render_highlighted_text()  → HTML with chip + percent
    └─ Plotly bar chart            → ranked horizontal bars
```

There are **three different LLM calls** in the pipeline, plus **embedding API calls** for similarity scoring. Knowing which is which is the key to understanding the cost and the math.

---

## Stage 1 — Turning free text into discrete features

`preprocess_input(age, sex, conditions, symptoms, model_name, api_key)` (app.py:72).

Shapley values need a **fixed set of features** ("players" in the game-theoretic sense). A single blob of free-text symptoms isn't a set of players — so we have to split it first.

We use `gpt-4o-mini` (LLM #1) to extract structured items:

1. The structured fields become single sentence-keys:
   - `age` → `"The patient is 65 years old."`
   - `sex` → `"The patient is male."`
   - `conditions` → `"Previous conditions: hypertension."` (only if non-empty)
2. The free-text symptom blob is sent with an extraction prompt asking for two arrays:
   - `symptoms`: each item a sentence like `"The patient has chest pain."`
   - `additional`: epidemiological/contextual facts (recent travel, exposure, etc.)
3. The arrays are flattened into numbered keys: `symptom_1`, `symptom_2`, `additional_1`, ...

**Why one LLM call here?** Splitting on punctuation gives garbage for clinical sentences in two languages. An LLM with `temperature=0.0` gives stable, semantically meaningful chunks that map well onto Shapley "players".

**Output of Stage 1** — a Python `dict` with N string-valued keys. N is typically 5–12 in our demo.

---

## Stage 2 — Computing Shapley values

This is the core XAI step. Driven by `run_analysis()` (app.py:209) and executed by `ShapleyAttribution.attribution()` (`src/llmSHAP/attribution_methods/shapley_attribution.py:52`).

### 2.1 Setup objects

| Object | Role |
|---|---|
| `DataHandler(data)` | Holds the feature dict; can produce a prompt string from any subset (coalition) of keys. |
| `BasicPromptCodec(system=...)` | Wraps each prompt with the medical-assistant system instruction. |
| `OpenAIInterface(model_name, max_tokens=300, temperature=0.0)` | The diagnostic LLM (LLM #2). `temperature=0.0` is essential — Shapley needs deterministic outputs. |
| `EmbeddingCosineSimilarity(api_url_endpoint="…/v1")` | Value function `v(·)` — uses OpenAI `text-embedding-3-small`. |
| `CounterfactualSampler()` | Picks which coalitions to evaluate (see 2.3). |

### 2.2 The Shapley value, conceptually

For a player *i* (a feature) and a value function *v* over coalitions *S* of features:

```
φ_i  =  Σ_{S ⊆ N\{i}}   w(S) · [ v(S ∪ {i}) − v(S) ]
```

Where the weight `w(S)` for the full Shapley formula is

```
w(S) = |S|! · (n − |S| − 1)! / n!
```

Each term `v(S ∪ {i}) − v(S)` is the **marginal contribution** of feature *i* when added to coalition *S*. The Shapley value is the weighted average of those marginal contributions across all coalitions that don't already contain *i*.

We never compute "true probability of diagnosis". Instead we use **semantic similarity to the full-input answer** as our value function (see 2.4).

### 2.3 Sampling — why we use `CounterfactualSampler`

A full enumeration over all `2^(n-1)` subsets is exponential. With 12 features that is 2 048 LLM calls per feature, ~24 576 total. Each call costs money and ~0.5–2 s.

`CounterfactualSampler` (`src/llmSHAP/attribution_methods/coalition_sampler.py:15`) does the cheapest possible approximation — it yields **exactly one coalition per feature**: *all other features except this one*, with weight `1.0`.

```python
def __call__(self, feature, keys):
    coalition = {k for k in keys if k != feature}
    yield coalition, 1.0
```

So our φ_i collapses to:

```
φ_i ≈  v(N) − v(N \ {i})
```

i.e. *"how much worse does the answer become when I remove this one feature, compared to the full-input answer?"* This is a **leave-one-out attribution**, mathematically a special case of Shapley with weight 1 on the singleton coalition. It is N+1 LLM calls instead of exponentially many — the trade-off that makes the demo run in seconds, not minutes.

### 2.4 The value function `v` — semantic similarity, not probability

`EmbeddingCosineSimilarity` (`src/llmSHAP/value_functions.py:102`):

1. Embed both texts (`text-embedding-3-small`, 1 536-dim vectors).
2. Cosine similarity between them:

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

The reference `base_generation` is the LLM's output **with all features present** (the "grand coalition"). For each coalition `S`, we measure how similar the diagnosis-with-S is to the full diagnosis.

So `v(S) ∈ [-1, 1]`, in practice `[0, 1]` for non-empty medical text. `v(N) ≈ 1.0` (compared to itself). The marginal contribution `v(S ∪ {i}) − v(S)` is positive when adding feature *i* makes the answer **more similar** to the full one — meaning *i* is a feature the model relied on. Negative values mean adding *i* pushes the model **away** from its full-input answer (suppressing).

Outputs are LRU-cached (max 2 000 pairs) so repeated comparisons are free.

### 2.5 Execution flow inside `attribution()`

For each feature `i` in the data dict:

1. Build the coalition set `S` from the sampler (one per feature with `CounterfactualSampler`).
2. In a thread pool (`num_threads=16`) submit two LLM calls per coalition: one for `v(S ∪ {i})` ≡ full input, one for `v(S)` ≡ leave-one-out.
3. The `_get_output` helper (`attribution_function.py:50`) deduplicates calls with the same coalition via a `Future`-backed cache (`use_cache=True`). The full-input call is reused across every feature.
4. `_compute_marginal_contribution` returns `weight * (v(base, with) − v(base, without))`. With CounterfactualSampler this is just `v(base, base) − v(base, leave_one_out)`.
5. `fsum(contributions)` is a numerically-stable sum over all sampled coalitions for that feature — gives `shapley_value`.
6. `_add_feature_score(i, shapley_value)` writes `{feature: {"value": <sentence>, "score": <φ_i>}}` into the result dict.

**Concrete LLM-call count in the demo** (with 8 features and `CounterfactualSampler`):

- 1 grand-coalition call (`v(N)`, cached and reused everywhere)
- 1 empty-coalition call (`v(∅)`, used for the `empty_baseline_value` reported on the result)
- 8 leave-one-out calls (one per feature, each cached so reuse is automatic)
- ≈ 9 unique embedding-API calls (one pair per unique generation pair)

So the heavy work is roughly **N+2 generation calls** plus **~N+2 embedding calls** — fast enough for a live demo.

### 2.6 Normalisation — the `*100%` we display

After attribution we have raw Shapley scores `φ_i`. They live on the cosine-similarity scale and are not directly interpretable as percentages. `_normalized_result()` (`attribution_function.py:45`) rescales:

```
score_i_normalized  =  φ_i  /  Σ_j |φ_j|
```

This guarantees `Σ|score_i_normalized| = 1`, so multiplying by 100 gives the **share-of-impact percentages** we show in the UI ("42%", "31%"). Sign is preserved: positive = contributed to the diagnosis, negative = pulled away from it.

The Attribution object that comes back from `.attribution()` then carries:

- `output` — the diagnosis text (the grand-coalition generation)
- `attribution` — a dict `{feature_key: {"value": sentence, "score": normalized_φ}}`
- `empty_baseline` — `v(base, ∅)`, the cosine-sim of the full-input answer to the answer-with-no-features
- `grand_coalition_value` — `v(base, base) ≈ 1.0`, sanity check

---

## Stage 3 — Rendering the highlighted chips

`render_highlighted_text(attribution)` (app.py:172).

1. Filter out internal keys (`__context__`) and empty values.
2. Compute `max_score = max(|score_i|)` for normalisation **across this single result** (so the strongest feature always reaches saturated colour, regardless of how big the absolute Shapley values were).
3. For each feature build a `<span class="word-chip">` with:
   - **background colour** from `score_to_color(score, max_score)`:
     - Compute `normalized = score / max_score` ∈ `[-1, 1]`.
     - If `> 0` → interpolate white → red. RGB is `(255, 255*(1−0.85·t), 255*(1−0.85·t))` with `t = normalized`. Alpha goes from `0.25` (faint) to `0.90` (bold).
     - If `< 0` → interpolate white → blue, symmetric formula.
     - If `= 0` → neutral grey `rgba(80,80,80,0.3)`.
   - **text colour** is white when `|normalized| > 0.3` (strong chips have white text), light grey otherwise.
   - **percentage label** `f"{|score|*100:.0f}%"` with a small red/blue super-script.
   - **key label** above the value (`SYMPTOM 1`, `AGE`, ...) for traceability.
4. Joined into a single HTML string and dropped into a dark `<div>`.

Result: every feature is one chip, sized like a word, colour-coded by sign, percentage stamped on it. A clinician can scan the layout and get the explanation in a second.

---

## Stage 4 — The Plotly bar chart

In `app.py:382-424`:

1. Take the same filtered `feature_items` dict.
2. Sort by `|score|` descending — biggest impact at the top.
3. Build a horizontal `go.Bar`:
   - x = `score * 100`  (the signed percent — bars extend right for positive, left for negative)
   - y = key labels (`symptom 1`, `age`, …) — full sentence is in `customdata` so it appears on hover
   - colour: red if positive, blue if negative
   - text label on each bar: `f"{|s|:.1f}%"`
4. `yaxis.autorange="reversed"` so the largest bar is at the top.

That's the second visualisation — same data, different layout. The chip view tells you *where* in the input each feature lives; the bar chart gives a clean ranking.

---

## Putting the math in one line per stage

| Stage | Math |
|---|---|
| Feature extraction | LLM-driven sentence segmentation → fixed feature set N |
| Shapley sampling | One coalition per feature: `S_i = N \ {i}` (CounterfactualSampler) |
| Marginal contribution | `Δ_i = v(N) − v(N\{i})` |
| Value function | `v(g₁, g₂) = cos(embed(g₁), embed(g₂))` ∈ `[-1, 1]` |
| Aggregation | `φ_i = 1 · Δ_i` (weight = 1 with this sampler) |
| Normalisation | `φ̂_i = φ_i / Σ_j |φ_j|` ⇒ Σ|φ̂_i| = 1 |
| Display | percent = `100·|φ̂_i|`, colour = sign(φ̂_i) interpolated red/blue by `|φ̂_i| / max_j|φ̂_j|` |

---

## What this gives the clinician

Every chip is a real, leave-one-out experiment: *"if I hide this fact from the LLM, how much does its diagnosis drift away from the full answer, measured in semantic-embedding space?"* The bigger the drift, the bigger the percent. The sign tells you whether the feature pulled the answer toward (red) or away from (blue) the full-input diagnosis.

That's it — no black box, just N+2 LLM calls and a cosine.
