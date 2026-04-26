# XAI-Care — Process Overview: From Symptoms to Explainable Diagnosis

## Overview

XAI-Care uses **llmSHAP** to provide physicians not just a diagnosis, but also a **granular explanation** of *which* patient facts drove the AI's conclusion.

The process consists of **four main steps**:

1. **Preprocessing** — Structure patient data
2. **Symptom Extraction** — Break down free text into meaningful features
3. **SHAP Attribution** — Calculate each factor's impact
4. **Diagnosis & Visualization** — Display results with color-coded explanations

---

## Steg 1: Förbehandling (Preprocessing)

### Vad händer här?

Istället för att behandla varje ord som separat nyckel (vilket skulle ge 50+ "faktorer" för en normal symptombeskrivning), **grupperar vi data** logiskt:

### Input från formuläret:
```
Ålder:                65 år
Kön:                  Man
Plats/Sjukhus:        Stockholm
Tidigare sjukdomar:   Diabetes, högt blodtryck
Symptombeskrivning:   "Radiating smärta i vänster arm, tryck över bröstet, 
                       illamående och kall svettning de senaste 2 timmarna"
```

### Output från förbehandling:
```json
{
  "age":              "The patient is 65 years old.",
  "sex":              "The patient is male.",
  "location":         "The patient is from Stockholm.",
  "conditions":       "Previous conditions: Diabetes, high blood pressure.",
  "symptom_1":        "The patient has radiating pain in the left arm.",
  "symptom_2":        "The patient has pressure over the chest.",
  "symptom_3":        "The patient has nausea.",
  "symptom_4":        "The patient has cold sweating for the past 2 hours."
}
```

### Hur gör vi det?

1. **Strukturerade fält** blir direkt till nycklar:
   - `age` → "The patient is 65 years old."
   - `sex` → "The patient is male."
   - `location` → "The patient is from Stockholm."
   - `conditions` → "Previous conditions: [...]"

2. **Friteksten (symptombeskrivning)** skickas till **samma LLM** (men utan llmSHAP) med instruktionen:
   > "Extract all distinct symptoms from the text. Return a JSON array of complete sentences, 
   > each starting with 'The patient has'. Keep medical context."

3. **LLM:n svarar** med en JSON-array av individuella symptommeningar.

4. **Vi indexerar** svaret som `symptom_1`, `symptom_2`, osv.

### Varför är detta viktigt?

- **Meningsfulla nycklar** istället för enskilda ord
- **Läkaren kan förstå** exakt vilka symptomer som mättes upp
- **llmSHAP får bättre data** — den jobbar med semantiska enheter, inte tokens

---

## Steg 2: Symptomextraktion — Detaljerna

### Prompten som skickas till LLM:

```
Extract all distinct symptoms from the text below. 
Return a JSON array of strings. Each string must be one complete sentence 
describing exactly one symptom, starting with 'The patient has' or 
'The patient is experiencing'. 
Respond in the same language as the input. 
Return ONLY valid JSON – no markdown, no explanation.

Text: [SYMPTOM DESCRIPTION HERE]
```

### Exempel: Input vs Output

**Input:**
```
Radiating smärta i vänster arm, tryck över bröstet, illamående och kall svettning 
de senaste 2 timmarna. Patienten har svårt att andas och är orolig.
```

**Output (JSON array):**
```json
[
  "The patient has radiating pain in the left arm.",
  "The patient has pressure over the chest.",
  "The patient has nausea.",
  "The patient has cold sweating for the past 2 hours.",
  "The patient has difficulty breathing.",
  "The patient appears anxious."
]
```

### Kodimplementering:

```python
def preprocess_input(...):
    # 1. Strukturerade fält blir direkt nycklar
    data = {
        "age": f"The patient is {age} years old.",
        "sex": f"The patient is {sex.lower()}.",
        "location": f"The patient is from {location}.",
    }
    
    # 2. Skicka symptombeskrivning till LLM för extraktion
    extraction_prompt = "Extract all distinct symptoms..."
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.0,  # Deterministic — samma input → samma output
        max_tokens=400,
    )
    
    # 3. Parsa JSON-responsen
    symptom_list = json.loads(response.choices[0].message.content)
    
    # 4. Lägg till varje symptom som separat nyckel
    for i, symptom in enumerate(symptom_list, start=1):
        data[f"symptom_{i}"] = symptom
    
    return data
```

---

## Steg 3: SHAP-Attribuering — Hur llmSHAP Beräknar Påverkan

### Vad är SHAP?

**SHAP** (SHapley Additive exPlanations) är en spelteorisk metod för att fördela en modells output "poäng" mellan varje input-faktor, baserat på hur mycket var och en **verkligen bidrog**.

### Processen i korthet:

1. **Baslinje** — Kör LLM:n med *alla* nycklar närvarande → **Diagnos A**

2. **Masking-experiment** — För *varje* nyckel:
   - Ta bort (maskera) den nyckeln → kör LLM:n igen → **Diagnos B**
   - Skillnad = hur mycket den nyckeln påverkade utgången

3. **Formell beräkning** — llmSHAP använder Shapley-värden från spelteori för att beräkna det "rättvisa" bidraget från varje nyckel (inte bara enkel differens)

4. **Resultat** — En poäng för varje nyckel mellan −1 och +1:
   - **Positiv värde** = Nyckeln *push:ar* mot diagnosenudgångspunkten
   - **Negativ värde** = Nyckeln *suppress:ar* diagnosenudgångspunkten
   - **Nära noll** = Neutral påverkan

### Konkret exempel:

**Scenario:** LLM diagnostiserar "akut koronart syndrom (hjärtinfarkt)"

| Nyckel | Värde | Tolkn |
|--------|-------|--------|
| `age` (65 år) | +0.12 | Moderat positivt — ålder är en riskfaktor |
| `sex` (man) | +0.08 | Svagt positivt — män har högre risk |
| `conditions` (diabetes) | +0.18 | **Starkt positivt** — diabetes är stor riskfaktor |
| `symptom_1` (radiating pain, left arm) | **+0.42** | **MYCKET STARKT positivt** — klassiskt hjärtinfarktsymptom |
| `symptom_2` (chest pressure) | **+0.31** | **Starkt positivt** — klassisk symptom |
| `symptom_3` (nausea) | +0.05 | Svagt positivt |
| `symptom_4` (cold sweating) | +0.19 | Positivt — hjärtinfarkt-signal |

**Slutsats från detta:** Radierande smärta i vänster arm och tryck över bröstet är de **mest avgörande faktorerna** för diagnosen.

### Vilken sampling-strategi använder vi?

Vi väljer automatiskt baserat på antal faktorer:

| Faktorer | Strategi | Förklaring |
|----------|----------|------------|
| ≤ 8 | **Full Enumeration** | Testar *alla* möjliga kombinationer (mest exakt, långsammare) |
| > 8 | **Random Sampling** | Samplar ~40% av kombinationerna (snabbare, ungefär lika exakt) |
| *Snabbläge* | **Counterfactual** | Leave-one-out: maskera bara en faktor åt gången (snabbast) |

---

## Steg 4: Diagnosis & Visualisering

### Diagnospromten (System Prompt till LLM):

```
You are a medical diagnostic assistant at an emergency department.
You receive patient information as a list of facts (some may be absent).
Give a concise diagnostic assessment in 2–3 sentences based only on what is present.
End with: 'Primary assessment: [diagnosis]'.
```

### Hur diagnosen framkommer:

1. **DataHandler bygger en prompt** med alla nycklar och deras värden:
   ```
   The patient is 65 years old.
   The patient is male.
   The patient is from Stockholm.
   Previous conditions: Diabetes, high blood pressure.
   The patient has radiating pain in the left arm.
   The patient has pressure over the chest.
   The patient has nausea.
   The patient has cold sweating for the past 2 hours.
   ```

2. **LLM:n läser detta** tillsammans med systempromten och genererar:
   ```
   The patient presents with classic acute coronary syndrome symptoms: 
   left arm radiating pain and substernal chest pressure. At 65 with diabetes 
   and hypertension, the risk profile supports immediate cardiac intervention.
   Primary assessment: Acute coronary syndrome / possible myocardial infarction.
   ```

3. **llmSHAP sparar denna diagnos** och det är vår **referenspunkt** för attribuering.

4. **Masking-experiment** — Sedan maskar vi *varje* nyckel och körs LLM igen:
   - Utan `symptom_1`: Diagnosen blir mer osäker? → högt SHAP-värde
   - Utan `age`: Diagnosen förändras knappt? → lågt SHAP-värde

### Visualiseringen:

**Färgkodning:**
- 🔴 **Röda/varmfärgade** chips = Positiv påverkan (push mot diagnosen)
- 🔵 **Blåa/kaldfärgade** chips = Negativ påverkan (suppresser diagnosen)
- ⚪ **Grå** chips = Neutral påverkan

**Presentationen:**
```
SYMPTOM 1: The patient has radiating pain in the left arm          42%
SEX: The patient is male.                                           8%
SYMPTOM 2: The patient has pressure over the chest.               31%
CONDITIONS: Previous conditions: Diabetes, high blood pressure.    18%
SYMPTOM 4: The patient has cold sweating for the past 2 hours.    19%
AGE: The patient is 65 years old.                                 12%
SYMPTOM 3: The patient has nausea.                                 5%
```

**Stapeldiagram:** Samma värden i fallande ordning för snabb insikt.

---

## Steg-för-steg: En Hel körning

### 1. Läkaren fyller i formulär:
```
Ålder: 65
Kön: Man
Plats: Stockholm
Tidigare: Diabetes, högt blodtryck
Symptom: "Radiating pain in left arm, chest pressure, nausea, cold sweating"
```

### 2. Förbehandling körs (< 1 sekund):
```python
data = preprocess_input(65, "Male", "Stockholm", "Diabetes, hypertension", 
                        "Radiating pain...", "gpt-4o-mini", api_key)
# Result:
# {
#   "age": "The patient is 65 years old.",
#   "sex": "The patient is male.",
#   ...
#   "symptom_1": "The patient has radiating pain in the left arm.",
#   "symptom_2": "The patient has pressure over the chest.",
#   ...
# }
```

### 3. llmSHAP initieras:
```python
handler = DataHandler(data)  # 7 nycklar totalt
sampler = FullEnumerationSampler(7)  # Testas alla 2^7 = 128 kombinationer

attribution_result = ShapleyAttribution(
    model=openai_interface,
    data_handler=handler,
    prompt_codec=system_prompt,
    sampler=sampler,
    use_cache=True,
    num_threads=16,
).attribution()
```

### 4. För varje kombination:
- **Iteration 1:** `[age, sex, location, conditions, symptom_1, symptom_2, symptom_3, symptom_4]` 
  → "Acute coronary syndrome" ✓

- **Iteration 2:** `[age, sex, location, conditions, symptom_2, symptom_3, symptom_4]` 
  (utan `symptom_1`) → "Possible cardiac issue?" (mer osäker)

- **Iteration 3:** `[age, sex, location, conditions, symptom_1, symptom_2, symptom_3, symptom_4]` 
  (utan `sex`) → "Acute coronary syndrome" (identisk)

- ... (totalt 128 iterationer)

### 5. Shapley-formeln beräknas:
```
SHAP[symptom_1] = (0.42 - andra bidrag) = +0.42
SHAP[symptom_2] = (0.31 - andra bidrag) = +0.31
SHAP[conditions] = (0.18 - andra bidrag) = +0.18
SHAP[symptom_4] = (0.19 - andra bidrag) = +0.19
SHAP[age] = (0.12 - andra bidrag) = +0.12
SHAP[sex] = (0.08 - andra bidrag) = +0.08
SHAP[symptom_3] = (0.05 - andra bidrag) = +0.05
```

### 6. UI:t visar resultaten:

```
┌─────────────────────────────────────────────┐
│ DIAGNOSIS                                   │
├─────────────────────────────────────────────┤
│ The patient presents with classic acute     │
│ coronary syndrome symptoms: left arm        │
│ radiating pain and substernal chest         │
│ pressure. At 65 with diabetes and           │
│ hypertension, the risk profile supports     │
│ immediate cardiac intervention.             │
│ Primary assessment: ACS/MI                  │
└─────────────────────────────────────────────┘

FEATURE IMPACT (XAI)

[🔴 SYMPTOM 1: radiating pain... 42%] 
[🔴 SYMPTOM 2: chest pressure... 31%]
[🔴 SYMPTOM 4: cold sweating... 19%]
[🔴 CONDITIONS: Diabetes... 18%]
[🔴 AGE: 65 years... 12%]
[🟠 SYMPTOM 3: nausea... 5%]
[🟠 SEX: male... 8%]

TOP CONTRIBUTING FEATURES (BAR CHART)
─────────────────────────────────────
SYMPTOM 1     ████████████████████ 42%
SYMPTOM 2     ███████████████ 31%
SYMPTOM 4     █████████ 19%
CONDITIONS    █████████ 18%
AGE           ██████ 12%
SEX           ████ 8%
SYMPTOM 3     ██ 5%
```

---

## Varför är detta viktigt för läkare?

### Problem utan XAI:
```
LLM: "Det är hjärtinfarkt."
Läkare: "Vad baserar du det på? Hur kan jag lita på det?"
```

### Lösning med XAI (llmSHAP):
```
LLM: "Det är hjärtinfarkt."
SHAP: "Främst på grund av:
  1. Radierad smärta i vänster arm (42% påverkan)
  2. Tryck över bröstet (31%)
  3. Kall svettning (19%)
  4. Tidigare diabetes (18%)"

Läkare: "Ja, det stämmer. Dessa är klassiska tecken. 
Jag kan agera baserat på denna förklaring."
```

### Fördelar:
1. **Transparens** — Läkaren ser exakt vilka faktorer som drev beslutet
2. **Granskning** — Kan ifrågasätta om nyckelns påverkan är medicinskt rimlig
3. **Kontroll** — Kan ignorera AI:n om den fokuserar på irrelevanta faktorer
4. **Tillit** — AI blir en *partner* istället för en "svart låda"
5. **Utbildning** — Nya läkare lär sig vilka faktorer som verkligen spelar roll

---

## Tekniska detaljer: Varför Shapley?

### Alternativ som vi **inte** använder:

**LIME (Lime-based explanations):**
- Enklare, snabbare
- Men: Lokala approximationer kan vara inexakta
- Problem: Två läkare med samma patientdata kan få olika förklaringar

**Attention weights:**
- Visar bara vad modellen "tittar på"
- Inte samma som orsakssamband

### Varför Shapley är bäst för medicin:

1. **Rättvis fördelning** — Alla faktorer får "kredit" för deras sanna bidrag
2. **Konsistent** — Samma input → samma SHAP-värden
3. **Verifierad** — Används i många höga regulatoriska miljöer (FDA, etc.)
4. **Teoristöd** — Baserad på spelteori, inte bara heuristik

---

## Sammanfattning: Processflödet

```
┌─────────────────────────────────────────┐
│ 1. Läkare fyller i formulär             │
│    (Strukturerade fält + symptomtext)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. Förbehandling (Preprocessing)        │
│    LLM extraherar semantiska symptom    │
│    från friteksten (utan llmSHAP)       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. SHAP-attribuering                    │
│    llmSHAP maskar varje nyckel          │
│    och beräknar påverkan på diagnos     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 4. Resultat & Visualisering             │
│    Färgkodade chips med procent          │
│    + stapeldiagram + rådata              │
└─────────────────────────────────────────┘
```

---

## Kodflödet (Implementering)

```python
# app.py:300
if analyze_btn:
    # 1. Preprocessing
    with st.spinner("Extracting features from symptoms..."):
        diagnosis, attribution, preprocessed = run_analysis(
            age, sex, location, conditions, symptoms,
            model_name, api_key, use_fast_mode
        )
    
    # 2. Show extracted features (transparency)
    with st.expander("Extracted features"):
        for key, value in preprocessed.items():
            st.markdown(f"**{key}:** {value}")
    
    # 3. Show diagnosis
    st.markdown("### Diagnosis")
    st.markdown(f'<div class="diagnosis-box">{diagnosis}</div>')
    
    # 4. Show SHAP attribution (colored chips)
    st.markdown("### Feature Impact (XAI)")
    highlighted_html = render_highlighted_text(attribution)
    st.markdown(f'<div...>{highlighted_html}</div>')
    
    # 5. Bar chart
    st.markdown("### Top Contributing Features")
    # ... plot sorted features by SHAP value
```

---

**Denna process är vårt bidrag till hackathonen:** Vi visar inte bara att AI kan diagnostisera, utan att det kan göra det **på ett sätt läkare kan förstå, verifiera och lita på**. Det är vägen från "svart låda" till "transparent AI" för vård.
