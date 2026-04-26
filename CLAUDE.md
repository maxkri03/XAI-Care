# 🎯 Hackathon Context: XAI-Care

Du är min AI-assistent och kodningspartner under detta hackathon. Vårt mål är att bygga en Proof of Concept (PoC) som maximerar poängen enligt hackathonets domarkriterier. 

Läs igenom denna kontext noggrant. Alla dina framtida svar och kodförslag måste linjera med projektets mål och hackathonets regler.

---

## 🚀 Vårt Projekt: XAI-Care
**Mål:** Implementera `llmSHAP` för att förklara LLM-baserade diagnostiska beslut på en akutmottagning/vårdcentral.  
**Syfte:** Göra AI-beslut förståeliga, granskningsbara och tillförlitliga för läkare och patienter.  
**Viktig avgränsning:** Vi fokuserar BARA på text/XAI (llmSHAP). **Ingen** Computer Vision.

### Vårt llmSHAP
Vi använder det **specifika llmSHAP-biblioteket** från: https://github.com/filipnaudot/llmSHAP
- Detta är vår XAI-motor för att bryta ner LLM-beslut till SHAP-värden per ord
- Vi bygger PoC:n runt detta bibliotek, inte något annat SHAP-alternativ

### Framtidsvision vs. Dagens PoC (Viktigt för "Real-World Applicability")
* **Visionen (Produktion):** Vi är väl medvetna om att en verklig framtida implementation kräver lokala AI-modeller upptränade på historisk, sekretessbelagd data för att garantera patientintegritet. 
* **Vår PoC (Hackathon):** Detta projekt är vår demo för att visa *hur* detta säkerställs i praktiken genom förklarbarhet (XAI).

### Dataflöde (PoC) & Konkret Exempel
* **Input:** Patientdata (ålder, plats, tidigare sjukdomar) samt en frispråkig beskrivning av symptom.
* **Process:** En LLM analyserar inputen. Därefter appliceras `llmSHAP` för att bryta ner och förklara exakt *vilka* delar av texten som drev modellen till sitt beslut.
* **Output & XAI-Värdet:**
  * ❌ *Utan llmSHAP:* En LLM säger "patienten har hjärtinfarkt" – men läkaren vet inte varför och kan inte lita på svaret.
  * ✅ *Med llmSHAP (Vår lösning):* Samma LLM säger "patienten har hjärtinfarkt", och UI:t visar omedelbart att det beror främst på orden *"utstrålande smärta i vänster arm"* (42% påverkan) och *"tryck över bröstet"* (31% påverkan).

---

## ⚖️ Hackathon - Domarkriterier (Viktigt!)
Vi utvärderas på två lika stora delar. Allt vi bygger och designar måste svara mot dessa:

1.  **Solution Quality (50%):** * Hur väl löser vår PoC problemet?
    * Är den funktionell och demonstrerbar?
    * *Fokus:* Vi måste få `llmSHAP` att fungera i praktiken och bygga en demo som tydligt visar procentuell påverkan på specifika ord (som i exemplet ovan).
2.  **Real-World Applicability (50%):** * Hur relevant är problemet idag?
    * Kan det rullas ut i verkligheten?
    * *Fokus:* Vårdpersonal behöver transparens. Vår insikt om lokala LLMs kombinerat med llmSHAP visar en extremt tydlig, säker och etisk väg från PoC till en produktionsklar miljö.

---

## 📋 Regler & Inlämningskrav
* **Open Source:** All kod ska upp på GitHub.
* **Working PoC:** Lösningen måste vara körbar och demonstrerbar ("Show, do not tell: A working demo is better than a perfect pitch").
* **submission.md:** Vi måste skriva en koncis markdown-fil till repot som täcker problem statement, solution approach, application och real-world impact.
* **Verktyg:** Fria val av språk/ramverk.

---

## 🧠 Instruktioner till dig (Claude)
När vi nu börjar arbeta, håll dig till följande principer:
1.  **Börja Enkelt (Start simple):** Föreslå den snabbaste och mest stabila vägen för att få en MVP/PoC att snurra under en helg. Undvik over-engineering.
2.  **Användarfokus (Think users):** När vi designar UI/Output, utgå alltid från exemplet "Med llmSHAP" ovan. Hur kan vi presentera detta visuellt (t.ex. färgkodad (highlighted) text) så att en läkare förstår värderingen (42% vs 31%) på en sekund?
3.  **Praktiskt över Hypotetiskt:** Prioritera kod som fungerar idag och fokuserar på den visuella XAI-demonstrationen.
4.  **Kod-Enkelheten Framför Allt:** Vi är inte datavetenskap-experter utan Pythonister med praktisk kodförståelse. All kod ska vara **läsbar, kommenterad och enkel** – inga avancerade ML-tricks eller komplex matematisk optimering. Föredra raka, tydliga lösningar framför eleganta men svåra abstrakt.

Bekräfta att du har förstått kontexten genom att kort föreslå en tech-stack (Python, specifika bibliotek för LLM och llmSHAP, samt ett enkelt UI-ramverk som Streamlit/Gradio för att markera ord med färg/procent) som vi bör använda för att bygga denna PoC i helgen!