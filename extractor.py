from openai import OpenAI
import config
import json
import re

class KnowledgeExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=config.DS_API_KEY, base_url=config.DEEPSEEK_API_URL)

    def brainstorm_keywords(self, formula):
        """
        [Upgrade] Use LLM to generate synonyms, mineral names, and specific abbreviations.
        It must be Strict. No generic family names.
        """
        formula = formula.strip()
        system_prompt = "You are a Crystallography Expert."
        
        user_prompt = f"""
        Target Material: {formula}
        
        Task: Generate a list of 3-5 strictly specific search keywords/aliases.
        
        **PRIORITY ORDER**:
        1. **Standard Abbreviations**: (e.g., Ba(Mg1/3Ta2/3)O3 -> "BMT"; Pb(Zr,Ti)O3 -> "PZT")
        2. **Specific Mineral Names**: (e.g., CaCuSi4O10 -> "Gillespite"; MgTiO3 -> "Geikielite")
        3. **Chemical Formula Variations**: (e.g., YBa2Cu3O7 -> "YBCO", "Y123")
        
        **NEGATIVE CONSTRAINTS (CRITICAL)**:
        - **NO Broad Classifiers**: Do NOT output generic terms like "Dielectric Material", "Ceramic", "Oxide", or "Perovskite" as standalone keywords.
        - **NO Structural Modifiers**: Strictly EXCLUDE suffixes or descriptors like "-based", "-type", "-like", or "structure". (e.g., Output "Gillespite", NOT "Gillespite-based" or "Gillespite structure").
        
        **Example Output**:
        - Input: 'CaCuSi4O10' -> ["CaCuSi4O10", "Gillespite", "Egyptian Blue", "MCuSi4O10", "CaO-CuO-SiO2"]
        - Input: 'Ba(Mg1/3Ta2/3)O3' -> ["Ba(Mg1/3Ta2/3)O3", "BMT", "Ba(Mg,Ta)O3"]
        - Input: 'TiO2' -> ["TiO2", "Rutile", "Anatase", "Brookite"]
        - Input: 'BaMoO4' -> ["BaMoO4", "Scheelite", "Barium molybdate"]
        
        Output Format: JSON list of strings only.
        """
        try:
            response = self.client.chat.completions.create(
                model=config.DS_MODEL, 
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1 # keep hallucination probability low
            )
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: content = content.replace("```json", "").replace("```", "")
            elif "```" in content: content = content.replace("```", "")
            
            keywords = json.loads(content)
            
            if not isinstance(keywords, list) or not keywords:
                return [formula]
            
            # Forcibly add the original formula back as a fallback
            if formula not in keywords:
                keywords.insert(0, formula)
                
            # Secondary cleanup: filter out overly short abbreviations (e.g. "A") or purely generic words
            filtered = []
            for k in keywords:
                k_clean = k.strip()
                # Again prevent the LLM from spuriously outputting "Perovskite"
                if k_clean.lower() in ["perovskite", "spinel", "garnet", "ceramic", "oxide", "solid solution"]:
                    continue
                if len(k_clean) >= 2:
                    filtered.append(k_clean)

            return list(set(filtered))
            
        except Exception as e:
            print(f"   ⚠️ Brainstorm Error: {e}")
            return [formula]

    def extract_property_values(self, formula, literature_context, summary=None):
        """
        [v13.0 Direct-Labeling] 
        Simplified logic: Just report what is seen.
        - If pure phase value is explicit -> Extract value, Label "Pure".
        - If series range is found -> Extract range, Label "Series Range (Formula)".
        """
        input_text_block = ""
        if summary:
            input_text_block += f"=== REPORT SUMMARY ===\n{summary}\n\n"
        
        input_text_block += f"=== RAW TEXT FRAGMENTS ===\n{literature_context}"

        system_prompt = (
            "You are a Data Extraction Assistant. "
            "Goal: Extract property values (εr, Q×f, τf) for the target material, "
            "including its pure phase, doped compositions, and solid solution series. "
            "Rule: Be honest and thorough. If data is from a doped composition or a solid "
            "solution series, extract it and label the condition with the specific "
            "composition/series. Do NOT restrict extraction to the pure phase only."
        )
        
        user_prompt = f"""
        **Target Material**: {formula}
        
        **Input Text**:
        {input_text_block}
        
        **EXTRACTION RULES**:
        1. **Pure Phase**: If the text explicitly says "Pure {formula} has εr = 5.7", extract "5.7" with condition "Pure".
        2. **Doped / Substituted Compositions** (CRITICAL - MUST EXTRACT):
           - If the text reports values for a doped or substituted composition, e.g.
             "La-doped {formula}", "{formula} doped with 1 mol% Mn", "Ba_x Sr_1-x TiO3 (x=0.3)",
             "A-site/B-site doping", extract the value and set condition to the SPECIFIC
             composition as written in the text (e.g., "La-doped", "x=0.3", "Ba_x Sr_1-x TiO3").
           - Extract ALL compositions reported, do not only keep the pure phase entry.
        3. **Solid Solution Series / Composition Ranges**:
           - If the text reports values for a series with varying composition x (e.g.,
             "Ba_x Sr_1-x TiO3", "(Ca_x Sr_1-x)CuSi4O10"), extract EACH explicitly stated
             composition (x=0.2, x=0.5, ...) as separate entries with their own values.
           - If the text reports a single RANGE for the series (e.g., "properties range from A to B"),
             extract the Range "A-B" with condition "Series Range (Series Formula)".
           - If the best or optimal composition is stated (e.g., "maximum Q×f at x=0.5"),
             extract that value with condition "x=0.5".
        4. **Multiple Data Points**: Include every distinct reported value per property;
           do not drop doped/series entries even if the pure phase value is also present.
        5. **Output Keys**: ONLY these three keys in the output JSON: "er", "qxf", "tcf".
           No other property keys (do NOT add tm, tan_delta, etc.).
           Return each key as an array; use [] when no data is found for that property.
           
        **Output Format (JSON)**:
        {{
            "er":  [ 
                {{"value": "5.7", "source": "[2]", "condition": "Pure"}},
                {{"value": "5.70-5.82", "source": "[3]", "condition": "Series Range (Ca_x Sr_1-x)CuSi4O10"}},
                {{"value": "6.1", "source": "[4]", "condition": "x=0.3"}}
            ],
            "qxf": [], 
            "tcf": []
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=config.DS_MODEL, 
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: content = content.replace("```json", "").replace("```", "")
            elif "```" in content: content = content.replace("```", "")
                
            data = json.loads(content)
            
            # Data Cleaning
            for k in ['er', 'qxf', 'tcf']:
                if k not in data or data[k] is None: data[k] = []
                elif isinstance(data[k], dict): data[k] = [data[k]]
                
                valid_items = []
                for item in data[k]:
                    # Remove model-knowledge-completion entries lacking literature support, preventing them from mixing into "extracted literature values"
                    if isinstance(item, dict) and item.get('source'):
                        _src_l = str(item['source']).lower()
                        if any(_t in _src_l for _t in (
                            'model knowledge', 'not in retrieved', 'model supplement',
                            'ai-generated', 'from model', 'language model', 'generated',
                        )):
                            continue
                    if 'value' in item: item['value'] = str(item['value'])
                    if 'source' in item:
                        src = str(item['source']).strip()
                        if src and not src.startswith("["):
                            match = re.search(r'\[.*?\]', src)
                            item['source'] = match.group(0) if match else f"[{src}]"
                    if 'condition' not in item or not item['condition']:
                        item['condition'] = "Unspecified"
                    else:
                        cond_clean = str(item['condition']).strip()
                        # Normalize pure-phase variants to "Pure" (the LLM may output variants such as "Pure BaTiO3")
                        if cond_clean.lower().startswith("pure"):
                            item['condition'] = "Pure"
                    valid_items.append(item)
                data[k] = valid_items

            return data
        except Exception as e: 
            print(f"   ⚠️ Extraction Error: {e}")
            return {"er": [], "qxf": [], "tcf": []}

    def generate_processing_guide(self, formula, literature_context):
        """Backward compatible: generate the concise report by default"""
        return self.generate_report(formula, literature_context, style="concise")

    def generate_report(self, formula, literature_context, style="concise"):
        """
        Generate a technical report.
        style: "concise" - concise (~300-500 words); "detailed" - detailed (~800-1200 words);
               "focused" - lightweight interpretation focusing only on microwave dielectric properties (~150-250 words, without structure/synthesis/application sections)
        """
        if style == "detailed":
            system_prompt = (
                "You are a Senior Materials Scientist. Write thorough, publication-ready "
                "literature reviews. Your reports should include mechanistic analysis, "
                "comparative discussion of different synthetic routes, and structure-property "
                "relationships. Never use conversational filler ('Of course', 'Here is', etc.). "
                "Do not add a top-level title like 'Technical Report: X' — the report heading "
                "is already shown elsewhere. Jump straight into content."
            )
            user_prompt = f"""
Material: {formula}
Reference Context: {literature_context}

Write a comprehensive technical synthesis (aim for 800-1200 words) covering:

1. Structure & Phase Information — crystal system, space group, lattice parameters,
   phase stability across temperature ranges, polymorphism, and how structure influences
   dielectric performance.
2. Synthesis Methods — detailed comparison of preparation routes (solid-state,
   sol-gel, hydrothermal, etc.), critical processing parameters (calcination temperature,
   sintering conditions), and how different routes affect phase purity and microstructure.
3. Dielectric Properties — εr, Q×f, τf values organized by pure vs. doped compositions.
   Discuss trends: how dopants/ substitutions shift properties, the underlying
   mechanisms (polarizability, tolerance factor, oxygen vacancies), and any
   reported structure-property correlations.
4. Applications & Outlook — specific microwave/RF applications where this material
   has been tested, performance comparisons with competing materials, and promising
   research directions.

Rules:
- Separate pure-phase from doped/series data with [Pure] / [Doped] labels
- Include quantitative values whenever available
- Discuss mechanisms behind property trends, not just list numbers
- Cite references inline as [1], [2] etc.
- No preamble, no sign-off, no "I hope this helps"
- Output valid Markdown, starting with ## headings
- Prefer data from the given Reference Context; if dielectric properties (er, Qxf, tcf) are NOT covered by the retrieved references (e.g. the material is studied only for optics/luminescence/fluorescence), do NOT say "references do not report dielectric data". Instead, supplement er/Qxf/tcf ranges for the pure phase and doped series using your own domain knowledge, and clearly label every model-supplemented section/paragraph as AI-generated: right after the section heading, insert a visible caution banner such as "> ⚠ AI-generated supplement — these figures are produced by the language model, not sourced from the cited references or any knowledge base, and may be subject to hallucination; treat as indicative only.", and additionally mark each supplied value with "[from model knowledge; not in retrieved references]"; never attach a fabricated [n] citation to model-knowledge data.
- If the references do cover dielectric properties, cite them as [n] as usual.
- For other sections (structure/synthesis/applications) lacking information, write a neutral placeholder like "No specific data in the retrieved references." as the body — never output the section heading with empty body.
- Keep each section moderately expanded but do not overrun.
- Do NOT output a standalone "References" / "Bibliography" list section. Only inline [n] citations are allowed in the body; the full reference list is rendered separately by the system.
"""
        elif style == "focused":
            system_prompt = (
                "You are a Materials Scientist focused on microwave dielectric ceramics. "
                "When the user only asks about dielectric properties, answer with a tight, "
                "performance-focused synthesis. Never use conversational filler "
                "('Of course', 'Here is', etc.). Do not add a top-level title like "
                "'Technical Report: X' — the heading is already shown elsewhere. "
                "Jump straight into content."
            )
            user_prompt = f"""
Material: {formula}
Reference Context: {literature_context}

Write a FOCUSED dielectric-performance synthesis (aim for ~150-250 words, valid Markdown,
starting with ## headings). Cover ONLY:

1. **Dielectric Properties** — εr, Q×f, τf values, clearly separated into pure-phase vs.
   doped/substituted/series compositions, citing references inline as [n].
2. A short mechanism/trend note (no more than 3-5 lines) only if directly supported by the
   references (e.g. dopant effects, polarizability, tolerance factor).

STRICTLY DO NOT write sections about:
- Crystal structure / phase details (space group, lattice parameters, phase stability)
- Synthesis / processing routes (calcination, sintering, preparation methods)
- Applications or outlook

While answering ONLY about dielectric performance, if a structure or synthesis fact is
strictly necessary to explain one dielectric trend, keep it to a single clause and never
expand it into its own section.

Rules:
- Clearly separate pure-phase data from doped/series data (use labels like [Pure] or [Doped]).
- Cite references inline as [1], [2] etc. Never attach a fabricated [n] citation to
  model-knowledge data.
- If the references do NOT cover dielectric properties (e.g. material studied only for
  optics/luminescence), you may supplement εr/Q×f/τf ranges using your own domain knowledge,
  but label every model-supplemented figure with the banner
  "> ⚠ AI-generated supplement — figures from the language model, not the cited references,
  may be subject to hallucination; treat as indicative only."
  and mark each supplied value with "[from model knowledge; not in retrieved references]".
- No preamble, no sign-off. Output valid Markdown starting with ## headings.

Example skeleton (adjust to actual data):
## Dielectric Properties
- **[Pure]** {formula}: εr ≈ … , Q×f ≈ … GHz, τf ≈ … ppm/°C [1]
- **[Doped]** … [2]
## Trend / Mechanism
- …
"""
        else:
            system_prompt = (
                "You are a Senior Materials Scientist. Write direct, factual technical reports. "
                "Never use conversational filler ('Of course', 'Here is', etc.). "
                "Do not add a top-level title like 'Technical Report: X' — the report heading "
                "is already shown elsewhere. Jump straight into content."
            )
            user_prompt = f"""
Material: {formula}
Reference Context: {literature_context}

Write a concise technical synthesis covering:
1. Structure & Phase Information — crystal structure, phase stability, polymorphism
2. Synthesis Methods — key preparation routes reported
3. Dielectric Properties — εr, Q×f, τf values with clear pure/doped distinction
4. Applications & Outlook

Rules:
- Clearly separate pure-phase data from doped/series data (use labels like [Pure] or [Doped])
- Cite references inline as [1], [2] etc.
- No preamble, no sign-off, no "I hope this helps"
- Output valid Markdown, starting with ## headings
- Prefer data from the given Reference Context; if dielectric properties (er, Qxf, tcf) are NOT covered by the retrieved references (e.g. the material is studied only for optics/luminescence/fluorescence), do NOT say "references do not report dielectric data". Instead, supplement er/Qxf/tcf ranges for the pure phase and doped series using your own domain knowledge, and clearly label every model-supplemented section/paragraph as AI-generated: right after the section heading, insert a visible caution banner such as "> ⚠ AI-generated supplement — these figures are produced by the language model, not sourced from the cited references or any knowledge base, and may be subject to hallucination; treat as indicative only.", and additionally mark each supplied value with "[from model knowledge; not in retrieved references]"; never attach a fabricated [n] citation to model-knowledge data.
- If the references do cover dielectric properties, cite them as [n] as usual.
- For other sections (structure/synthesis/applications) lacking information, write a neutral placeholder like "No specific data in the retrieved references." as the body — never output the section heading with empty body.
- Keep it brief: 2-4 bullet points per section, 300-500 words total.
- Do NOT output a standalone "References" / "Bibliography" list section. Only inline [n] citations are allowed in the body; the full reference list is rendered separately by the system.
"""
        try:
            response = self.client.chat.completions.create(
                model=config.DS_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2048 if style == "detailed" else 1024,
            )
            return response.choices[0].message.content
        except:
            return "LLM Error"