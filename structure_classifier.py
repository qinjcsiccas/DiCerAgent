"""
Intelligent structure type classifier - uses a large language model to determine material structure type
"""

import requests
import config

def call_llm(prompt):
    """Call DeepSeek API"""
    if not config.DS_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {config.DS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.DS_MODEL,
        "messages": [
            {"role": "system", "content": "You are a materials science expert. Analyze the chemical formula and determine the structure type."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            f"{config.DEEPSEEK_API_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except:
        pass
    return None

def analyze_structure_with_llm(formula: str, structure_info: str = "") -> dict:
    """
    Analyze the material structure type using a large language model
    
    Args:
        formula: chemical formula, e.g. "BaTiO3", "CaWO4"
        structure_info: optional crystal structure information
    
    Returns:
        dict: {
            "structure_type": "ABO3" / "ABO4" / "spinel" / "perovskite" / "unknown",
            "confidence": 0.0-1.0,
            "reasoning": "analysis reasoning"
        }
    """
    
    # Known structure type templates
    structure_prompt = f"""Analyze the following material and determine its crystal structure type.

Chemical Formula: {formula}
Structure Information: {structure_info}

Known structure types:
- ABO3: Perovskite structure (e.g., BaTiO3, SrTiO3, CaTiO3)
- ABO4: Tungsten bronze structure (e.g., CaWO4, SrWO4, BaWO4)
- AB2O4: Spinel structure (e.g., MgAl2O4, ZnFe2O4)
- ABO2: Corundum structure (e.g., TiO2 is not ABO2, but alpha-Al2O3 is corundum)
- SiO2: Various polymorphs (quartz, cristobalite, etc.)
- AX: Rock salt (e.g., NaCl, MgO)
- AX2: Fluorite (e.g., CaF2, UO2)
- Perovskite: General ABO3
- Tungsten Bronze: General ABO4
- Unknown: Cannot determine

Please respond in JSON format:
{{"structure_type": "ABO3", "confidence": 0.95, "reasoning": "The formula BaTiO3 is a classic perovskite structure with A-site Ba2+, B-site Ti4+, and O2- in cubic arrangement"}}

If uncertain, set confidence to lower value."""

    result = call_llm(structure_prompt)
    
    if not result:
        return {"structure_type": "Unknown", "confidence": 0.0, "reasoning": "LLM not available"}
    
    # Try to parse JSON
    import json
    import re
    
    try:
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # If parsing fails, return unknown
    return {"structure_type": "Unknown", "confidence": 0.0, "reasoning": result}

def determine_structure_type(formula: str, structure_info: str = "", use_llm: bool = True) -> str:
    """
    Determine the structure type comprehensively (rules + LLM)
    
    1. First use rules for quick judgment
    2. If uncertain, use the LLM
    """
    # Quick rule-based judgment
    result = rule_based_check(formula)
    if result != "Unknown":
        return result
    
    # Use LLM
    if use_llm:
        llm_result = analyze_structure_with_llm(formula, structure_info)
        return llm_result.get("structure_type", "Unknown")
    
    return "Unknown"

def rule_based_check(formula: str) -> str:
    """Quick rule-based judgment based on the chemical formula"""
    import re
    
    # Remove numbers (but keep numbers inside the formula)
    # First handle numbers inside parentheses
    formula_clean = formula
    
    # Use pymatgen for more accurate parsing
    try:
        from pymatgen.core import Composition
        comp = Composition(formula)
        el_dict = comp.get_el_amt_dict()
    except:
        # If parsing fails, use a simple method
        elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        el_dict = {}
        for el, num in elements:
            if el:
                count = int(num) if num else 1
                el_dict[el] = el_dict.get(el, 0) + count
    
    if 'O' not in el_dict:
        return "Unknown"
    
    o_amt = el_dict.get('O', 0)
    if o_amt == 0:
        return "Unknown"
    
    # Compute cation-to-oxygen ratio
    cation_sum = sum(v for k, v in el_dict.items() if k != 'O')
    
    # ABO3: cation:O ≈ 2:3 (ratio ≈ 0.67, cation_sum ≈ 2)
    if o_amt == 3 and 1.5 < cation_sum < 2.5:
        return "ABO3"
    
    # ABO4: cation:O ≈ 2:4 = 1:2 (ratio = 0.5, cation_sum ≈ 2)
    # Common examples: CaWO4, SrWO4, BaWO4
    if o_amt == 4 and 1.5 < cation_sum < 2.5:
        return "ABO4"
    
    # AB2O4 (Spinel): cation:O ≈ 3:4 (ratio = 0.75, cation_sum = 3)
    if o_amt == 4 and 2.5 < cation_sum < 3.5:
        return "AB2O4"
    
    return "Unknown"

if __name__ == "__main__":
    # Test
    test_formulas = ["BaTiO3", "CaWO4", "SrTiO3", "MgO", "SiO2", "Fe3O4"]
    for f in test_formulas:
        result = determine_structure_type(f)
        print(f"{f}: {result}")
