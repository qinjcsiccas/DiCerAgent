"""
LLM Client
------------------------
Unified DeepSeek API call wrapper, including System Prompt templates for each method.
"""

import os, sys, json, re, time, requests, configparser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.config import (
    DS_API_KEY, DEEPSEEK_URL, LLM_MODEL, LLM_TIMEOUT,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_MAX_RETRIES,
)


# ============================================================
# System Prompts - eight methods
# ============================================================

SYS_ZS = (
    "You are an inorganic materials expert. Given a chemical formula, "
    'predict its property value. Output JSON: {"value": <number>, "confidence": <0-1>, '
    '"reasoning": "<2 sentences>"}'
)

SYS_ML_ADJUST = (
    "You are an inorganic materials expert. Given a material and its "
    "ML prediction from crystal structure features, refine the prediction based on your "
    'domain knowledge about composition-structure-property relationships. '
    'Output JSON: {"value": <number>, "confidence": <0-1>, "reasoning": "<2 sentences>"}'
)

SYS_LIT_FULL = (
    "You are an inorganic materials expert. Given a target material "
    "(formula) and similar materials from literature with REAL measured values, "
    'estimate the target property. Output JSON: {"value": <number>, "confidence": <0-1>, '
    '"reasoning": "<2 sentences>"}'
)

SYS_ML_LIT_FULL = (
    "You are an inorganic materials expert. Given a target material "
    "(formula + ML prediction from crystal structure) and similar materials from "
    "literature with REAL measured values, estimate the target property by fusing "
    'ML prediction with literature evidence. Output JSON: {"value": <number>, '
    '"confidence": <0-1>, "reasoning": "<2 sentences>"}'
)

SYS_ML_LIT_PROP = (
    "You are an inorganic materials expert. Given a target material with "
    "its ML prediction (from crystal structure features) and literature performance data "
    "from similar materials, compare and reconcile the ML prediction with real-world literature "
    "evidence, then make a refined prediction. Output JSON: {\"value\": <number>, "
    "\"confidence\": <0-1>, \"reasoning\": \"<2 sentences>\"}"
)


def _call_deepseek(system_prompt, user_prompt):
    """Low-level API call with retry (exponential backoff on failure; returns None on failure, handled explicitly by the caller)"""
    if not DS_API_KEY:
        print("   [LLM] DS_API_KEY not configured; LLM fallback disabled.", flush=True)
        return None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            r = requests.post(
                f"{DEEPSEEK_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                },
                timeout=LLM_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            print(
                f"   [LLM] HTTP {r.status_code} (attempt {attempt + 1}/{LLM_MAX_RETRIES})",
                flush=True,
            )
        except Exception as e:
            print(
                f"   [LLM] request error: {type(e).__name__}: {e} "
                f"(attempt {attempt + 1}/{LLM_MAX_RETRIES})",
                flush=True,
            )
        # Exponential backoff: 1s -> 2s -> 4s ... (capped at 8s), to avoid rate-limit/timeout storms amplifying missing values
        if attempt < LLM_MAX_RETRIES - 1:
            time.sleep(min(2 ** attempt, 8))
    return None


def _in_range(value, value_range):
    """Physical range check: returns (ok, reason)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, "non_numeric"
    if value_range is None:
        return True, ""
    lo, hi = value_range
    if v < lo or v > hi:
        return False, f"out_of_range({v} not in [{lo}, {hi}])"
    return True, ""


def llm_json(system_prompt, user_prompt, default_value, value_range=None):
    """Call the LLM and parse the JSON result.

    The returned dict always contains a status field (backward compatible: old
    callers can ignore the new fields):
      - "ok":           parsing succeeded and (if value_range is provided) passed the physical range check
      - "out_of_range": a numeric value was parsed but falls outside the physical range (confidence=0.0, not trusted)
      - "error":        API failure / parse failure (value is the default placeholder, confidence=0.0)

    Fix points: F1 (failures no longer silently drop values), F2 (LLM outputs
    whole-step/unreasonable values no longer pass through without validation).
    """
    text = _call_deepseek(system_prompt, user_prompt)
    if not text:
        return {
            "value": float(default_value),
            "confidence": 0.0,
            "status": "error",
            "reason": "api_failed_or_empty",
        }
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "value" in obj:
                ok, reason = _in_range(obj.get("value"), value_range)
                if ok:
                    return {
                        "value": float(obj["value"]),
                        "confidence": float(obj.get("confidence", 0.3)),
                        "reasoning": str(obj.get("reasoning", "")),
                        "status": "ok",
                    }
                return {
                    "value": float(obj["value"]),
                    "confidence": 0.0,
                    "status": "out_of_range",
                    "reason": reason,
                    "raw": text[:200],
                }
        except Exception:
            pass
    # Try extracting a number
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        ok, reason = _in_range(nums[0], value_range)
        if ok:
            return {"value": float(nums[0]), "confidence": 0.3, "status": "ok"}
        return {
            "value": float(nums[0]),
            "confidence": 0.0,
            "status": "out_of_range",
            "reason": reason,
            "raw": text[:200],
        }
    return {
        "value": float(default_value),
        "confidence": 0.0,
        "status": "error",
        "reason": "parse_failed",
        "raw": text[:200],
    }


# ============================================================
# Prompt Builders - each method
# ============================================================

def build_prompt_zs(formula, prop_name, prop_unit="", prop_desc=None):
    """M4: LLM zero-shot reasoning"""
    desc = prop_desc or prop_name
    return (
        f"Chemical formula: {formula}\n\n"
        f"Predict the {desc}"
        f"{' (' + prop_unit + ')' if prop_unit else ''}"
        f" of this material. "
        f"Return JSON with 'value' field."
    )


def build_prompt_ml(formula, ml_pred, prop_name, prop_unit="", prop_desc=None):
    """M5: ML + LLM enhancement (no literature)"""
    desc = prop_desc or prop_name
    ml_str = f"{ml_pred:.4f}" if ml_pred is not None else "not available"
    parts = [
        f"Target material: {formula}",
        f"ML model prediction (from crystal structure): {ml_str}",
    ]
    parts.append(
        f"\nRefine the {desc}"
        f"{' (' + prop_unit + ')' if prop_unit else ''}"
        f" prediction based on your domain knowledge. Return JSON."
    )
    return "\n".join(parts)


def build_prompt_full(formula, neighbors, prop_name, prop_unit="", prop_desc=None):
    """M6: LLM-Full (pure literature reasoning) - all fields, same neighbor information volume as LLM-ML-Full"""
    desc = prop_desc or prop_name
    parts = [
        f"## Target Material\nFormula: {formula}",
    ]
    parts.append(f"\n## Knowledge Retrieval Data - Real Experimental Data ({len(neighbors)} similar materials)")
    for i, nb in enumerate(neighbors):
        parts.append(f"\n### Neighbor {i + 1}: {nb.get('formula', '?')}")
        parts.append(f"  • Measured {desc} = {nb.get('value', '?'):.2f}")
        parts.append(f"  • Cosine similarity = {nb.get('similarity', 0):.4f}")
        for field, label in [
            ("synthesis_route", "Synthesis route"),
            ("raw_materials", "Raw materials"),
            ("forming_method", "Forming method"),
            ("sintering_temp_C", "Sintering temperature (°C)"),
            ("sintering_time_h", "Sintering time (h)"),
            ("sintering_atmosphere", "Sintering atmosphere"),
            ("sintering_additives", "Sintering additives"),
            ("calcination_temp_C", "Calcination temperature (°C)"),
            ("calcination_time_h", "Calcination time (h)"),
            ("crystal_system", "Crystal system"),
            ("space_group", "Space group"),
            ("cell_volume_A3", "Cell volume (Å³)"),
            ("is_single_phase", "Single phase"),
            ("secondary_phases", "Secondary phases"),
            ("grain_size_um", "Grain size (µm)"),
            ("grain_morphology", "Grain morphology"),
            ("porosity_pct", "Porosity (%)"),
            ("dopants", "Dopants"),
            ("theoretical_density", "Theoretical density (g/cm³)"),
            ("bulk_density", "Bulk density (g/cm³)"),
            ("freq_GHz", "Measurement frequency (GHz)"),
        ]:
            val = nb.get(field)
            if val is not None:
                if field == "is_single_phase":
                    val = "Yes" if val else "No"
                parts.append(f"  • {label}: {val}")
    parts.append(
        f"\nBased on these literature results, estimate the {desc}"
        f"{' (' + prop_unit + ')' if prop_unit else ''}"
        f" of the target material. Return JSON."
    )
    return "\n".join(parts)


def build_prompt_ml_full(formula, ml_pred, neighbors, prop_name, prop_unit="", prop_desc=None):
    """M8: ML + literature fusion (strongest strategy)"""
    desc = prop_desc or prop_name
    ml_str = f"{desc} = {ml_pred:.4f}" if ml_pred is not None else f"{desc} = not available"
    parts = [
        f"## Target Material\nFormula: {formula}",
        f"ML prediction (from crystal structure): {ml_str}",
    ]
    parts.append(f"\n## Knowledge Retrieval Data - Real Experimental Data ({len(neighbors)} similar materials)")
    for i, nb in enumerate(neighbors):
        parts.append(f"\n### Neighbor {i + 1}: {nb.get('formula', '?')}")
        parts.append(f"  • Measured {desc} = {nb.get('value', '?'):.2f}")
        parts.append(f"  • Cosine similarity = {nb.get('similarity', 0):.4f}")
        for field, label in [
            ("synthesis_route", "Synthesis route"),
            ("raw_materials", "Raw materials"),
            ("forming_method", "Forming method"),
            ("sintering_temp_C", "Sintering temperature (°C)"),
            ("sintering_time_h", "Sintering time (h)"),
            ("sintering_atmosphere", "Sintering atmosphere"),
            ("sintering_additives", "Sintering additives"),
            ("calcination_temp_C", "Calcination temperature (°C)"),
            ("calcination_time_h", "Calcination time (h)"),
            ("crystal_system", "Crystal system"),
            ("space_group", "Space group"),
            ("cell_volume_A3", "Cell volume (Å³)"),
            ("is_single_phase", "Single phase"),
            ("secondary_phases", "Secondary phases"),
            ("grain_size_um", "Grain size (µm)"),
            ("grain_morphology", "Grain morphology"),
            ("porosity_pct", "Porosity (%)"),
            ("dopants", "Dopants"),
            ("theoretical_density", "Theoretical density (g/cm³)"),
            ("bulk_density", "Bulk density (g/cm³)"),
            ("freq_GHz", "Measurement frequency (GHz)"),
        ]:
            val = nb.get(field)
            if val is not None:
                if field == "is_single_phase":
                    val = "Yes" if val else "No"
                parts.append(f"  • {label}: {val}")
    parts.append(
        f"\nFuse the ML prediction with the literature evidence above. "
        f"Best estimate of {desc}"
        f"{' (' + prop_unit + ')' if prop_unit else ''}"
        f"? Return JSON."
    )
    return "\n".join(parts)


def build_prompt_ml_prop(formula, ml_pred, neighbors, prop_name, prop_unit="", prop_desc=None):
    """M7: LLM-ML-Prop (ML prediction + knowledge neighbor performance data) - formula and performance values only, no other fields"""
    desc = prop_desc or prop_name
    ml_str = f"{desc} = {ml_pred:.4f}" if ml_pred is not None else f"{desc} = not available"
    parts = [
        f"## Target Material\nFormula: {formula}",
        f"ML prediction (from crystal structure): {ml_str}",
    ]
    parts.append(f"\n## Literature Neighbors - Performance Data Only")
    for i, nb in enumerate(neighbors[:8]):
        parts.append(f"  {i + 1}. {nb['formula']}: {desc}={nb['value']:.2f}")
    parts.append(
        f"\nReconcile the ML prediction with the literature evidence above. "
        f"Best estimate of {desc}"
        f"{' (' + prop_unit + ')' if prop_unit else ''}"
        f"? Return JSON."
    )
    return "\n".join(parts)
