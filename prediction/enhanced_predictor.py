"""
Enhanced Predictor - Eight-Method Adaptive Routing
===============================
EnhancedPredictor: unified prediction entry point that automatically routes to the
best method combination based on available data.

Input: chemical formula + optional structure data + optional user free text + existing ML predictions
Output: multi-method annotated prediction results (with confidence, reasoning basis, knowledge neighbor summary)

Routing logic:
  Has structure -> ML main output + optional LLM enhancement
    - Has literature -> LLM-ML-Full (ML + literature fusion; on failure falls back to LLM-ML)
    - No literature -> LLM-ML (ML + LLM domain knowledge correction)
  No structure -> literature/LLM degradation
    - Has literature -> KNN + LLM-Full
    - No literature -> LLM-ZS
"""

import os
import sys
import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_current_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from prediction.config import (
    ELEMENTS, PROPERTY_CONFIG, K_NEIGHBORS,
    MIN_KR_NEIGHBORS, MIN_NEIGHBOR_SIMILARITY,
)
from enhanced_shared.knowledge_retrieval import load_kr_index, KnowledgeRetriever, formula_to_vector
from enhanced_shared.llm_client import (
    llm_json,
    SYS_ZS, SYS_ML_ADJUST, SYS_LIT_FULL, SYS_ML_LIT_FULL, SYS_ML_LIT_PROP,
    build_prompt_zs, build_prompt_ml, build_prompt_full,
    build_prompt_ml_full, build_prompt_ml_prop,
)
from plugins.base import get_property_info


def _safe_float(v):
    """Safely convert to float; None / NaN / invalid strings return None."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(fv):
        return None
    return fv


class EnhancedPredictor:
    """Multi-method adaptive-routing predictor."""

    def __init__(self, config=None):
        self._kr_loaded = False
        self._vectors = None
        self._prop_values = None
        self._meta = None
        self._retrievers = {}     # prop_name -> KnowledgeRetriever

    # ================================================================
    # Main entry
    # ================================================================

    def predict(self, formula, target_properties, user_context=None,
                has_structure=False, ml_predictions=None,
                enabled_methods=None):
        """Main prediction entry.

        Parameters
        ----------
        formula : str
            Normalized chemical formula
        target_properties : list[str]
            Target properties, e.g. ['er', 'qf', 'tcf']
        user_context : str or None
            User-provided free text (processing, microstructure, etc.)
        has_structure : bool
            Whether crystal structure data is already available (MP/COD/local file, any source)
        ml_predictions : dict or None
            ML prediction values already computed by plugins, e.g. {'er': 12.34, 'qf': 50000}
        enabled_methods : list[str] or None
            User-selected method list, e.g. ['ML', 'LLM-ML-Full'].
            None uses automatic routing; a non-empty list runs only the explicitly
            checked methods, silently skipping methods with insufficient data.

        Returns
        -------
        dict : {'results': [...], 'route_info': {...}}
        """
        if ml_predictions is None:
            ml_predictions = {}

        self._ensure_kr_loaded()

        # ================================================================
        # Manual mode: user explicitly checked methods -> schedule on demand, silently skip if data is insufficient
        # ================================================================
        if enabled_methods:
            return self._predict_manual(
                formula, target_properties, user_context,
                has_structure, ml_predictions, enabled_methods,
            )

        # ================================================================
        # Automatic mode: eight-method adaptive routing (current default behavior)
        # ================================================================
        all_results = []
        neighbors_by_prop = {}

        for prop_name in target_properties:
            _resolved_key, prop_cfg = self._resolve_prop_config(prop_name)
            prop_unit = prop_cfg.get("unit", "")
            default_val = prop_cfg.get("default_pred", 0)
            kr_field = prop_cfg.get("kr_field")

            # Knowledge retrieval: only take the literature path when kr_field is configured
            has_lit = False
            neighbors = []
            has_kr = False
            if kr_field is not None:
                neighbors = self._retrieve_lit(formula, _resolved_key)
                has_kr = self._check_knowledge_quality(neighbors)
            neighbors_by_prop[prop_name] = neighbors[:K_NEIGHBORS]

            ml_val = ml_predictions.get(prop_name)

            if has_structure:
                # ------ Has structure ------
                if ml_val is not None:
                    all_results.append({
                        "target": prop_name, "method": "ML",
                        "value": ml_val, "unit": prop_unit,
                        "confidence": "high",
                    })

                if ml_val is not None:
                    # ML has a prediction -> use methods with ML
                    if has_kr:
                        llm_res = self._call_llm_for_method(
                            "LLM-ML-Full", formula, prop_name, prop_unit,
                            ml_pred=ml_val, neighbors=neighbors[:K_NEIGHBORS],
                            user_context=user_context, default_val=default_val,
                        )
                        if llm_res:
                            all_results.append({
                                "target": prop_name, "method": "LLM-ML-Full",
                                "value": llm_res["value"], "unit": prop_unit,
                                "confidence": self._conf_label(llm_res.get("confidence", 0)),
                                "reasoning": llm_res.get("reasoning", ""),
                                "neighbor_summary": self._summarize_neighbors(neighbors[:3]),
                            })
                    else:
                        llm_res = self._call_llm_for_method(
                            "LLM-ML", formula, prop_name, prop_unit,
                            ml_pred=ml_val, user_context=user_context,
                            default_val=default_val,
                        )
                        if llm_res:
                            all_results.append({
                                "target": prop_name, "method": "LLM-ML",
                                "value": llm_res["value"], "unit": prop_unit,
                                "confidence": self._conf_label(llm_res.get("confidence", 0)),
                                "reasoning": llm_res.get("reasoning", ""),
                            })
                else:
                    # ML has no prediction -> forbid method names with the ML suffix
                    if has_kr:
                        llm_res = self._call_llm_for_method(
                            "LLM-Full", formula, prop_name, prop_unit,
                            neighbors=neighbors[:K_NEIGHBORS],
                            user_context=user_context, default_val=default_val,
                        )
                        if llm_res:
                            all_results.append({
                                "target": prop_name, "method": "LLM-Full",
                                "value": llm_res["value"], "unit": prop_unit,
                                "confidence": self._conf_label(llm_res.get("confidence", 0)),
                                "reasoning": llm_res.get("reasoning", ""),
                                "neighbor_summary": self._summarize_neighbors(neighbors[:3]),
                            })
                    else:
                        llm_res = self._call_llm_for_method(
                            "LLM-ZS", formula, prop_name, prop_unit,
                            user_context=user_context, default_val=default_val,
                        )
                        if llm_res:
                            all_results.append({
                                "target": prop_name, "method": "LLM-ZS",
                                "value": llm_res["value"], "unit": prop_unit,
                                "confidence": self._conf_label(llm_res.get("confidence", 0)),
                                "reasoning": llm_res.get("reasoning", ""),
                            })
            else:
                # ------ No structure data ------
                if has_kr:
                    llm_res = self._call_llm_for_method(
                        "LLM-Full", formula, prop_name, prop_unit,
                        neighbors=neighbors[:K_NEIGHBORS],
                        user_context=user_context, default_val=default_val,
                    )
                    if llm_res:
                        all_results.append({
                            "target": prop_name, "method": "LLM-Full",
                            "value": llm_res["value"], "unit": prop_unit,
                            "confidence": self._conf_label(llm_res.get("confidence", 0)),
                            "reasoning": llm_res.get("reasoning", ""),
                            "neighbor_summary": self._summarize_neighbors(neighbors[:3]),
                        })
                else:
                    # Last resort: Zero-Shot (reached only when there is no structure and no literature)
                    llm_res = self._call_llm_for_method(
                        "LLM-ZS", formula, prop_name, prop_unit,
                        user_context=user_context, default_val=default_val,
                    )
                    if llm_res:
                        all_results.append({
                            "target": prop_name, "method": "LLM-ZS",
                            "value": llm_res["value"], "unit": prop_unit,
                            "confidence": self._conf_label(llm_res.get("confidence", 0)),
                            "reasoning": llm_res.get("reasoning", ""),
                        })

        route_info = {
            "has_structure": has_structure,
            "neighbor_counts": {
                p: len(neighbors_by_prop.get(p, [])) for p in target_properties
            },
            "has_literature": {
                p: self._check_knowledge_quality(neighbors_by_prop.get(p, []))
                for p in target_properties
            },
        }

        return {"results": all_results, "route_info": route_info}

    # ================================================================
    # Internal methods
    # ================================================================

    def _predict_manual(self, formula, target_properties, user_context,
                        has_structure, ml_predictions, enabled_methods):
        """Manual mode: run only user-checked methods, silently skip when data is insufficient."""
        all_results = []
        neighbors_by_prop = {}

        # Data availability labels
        AVAIL = {
            "ML":          lambda p: ml_predictions.get(p) is not None,
            "KNN":         lambda p: self._check_knowledge_quality(
                neighbors_by_prop.get(p, [])),
            "LLM-ZS":      lambda p: True,  # no dependencies
            "LLM-ML":      lambda p: ml_predictions.get(p) is not None,
            "LLM-Full":    lambda p: self._check_knowledge_quality(
                neighbors_by_prop.get(p, [])),
            "LLM-ML-Full": lambda p: (
                ml_predictions.get(p) is not None
                and self._check_knowledge_quality(neighbors_by_prop.get(p, []))
            ),
            "LLM-ML-Prop": lambda p: (
                ml_predictions.get(p) is not None
                and self._check_knowledge_quality(neighbors_by_prop.get(p, []))
            ),
            "ML-KNN":      lambda p: (
                ml_predictions.get(p) is not None
                and self._check_knowledge_quality(neighbors_by_prop.get(p, []))
            ),
        }

        for prop_name in target_properties:
            _resolved_key, prop_cfg = self._resolve_prop_config(prop_name)
            prop_unit = prop_cfg.get("unit", "")
            default_val = prop_cfg.get("default_pred", 0)
            kr_field = prop_cfg.get("kr_field")

            # Knowledge retrieval: regardless of whether the user selects literature methods, prepare here first
            neighbors = []
            if kr_field is not None:
                neighbors = self._retrieve_lit(formula, _resolved_key)
            neighbors_by_prop[prop_name] = neighbors[:K_NEIGHBORS]

            ml_val = ml_predictions.get(prop_name)

            for method in enabled_methods:
                if method not in AVAIL:
                    continue
                if not AVAIL[method](prop_name):
                    continue  # insufficient data, silently skip

                # ----- Pure ML -----
                if method == "ML":
                    all_results.append({
                        "target": prop_name, "method": "ML",
                        "value": ml_val, "unit": prop_unit,
                        "confidence": "high",
                    })

                # ----- Pure KNN -----
                elif method == "KNN":
                    knn_val = self._knn_predict(
                        neighbors[:K_NEIGHBORS], prop_name, prop_cfg)
                    if knn_val is not None:
                        all_results.append({
                            "target": prop_name, "method": "KNN",
                            "value": knn_val, "unit": prop_unit,
                            "confidence": "medium",
                            "neighbor_summary": self._summarize_neighbors(
                                neighbors[:3]),
                        })

                # ----- ML-KNN fusion -----
                elif method == "ML-KNN":
                    knn_val = self._knn_predict(
                        neighbors[:K_NEIGHBORS], prop_name, prop_cfg)
                    if ml_val is not None and knn_val is not None:
                        fused = (float(ml_val) + float(knn_val)) / 2.0
                        all_results.append({
                            "target": prop_name, "method": "ML-KNN",
                            "value": fused, "unit": prop_unit,
                            "confidence": "medium",
                            "neighbor_summary": self._summarize_neighbors(
                                neighbors[:3]),
                        })

                # ----- LLM methods -----
                elif method.startswith("LLM"):
                    llm_res = self._call_llm_for_method(
                        method, formula, prop_name, prop_unit,
                        ml_pred=ml_val,
                        neighbors=neighbors[:K_NEIGHBORS] if "Full" in method or "Prop" in method else None,
                        user_context=user_context,
                        default_val=default_val,
                    )
                    if llm_res:
                        entry = {
                            "target": prop_name,
                            "method": method,
                            "value": llm_res["value"],
                            "unit": prop_unit,
                            "confidence": self._conf_label(
                                llm_res.get("confidence", 0)),
                            "reasoning": llm_res.get("reasoning", ""),
                        }
                        if "Full" in method or "Prop" in method:
                            entry["neighbor_summary"] = self._summarize_neighbors(
                                neighbors[:3])
                        all_results.append(entry)

        route_info = {
            "has_structure": has_structure,
            "manual_methods": enabled_methods,
            "neighbor_counts": {
                p: len(neighbors_by_prop.get(p, [])) for p in target_properties
            },
            "has_literature": {
                p: self._check_knowledge_quality(neighbors_by_prop.get(p, []))
                for p in target_properties
            },
        }

        return {"results": all_results, "route_info": route_info}

    def _ensure_kr_loaded(self):
        """Lazily load the literature index."""
        if self._kr_loaded:
            return
        print("   [Lit] Loading literature index...", flush=True)
        vectors, prop_values, meta = load_kr_index()
        self._vectors = vectors
        self._prop_values = prop_values
        self._meta = meta
        self._retrievers = {}  # created on demand
        self._kr_loaded = True

    def _get_retriever(self, prop_name):
        """Get or create the retriever for the specified property."""
        if prop_name not in self._retrievers:
            self._retrievers[prop_name] = KnowledgeRetriever(
                self._vectors, self._meta, self._prop_values,
                target_property=prop_name,
            )
        return self._retrievers[prop_name]

    def _resolve_prop_config(self, prop_name):
        """Resolve property name -> (resolved_key, config). Compatible with tcf/tcf_abo3 aliases."""
        if prop_name in PROPERTY_CONFIG:
            return prop_name, PROPERTY_CONFIG[prop_name]
        for key in sorted(PROPERTY_CONFIG.keys(),
                          key=lambda k: len(k), reverse=True):
            if key.startswith(prop_name):
                return key, PROPERTY_CONFIG[key]
        return prop_name, {"default_pred": 0.0, "unit": ""}

    def _retrieve_lit(self, formula, prop_name):
        """Retrieve knowledge neighbors."""
        try:
            retriever = self._get_retriever(prop_name)
            return retriever.retrieve(formula, k=K_NEIGHBORS * 3)
        except Exception as e:
            print(f"   [!] Lit retrieval failed for {prop_name}: {e}")
            return []

    def _check_knowledge_quality(self, neighbors):
        """Distinguish KR quality from availability (F5): triple threshold of count + similarity + valid numeric neighbors.

        Literature data is considered usable (can support KNN / LLM-Full reasoning)
        only when there are >= MIN_KR_NEIGHBORS neighbors with similarity>0 and a
        usable value, and the maximum similarity meets the threshold.
        """
        if len(neighbors) < MIN_KR_NEIGHBORS:
            return False
        valid = [
            n for n in neighbors
            if n.get("similarity", 0) > 0 and _safe_float(n.get("value")) is not None
        ]
        if len(valid) < MIN_KR_NEIGHBORS:
            return False
        max_sim = max((n.get("similarity", 0) for n in valid), default=0)
        return max_sim >= MIN_NEIGHBOR_SIMILARITY

    def _knn_predict(self, neighbors, prop_name, prop_cfg):
        """Cosine-similarity weighted KNN. Filters NaN / None values."""
        if not neighbors:
            return None
        default_val = prop_cfg.get("default_pred", 0)
        weights = []
        values = []
        for n in neighbors:
            sim = n.get("similarity", 0)
            raw_val = n.get("value")
            if sim <= 0 or raw_val is None:
                continue
            try:
                v = float(raw_val)
                if np.isnan(v):
                    continue
            except (ValueError, TypeError):
                continue
            weights.append(sim)
            values.append(v)
        if not weights:
            return None  # all neighbors are NaN, do not predict
        weights = np.array(weights)
        values = np.array(values)
        return float(np.average(values, weights=weights))

    def _call_llm_for_method(self, method, formula, prop_name, prop_unit,
                             ml_pred=None, neighbors=None, user_context=None,
                             default_val=0.0):
        """Unified LLM call entry."""
        method_map = {
            "LLM-ZS":       (SYS_ZS, build_prompt_zs),
            "LLM-ML":       (SYS_ML_ADJUST, build_prompt_ml),
            "LLM-Full":     (SYS_LIT_FULL, build_prompt_full),
            "LLM-ML-Full":  (SYS_ML_LIT_FULL, build_prompt_ml_full),
            "LLM-ML-Prop":  (SYS_ML_LIT_PROP, build_prompt_ml_prop),
        }
        if method not in method_map:
            return None

        # Resolve property description (e.g. "Melting Temperature" / "Dielectric Constant")
        prop_info = get_property_info(prop_name)
        prop_desc = prop_info.name if prop_info.name != prop_name else prop_name

        sys_prompt, builder = method_map[method]

        # Build the user prompt
        if method == "LLM-ZS":
            user_prompt = builder(formula, prop_name, prop_unit, prop_desc=prop_desc)
        elif method == "LLM-ML":
            user_prompt = builder(formula, ml_pred, prop_name, prop_unit, prop_desc=prop_desc)
        elif method in ("LLM-Full", "LLM-ML-Full", "LLM-ML-Prop"):
            if method == "LLM-Full":
                user_prompt = builder(formula, neighbors or [], prop_name, prop_unit, prop_desc=prop_desc)
            elif method == "LLM-ML-Prop":
                user_prompt = builder(formula, ml_pred, neighbors or [], prop_name, prop_unit, prop_desc=prop_desc)
            else:
                user_prompt = builder(formula, ml_pred, neighbors or [], prop_name, prop_unit, prop_desc=prop_desc)
        else:
            return None

        # Inject user context
        if user_context:
            user_prompt += f"\n\nAdditional context from user: {user_context}"

        # F2: output physical range validation - validate LLM-returned values against the property range
        _rk, _cfg = self._resolve_prop_config(prop_name)
        value_range = _cfg.get("range")

        try:
            res = llm_json(sys_prompt, user_prompt, default_val,
                           value_range=value_range)
        except Exception as e:
            print(f"   [!] LLM {method} failed: {e}")
            return None

        # F1/F3: explicit failure flag + confidence filtering - default placeholder fake values (confidence=0)
        # are never returned to the upper layer, to avoid being written to CSV as valid predictions
        status = res.get("status", "ok")
        conf = res.get("confidence", 0.0)
        if status in ("error", "out_of_range") or conf is None or float(conf) <= 0:
            print(
                f"   [!] LLM {method} result rejected "
                f"(status={status}, confidence={conf}, value={res.get('value')})",
                flush=True,
            )
            return None
        return res

    def _conf_label(self, score):
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        return "low"

    def _summarize_neighbors(self, neighbors):
        return [
            {
                "formula": n.get("formula", "?"),
                "value": n.get("value"),
                "similarity": round(n.get("similarity", 0), 4),
                "crystal_system": n.get("crystal_system"),
                "year": n.get("year"),
            }
            for n in neighbors[:3]
        ]


# ================================================================
# Condition detectors (for agent use)
# ================================================================

def detect_has_ml(cif_path=None, feature_cols=None):
    if cif_path is None:
        return False, "No CIF file path provided"
    if not os.path.exists(cif_path):
        return False, f"CIF file not found: {cif_path}"
    return True, "CIF file available"


def detect_has_knowledge(formula, kr_retriever, min_neighbors=MIN_KR_NEIGHBORS):
    try:
        neighbors = kr_retriever.retrieve(formula, k=min_neighbors)
        if len(neighbors) < min_neighbors:
            return False, f"Insufficient neighbors ({len(neighbors)} < {min_neighbors})", 0
        return True, "Sufficient literature data", len(neighbors)
    except Exception as e:
        return False, f"Retrieval error: {e}", 0
