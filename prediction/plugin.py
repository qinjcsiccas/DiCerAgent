"""
Enhanced Prediction Plugin - DiCerAgent Plugin protocol adapter
==========================================
Wraps EnhancedPredictor as a standard DiCerAgent plugin,
registers it into global_registry, coexisting with existing ML plugins.

Package path: prediction.plugin
"""

from typing import Dict, Any, List
import os

from prediction.enhanced_predictor import EnhancedPredictor


class EnhancedFeatureCalculator:
    """Feature calculator adapter: CIF path -> unified feature dict."""

    def __init__(self, enhanced_predictor: EnhancedPredictor):
        self._predictor = enhanced_predictor

    def calculate(self, structure, base_features: Dict[str, float]) -> Dict[str, float]:
        """Compute enhanced features from a CIF structure.

        Extract general structure-derived features from the pymatgen Structure
        (formula, lattice parameters, space group, density, volume, etc.), merge
        them with the base features and return; returns {} when structure is None
        (keeping compatibility with old callers). F4: original TODO empty
        implementation -> real structure feature extraction.
        """
        if structure is None:
            return {}
        out: Dict[str, float] = {}
        try:
            comp = structure.composition
            out["formula"] = comp.reduced_formula
            out["Formula"] = comp.reduced_formula
            out["num_atoms"] = float(comp.num_atoms)
            out["weight"] = float(comp.weight)
            lattice = structure.lattice
            out["lattice_a"] = float(lattice.a)
            out["lattice_b"] = float(lattice.b)
            out["lattice_c"] = float(lattice.c)
            out["lattice_alpha"] = float(lattice.alpha)
            out["lattice_beta"] = float(lattice.beta)
            out["lattice_gamma"] = float(lattice.gamma)
            out["volume"] = float(lattice.volume)
            out["density"] = float(structure.density)
            try:
                sg = structure.get_space_group_info()
                out["space_group"] = sg[0] if sg else "Unknown"
            except Exception:
                out["space_group"] = "Unknown"
        except Exception as e:
            print(
                f"   [!] EnhancedFeatureCalculator.calculate failed: {e}",
                flush=True,
            )
            return {}
        merged = dict(base_features or {})
        merged.update(out)
        return merged


class EnhancedPredictorAdapter:
    """Prediction adapter: wraps EnhancedPredictor into the Plugin Predictor protocol."""

    def __init__(self, enhanced_predictor: EnhancedPredictor):
        self._predictor = enhanced_predictor
        self._user_context = None
        self._cif_path = None

    def set_context(self, user_context: str = None, cif_path: str = None):
        """Set the user context and CIF path."""
        self._user_context = user_context
        self._cif_path = cif_path

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Run enhanced prediction.

        Extract formula and structure availability from features, call
        EnhancedPredictor.predict(), and return a structure consumable by
        analyze_single:
          {"enhanced_results": [...], "route_info": {...}}
        Returns {} when formula is missing or the call fails (keeping
        compatibility with old callers).
        F4: original TODO empty implementation -> real call to the main-chain
        enhanced predictor.
        """
        if not features:
            return {}
        formula = features.get("formula") or features.get("Formula")
        if not formula:
            return {}

        # Structure availability: explicit cif_path (exists locally) or already
        # containing structure-derived features counts as having structure
        has_structure = False
        cif_path = self._cif_path or features.get("cif_path")
        if cif_path and os.path.exists(str(cif_path)):
            has_structure = True
        if not has_structure and any(
            k in features
            for k in ("lattice_a", "lattice_b", "lattice_c", "volume",
                      "space_group", "density")
        ):
            has_structure = True

        target_props = features.get("target_properties")
        if not target_props:
            target_props = ["er", "qf", "tcf", "tm"]
        ml_predictions = {}
        for tp in target_props:
            if tp in features:
                ml_predictions[tp] = features[tp]

        try:
            out = self._predictor.predict(
                formula=str(formula),
                target_properties=list(target_props),
                user_context=self._user_context,
                has_structure=has_structure,
                ml_predictions=ml_predictions or None,
                enabled_methods=None,
            )
            if out:
                return {
                    "enhanced_results": out.get("results", []),
                    "route_info": out.get("route_info", {}),
                }
        except Exception as e:
            print(
                f"   [!] EnhancedPredictorAdapter.predict failed: {e}",
                flush=True,
            )
        return {}

    def get_required_features(self) -> List[str]:
        """Required input features."""
        return ["formula", "cif_path"]


# ================================================================
# Plugin registration
# ================================================================

def create_plugin():
    """Create an EnhancedPredictionPlugin instance and register it into global_registry.

    Returns
    -------
    Plugin
    """
    from plugins.base import Plugin, ModelMetadata, ModelType
    from plugins.registry import global_registry

    predictor = EnhancedPredictor()

    plugin = Plugin(
        id="enhanced_prediction",
        structure_type="general",
        feature_calculator=EnhancedFeatureCalculator(predictor),
        predictor=EnhancedPredictorAdapter(predictor),
        metadata=ModelMetadata(
            name="Enhanced Prediction (8-Method Router)",
            version="2.0",
            author="DiCerAgent",
            description=(
                "Eight-method adaptive-routing prediction: ML + knowledge retrieval + LLM fusion."
                "Supports free-text context injection (processing/microstructure/dopants etc.)."
                "Coexists with the existing single-method ML plugin, providing enhanced inference results."
            ),
            supported_structures=["ABO3", "ABO4", "General"],
            model_type=ModelType.PYTHON,
            scope="enhanced",
        ),
        enabled=True,
        scope="enhanced",
    )

    global_registry.register(plugin)
    return plugin
