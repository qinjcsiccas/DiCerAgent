import os
import joblib
import pandas as pd
import numpy as np
from plugins.base import BasePredictor, BaseFeatureCalculator, ModelMetadata, ModelType
from plugins.registry import register_plugin, Plugin
import config

metadata = ModelMetadata(
    name="SVR Dielectric Constant Predictor",
    version="1.0.0",
    author="Built-in",
    description="Support Vector Regression model for predicting dielectric constant (er)",
    supported_structures=["general"],
    model_type=ModelType.PYTHON,
    dependencies=["scikit-learn", "joblib"]
)

class ERFeatureCalculator(BaseFeatureCalculator):
    """Dielectric-constant feature calculator - uses basic features, no extra computation needed"""
    
    metadata = metadata
    
    def calculate(self, structure, base_features: dict) -> dict:
        """Directly return the basic features"""
        return base_features
    
    def get_required_base_features(self) -> list:
        return ["Mass", "Va", "P_MLR_pv", "ASD", "Nbr_dist_var", "Bond_abs_dev"]

class ERPredictor(BasePredictor):
    """Dielectric-constant predictor"""
    
    metadata = metadata
    _model = None
    
    def __init__(self):
        if ERPredictor._model is None:
            if os.path.exists(config.PATH_SVR_MODEL):
                ERPredictor._model = joblib.load(config.PATH_SVR_MODEL)
            else:
                print(f"[X] Warning: SVR Model not found at {config.PATH_SVR_MODEL}")
    
    def predict(self, features: dict) -> dict:
        if not ERPredictor._model:
            return {"er_mean": None, "er_dev": None}
        
        feature_cols = ["Mass", "Va", "P_MLR_pv", "ASD", "Nbr_dist_var", "Bond_abs_dev"]
        # Check that all required features are present
        missing = [c for c in feature_cols if c not in features or features.get(c) is None]
        if missing:
            return {"er_mean": None, "er_dev": None}

        try:
            X_df = pd.DataFrame([features])[feature_cols]
            if hasattr(ERPredictor._model, "estimators_"):
                all_preds = [est.predict(X_df.values)[0] for est in ERPredictor._model.estimators_]
                return {"er_mean": float(np.mean(all_preds)), "er_dev": float(np.std(all_preds))}
            else:
                pred = ERPredictor._model.predict(X_df.values)[0]
                return {"er_mean": float(pred), "er_dev": 0.0}
        except Exception as e:
            print(f"      [er_predictor] Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {"er_mean": None, "er_dev": None}
    
    def get_required_features(self) -> list:
        return ["Mass", "Va", "P_MLR_pv", "ASD", "Nbr_dist_var", "Bond_abs_dev"]

from plugins.registry import register_plugin, Plugin

register_plugin(Plugin(
    id="er_predictor",
    structure_type="general",
    feature_calculator=ERFeatureCalculator(),
    predictor=ERPredictor()
))

print("[*] er_predictor plugin loaded")
