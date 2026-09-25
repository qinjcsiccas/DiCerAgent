import os
import joblib
import pandas as pd
from plugins.base import BasePredictor, BaseFeatureCalculator, ModelMetadata, ModelType
import config

metadata = ModelMetadata(
    name="ABO3 TCF Predictor (XGBoost)",
    version="1.0.0",
    author="Built-in",
    description="XGBoost model for predicting ABO3 temperature coefficient of frequency (TCF)",
    supported_structures=["ABO3"],
    model_type=ModelType.PYTHON,
    dependencies=["xgboost", "joblib"]
)

class ABO3TCFFeatureCalculator(BaseFeatureCalculator):
    """ABO3 TCF feature calculator"""
    
    metadata = metadata
    
    def calculate(self, structure, base_features: dict) -> dict:
        from features import FeatureCalculator
        from data_loader import DataLoader
        
        loader = DataLoader()
        calculator = FeatureCalculator(loader)
        
        z_factor = structure.composition.num_atoms / 5.0
        m = structure.composition.weight / z_factor
        pm = base_features.get("P_MLR_pv", 0) * (structure.volume / z_factor)
        
        valences = calculator.get_valences_ordered(structure)
        
        cation_bond_lengths = []
        total_vi_cell = 0.0
        
        for i, site in enumerate(structure):
            el = site.specie.symbol
            val = int(round(valences[i]))
            
            radius = 1.40 if el == 'O' else 0.0
            if el != 'O':
                from pymatgen.core import Species
                r_12 = calculator.get_specific_shannon_radius_ordered(el, val, "XII")
                r_6 = calculator.get_specific_shannon_radius_ordered(el, val, "VI")
                if r_12 > 0.90:
                    radius = r_12
                elif r_6 > 0:
                    radius = r_6
                else:
                    try:
                        radius = Species(el, val).average_ionic_radius
                    except:
                        radius = 0.6
            total_vi_cell += (4/3) * 3.14159 * (radius ** 3)
            
            if el != "O":
                nn_info = calculator.cnn.get_nn_info(structure, i)
                dists = [site.distance(structure[n['site_index']], jimage=n['image']) 
                         for n in nn_info if structure[n['site_index']].specie.symbol == 'O']
                if dists:
                    cation_bond_lengths.append(sum(dists) / len(dists))
        
        Vi = total_vi_cell / z_factor
        
        if len(cation_bond_lengths) >= 2:
            cation_bond_lengths.sort(reverse=True)
            mid = len(cation_bond_lengths) // 2
            tt = sum(cation_bond_lengths[:mid]) / (1.414 * sum(cation_bond_lengths[mid:])) if sum(cation_bond_lengths[mid:]) > 0 else 1.0
        else:
            tt = 1.0
        
        return {"m": m, "Vi": Vi, "tt": tt, "pm": pm}
    
    def get_required_base_features(self) -> list:
        return ["P_MLR_pv"]

class ABO3TCFPredictor(BasePredictor):
    """ABO3 TCF predictor - XGBoost"""
    
    metadata = metadata
    _model = None
    
    def __init__(self):
        if ABO3TCFPredictor._model is None:
            if os.path.exists(config.PATH_ABO3_TCF_MODEL):
                try:
                    ABO3TCFPredictor._model = joblib.load(config.PATH_ABO3_TCF_MODEL)
                except Exception as e:
                    print(f"[X] Error loading XGBoost model: {e}")
            else:
                print(f"[X] Warning: ABO3 Model not found at {config.PATH_ABO3_TCF_MODEL}")
    
    def predict(self, features: dict) -> dict:
        if not ABO3TCFPredictor._model:
            return {"tcf_mean": None, "tcf_dev": None}
        
        feature_cols = ["m", "Vi", "tt", "pm"]
        
        try:
            X_df = pd.DataFrame([features])[feature_cols]
            pred = ABO3TCFPredictor._model.predict(X_df)
            # Estimate the standard deviation: use 10% of the absolute predicted value as the uncertainty
            estimated_dev = abs(float(pred[0])) * 0.1 if pred[0] != 0 else 5.0
            return {"tcf_mean": float(pred[0]), "tcf_dev": estimated_dev}
        except Exception as e:
            print(f"[X] ABO3 TCF Prediction Error: {e}")
            return {"tcf_mean": None, "tcf_dev": None}
    
    def get_required_features(self) -> list:
        return ["m", "Vi", "tt", "pm"]

from plugins.registry import register_plugin, Plugin

register_plugin(Plugin(
    id="abo3_tcf_predictor",
    structure_type="ABO3",
    feature_calculator=ABO3TCFFeatureCalculator(),
    predictor=ABO3TCFPredictor()
))
